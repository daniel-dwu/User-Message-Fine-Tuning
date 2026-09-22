"""LLM judges with typed output.

Three shapes, chosen by how the judge prompt was written:

* ``StructuredJudge`` forces a tool/function call against a JSON schema, so
  labels come back typed rather than parsed out of prose. Used where the
  rubric is a fixed enum (belief buckets, apple/orange, Betley scores).
* ``JsonJudge`` sends a hand-written prompt template that asks for a JSON
  object. The prompt file is the single source of truth, so it can be edited
  without touching code. Used by the degradation rubric.
* ``TextJudge`` returns free text, for grading prompts that answer inside
  ``<answer>`` tags (the believe-it-or-not belief evals), and doubles as the
  adversary in the multi-turn debate eval.

Provider is chosen from the model name (``gpt-*``/``o*`` → OpenAI, otherwise
Anthropic), so one vendor's billing outage does not stop an experiment.

Failures return ``None`` after retries. Callers must DROP those completions and
count them, never score them as clean — a judge outage must not look like an
improvement.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any

logger = logging.getLogger(__name__)

_OPENAI_PREFIXES = ("gpt", "o1", "o3", "o4", "chatgpt")
# Reasoning-capable OpenAI models reject function tools unless
# reasoning_effort is explicitly "none"; older models reject the parameter
# itself, so it is sent only for these families and dropped if refused.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_openai_model(model: str) -> bool:
    return model.lower().startswith(_OPENAI_PREFIXES)


def _needs_reasoning_effort_none(model: str) -> bool:
    return model.lower().startswith(_REASONING_PREFIXES)


def _backoff(attempt: int) -> float:
    return 2**attempt + random.uniform(0, 1)


class StructuredJudge:
    """Forced-tool judge returning a dict matching ``properties``."""

    def __init__(
        self,
        model: str,
        system: str,
        properties: dict[str, Any],
        tool_name: str,
        tool_description: str,
        concurrency: int = 20,
        max_retries: int = 4,
        max_tokens: int = 600,
        temperature: float | None = None,
    ):
        self.model = model
        self.system = system
        self.properties = properties
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        # None keeps the provider default; classification judges pass 0 so
        # borderline cases do not flip between runs.
        self.temperature = temperature
        self.n_failures = 0
        self._sem = asyncio.Semaphore(concurrency)
        self._openai = is_openai_model(model)
        self._reasoning_effort_none = _needs_reasoning_effort_none(model)
        if self._openai:
            from openai import AsyncOpenAI

            self._client: Any = AsyncOpenAI()
        else:
            import anthropic

            self._client = anthropic.AsyncAnthropic()

    async def _call_openai(self, user_message: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": user_message},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": self.tool_name,
                        "description": self.tool_description,
                        "parameters": {
                            "type": "object",
                            "properties": self.properties,
                            "required": list(self.properties),
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": self.tool_name}},
        )
        if self._reasoning_effort_none:
            kwargs["reasoning_effort"] = "none"
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - parameter negotiation, then re-raise
            dropped = False
            if "reasoning_effort" in str(e) and self._reasoning_effort_none:
                self._reasoning_effort_none = False
                kwargs.pop("reasoning_effort", None)
                dropped = True
            if "temperature" in str(e) and self.temperature is not None:
                self.temperature = None
                kwargs.pop("temperature", None)
                dropped = True
            if not dropped:
                raise
            resp = await self._client.chat.completions.create(**kwargs)
        calls = resp.choices[0].message.tool_calls
        if not calls:
            raise ValueError("judge returned no tool call")
        return json.loads(calls[0].function.arguments)

    async def _call_anthropic(self, user_message: str) -> dict[str, Any]:
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=[
                {
                    "name": self.tool_name,
                    "description": self.tool_description,
                    "input_schema": {
                        "type": "object",
                        "properties": self.properties,
                        "required": list(self.properties),
                    },
                }
            ],
            tool_choice={"type": "tool", "name": self.tool_name},
            messages=[{"role": "user", "content": user_message}],
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise ValueError("judge response missing tool_use block")

    async def label(self, user_message: str) -> dict[str, Any] | None:
        async with self._sem:
            for attempt in range(self.max_retries):
                try:
                    if self._openai:
                        return await self._call_openai(user_message)
                    return await self._call_anthropic(user_message)
                except Exception as e:  # noqa: BLE001 - degrade, count, continue
                    if attempt == self.max_retries - 1:
                        self.n_failures += 1
                        logger.warning("judge failed, dropping completion: %s", e)
                        return None
                    await asyncio.sleep(_backoff(attempt))
        return None


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object out of a judge reply.

    Judges wrap JSON in fences, prepend prose, or copy a trailing comma out of
    the prompt's example block often enough that bare ``json.loads`` is not
    reliable. Strict parse first, then the outermost ``{...}`` span, then a
    trailing-comma repair.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("`")
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    span = s[start : end + 1]
    try:
        return json.loads(span)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", span))


class JsonJudge:
    """Judge driven by a hand-written prompt that asks for a JSON object."""

    def __init__(
        self,
        model: str,
        concurrency: int = 20,
        max_retries: int = 4,
        max_tokens: int = 800,
    ):
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.n_failures = 0
        self._sem = asyncio.Semaphore(concurrency)
        self._openai = is_openai_model(model)
        if self._openai:
            from openai import AsyncOpenAI

            self._client: Any = AsyncOpenAI()
        else:
            import anthropic

            self._client = anthropic.AsyncAnthropic()

    async def _call(self, user_message: str) -> dict[str, Any]:
        if self._openai:
            resp = await self._client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": user_message}],
            )
            return extract_json_object(resp.choices[0].message.content or "")
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": user_message}],
        )
        return extract_json_object("".join(b.text for b in resp.content if b.type == "text"))

    async def label(self, user_message: str) -> dict[str, Any] | None:
        async with self._sem:
            for attempt in range(self.max_retries):
                try:
                    return await self._call(user_message)
                except Exception as e:  # noqa: BLE001 - degrade, count, continue
                    if attempt == self.max_retries - 1:
                        self.n_failures += 1
                        logger.warning("judge failed, dropping completion: %s", e)
                        return None
                    await asyncio.sleep(_backoff(attempt))
        return None


class TextJudge:
    """Free-text judge: ``grade`` for rubric prompts, ``chat`` for an adversary turn.

    Retries transient failures with backoff and re-raises on the last attempt:
    a belief eval writes its output only at the very end, so one dropped grade
    would be visible as a missing verdict, not a silently clean score.
    """

    def __init__(self, model: str, concurrency: int = 16, max_tokens: int = 2000):
        self.model = model
        self.max_tokens = max_tokens
        self._sem = asyncio.Semaphore(concurrency)
        self._openai = is_openai_model(model)
        if self._openai:
            from openai import AsyncOpenAI

            self._client: Any = AsyncOpenAI()
        else:
            import anthropic

            self._client = anthropic.AsyncAnthropic()

    async def _call(self, prompt: str, max_tokens: int) -> str:
        if self._openai:
            resp = await self._client.chat.completions.create(
                model=self.model,
                max_completion_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    async def _with_retries(self, prompt: str, max_tokens: int, attempts: int = 6) -> str:
        async with self._sem:
            for attempt in range(attempts):
                try:
                    return await self._call(prompt, max_tokens)
                except Exception:  # noqa: BLE001 - transient API errors
                    if attempt == attempts - 1:
                        raise
                    await asyncio.sleep(_backoff(attempt))
        raise RuntimeError("unreachable")

    async def grade(self, prompt: str) -> str:
        return await self._with_retries(prompt, self.max_tokens)

    async def chat(self, prompt: str, max_tokens: int = 500) -> str:
        return await self._with_retries(prompt, max_tokens)
