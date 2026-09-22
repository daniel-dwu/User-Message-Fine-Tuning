"""Sampling from a base model or a Tinker checkpoint through the chat template.

Every eval here renders a conversation with the cookbook renderer, samples one
completion, and parses the assistant text back out. This wraps that pattern
with bounded concurrency and per-call ``max_tokens`` / ``temperature``
overrides (the belief evals use different caps per eval). Renderer defaults to
the non-thinking Qwen3 template used throughout.
"""

from __future__ import annotations

import asyncio

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf.chat_format import RENDERER_NAME

Message = renderers.Message


def sampling_client(model_name: str, checkpoint: str | None) -> tinker.SamplingClient:
    """Base model when ``checkpoint`` is None, else the given sampler path."""
    service = tinker.ServiceClient()
    if checkpoint is None:
        return service.create_sampling_client(base_model=model_name)
    return service.create_sampling_client(model_path=checkpoint)


class Sampler:
    """Chat completions with a concurrency cap and per-call overrides."""

    def __init__(
        self,
        model_name: str,
        checkpoint: str | None,
        max_tokens: int,
        temperature: float = 1.0,
        concurrency: int = 16,
        renderer_name: str = RENDERER_NAME,
    ):
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.renderer = renderers.get_renderer(renderer_name, tokenizer=get_tokenizer(model_name))
        self.client = sampling_client(model_name, checkpoint)
        self._sem = asyncio.Semaphore(concurrency)

    async def sample_with_cap(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, bool]:
        """(assistant text, whether generation hit the token cap)."""
        cap = self.max_tokens if max_tokens is None else max_tokens
        prompt = self.renderer.build_generation_prompt(messages)
        params = tinker.SamplingParams(
            max_tokens=cap,
            temperature=self.temperature if temperature is None else temperature,
            stop=self.renderer.get_stop_sequences(),
        )
        async with self._sem:
            result = await self.client.sample_async(prompt, num_samples=1, sampling_params=params)
        tokens = result.sequences[0].tokens
        message, _termination = self.renderer.parse_response(tokens)
        return message["content"], len(tokens) >= cap

    async def sample(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        text, _hit_cap = await self.sample_with_cap(messages, max_tokens, temperature)
        return text

    async def sample_many(self, conversations: list[list[Message]], **kw) -> list[str]:
        return list(await asyncio.gather(*[self.sample(m, **kw) for m in conversations]))


def user_turn(content: str, system: str | None = None) -> list[Message]:
    """The one-turn conversation every first-turn eval sends."""
    messages: list[Message] = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=content))
    return messages


def as_messages(rows: list[dict]) -> list[Message]:
    return [Message(role=r["role"], content=r["content"]) for r in rows]
