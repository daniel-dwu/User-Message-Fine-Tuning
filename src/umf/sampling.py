"""Sampling from a base model or a Tinker checkpoint through the chat template.

A thin wrapper over the cookbook's ``TinkerMessageCompleter`` that adds the two
things every eval here needs: bounded concurrency, and a checkpoint/base-model
switch that resolves ``tinker://`` sampler paths and plain model names the same
way. Renderer defaults to the non-thinking Qwen3 template used throughout.
"""

from __future__ import annotations

import asyncio

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerMessageCompleter
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
    """Chat completions with a concurrency cap.

    ``sample`` returns the assistant text; ``sample_many`` fans out over a list
    of message lists and preserves order.
    """

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
        tokenizer = get_tokenizer(model_name)
        self.renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
        self._completer = TinkerMessageCompleter(
            sampling_client(model_name, checkpoint),
            self.renderer,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._sem = asyncio.Semaphore(concurrency)

    async def sample(self, messages: list[Message]) -> str:
        async with self._sem:
            reply = await self._completer(messages)
        return reply["content"]

    async def sample_many(self, conversations: list[list[Message]]) -> list[str]:
        return list(await asyncio.gather(*[self.sample(m) for m in conversations]))


def user_turn(content: str, system: str | None = None) -> list[Message]:
    """The one-turn conversation every first-turn eval sends."""
    messages: list[Message] = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=content))
    return messages
