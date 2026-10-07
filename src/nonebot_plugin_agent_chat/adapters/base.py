from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any, Protocol, runtime_checkable

from ..errors import ProviderError
from ..models import AgentMessage, ModelTurn, ProviderProfile, Source
from ..tools import ToolSpec


def dump_model(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    method = getattr(value, "model_dump", None)
    if method is not None:
        return method(exclude_none=True)
    method = getattr(value, "dict", None)
    if method is not None:
        return method(exclude_none=True)
    return {}


def read_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def walk_mappings(value: Any) -> Iterator[dict[str, Any]]:
    """Yield every mapping nested inside provider SDK objects and JSON containers."""

    if isinstance(value, list):
        for child in value:
            yield from walk_mappings(child)
        return
    if not isinstance(value, dict):
        value = dump_model(value)
    if not value:
        return
    yield value
    for child in value.values():
        if isinstance(child, (dict, list)) or hasattr(child, "model_dump"):
            yield from walk_mappings(child)


def unique_sources(sources: Iterable[Source]) -> list[Source]:
    result: list[Source] = []
    seen = set()
    for source in sources:
        if not source.url or source.url in seen:
            continue
        seen.add(source.url)
        result.append(source)
    return result


def classify_provider_exception(
    exc: BaseException,
    *,
    emitted_text: bool,
) -> ProviderError:
    status_code = getattr(exc, "status_code", None)
    class_name = type(exc).__name__
    lowered = class_name.lower()
    retriable = (
        isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError))
        or "connection" in lowered
        or "timeout" in lowered
        or "ratelimit" in lowered
        or status_code in (408, 429)
        or (isinstance(status_code, int) and 500 <= status_code <= 599)
    )
    message = f"{class_name}"
    if status_code is not None:
        message += f" (HTTP {status_code})"
    return ProviderError(
        message,
        retriable=retriable,
        emitted_text=emitted_text,
        status_code=status_code,
        error_type=class_name,
    )


class ProtocolAdapter(ABC):
    def __init__(self, profile: ProviderProfile, api_key: str) -> None:
        self.profile = profile
        del api_key  # Credentials live only in the provider SDK client.

    @abstractmethod
    async def stream_turn(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSpec],
        system_prompt: str,
    ) -> ModelTurn:
        raise NotImplementedError

    async def close(self) -> None:
        return None


@runtime_checkable
class HostedSearchAdapter(Protocol):
    """Capability implemented only by adapters with provider-owned search."""

    async def stream_turn_with_search_budget(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSpec],
        system_prompt: str,
        *,
        remaining_searches: int,
    ) -> ModelTurn: ...
