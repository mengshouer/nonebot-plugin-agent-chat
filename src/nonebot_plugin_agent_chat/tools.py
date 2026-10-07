from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .errors import ConfigurationError, ProfileCredentialError, ToolExecutionError
from .models import ProviderProfile, SearchMode, Source
from .profiles import ProfileRegistry


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    num_results: int = Field(default=5, ge=1, le=10)


class ToolRisk(str, Enum):
    READ_ONLY = "read_only"
    CONFIRMATION = "confirmation"
    ADMIN_ONLY = "admin_only"


@dataclass(frozen=True)
class ToolContext:
    profile_name: str
    profile: ProviderProfile
    subject_key: str = ""
    context_key: str = ""
    secret_resolver: Callable[[str], str | None] = os.getenv

    def secret(self, name: str) -> str | None:
        return self.secret_resolver(name)


@dataclass
class ToolOutput:
    content: str
    sources: list[Source] = field(default_factory=list)


ToolHandler = Callable[[BaseModel, ToolContext], Awaitable[ToolOutput]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    risk: ToolRisk = ToolRisk.READ_ONLY
    timeout_seconds: float | None = None

    def parameters_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return schema


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, *, replace: bool = False) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", spec.name):
            raise ConfigurationError(f"Invalid tool name: {spec.name}")
        if spec.name in self._tools and not replace:
            raise ConfigurationError(f"Tool already registered: {spec.name}")
        if spec.timeout_seconds is not None and spec.timeout_seconds <= 0:
            raise ConfigurationError(f"Tool timeout must be positive: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolExecutionError(f"Unknown tool: {name}") from exc

    def for_profile(self, profile: ProviderProfile) -> list[ToolSpec]:
        names = list(profile.enabled_tools)
        if profile.search_mode == SearchMode.EXA and "web_search" not in names:
            names.insert(0, "web_search")

        selected: list[ToolSpec] = []
        for name in names:
            spec = self.get(name)
            if spec.risk != ToolRisk.READ_ONLY:
                raise ConfigurationError(
                    "MVP only permits read-only tools; "
                    f"{name} requires {spec.risk.value}"
                )
            selected.append(spec)
        return selected

    async def execute(
        self,
        name: str,
        arguments_json: str,
        context: ToolContext,
        *,
        allowed_names: set[str] | None = None,
    ) -> ToolOutput:
        spec = self.get(name)
        if allowed_names is not None and name not in allowed_names:
            raise ToolExecutionError(f"Tool is not enabled for this profile: {name}")
        if spec.risk != ToolRisk.READ_ONLY:
            raise ToolExecutionError(
                f"Tool is not executable in MVP: {name} ({spec.risk.value})"
            )
        try:
            raw = json.loads(arguments_json or "{}")
            arguments = spec.input_model.model_validate(raw)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise ToolExecutionError(f"Invalid arguments for {name}: {exc}") from exc
        return await spec.handler(arguments, context)


registry = ToolRegistry()


def register_tool(spec: ToolSpec, *, replace: bool = False) -> None:
    """Public extension point for other NoneBot plugins."""

    registry.register(spec, replace=replace)


async def _close_exa(client: Any) -> None:
    candidates = [client, getattr(client, "_client", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        for method_name in ("aclose", "close"):
            method = getattr(candidate, method_name, None)
            if method is None:
                continue
            result = method()
            if inspect.isawaitable(result):
                await result
            return


async def exa_web_search(
    arguments: BaseModel,
    context: ToolContext,
) -> ToolOutput:
    try:
        from exa_py import AsyncExa
    except ImportError as exc:
        raise ToolExecutionError("Exa search requires the exa-py package") from exc

    parsed = WebSearchInput.model_validate(arguments)
    profile = context.profile
    try:
        # Inline credential first, then the environment variable.
        api_key = ProfileRegistry.resolve_exa_api_key(profile, context.secret)
    except ProfileCredentialError as exc:
        raise ToolExecutionError(str(exc)) from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if profile.exa_base_url:
        kwargs["api_base"] = profile.exa_base_url
    client = AsyncExa(**kwargs)
    try:
        response = await client.search(
            parsed.query,
            num_results=min(parsed.num_results, profile.exa_num_results),
            contents={"highlights": True},
        )
    except Exception as exc:
        raise ToolExecutionError(f"Exa search failed: {type(exc).__name__}") from exc
    finally:
        await _close_exa(client)

    response_results = getattr(response, "results", None)
    if response_results is None and isinstance(response, dict):
        response_results = response.get("results", [])

    normalized: list[dict[str, str]] = []
    sources: list[Source] = []
    remaining_chars = 12000
    for item in response_results or []:
        if isinstance(item, dict):
            title = str(item.get("title") or "")
            url = str(item.get("url") or item.get("id") or "")
            highlights = item.get("highlights") or []
            text = item.get("text") or ""
        else:
            title = str(getattr(item, "title", "") or "")
            url = str(getattr(item, "url", "") or getattr(item, "id", "") or "")
            highlights = getattr(item, "highlights", None) or []
            text = getattr(item, "text", "") or ""

        if isinstance(highlights, str):
            content = highlights
        elif highlights:
            content = "\n".join(str(value) for value in highlights)
        else:
            content = str(text)
        content = content[: min(4000, remaining_chars)]
        remaining_chars -= len(content)
        normalized.append({"title": title, "url": url, "content": content})
        if url.startswith(("http://", "https://")):
            sources.append(Source(url=url, title=title))
        if remaining_chars <= 0:
            break

    return ToolOutput(
        content=json.dumps({"results": normalized}, ensure_ascii=False),
        sources=sources,
    )


register_tool(
    ToolSpec(
        name="web_search",
        description=(
            "Search the live web for current or factual information. "
            "Use concise search queries and cite returned URLs."
        ),
        input_model=WebSearchInput,
        handler=exa_web_search,
        risk=ToolRisk.READ_ONLY,
    )
)
