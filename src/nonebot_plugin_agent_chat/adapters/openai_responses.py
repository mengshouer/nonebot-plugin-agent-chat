from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from openai import AsyncOpenAI

from ..errors import ProviderError
from ..models import (
    AgentMessage,
    MessageRole,
    ModelTurn,
    ReasoningEffort,
    SearchMode,
    Source,
    ToolCall,
    Usage,
)
from ..tools import ToolSpec
from .base import (
    ProtocolAdapter,
    classify_provider_exception,
    dump_model,
    read_value,
    unique_sources,
    walk_mappings,
)


class OpenAIResponsesAdapter(ProtocolAdapter):
    def __init__(self, profile: Any, api_key: str) -> None:
        super().__init__(profile, api_key)
        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        self.client = AsyncOpenAI(**kwargs)

    @staticmethod
    def _input(messages: list[AgentMessage]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for message in messages:
            if message.role is MessageRole.USER:
                content: list[dict[str, Any]] = []
                if message.text:
                    content.append({"type": "input_text", "text": message.text})
                for image in message.images:
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": image.data_url(),
                            "detail": "auto",
                        }
                    )
                output.append({"role": "user", "content": content})
            elif message.role is MessageRole.ASSISTANT:
                if message.native_items:
                    output.extend(message.native_items)
                    continue
                if message.text:
                    output.append({"role": "assistant", "content": message.text})
                for call in message.tool_calls:
                    output.append(
                        {
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                    )
            elif message.role is MessageRole.TOOL:
                output.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.text,
                    }
                )
        return output

    @staticmethod
    def _function_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema(),
                "strict": False,
            }
            for tool in tools
        ]

    def _reasoning(self) -> dict[str, str] | None:
        effort = self.profile.reasoning_effort
        if effort == ReasoningEffort.PROVIDER_DEFAULT:
            return None
        if effort == ReasoningEffort.OFF and not self.profile.capabilities.reasoning:
            return None
        mapped = "none" if effort == ReasoningEffort.OFF else effort.value
        result = {"effort": mapped}
        if mapped != "none":
            result["summary"] = "auto"
        return result

    @staticmethod
    def _sources_from_value(value: Any) -> list[Source]:
        sources: list[Source] = []
        for item in walk_mappings(value):
            item_type = str(item.get("type") or "").lower()
            url = item.get("url")
            if (
                isinstance(url, str)
                and url.startswith(("http://", "https://"))
                and (
                    "citation" in item_type or "source" in item_type or "title" in item
                )
            ):
                sources.append(Source(url=url, title=str(item.get("title") or "")))
        return unique_sources(sources)

    @staticmethod
    def _text_from_items(items: Iterable[dict[str, Any]]) -> str:
        parts: list[str] = []
        for item in items:
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    content = dump_model(content)
                if content.get("type") in ("output_text", "refusal"):
                    text = content.get("text") or content.get("refusal")
                    if text:
                        parts.append(str(text))
        return "".join(parts)

    @staticmethod
    def _hosted_search_count(items: Iterable[dict[str, Any]]) -> int:
        return sum(1 for item in items if item.get("type") == "web_search_call")

    @staticmethod
    def _usage_from_response(response: Any) -> Usage:
        response_usage = read_value(response, "usage")
        if response_usage is None:
            return Usage()
        output_details = read_value(response_usage, "output_tokens_details")
        return Usage(
            input_tokens=int(read_value(response_usage, "input_tokens", 0) or 0),
            output_tokens=int(read_value(response_usage, "output_tokens", 0) or 0),
            total_tokens=int(read_value(response_usage, "total_tokens", 0) or 0),
            reasoning_tokens=int(
                read_value(output_details, "reasoning_tokens", 0) or 0
            ),
        )

    @staticmethod
    def _tool_calls(items: Iterable[dict[str, Any]]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        seen = set()
        for item in items:
            if item.get("type") != "function_call":
                continue
            call_id = str(item.get("call_id") or item.get("id") or "")
            if not call_id or call_id in seen:
                continue
            seen.add(call_id)
            calls.append(
                ToolCall(
                    id=call_id,
                    name=str(item.get("name") or ""),
                    arguments=str(item.get("arguments") or "{}"),
                )
            )
        return [call for call in calls if call.name]

    async def stream_turn(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSpec],
        system_prompt: str,
    ) -> ModelTurn:
        return await self._stream_turn(messages, tools, system_prompt)

    async def stream_turn_with_search_budget(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSpec],
        system_prompt: str,
        *,
        remaining_searches: int,
    ) -> ModelTurn:
        return await self._stream_turn(
            messages,
            tools,
            system_prompt,
            remaining_searches=remaining_searches,
        )

    async def _stream_turn(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSpec],
        system_prompt: str,
        *,
        remaining_searches: int | None = None,
    ) -> ModelTurn:
        request_tools = self._function_tools(tools)
        include: list[str] = []
        hosted_search_enabled = (
            self.profile.search_mode == SearchMode.BUILTIN_WEB_SEARCH
            and (remaining_searches is None or remaining_searches > 0)
        )
        if hosted_search_enabled:
            request_tools.insert(0, {"type": "web_search"})
            include.append("web_search_call.action.sources")
        if not self.profile.responses_store and self.profile.capabilities.reasoning:
            include.append("reasoning.encrypted_content")

        request: dict[str, Any] = {
            "model": self.profile.model,
            "input": self._input(messages),
            "stream": True,
            "store": self.profile.responses_store,
            "max_output_tokens": self.profile.max_output_tokens,
            "extra_headers": self.profile.extra_headers or None,
            "extra_query": self.profile.extra_query or None,
            "extra_body": self.profile.extra_body or None,
            "timeout": self.profile.request_timeout_seconds,
        }
        if system_prompt:
            request["instructions"] = system_prompt
        if self.profile.temperature is not None:
            request["temperature"] = self.profile.temperature
        reasoning = self._reasoning()
        if reasoning is not None:
            request["reasoning"] = reasoning
        if request_tools:
            request["tools"] = request_tools
            request["tool_choice"] = "auto"
            request["parallel_tool_calls"] = False
        if hosted_search_enabled:
            request["max_tool_calls"] = self.profile.max_builtin_tool_calls
            if remaining_searches is not None:
                request["max_tool_calls"] = min(
                    request["max_tool_calls"],
                    remaining_searches,
                )
        if include:
            request["include"] = include

        text_parts: list[str] = []
        refusal_parts: list[str] = []
        native_items: list[dict[str, Any]] = []
        sources: list[Source] = []
        usage = Usage()
        finish_reason: str | None = None
        stream: Any = None

        try:
            stream = await self.client.responses.create(**request)
            async for event in stream:
                event_type = str(read_value(event, "type", ""))
                if event_type == "response.output_text.delta":
                    delta = str(read_value(event, "delta", "") or "")
                    if delta:
                        text_parts.append(delta)
                elif event_type == "response.refusal.delta":
                    delta = str(read_value(event, "delta", "") or "")
                    if delta:
                        refusal_parts.append(delta)
                elif event_type == "response.output_text.annotation.added":
                    annotation = read_value(event, "annotation")
                    sources.extend(self._sources_from_value(annotation))
                elif event_type == "response.output_item.done":
                    item = dump_model(read_value(event, "item"))
                    if item:
                        native_items.append(item)
                elif event_type == "response.completed":
                    response = read_value(event, "response")
                    response_items = read_value(response, "output", []) or []
                    dumped_items = [dump_model(item) for item in response_items]
                    completed_items = [item for item in dumped_items if item]
                    # Some compatible gateways leave response.output empty and
                    # provide all items only through output_item.done events.
                    if completed_items:
                        native_items = completed_items
                    usage = self._usage_from_response(response)
                    sources.extend(self._sources_from_value(response_items))
                    finish_reason = "completed"
                elif event_type == "response.incomplete":
                    response = read_value(event, "response")
                    response_items = read_value(response, "output", []) or []
                    dumped_items = [dump_model(item) for item in response_items]
                    incomplete_items = [item for item in dumped_items if item]
                    if incomplete_items:
                        native_items = incomplete_items
                    usage = self._usage_from_response(response)
                    sources.extend(self._sources_from_value(response_items))
                    details = read_value(response, "incomplete_details")
                    finish_reason = str(
                        read_value(details, "reason", "incomplete") or "incomplete"
                    )
                elif event_type in ("response.failed", "error"):
                    error = read_value(event, "error") or read_value(
                        read_value(event, "response"), "error"
                    )
                    code = str(read_value(error, "code", "") or "")
                    retriable = code in {
                        "server_error",
                        "rate_limit_exceeded",
                        "timeout",
                    }
                    raise ProviderError(
                        f"OpenAI Responses stream error: {code or 'unknown'}",
                        retriable=retriable,
                        emitted_text=bool(text_parts or refusal_parts),
                        error_type=code or "response_failed",
                    )
        except ProviderError:
            raise
        except Exception as exc:
            raise classify_provider_exception(
                exc, emitted_text=bool(text_parts or refusal_parts)
            ) from exc
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if close is not None:
                    await close()

        text = "".join(text_parts or refusal_parts)
        if not text:
            text = self._text_from_items(native_items)
        sources.extend(self._sources_from_value(native_items))
        tool_calls = self._tool_calls(native_items)

        return ModelTurn(
            text=text,
            tool_calls=tool_calls,
            native_items=native_items,
            sources=unique_sources(sources),
            usage=usage,
            hosted_searches=self._hosted_search_count(native_items),
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        await self.client.close()
