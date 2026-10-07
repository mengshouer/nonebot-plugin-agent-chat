from __future__ import annotations

import json
from typing import Any

from anthropic import AsyncAnthropic

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


class AnthropicMessagesAdapter(ProtocolAdapter):
    def __init__(self, profile: Any, api_key: str) -> None:
        super().__init__(profile, api_key)
        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        self.client = AsyncAnthropic(**kwargs)

    @staticmethod
    def _messages(messages: list[AgentMessage]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for message in messages:
            if message.role is MessageRole.USER:
                content: list[dict[str, Any]] = []
                if message.text:
                    content.append({"type": "text", "text": message.text})
                for image in message.images:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image.media_type,
                                "data": image.base64_data(),
                            },
                        }
                    )
                output.append({"role": "user", "content": content})
            elif message.role is MessageRole.ASSISTANT:
                if message.native_items:
                    content = message.native_items
                else:
                    content = []
                    if message.text:
                        content.append({"type": "text", "text": message.text})
                    content.extend(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": json.loads(call.arguments or "{}"),
                        }
                        for call in message.tool_calls
                    )
                output.append({"role": "assistant", "content": content})
            elif message.role is MessageRole.TOOL:
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.text,
                    "is_error": message.tool_error,
                }
                if (
                    output
                    and output[-1].get("role") == "user"
                    and all(
                        item.get("type") == "tool_result"
                        for item in output[-1].get("content", [])
                    )
                ):
                    output[-1]["content"].append(block)
                else:
                    output.append({"role": "user", "content": [block]})
        return output

    @staticmethod
    def _tool_definitions(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters_schema(),
            }
            for tool in tools
        ]

    @staticmethod
    def _sources_from_value(value: Any) -> list[Source]:
        sources: list[Source] = []
        for item in walk_mappings(value):
            item_type = str(item.get("type") or "").lower()
            url = item.get("url")
            if (
                item_type in {"web_search_result", "web_search_result_location"}
                and isinstance(url, str)
                and url.startswith(("http://", "https://"))
            ):
                sources.append(Source(url=url, title=str(item.get("title") or "")))
        return unique_sources(sources)

    @staticmethod
    def _has_search_results(block: dict[str, Any]) -> bool:
        """True when a result block carries actual results, not an error object."""

        content = block.get("content")
        if isinstance(content, dict) or hasattr(content, "model_dump"):
            content = dump_model(content)
        if not isinstance(content, list):
            return False
        for item in content:
            if not isinstance(item, dict):
                item = dump_model(item)
            if isinstance(item, dict) and item.get("type") == "web_search_result":
                return True
        return False

    @staticmethod
    def _web_search_requests(value: Any) -> int:
        server_usage = read_value(value, "server_tool_use")
        return int(read_value(server_usage, "web_search_requests", 0) or 0)

    def _thinking(self) -> dict[str, Any]:
        effort = self.profile.reasoning_effort
        if effort == ReasoningEffort.PROVIDER_DEFAULT:
            return {}
        if effort == ReasoningEffort.OFF:
            if not self.profile.capabilities.reasoning:
                return {}
            return {"thinking": {"type": "disabled"}}
        return {
            "thinking": {"type": "adaptive", "display": "omitted"},
            "output_config": {"effort": effort.value},
        }

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
        request: dict[str, Any] = {
            "model": self.profile.model,
            "messages": self._messages(messages),
            "max_tokens": self.profile.max_output_tokens,
            "stream": True,
            "extra_headers": self.profile.extra_headers or None,
            "extra_query": self.profile.extra_query or None,
            "extra_body": self.profile.extra_body or None,
            "timeout": self.profile.request_timeout_seconds,
        }
        if system_prompt:
            request["system"] = system_prompt
        if self.profile.temperature is not None:
            request["temperature"] = self.profile.temperature

        request_tools = self._tool_definitions(tools)
        hosted_search_enabled = (
            self.profile.search_mode == SearchMode.BUILTIN_WEB_SEARCH
            and (remaining_searches is None or remaining_searches > 0)
        )
        if hosted_search_enabled:
            max_uses = self.profile.max_builtin_tool_calls
            if remaining_searches is not None:
                max_uses = min(max_uses, remaining_searches)
            request_tools.insert(
                0,
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": max_uses,
                },
            )
        if request_tools:
            request["tools"] = request_tools
            request["tool_choice"] = {"type": "auto"}
        request.update(self._thinking())

        blocks: dict[int, dict[str, Any]] = {}
        json_deltas: dict[int, str] = {}
        invalid_tool_json: set[int] = set()
        text_parts: list[str] = []
        sources: list[Source] = []
        server_search_blocks = 0
        reported_searches = 0
        result_search_blocks = 0
        usage = Usage()
        finish_reason: str | None = None
        stream: Any = None

        try:
            stream = await self.client.messages.create(**request)
            async for event in stream:
                event_type = str(read_value(event, "type", ""))
                if event_type == "message_start":
                    message = read_value(event, "message")
                    event_usage = read_value(message, "usage")
                    usage.input_tokens = int(
                        read_value(event_usage, "input_tokens", 0) or 0
                    )
                    reported_searches = max(
                        reported_searches,
                        self._web_search_requests(event_usage),
                    )
                elif event_type == "content_block_start":
                    index = int(read_value(event, "index", 0) or 0)
                    block = dump_model(read_value(event, "content_block"))
                    if block:
                        blocks[index] = block
                        block_type = block.get("type")
                        if block_type in {"tool_use", "server_tool_use"}:
                            json_deltas[index] = ""
                        if (
                            block_type == "server_tool_use"
                            and block.get("name") == "web_search"
                        ):
                            server_search_blocks += 1
                        elif (
                            block_type == "web_search_tool_result"
                            and self._has_search_results(block)
                        ):
                            # A gateway may forward result blocks while omitting
                            # both usage counters and server_tool_use blocks; the
                            # presence of real results still proves a search ran.
                            result_search_blocks += 1
                        sources.extend(self._sources_from_value(block))
                elif event_type == "content_block_delta":
                    index = int(read_value(event, "index", 0) or 0)
                    delta = read_value(event, "delta")
                    delta_type = str(read_value(delta, "type", ""))
                    block = blocks.setdefault(index, {})
                    if delta_type == "text_delta":
                        text = str(read_value(delta, "text", "") or "")
                        if text:
                            block["text"] = str(block.get("text") or "") + text
                            text_parts.append(text)
                    elif delta_type == "thinking_delta":
                        thinking = str(read_value(delta, "thinking", "") or "")
                        block["thinking"] = str(block.get("thinking") or "") + thinking
                    elif delta_type == "signature_delta":
                        signature = str(read_value(delta, "signature", "") or "")
                        block["signature"] = (
                            str(block.get("signature") or "") + signature
                        )
                    elif delta_type == "input_json_delta":
                        partial = str(read_value(delta, "partial_json", "") or "")
                        json_deltas[index] = json_deltas.get(index, "") + partial
                    elif delta_type == "citations_delta":
                        citation = dump_model(read_value(delta, "citation"))
                        if citation:
                            citations = block.get("citations")
                            if not isinstance(citations, list):
                                citations = []
                                block["citations"] = citations
                            citations.append(citation)
                            sources.extend(self._sources_from_value(citation))
                elif event_type == "content_block_stop":
                    index = int(read_value(event, "index", 0) or 0)
                    block = blocks.get(index, {})
                    if block.get("type") in {
                        "tool_use",
                        "server_tool_use",
                    } and json_deltas.get(index):
                        try:
                            block["input"] = json.loads(json_deltas[index])
                        except json.JSONDecodeError:
                            if block.get("type") == "tool_use":
                                invalid_tool_json.add(index)
                elif event_type == "error":
                    error = read_value(event, "error")
                    error_type = str(read_value(error, "type", "") or "")
                    raise ProviderError(
                        f"Anthropic stream error: {error_type or 'unknown'}",
                        retriable=error_type
                        in {"overloaded_error", "rate_limit_error", "api_error"},
                        emitted_text=bool(text_parts),
                        error_type=error_type or "anthropic_stream_error",
                    )
                elif event_type == "message_delta":
                    delta = read_value(event, "delta")
                    finish_reason = read_value(delta, "stop_reason") or finish_reason
                    event_usage = read_value(event, "usage")
                    output_tokens = int(
                        read_value(event_usage, "output_tokens", 0) or 0
                    )
                    usage.output_tokens = output_tokens
                    usage.total_tokens = usage.input_tokens + usage.output_tokens
                    reported_searches = max(
                        reported_searches,
                        self._web_search_requests(event_usage),
                    )
        except ProviderError:
            raise
        except Exception as exc:
            raise classify_provider_exception(
                exc, emitted_text=bool(text_parts)
            ) from exc
        finally:
            if stream is not None:
                close = getattr(stream, "close", None)
                if close is not None:
                    await close()

        native_blocks = [blocks[index] for index in sorted(blocks)]
        sources.extend(self._sources_from_value(native_blocks))
        tool_calls: list[ToolCall] = []
        for index, block in sorted(blocks.items()):
            if block.get("type") != "tool_use":
                continue
            tool_input = block.get("input")
            if index in invalid_tool_json:
                arguments = json_deltas[index]
            else:
                if tool_input is None and json_deltas.get(index):
                    try:
                        tool_input = json.loads(json_deltas[index])
                    except json.JSONDecodeError:
                        invalid_tool_json.add(index)
                        arguments = json_deltas[index]
                    else:
                        arguments = json.dumps(tool_input or {}, ensure_ascii=False)
                else:
                    arguments = json.dumps(tool_input or {}, ensure_ascii=False)
            call = ToolCall(
                id=str(block.get("id") or f"tool_{index}"),
                name=str(block.get("name") or ""),
                arguments=arguments,
            )
            if call.name:
                tool_calls.append(call)

        return ModelTurn(
            text="".join(text_parts),
            tool_calls=tool_calls,
            native_items=native_blocks,
            sources=unique_sources(sources),
            usage=usage,
            hosted_searches=max(
                server_search_blocks,
                reported_searches,
                result_search_blocks,
            ),
            finish_reason=(
                "invalid_tool_arguments" if invalid_tool_json else finish_reason
            ),
        )

    async def close(self) -> None:
        await self.client.close()
