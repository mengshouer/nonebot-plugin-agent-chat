from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from ..errors import ProviderError
from ..models import (
    AgentMessage,
    MessageRole,
    ModelTurn,
    ReasoningEffort,
    ToolCall,
    Usage,
)
from ..tools import ToolSpec
from .base import ProtocolAdapter, classify_provider_exception, read_value


class ThinkTagFilter:
    def __init__(self) -> None:
        self.inside = False
        self.pending = ""

    @staticmethod
    def _partial_suffix_length(value: str, marker: str) -> int:
        maximum = min(len(value), len(marker) - 1)
        for size in range(maximum, 0, -1):
            if value.endswith(marker[:size]):
                return size
        return 0

    def feed(self, value: str) -> str:
        self.pending += value
        visible: list[str] = []
        while self.pending:
            marker = "</think>" if self.inside else "<think>"
            index = self.pending.find(marker)
            if index >= 0:
                if not self.inside:
                    visible.append(self.pending[:index])
                self.pending = self.pending[index + len(marker) :]
                self.inside = not self.inside
                continue
            keep = self._partial_suffix_length(self.pending, marker)
            safe = self.pending[:-keep] if keep else self.pending
            if not self.inside:
                visible.append(safe)
            self.pending = self.pending[-keep:] if keep else ""
            break
        return "".join(visible)

    def finish(self) -> str:
        value = self.pending if not self.inside else ""
        self.pending = ""
        return value


class OpenAIChatAdapter(ProtocolAdapter):
    def __init__(self, profile: Any, api_key: str) -> None:
        super().__init__(profile, api_key)
        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        self.client = AsyncOpenAI(**kwargs)

    @staticmethod
    def _messages(
        messages: list[AgentMessage], system_prompt: str
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        if system_prompt:
            output.append({"role": "system", "content": system_prompt})

        for message in messages:
            if message.role is MessageRole.USER:
                if message.images:
                    content: list[dict[str, Any]] = []
                    if message.text:
                        content.append({"type": "text", "text": message.text})
                    for image in message.images:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": image.data_url()},
                            }
                        )
                    output.append({"role": "user", "content": content})
                else:
                    output.append({"role": "user", "content": message.text})
            elif message.role is MessageRole.ASSISTANT:
                item: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.text or None,
                }
                for native in message.native_items:
                    reasoning_content = native.get("reasoning_content")
                    if reasoning_content:
                        # Compatible reasoning models require this opaque block
                        # for tool continuation. It is never surfaced to users.
                        item["reasoning_content"] = reasoning_content
                        break
                if message.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                        for call in message.tool_calls
                    ]
                output.append(item)
            elif message.role is MessageRole.TOOL:
                output.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.text,
                    }
                )
        return output

    @staticmethod
    def _tool_definitions(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema(),
                },
            }
            for tool in tools
        ]

    def _reasoning_effort(self) -> str | None:
        effort = self.profile.reasoning_effort
        if effort == ReasoningEffort.PROVIDER_DEFAULT:
            return None
        if effort == ReasoningEffort.OFF:
            return "none" if self.profile.capabilities.reasoning else None
        return effort.value

    async def stream_turn(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSpec],
        system_prompt: str,
    ) -> ModelTurn:
        request: dict[str, Any] = {
            "model": self.profile.model,
            "messages": self._messages(messages, system_prompt),
            "stream": True,
            (
                "max_completion_tokens"
                if self.profile.capabilities.reasoning
                else "max_tokens"
            ): self.profile.max_output_tokens,
            "extra_headers": self.profile.extra_headers or None,
            "extra_query": self.profile.extra_query or None,
            "extra_body": self.profile.extra_body or None,
            "timeout": self.profile.request_timeout_seconds,
        }
        if not self.profile.base_url or "api.openai.com" in self.profile.base_url:
            request["stream_options"] = {"include_usage": True}
        if self.profile.temperature is not None:
            request["temperature"] = self.profile.temperature
        effort = self._reasoning_effort()
        if effort is not None:
            request["reasoning_effort"] = effort
        if tools:
            request["tools"] = self._tool_definitions(tools)
            request["tool_choice"] = "auto"
            request["parallel_tool_calls"] = False

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        usage = Usage()
        think_filter = ThinkTagFilter()
        stream: Any = None
        try:
            stream = await self.client.chat.completions.create(**request)
            async for chunk in stream:
                chunk_usage = read_value(chunk, "usage")
                if chunk_usage is not None:
                    completion_details = read_value(
                        chunk_usage, "completion_tokens_details"
                    )
                    usage = Usage(
                        input_tokens=int(
                            read_value(chunk_usage, "prompt_tokens", 0) or 0
                        ),
                        output_tokens=int(
                            read_value(chunk_usage, "completion_tokens", 0) or 0
                        ),
                        total_tokens=int(
                            read_value(chunk_usage, "total_tokens", 0) or 0
                        ),
                        reasoning_tokens=int(
                            read_value(completion_details, "reasoning_tokens", 0) or 0
                        ),
                    )

                choices = read_value(chunk, "choices", []) or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = read_value(choice, "finish_reason") or finish_reason
                delta = read_value(choice, "delta")
                reasoning_content = read_value(delta, "reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content:
                    reasoning_parts.append(reasoning_content)
                content = read_value(delta, "content")
                if isinstance(content, str) and content:
                    visible = think_filter.feed(content)
                    if visible:
                        text_parts.append(visible)

                for call_delta in read_value(delta, "tool_calls", []) or []:
                    index = int(read_value(call_delta, "index", 0) or 0)
                    current = calls.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    call_id = read_value(call_delta, "id")
                    if call_id:
                        current["id"] = str(call_id)
                    function = read_value(call_delta, "function")
                    name = read_value(function, "name")
                    arguments = read_value(function, "arguments")
                    if name:
                        current["name"] += str(name)
                    if arguments:
                        current["arguments"] += str(arguments)
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

        tail = think_filter.finish()
        if tail:
            text_parts.append(tail)

        tool_calls = [
            ToolCall(
                id=value["id"] or f"call_{index}",
                name=value["name"],
                arguments=value["arguments"] or "{}",
            )
            for index, value in sorted(calls.items())
            if value["name"]
        ]

        native_items = []
        if reasoning_parts:
            native_items.append({"reasoning_content": "".join(reasoning_parts)})
        return ModelTurn(
            text="".join(text_parts),
            tool_calls=tool_calls,
            native_items=native_items,
            usage=usage,
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        await self.client.close()
