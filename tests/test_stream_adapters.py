import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from nonebot_plugin_agent_chat.adapters.anthropic import AnthropicMessagesAdapter
from nonebot_plugin_agent_chat.adapters.openai_chat import (
    OpenAIChatAdapter,
    ThinkTagFilter,
)
from nonebot_plugin_agent_chat.adapters.openai_responses import OpenAIResponsesAdapter
from nonebot_plugin_agent_chat.models import AgentMessage, ProviderProfile


class FakeStream:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.closed = False

    def __aiter__(self):
        self.iterator = iter(self.events)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self) -> None:
        self.closed = True


class ThinkTagFilterTests(unittest.TestCase):
    def test_split_tags_and_thinking_are_hidden(self) -> None:
        parser = ThinkTagFilter()
        visible = [
            parser.feed("before<th"),
            parser.feed("ink>secret"),
            parser.feed("</thi"),
            parser.feed("nk>after"),
            parser.finish(),
        ]
        self.assertEqual("".join(visible), "beforeafter")


class StreamAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_stream_collects_text_and_tool_call(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "openai-completions", "model": "test"}
        )
        adapter = OpenAIChatAdapter(profile, "test")
        real_client = adapter.client
        stream = FakeStream(
            [
                {
                    "choices": [
                        {
                            "delta": {"reasoning_content": "opaque-thought"},
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {"content": "hello "},
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "function": {
                                            "name": "look",
                                            "arguments": '{"q":',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"x"}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 7,
                        "total_tokens": 17,
                        "completion_tokens_details": {"reasoning_tokens": 5},
                    },
                },
            ]
        )
        create = AsyncMock(return_value=stream)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        adapter.client = cast(Any, fake_client)

        turn = await adapter.stream_turn(
            [AgentMessage(role="user", text="question")], [], "system"
        )

        self.assertEqual(turn.text, "hello ")
        self.assertEqual(turn.native_items, [{"reasoning_content": "opaque-thought"}])
        self.assertEqual(turn.tool_calls[0].arguments, '{"q":"x"}')
        self.assertEqual(turn.usage.reasoning_tokens, 5)
        self.assertTrue(stream.closed)
        self.assertIsNotNone(create.await_args)
        assert create.await_args is not None
        request = create.await_args.kwargs
        self.assertIn("max_tokens", request)
        self.assertNotIn("max_completion_tokens", request)
        continuation = adapter._messages(
            [
                AgentMessage(
                    role="assistant",
                    tool_calls=turn.tool_calls,
                    native_items=turn.native_items,
                )
            ],
            "",
        )
        self.assertEqual(continuation[0]["reasoning_content"], "opaque-thought")
        await real_client.close()

    async def test_responses_stream_collects_completed_output_and_sources(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-responses",
                "model": "test",
                "capabilities": {"tools": True},
                "search_mode": "builtin_web_search",
            }
        )
        adapter = OpenAIResponsesAdapter(profile, "test")
        real_client = adapter.client
        output = [
            {"type": "web_search_call", "id": "search-1", "status": "completed"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "answer",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com",
                                "title": "Example",
                            }
                        ],
                    }
                ],
            },
        ]
        stream = FakeStream(
            [
                {"type": "response.output_text.delta", "delta": "answer"},
                {
                    "type": "response.completed",
                    "response": {
                        "output": output,
                        "usage": {
                            "input_tokens": 2,
                            "output_tokens": 1,
                            "total_tokens": 3,
                            "output_tokens_details": {"reasoning_tokens": 1},
                        },
                    },
                },
            ]
        )
        create = AsyncMock(return_value=stream)
        fake_client = SimpleNamespace(responses=SimpleNamespace(create=create))
        adapter.client = cast(Any, fake_client)

        turn = await adapter.stream_turn_with_search_budget(
            [AgentMessage(role="user", text="question")],
            [],
            "system",
            remaining_searches=2,
        )

        self.assertEqual(turn.text, "answer")
        self.assertEqual(turn.usage.total_tokens, 3)
        self.assertEqual(turn.usage.reasoning_tokens, 1)
        self.assertEqual(turn.sources[0].url, "https://example.com")
        self.assertEqual(turn.hosted_searches, 1)
        self.assertIsNotNone(create.await_args)
        assert create.await_args is not None
        request = create.await_args.kwargs
        self.assertEqual(request["tools"], [{"type": "web_search"}])
        self.assertEqual(request["max_tool_calls"], 2)
        # Privacy-first default: the conversation is not stored server-side.
        self.assertFalse(request["store"])
        await real_client.close()

    async def test_responses_store_requires_an_explicit_opt_in(self) -> None:
        stored = ProviderProfile.model_validate(
            {
                "protocol": "openai-responses",
                "model": "test",
                "responses_store": True,
            }
        )
        adapter = OpenAIResponsesAdapter(stored, "test")
        real_client = adapter.client
        stream = FakeStream(
            [{"type": "response.completed", "response": {"output": []}}]
        )
        create = AsyncMock(return_value=stream)
        adapter.client = cast(
            Any, SimpleNamespace(responses=SimpleNamespace(create=create))
        )

        await adapter.stream_turn([], [], "system")

        assert create.await_args is not None
        self.assertTrue(create.await_args.kwargs["store"])
        await real_client.close()

    async def test_responses_omits_hosted_search_when_budget_is_empty(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-responses",
                "model": "test",
                "search_mode": "builtin_web_search",
            }
        )
        adapter = OpenAIResponsesAdapter(profile, "test")
        real_client = adapter.client
        stream = FakeStream(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "output": [
                            {
                                "type": "message",
                                "content": [{"type": "output_text", "text": "answer"}],
                            }
                        ],
                        "usage": {},
                    },
                }
            ]
        )
        create = AsyncMock(return_value=stream)
        adapter.client = cast(
            Any,
            SimpleNamespace(responses=SimpleNamespace(create=create)),
        )

        turn = await adapter.stream_turn_with_search_budget(
            [AgentMessage(role="user", text="question")],
            [],
            "system",
            remaining_searches=0,
        )

        self.assertEqual(turn.text, "answer")
        assert create.await_args is not None
        request = create.await_args.kwargs
        self.assertNotIn("tools", request)
        self.assertNotIn("max_tool_calls", request)
        await real_client.close()

    async def test_responses_keeps_done_items_when_completed_output_is_empty(
        self,
    ) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-responses",
                "model": "test",
                "capabilities": {"tools": True},
            }
        )
        adapter = OpenAIResponsesAdapter(profile, "test")
        real_client = adapter.client
        stream = FakeStream(
            [
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://example.com",
                                        "title": "Example",
                                    }
                                ],
                            }
                        ],
                    },
                },
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "lookup",
                        "arguments": '{"q":"x"}',
                    },
                },
                {
                    "type": "response.completed",
                    "response": {"output": [], "usage": {}},
                },
            ]
        )
        create = AsyncMock(return_value=stream)
        adapter.client = cast(
            Any,
            SimpleNamespace(responses=SimpleNamespace(create=create)),
        )

        turn = await adapter.stream_turn(
            [AgentMessage(role="user", text="question")], [], "system"
        )

        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].name, "lookup")
        self.assertEqual(turn.native_items[1]["call_id"], "call-1")
        self.assertEqual(turn.sources[0].url, "https://example.com")
        await real_client.close()

    async def test_anthropic_marks_malformed_tool_json_incomplete(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "capabilities": {"tools": True},
            }
        )
        adapter = AnthropicMessagesAdapter(profile, "test")
        real_client = adapter.client
        stream = FakeStream(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "lookup",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"query":',
                    },
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 1},
                },
            ]
        )
        adapter.client = cast(
            Any,
            SimpleNamespace(
                messages=SimpleNamespace(create=AsyncMock(return_value=stream))
            ),
        )

        turn = await adapter.stream_turn(
            [AgentMessage(role="user", text="question")], [], "system"
        )

        self.assertEqual(turn.finish_reason, "invalid_tool_arguments")
        self.assertEqual(turn.tool_calls[0].arguments, '{"query":')
        await real_client.close()

    async def test_anthropic_stream_preserves_signed_thinking_for_continuation(
        self,
    ) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "capabilities": {"reasoning": True},
                "reasoning_effort": "high",
            }
        )
        adapter = AnthropicMessagesAdapter(profile, "test")
        real_client = adapter.client
        stream = FakeStream(
            [
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 4}},
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "thought"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "signed"},
                },
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "answer"},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 2},
                },
            ]
        )
        create = AsyncMock(return_value=stream)
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create))
        adapter.client = cast(Any, fake_client)

        turn = await adapter.stream_turn(
            [AgentMessage(role="user", text="question")], [], "system"
        )

        self.assertEqual(turn.text, "answer")
        self.assertEqual(turn.native_items[0]["thinking"], "thought")
        self.assertEqual(turn.native_items[0]["signature"], "signed")
        self.assertIsNotNone(create.await_args)
        assert create.await_args is not None
        request = create.await_args.kwargs
        self.assertEqual(
            request["thinking"], {"type": "adaptive", "display": "omitted"}
        )
        self.assertEqual(request["output_config"], {"effort": "high"})
        await real_client.close()


if __name__ == "__main__":
    unittest.main()
