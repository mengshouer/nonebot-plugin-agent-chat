import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from nonebot_plugin_agent_chat.adapters.anthropic import AnthropicMessagesAdapter
from nonebot_plugin_agent_chat.models import AgentMessage, ProviderProfile


class FakeStream:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def __aiter__(self):
        self.iterator = iter(self.events)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self) -> None:
        return None


class AnthropicHostedSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_collects_search_usage_sources_and_native_blocks(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "search_mode": "builtin_web_search",
                "max_builtin_tool_calls": 4,
            }
        )
        adapter = AnthropicMessagesAdapter(profile, "test")
        real_client = adapter.client
        stream = FakeStream(
            [
                {
                    "type": "message_start",
                    "message": {
                        "usage": {
                            "input_tokens": 4,
                            "server_tool_use": {"web_search_requests": 0},
                        }
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srv-1",
                        "name": "web_search",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"query":"current news"}',
                    },
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srv-1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.com/news",
                                "title": "Example News",
                                "encrypted_content": "opaque",
                            }
                        ],
                    },
                },
                {"type": "content_block_stop", "index": 1},
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {
                        "type": "text",
                        "text": "",
                        "citations": [],
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "text_delta", "text": "answer"},
                },
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {
                        "type": "citations_delta",
                        "citation": {
                            "type": "web_search_result_location",
                            "url": "https://example.com/news",
                            "title": "Example News",
                            "encrypted_index": "opaque-index",
                            "cited_text": "example",
                        },
                    },
                },
                {"type": "content_block_stop", "index": 2},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {
                        "output_tokens": 3,
                        "server_tool_use": {"web_search_requests": 1},
                    },
                },
            ]
        )
        create = AsyncMock(return_value=stream)
        adapter.client = cast(
            Any,
            SimpleNamespace(messages=SimpleNamespace(create=create)),
        )

        turn = await adapter.stream_turn_with_search_budget(
            [AgentMessage(role="user", text="question")],
            [],
            "system",
            remaining_searches=2,
        )

        self.assertEqual(turn.text, "answer")
        self.assertEqual(turn.hosted_searches, 1)
        self.assertEqual(turn.usage.total_tokens, 7)
        self.assertEqual(
            [source.url for source in turn.sources], ["https://example.com/news"]
        )
        self.assertEqual(turn.native_items[0]["input"], {"query": "current news"})
        self.assertEqual(
            turn.native_items[1]["content"][0]["encrypted_content"],
            "opaque",
        )
        self.assertEqual(
            turn.native_items[2]["citations"][0]["encrypted_index"],
            "opaque-index",
        )
        self.assertEqual(turn.tool_calls, [])
        assert create.await_args is not None
        self.assertEqual(
            create.await_args.kwargs["tools"],
            [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 2,
                }
            ],
        )
        await real_client.close()

    async def test_budget_zero_omits_server_search_tool(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "search_mode": "builtin_web_search",
            }
        )
        adapter = AnthropicMessagesAdapter(profile, "test")
        real_client = adapter.client
        stream = FakeStream(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "answer"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 1},
                },
            ]
        )
        create = AsyncMock(return_value=stream)
        adapter.client = cast(
            Any,
            SimpleNamespace(messages=SimpleNamespace(create=create)),
        )

        turn = await adapter.stream_turn_with_search_budget(
            [AgentMessage(role="user", text="question")],
            [],
            "system",
            remaining_searches=0,
        )

        self.assertEqual(turn.text, "answer")
        assert create.await_args is not None
        self.assertNotIn("tools", create.await_args.kwargs)
        await real_client.close()

    async def test_result_blocks_alone_still_charge_the_budget(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "search_mode": "builtin_web_search",
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
                        "type": "web_search_tool_result",
                        "tool_use_id": "srv-1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.com/a",
                                "title": "A",
                                "encrypted_content": "opaque",
                            }
                        ],
                    },
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "pause_turn"},
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

        turn = await adapter.stream_turn_with_search_budget(
            [AgentMessage(role="user", text="question")],
            [],
            "system",
            remaining_searches=3,
        )

        self.assertEqual(turn.hosted_searches, 1)
        self.assertEqual(turn.finish_reason, "pause_turn")
        self.assertEqual(
            [source.url for source in turn.sources], ["https://example.com/a"]
        )
        await real_client.close()

    async def test_error_result_blocks_do_not_charge_the_budget(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "search_mode": "builtin_web_search",
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
                        "type": "web_search_tool_result",
                        "tool_use_id": "srv-1",
                        "content": {
                            "type": "web_search_tool_result_error",
                            "error_code": "max_uses_exceeded",
                        },
                    },
                },
                {"type": "content_block_stop", "index": 0},
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
                {"type": "content_block_stop", "index": 1},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 2},
                },
            ]
        )
        adapter.client = cast(
            Any,
            SimpleNamespace(
                messages=SimpleNamespace(create=AsyncMock(return_value=stream))
            ),
        )

        turn = await adapter.stream_turn_with_search_budget(
            [AgentMessage(role="user", text="question")],
            [],
            "system",
            remaining_searches=3,
        )

        self.assertEqual(turn.hosted_searches, 0)
        self.assertEqual(turn.sources, [])
        await real_client.close()


if __name__ == "__main__":
    unittest.main()
