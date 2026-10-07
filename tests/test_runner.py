import unittest

from pydantic import BaseModel

from nonebot_plugin_agent_chat.adapters.base import ProtocolAdapter
from nonebot_plugin_agent_chat.errors import ProviderError, ProviderIncompleteError
from nonebot_plugin_agent_chat.hooks import HookRegistry, RunContext
from nonebot_plugin_agent_chat.models import (
    AgentMessage,
    ModelTurn,
    ProviderProfile,
    Source,
    ToolCall,
    Usage,
)
from nonebot_plugin_agent_chat.runner import AgentRunner, RunnerBudget, RunnerLimits
from nonebot_plugin_agent_chat.tools import (
    ToolContext,
    ToolOutput,
    ToolRegistry,
    ToolRisk,
    ToolSpec,
)


class SearchInput(BaseModel):
    query: str


class FakeAdapter(ProtocolAdapter):
    def __init__(
        self,
        turns: list[ModelTurn | BaseException],
        profile: ProviderProfile,
    ) -> None:
        super().__init__(profile, "test")
        self.turns = list(turns)
        self.histories: list[list[AgentMessage]] = []
        self.remaining_searches: list[int | None] = []

    async def stream_turn(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSpec],
        system_prompt: str,
    ) -> ModelTurn:
        return self._next(messages, None)

    async def stream_turn_with_search_budget(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSpec],
        system_prompt: str,
        *,
        remaining_searches: int,
    ) -> ModelTurn:
        return self._next(messages, remaining_searches)

    def _next(
        self,
        messages: list[AgentMessage],
        remaining_searches: int | None,
    ) -> ModelTurn:
        self.histories.append(list(messages))
        self.remaining_searches.append(remaining_searches)
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_loop_and_usage(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-completions",
                "model": "test",
                "search_mode": "off",
            }
        )
        calls = []

        async def handler(arguments: BaseModel, context: ToolContext) -> ToolOutput:
            parsed = SearchInput.model_validate(arguments)
            calls.append(parsed.query)
            return ToolOutput(
                content='{"answer":"result"}',
                sources=[Source(url="https://example.com", title="Example")],
            )

        spec = ToolSpec("lookup", "Look up data", SearchInput, handler)
        tools = ToolRegistry()
        tools.register(spec)
        adapter = FakeAdapter(
            [
                ModelTurn(
                    text="",
                    tool_calls=[ToolCall("call-1", "lookup", '{"query":"q"}')],
                    usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
                ),
                ModelTurn(
                    text="final",
                    usage=Usage(input_tokens=4, output_tokens=2, total_tokens=6),
                ),
            ],
            profile,
        )
        runner = AgentRunner(
            adapter,
            tools,
            [spec],
            ToolContext("test", profile),
            RunnerLimits(),
        )

        result = await runner.run([AgentMessage(role="user", text="question")])

        self.assertEqual(result.text, "final")
        self.assertEqual(calls, ["q"])
        self.assertEqual(result.local_tool_calls, 1)
        self.assertEqual(result.usage.total_tokens, 9)
        self.assertEqual(result.sources[0].url, "https://example.com")
        second_history = adapter.histories[1]
        self.assertEqual(second_history[-1].role, "tool")
        self.assertEqual(second_history[-1].tool_call_id, "call-1")

    async def test_before_tool_hook_runs(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "openai-completions", "model": "test"}
        )
        seen = []

        async def handler(arguments: BaseModel, context: ToolContext) -> ToolOutput:
            return ToolOutput(content="ok")

        async def before_tool(call: ToolCall, context: RunContext) -> None:
            seen.append((call.name, context.profile_name))

        spec = ToolSpec("lookup", "Look up data", SearchInput, handler)
        tools = ToolRegistry()
        tools.register(spec)
        hooks = HookRegistry()
        hooks.before_tool.append(before_tool)
        adapter = FakeAdapter(
            [
                ModelTurn(
                    text="",
                    tool_calls=[ToolCall("call-1", "lookup", '{"query":"q"}')],
                ),
                ModelTurn(text="done"),
            ],
            profile,
        )
        runner = AgentRunner(
            adapter,
            tools,
            [spec],
            ToolContext("test", profile),
            RunnerLimits(),
            hooks=hooks,
            run_context=RunContext("test", profile),
        )

        await runner.run([AgentMessage(role="user", text="question")])
        self.assertEqual(seen, [("lookup", "test")])

    async def test_before_tool_hook_can_reject_the_run(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "openai-completions", "model": "test"}
        )
        called = []

        async def handler(arguments: BaseModel, context: ToolContext) -> ToolOutput:
            called.append(True)
            return ToolOutput(content="ok")

        async def reject(call: ToolCall, context: RunContext) -> None:
            raise RuntimeError("blocked by policy")

        spec = ToolSpec("lookup", "Look up data", SearchInput, handler)
        tools = ToolRegistry()
        tools.register(spec)
        hooks = HookRegistry()
        hooks.before_tool.append(reject)
        runner = AgentRunner(
            FakeAdapter(
                [
                    ModelTurn(
                        text="",
                        tool_calls=[ToolCall("call-1", "lookup", '{"query":"q"}')],
                    )
                ],
                profile,
            ),
            tools,
            [spec],
            ToolContext("test", profile),
            RunnerLimits(),
            hooks=hooks,
            run_context=RunContext("test", profile),
        )

        with self.assertRaisesRegex(RuntimeError, "blocked by policy"):
            await runner.run([AgentMessage(role="user", text="question")])
        self.assertFalse(called)

    async def test_provider_failure_after_tool_is_not_retriable(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "openai-completions", "model": "test"}
        )

        async def handler(arguments: BaseModel, context: ToolContext) -> ToolOutput:
            return ToolOutput(content="ok")

        spec = ToolSpec("lookup", "Look up data", SearchInput, handler)
        tools = ToolRegistry()
        tools.register(spec)
        provider_error = ProviderError("temporary", retriable=True)
        runner = AgentRunner(
            FakeAdapter(
                [
                    ModelTurn(
                        text="",
                        tool_calls=[ToolCall("call-1", "lookup", '{"query":"q"}')],
                    ),
                    provider_error,
                ],
                profile,
            ),
            tools,
            [spec],
            ToolContext("test", profile),
            RunnerLimits(),
        )

        with self.assertRaises(ProviderError):
            await runner.run([AgentMessage(role="user", text="question")])
        self.assertFalse(provider_error.retriable)

    async def test_successful_hosted_search_settles_actual_usage(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-responses",
                "model": "test",
                "search_mode": "builtin_web_search",
            }
        )
        adapter = FakeAdapter(
            [ModelTurn(text="answer", hosted_searches=1)],
            profile,
        )
        runner = AgentRunner(
            adapter,
            ToolRegistry(),
            [],
            ToolContext("test", profile),
            RunnerLimits(max_searches=3),
        )

        result = await runner.run([AgentMessage(role="user", text="question")])

        self.assertEqual(adapter.remaining_searches, [3])
        self.assertEqual(result.searches, 1)

    async def test_anthropic_pause_turn_continues_with_native_blocks(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "search_mode": "builtin_web_search",
            }
        )
        paused_blocks = [
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srv-1",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com",
                        "title": "Example",
                        "encrypted_content": "opaque",
                    }
                ],
            }
        ]
        adapter = FakeAdapter(
            [
                ModelTurn(
                    text="",
                    native_items=paused_blocks,
                    hosted_searches=1,
                    finish_reason="pause_turn",
                ),
                ModelTurn(text="answer"),
            ],
            profile,
        )
        runner = AgentRunner(
            adapter,
            ToolRegistry(),
            [],
            ToolContext("test", profile),
            RunnerLimits(max_searches=3),
        )

        result = await runner.run([AgentMessage(role="user", text="question")])

        self.assertEqual(result.text, "answer")
        self.assertEqual(result.model_turns, 2)
        self.assertEqual(result.searches, 1)
        self.assertEqual(adapter.remaining_searches, [3, 2])
        self.assertEqual(adapter.histories[1][-1].native_items, paused_blocks)

    async def test_anthropic_pause_turn_at_search_limit_still_answers(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "search_mode": "builtin_web_search",
            }
        )
        adapter = FakeAdapter(
            [
                ModelTurn(
                    text="",
                    native_items=[{"type": "web_search_tool_result"}],
                    hosted_searches=1,
                    finish_reason="pause_turn",
                ),
                ModelTurn(text="answer from search results"),
            ],
            profile,
        )
        runner = AgentRunner(
            adapter,
            ToolRegistry(),
            [],
            ToolContext("test", profile),
            RunnerLimits(max_searches=1),
        )

        result = await runner.run([AgentMessage(role="user", text="question")])

        self.assertEqual(result.text, "answer from search results")
        self.assertEqual(result.searches, 1)
        self.assertEqual(adapter.remaining_searches, [1, 0])

    async def test_anthropic_pause_turn_respects_model_turn_limit(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "search_mode": "builtin_web_search",
            }
        )
        adapter = FakeAdapter(
            [
                ModelTurn(
                    text="",
                    native_items=[{"type": "web_search_tool_result"}],
                    hosted_searches=1,
                    finish_reason="pause_turn",
                )
            ],
            profile,
        )
        runner = AgentRunner(
            adapter,
            ToolRegistry(),
            [],
            ToolContext("test", profile),
            RunnerLimits(max_searches=3, max_model_turns=1),
        )

        with self.assertRaises(ProviderIncompleteError) as captured:
            await runner.run([AgentMessage(role="user", text="question")])

        self.assertEqual(captured.exception.error_type, "model_turn_limit")
        self.assertEqual(adapter.remaining_searches, [3])

    async def test_anthropic_pause_turn_rejects_malformed_turn(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "search_mode": "builtin_web_search",
            }
        )
        adapter = FakeAdapter(
            [ModelTurn(text="", native_items=[], finish_reason="pause_turn")],
            profile,
        )
        runner = AgentRunner(
            adapter,
            ToolRegistry(),
            [],
            ToolContext("test", profile),
            RunnerLimits(max_searches=3),
        )

        with self.assertRaises(ProviderIncompleteError) as captured:
            await runner.run([AgentMessage(role="user", text="question")])

        self.assertEqual(captured.exception.error_type, "invalid_pause_turn")

    async def test_failed_hosted_search_attempt_keeps_reserved_budget(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-responses",
                "model": "test",
                "search_mode": "builtin_web_search",
            }
        )
        budget = RunnerBudget()
        first_adapter = FakeAdapter(
            [ProviderError("temporary", retriable=True)],
            profile,
        )
        first = AgentRunner(
            first_adapter,
            ToolRegistry(),
            [],
            ToolContext("test", profile),
            RunnerLimits(max_searches=3),
            budget=budget,
        )
        with self.assertRaises(ProviderError):
            await first.run([AgentMessage(role="user", text="question")])

        second_adapter = FakeAdapter([ModelTurn(text="answer")], profile)
        second = AgentRunner(
            second_adapter,
            ToolRegistry(),
            [],
            ToolContext("test", profile),
            RunnerLimits(max_searches=3),
            budget=budget,
        )
        result = await second.run([AgentMessage(role="user", text="question")])

        self.assertEqual(first_adapter.remaining_searches, [3])
        self.assertEqual(second_adapter.remaining_searches, [0])
        self.assertEqual(result.searches, 3)

    async def test_search_budget_is_shared_across_runner_retries(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-responses",
                "model": "test",
                "search_mode": "builtin_web_search",
            }
        )
        budget = RunnerBudget()
        first_adapter = FakeAdapter(
            [
                ModelTurn(
                    text="",
                    tool_calls=[ToolCall("unknown", "unknown", "{}")],
                    hosted_searches=2,
                ),
                ProviderError("temporary", retriable=True),
            ],
            profile,
        )
        first = AgentRunner(
            first_adapter,
            ToolRegistry(),
            [],
            ToolContext("test", profile),
            RunnerLimits(max_searches=3),
            budget=budget,
        )
        with self.assertRaises(ProviderError):
            await first.run([AgentMessage(role="user", text="question")])

        second_adapter = FakeAdapter([ModelTurn(text="answer")], profile)
        second = AgentRunner(
            second_adapter,
            ToolRegistry(),
            [],
            ToolContext("test", profile),
            RunnerLimits(max_searches=3),
            budget=budget,
        )
        result = await second.run([AgentMessage(role="user", text="question")])

        # The retry sees only the allowance left after the failed attempt, whose
        # hosted-search reservation is intentionally retained (see
        # test_failed_hosted_search_attempt_keeps_reserved_budget). Accounting is
        # therefore conservative: it reports the reserved cap rather than the two
        # observed searches, and can never exceed the per-run limit.
        self.assertEqual(second_adapter.remaining_searches, [0])
        self.assertEqual(result.searches, 3)
        self.assertLessEqual(result.searches, 3)
        self.assertEqual(result.model_turns, 3)

    async def test_excess_parallel_searches_return_tool_error_then_answer(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "openai-completions", "model": "test"}
        )
        calls = []

        async def handler(arguments: BaseModel, context: ToolContext) -> ToolOutput:
            calls.append(arguments)
            return ToolOutput(content="result")

        spec = ToolSpec("web_search", "Search", SearchInput, handler)
        tools = ToolRegistry()
        tools.register(spec)
        adapter = FakeAdapter(
            [
                ModelTurn(
                    text="",
                    tool_calls=[
                        ToolCall(f"call-{index}", "web_search", '{"query":"q"}')
                        for index in range(4)
                    ],
                ),
                ModelTurn(text="answer after searches"),
            ],
            profile,
        )
        runner = AgentRunner(
            adapter,
            tools,
            [spec],
            ToolContext("test", profile),
            RunnerLimits(max_searches=3),
        )

        result = await runner.run([AgentMessage(role="user", text="question")])

        self.assertEqual(result.text, "answer after searches")
        self.assertEqual(len(calls), 3)
        self.assertEqual(result.searches, 3)
        self.assertIn("SearchLimitReached", adapter.histories[1][-1].text)

    async def test_search_budget_blocks_local_search_before_execution(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "openai-completions", "model": "test"}
        )
        called = []

        async def handler(arguments: BaseModel, context: ToolContext) -> ToolOutput:
            called.append(True)
            return ToolOutput(content="ok")

        spec = ToolSpec("web_search", "Search", SearchInput, handler)
        tools = ToolRegistry()
        tools.register(spec)
        adapter = FakeAdapter(
            [
                ModelTurn(
                    text="",
                    tool_calls=[ToolCall("call-1", "web_search", '{"query":"q"}')],
                ),
                ModelTurn(text="answer from existing information"),
            ],
            profile,
        )
        runner = AgentRunner(
            adapter,
            tools,
            [spec],
            ToolContext("test", profile),
            RunnerLimits(max_searches=0),
        )

        result = await runner.run([AgentMessage(role="user", text="question")])

        self.assertEqual(result.text, "answer from existing information")
        self.assertFalse(called)
        self.assertIn("SearchLimitReached", adapter.histories[1][-1].text)
        # A local-function search profile gets no provider search allowance.
        self.assertEqual(adapter.remaining_searches, [None, None])

    async def test_malformed_tool_arguments_are_never_executed(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "anthropic-messages", "model": "test"}
        )
        called = []

        async def handler(arguments: BaseModel, context: ToolContext) -> ToolOutput:
            called.append(True)
            return ToolOutput(content="ok")

        spec = ToolSpec("lookup", "Look up data", SearchInput, handler)
        tools = ToolRegistry()
        tools.register(spec)
        runner = AgentRunner(
            FakeAdapter(
                [
                    ModelTurn(
                        text="",
                        tool_calls=[ToolCall("call-1", "lookup", '{"query":')],
                        finish_reason="invalid_tool_arguments",
                    )
                ],
                profile,
            ),
            tools,
            [spec],
            ToolContext("test", profile),
            RunnerLimits(),
        )

        with self.assertRaises(ProviderIncompleteError) as captured:
            await runner.run([AgentMessage(role="user", text="question")])
        self.assertEqual(captured.exception.error_type, "truncated_tool_call")
        self.assertFalse(called)

    async def test_final_turn_does_not_execute_tool(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "openai-completions", "model": "test"}
        )
        called = []

        async def handler(arguments: BaseModel, context: ToolContext) -> ToolOutput:
            called.append(True)
            return ToolOutput(content="ok")

        spec = ToolSpec("lookup", "Look up data", SearchInput, handler)
        tools = ToolRegistry()
        tools.register(spec)
        runner = AgentRunner(
            FakeAdapter(
                [
                    ModelTurn(
                        text="",
                        tool_calls=[ToolCall("call-1", "lookup", '{"query":"q"}')],
                    )
                ],
                profile,
            ),
            tools,
            [spec],
            ToolContext("test", profile),
            RunnerLimits(max_model_turns=1),
        )

        with self.assertRaises(ProviderIncompleteError):
            await runner.run([AgentMessage(role="user", text="question")])
        self.assertFalse(called)

    async def test_truncated_final_text_is_preserved_with_notice(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "openai-completions", "model": "test"}
        )
        runner = AgentRunner(
            FakeAdapter([ModelTurn("partial", finish_reason="length")], profile),
            ToolRegistry(),
            [],
            ToolContext("test", profile),
            RunnerLimits(),
        )

        result = await runner.run([AgentMessage(role="user", text="question")])
        self.assertIn("partial", result.text)
        self.assertIn("截断", result.text)

    async def test_unadvertised_high_risk_tool_is_not_executed(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "openai-completions", "model": "test"}
        )
        called = []

        async def handler(arguments: BaseModel, context: ToolContext) -> ToolOutput:
            called.append(True)
            return ToolOutput(content="should-not-run")

        tools = ToolRegistry()
        tools.register(
            ToolSpec(
                "admin_action",
                "Administrative action",
                SearchInput,
                handler,
                risk=ToolRisk.ADMIN_ONLY,
            )
        )
        adapter = FakeAdapter(
            [
                ModelTurn(
                    text="",
                    tool_calls=[ToolCall("call-1", "admin_action", '{"query":"q"}')],
                ),
                ModelTurn(text="done"),
            ],
            profile,
        )
        runner = AgentRunner(
            adapter,
            tools,
            [],
            ToolContext("test", profile),
            RunnerLimits(),
        )

        result = await runner.run([AgentMessage(role="user", text="question")])

        self.assertEqual(result.text, "done")
        self.assertFalse(called)
        self.assertTrue(adapter.histories[1][-1].tool_error)

    async def test_tool_failure_is_returned_to_model(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "openai-completions", "model": "test"}
        )

        async def handler(arguments: BaseModel, context: ToolContext) -> ToolOutput:
            raise RuntimeError("secret failure detail")

        spec = ToolSpec("lookup", "Look up data", SearchInput, handler)
        tools = ToolRegistry()
        tools.register(spec)
        adapter = FakeAdapter(
            [
                ModelTurn(
                    text="",
                    tool_calls=[ToolCall("call-1", "lookup", '{"query":"q"}')],
                ),
                ModelTurn(text="fallback answer"),
            ],
            profile,
        )
        runner = AgentRunner(
            adapter,
            tools,
            [spec],
            ToolContext("test", profile),
            RunnerLimits(),
        )

        result = await runner.run([AgentMessage(role="user", text="question")])

        self.assertEqual(result.text, "fallback answer")
        tool_message = adapter.histories[1][-1]
        self.assertIn("RuntimeError", tool_message.text)
        self.assertNotIn("secret failure detail", tool_message.text)
        self.assertTrue(tool_message.tool_error)


if __name__ == "__main__":
    unittest.main()
