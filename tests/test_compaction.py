"""Tests for Agent Room rolling context compaction and overflow recovery."""

from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from nonebot_plugin_agent_chat.adapters.base import (
    ProtocolAdapter,
    classify_provider_exception,
)
from nonebot_plugin_agent_chat.compaction import (
    COMPACTION_SUMMARY_ACK,
    generate_compaction_summary,
    plan_compaction,
    profile_for_summarization,
)
from nonebot_plugin_agent_chat.config import Config
from nonebot_plugin_agent_chat.errors import (
    ProviderError,
    ProviderIncompleteError,
    RoomError,
)
from nonebot_plugin_agent_chat.input import CollectedInput
from nonebot_plugin_agent_chat.models import (
    AgentMessage,
    MessageRole,
    ModelTurn,
    ProviderProfile,
    ReasoningEffort,
    RunResult,
    SearchMode,
    Usage,
)
from nonebot_plugin_agent_chat.paths import StoragePaths
from nonebot_plugin_agent_chat.runner import AgentRunner, RunnerLimits
from nonebot_plugin_agent_chat.service import AgentChatService
from nonebot_plugin_agent_chat.storage import AgentRoom, StoredMessage
from nonebot_plugin_agent_chat.tools import ToolContext, ToolRegistry, ToolSpec


class _FakeSummaryAdapter(ProtocolAdapter):
    def __init__(
        self,
        profile: ProviderProfile,
        api_key: str,
        *,
        summary_text: str = "## Goal & Topic\nCompacted summary",
        finish_reason: str = "stop",
        seen_prompts: list[str] | None = None,
        seen_profiles: list[ProviderProfile] | None = None,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(profile, api_key)
        self._summary_text = summary_text
        self._finish_reason = finish_reason
        self._seen_prompts = seen_prompts
        self._seen_profiles = seen_profiles
        self._error = error

    async def stream_turn(
        self,
        messages: list[AgentMessage],
        tools: list[ToolSpec],
        system_prompt: str,
    ) -> ModelTurn:
        if self._seen_profiles is not None:
            self._seen_profiles.append(self.profile)
        if self._seen_prompts is not None:
            self._seen_prompts.append(messages[0].text)
        if self._error is not None:
            raise self._error
        return ModelTurn(
            text=self._summary_text,
            usage=Usage(input_tokens=50, output_tokens=20, total_tokens=70),
            finish_reason=self._finish_reason,
        )


class CompactionUnitTests(unittest.IsolatedAsyncioTestCase):
    def test_overflow_classification_detects_provider_messages_and_ignores_429(
        self,
    ) -> None:
        class FakeBadRequestError(Exception):
            status_code = 400

        overflow_err = classify_provider_exception(
            FakeBadRequestError("prompt is too long: 210000 tokens > 200000 maximum"),
            emitted_text=False,
        )
        self.assertTrue(overflow_err.context_overflow)
        self.assertNotIn("210000", str(overflow_err))

        class FakeRateLimitError(Exception):
            status_code = 429

        rate_err = classify_provider_exception(
            FakeRateLimitError("Rate limit: too many tokens per minute"),
            emitted_text=False,
        )
        self.assertFalse(rate_err.context_overflow)

    def test_profile_for_summarization_scales_output_and_disables_tools(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-responses",
                "model": "gpt-5",
                "context_window": 4096,
                "max_output_tokens": 4096,
                "search_mode": "builtin_web_search",
                "reasoning_effort": "high",
                "capabilities": {"tools": True, "vision": True, "reasoning": True},
            }
        )
        stripped = profile_for_summarization(profile)
        self.assertFalse(stripped.capabilities.tools)
        self.assertFalse(stripped.capabilities.vision)
        self.assertEqual(stripped.search_mode, SearchMode.OFF)
        self.assertEqual(stripped.reasoning_effort, ReasoningEffort.OFF)
        self.assertEqual(stripped.max_output_tokens, 819)

    def test_plan_compaction_keeps_recent_tail_and_summarizes_older_turns(self) -> None:
        room = AgentRoom(
            id="r1",
            context_key="ctx",
            name="room",
            profile="primary",
            status="active",
            created_at=1,
            updated_at=1,
            context_tokens=3600,
        )
        messages = [
            StoredMessage(id=1, role="user", text="u0 " * 600),
            StoredMessage(id=2, role="assistant", text="a0 " * 600),
            StoredMessage(id=3, role="user", text="u1 " * 600),
            StoredMessage(id=4, role="assistant", text="a1 " * 600),
            StoredMessage(id=5, role="user", text="u2 short"),
            StoredMessage(id=6, role="assistant", text="a2 short"),
        ]
        self.assertIsNone(
            plan_compaction(
                room,
                messages,
                context_window=10_000,
                threshold_percent=80,
            )
        )
        plan = plan_compaction(
            room,
            messages,
            context_window=4096,
            threshold_percent=80,
        )
        assert plan is not None
        self.assertEqual([m.id for m in plan.messages_to_summarize], [1, 2, 3, 4])
        self.assertEqual([m.id for m in plan.kept_messages], [5, 6])
        self.assertEqual(plan.remove_ids, [1, 2, 3, 4])

    def test_plan_compaction_never_removes_messages_excluded_from_transcript(
        self,
    ) -> None:
        """Messages that do not fit the summary input budget stay in kept_messages."""

        room = AgentRoom(
            id="r1",
            context_key="ctx",
            name="room",
            profile="primary",
            status="active",
            created_at=1,
            updated_at=1,
            context_tokens=4000,
        )
        messages = [
            StoredMessage(id=1, role="user", text="first_exchange " * 300),
            StoredMessage(id=2, role="assistant", text="first_reply " * 300),
            StoredMessage(id=3, role="user", text="second_exchange " * 300),
            StoredMessage(id=4, role="assistant", text="second_reply " * 300),
            StoredMessage(id=5, role="user", text="third_exchange"),
            StoredMessage(id=6, role="assistant", text="third_reply"),
        ]
        plan = plan_compaction(
            room,
            messages,
            context_window=4096,
            threshold_percent=80,
        )
        assert plan is not None
        # Only the first exchange fits into the tight 4096-window transcript budget;
        # the second exchange must remain in kept_messages rather than being deleted.
        self.assertEqual(plan.remove_ids, [1, 2])
        self.assertEqual([m.id for m in plan.kept_messages], [3, 4, 5, 6])
        self.assertIn("first_exchange", plan.transcript)
        self.assertNotIn("second_exchange", plan.transcript)

    async def test_truncated_summary_finish_reasons_are_rejected(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "anthropic-messages", "model": "claude"}
        )
        room = AgentRoom(
            "r", "ctx", "room", "p", "active", 1, 1, context_tokens=200_000
        )
        plan = plan_compaction(
            room,
            [
                StoredMessage(id=1, role="user", text="q1"),
                StoredMessage(id=2, role="assistant", text="a1"),
                StoredMessage(id=3, role="user", text="q2"),
                StoredMessage(id=4, role="assistant", text="a2"),
            ],
            context_window=200_000,
            threshold_percent=80,
        )
        assert plan is not None
        for reason in ("length", "max_tokens", "max_output_tokens", "incomplete"):
            with (
                self.subTest(reason=reason),
                patch(
                    "nonebot_plugin_agent_chat.compaction.create_adapter",
                    side_effect=lambda p, k, r=reason: _FakeSummaryAdapter(
                        p, k, finish_reason=r
                    ),
                ),
                self.assertRaises(ProviderIncompleteError),
            ):
                await generate_compaction_summary(plan, profile, "secret")

    async def test_runner_treats_model_context_window_exceeded_as_overflow(
        self,
    ) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "anthropic-messages", "model": "claude"}
        )
        for text, expect_emitted in (("", False), ("partial text", True)):
            with self.subTest(text=text):
                adapter = _FakeSummaryAdapter(
                    profile,
                    "secret",
                    summary_text=text,
                    finish_reason="model_context_window_exceeded",
                )
                runner = AgentRunner(
                    adapter=adapter,
                    tools=ToolRegistry(),
                    tool_specs=[],
                    tool_context=ToolContext("p", profile),
                    limits=RunnerLimits(max_model_turns=2),
                )
                with self.assertRaises(ProviderError) as caught:
                    await runner.run([AgentMessage(role=MessageRole.USER, text="hi")])
                self.assertTrue(caught.exception.context_overflow)
                self.assertEqual(caught.exception.emitted_text, expect_emitted)


class RoomCompactionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        (self.profiles / "primary.json").write_text(
            json.dumps(
                {
                    "protocol": "openai-completions",
                    "model": "primary-model",
                    "api_key_env": "TEST_PRIMARY_KEY",
                    "context_window": 4096,
                    "capabilities": {"vision": True, "tools": True, "reasoning": True},
                }
            ),
            encoding="utf-8",
        )
        os.environ["TEST_PRIMARY_KEY"] = "primary-secret"
        self.service = AgentChatService(
            Config(
                agent_chat_default_profile="primary",
                agent_chat_room_enabled=True,
                agent_chat_room_compact_threshold_percent=80,
                agent_chat_user_cooldown_seconds=0,
                agent_chat_cleanup_interval_seconds=0,
            ),
            paths=StoragePaths(self.root / "data", self.profiles),
        )
        await self.service.initialize()

    async def asyncTearDown(self) -> None:
        await self.service.close()
        self.temp.cleanup()

    async def test_post_turn_auto_compaction_injects_summary_on_next_ask(self) -> None:
        await self.service.create_room("ctx-auto", "work")
        seen_histories: list[list[AgentMessage]] = []
        seen_summary_prompts: list[str] = []

        async def fake_run_profile(
            service: AgentChatService,
            loaded: object,
            messages: list[AgentMessage],
            **kwargs: object,
        ) -> RunResult:
            seen_histories.append(list(messages))
            turn_number = len(seen_histories)
            used = 1500 if turn_number == 1 else 3500
            return RunResult(
                text=f"answer-{turn_number}",
                sources=[],
                usage=Usage(
                    input_tokens=used, output_tokens=100, total_tokens=used + 100
                ),
                model_turns=1,
                local_tool_calls=0,
                searches=0,
                actual_profile="primary",
            )

        with (
            patch.object(
                self.service,
                "_run_profile",
                new=types.MethodType(fake_run_profile, self.service),
            ),
            patch(
                "nonebot_plugin_agent_chat.compaction.create_adapter",
                side_effect=lambda profile, key: _FakeSummaryAdapter(
                    profile,
                    key,
                    summary_text="## Goal & Topic\nSummary of turn 1",
                    seen_prompts=seen_summary_prompts,
                ),
            ),
        ):
            await self.service.ask_room(
                CollectedInput("question 1"),
                subject_key="u1",
                context_key="ctx-auto",
            )
            room_after_1 = await self.service.room_status("ctx-auto")
            assert room_after_1 is not None
            self.assertEqual(room_after_1.compactions, 0)

            await self.service.ask_room(
                CollectedInput("question 2"),
                subject_key="u1",
                context_key="ctx-auto",
            )
            room_after_2 = await self.service.room_status("ctx-auto")
            assert room_after_2 is not None
            self.assertEqual(room_after_2.compactions, 1)
            self.assertIn("Summary of turn 1", room_after_2.summary)
            self.assertEqual(len(seen_summary_prompts), 1)
            self.assertIn("question 1", seen_summary_prompts[0])

            await self.service.ask_room(
                CollectedInput("question 3"),
                subject_key="u1",
                context_key="ctx-auto",
            )

        third_call_messages = seen_histories[2]
        self.assertIn("<summary>", third_call_messages[0].text)
        self.assertIn("Summary of turn 1", third_call_messages[0].text)
        self.assertEqual(third_call_messages[1].text, COMPACTION_SUMMARY_ACK)
        self.assertEqual(third_call_messages[2].text, "question 2")
        self.assertEqual(third_call_messages[3].text, "answer-2")
        self.assertEqual(third_call_messages[4].text, "question 3")

    async def test_overflow_recovery_resets_turn_budget_and_separates_context_tokens(
        self,
    ) -> None:
        self.service.config.agent_chat_max_model_turns = 1
        await self.service.create_room("ctx-overflow", "work")
        calls = 0

        async def fake_run_profile(
            service: AgentChatService,
            loaded: object,
            messages: list[AgentMessage],
            **kwargs: object,
        ) -> RunResult:
            nonlocal calls
            calls += 1
            budget = kwargs["budget"]
            self.assertEqual(budget.model_turns, 0)
            budget.model_turns += 1
            if calls == 2:
                raise ProviderError(
                    "BadRequestError (HTTP 400)",
                    retriable=False,
                    emitted_text=False,
                    status_code=400,
                    error_type="BadRequestError",
                    context_overflow=True,
                )
            return RunResult(
                text=f"ok-{calls}",
                sources=[],
                usage=Usage(input_tokens=400, output_tokens=50, total_tokens=450),
                model_turns=1,
                local_tool_calls=0,
                searches=0,
                actual_profile="primary",
            )

        with (
            patch.object(
                self.service,
                "_run_profile",
                new=types.MethodType(fake_run_profile, self.service),
            ),
            patch(
                "nonebot_plugin_agent_chat.compaction.create_adapter",
                side_effect=lambda profile, key: _FakeSummaryAdapter(
                    profile,
                    key,
                    summary_text="## Goal & Topic\nRecovered summary",
                ),
            ),
        ):
            await self.service.ask_room(
                CollectedInput("turn 1"),
                subject_key="u1",
                context_key="ctx-overflow",
            )
            result = await self.service.ask_room(
                CollectedInput("turn 2 after overflow"),
                subject_key="u1",
                context_key="ctx-overflow",
            )

        self.assertEqual(result.text, "ok-3")
        # RunResult usage includes both the answer (450) and the summary call (70)...
        self.assertEqual(result.usage.total_tokens, 520)
        room = await self.service.room_status("ctx-overflow")
        assert room is not None
        self.assertEqual(room.compactions, 1)
        # ...but room.context_tokens records only the answer's context (450).
        self.assertEqual(room.context_tokens, 450)

    async def test_manual_compact_room_and_missing_profile_status(self) -> None:
        await self.service.create_room("ctx-manual", "design")
        with self.assertRaisesRegex(RoomError, "没有可压缩"):
            await self.service.compact_room("ctx-manual")

        async def fake_run_profile(
            service: AgentChatService,
            loaded: object,
            messages: list[AgentMessage],
            **kwargs: object,
        ) -> RunResult:
            return RunResult(
                text="architecture answer",
                sources=[],
                usage=Usage(input_tokens=600, output_tokens=100, total_tokens=700),
                model_turns=1,
                local_tool_calls=0,
                searches=0,
                actual_profile="primary",
            )

        seen_prompts: list[str] = []
        with (
            patch.object(
                self.service,
                "_run_profile",
                new=types.MethodType(fake_run_profile, self.service),
            ),
            patch(
                "nonebot_plugin_agent_chat.compaction.create_adapter",
                side_effect=lambda profile, key: _FakeSummaryAdapter(
                    profile,
                    key,
                    summary_text="## Goal & Topic\nSchema decisions",
                    seen_prompts=seen_prompts,
                ),
            ),
        ):
            await self.service.ask_room(
                CollectedInput("let us design the schema"),
                subject_key="u1",
                context_key="ctx-manual",
            )
            outcome = await self.service.compact_room(
                "ctx-manual",
                "keep table names",
                subject_key="u1",
            )

        self.assertEqual(outcome.summarized_messages, 2)
        self.assertEqual(outcome.kept_messages, 0)
        self.assertIn("Additional focus: keep table names", seen_prompts[0])
        status_text = await self.service.describe_room_status("ctx-manual")
        assert status_text is not None
        self.assertIn("compactions: 1", status_text)
        self.assertIn("4.1k tokens", status_text)

        # If the bound profile is gone, status reports it instead of faking 200k.
        room = await self.service.room_status("ctx-manual")
        assert room is not None
        await self.service.store.set_room_profile(room.id, "deleted-profile")
        broken_status = await self.service.describe_room_status("ctx-manual")
        assert broken_status is not None
        self.assertIn("deleted-profile（不存在）", broken_status)
        self.assertIn("Context: 不可用", broken_status)

    async def test_auto_compaction_failure_and_unexpected_errors_preserve_answer(
        self,
    ) -> None:
        await self.service.create_room("ctx-fallback", "resilient")

        async def fake_run_profile(
            service: AgentChatService,
            loaded: object,
            messages: list[AgentMessage],
            **kwargs: object,
        ) -> RunResult:
            return RunResult(
                text="still works",
                sources=[],
                usage=Usage(input_tokens=3600, output_tokens=100, total_tokens=3700),
                model_turns=1,
                local_tool_calls=0,
                searches=0,
                actual_profile="primary",
            )

        with (
            patch.object(
                self.service,
                "_run_profile",
                new=types.MethodType(fake_run_profile, self.service),
            ),
            patch(
                "nonebot_plugin_agent_chat.compaction.create_adapter",
                side_effect=lambda profile, key: _FakeSummaryAdapter(
                    profile,
                    key,
                    error=ProviderError("gateway down", error_type="APIError"),
                ),
            ),
        ):
            await self.service.ask_room(
                CollectedInput("q1"),
                subject_key="u1",
                context_key="ctx-fallback",
            )
            result = await self.service.ask_room(
                CollectedInput("q2"),
                subject_key="u1",
                context_key="ctx-fallback",
            )

        self.assertEqual(result.text, "still works")
        remaining = await self.service.store.room_history(
            (await self.service.room_status("ctx-fallback")).id  # type: ignore[union-attr]
        )
        self.assertEqual([m.text for m in remaining], ["q2", "still works"])

        # Even an unexpected runtime timeout in post-turn compaction must not
        # discard an already-persisted answer.
        with (
            patch.object(
                self.service,
                "_run_profile",
                new=types.MethodType(fake_run_profile, self.service),
            ),
            patch.object(
                self.service,
                "_compact_room_locked",
                side_effect=TimeoutError("post-turn timeout"),
            ),
        ):
            result_3 = await self.service.ask_room(
                CollectedInput("q3"),
                subject_key="u1",
                context_key="ctx-fallback",
            )
        self.assertEqual(result_3.text, "still works")


if __name__ == "__main__":
    unittest.main()
