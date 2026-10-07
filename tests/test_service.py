import asyncio
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from nonebot_plugin_agent_chat import config_editor, env_file
from nonebot_plugin_agent_chat.config import Config
from nonebot_plugin_agent_chat.errors import (
    BusyError,
    ConfigurationError,
    InputError,
    ProviderError,
    RoomError,
)
from nonebot_plugin_agent_chat.input import CollectedInput
from nonebot_plugin_agent_chat.models import AgentImage, RunResult, Usage
from nonebot_plugin_agent_chat.service import AgentChatService
from nonebot_plugin_agent_chat.tools import (
    ToolOutput,
    ToolRegistry,
    ToolRisk,
    ToolSpec,
)


class _EchoArguments(BaseModel):
    value: str = ""


async def _echo_tool(arguments: BaseModel, context: object) -> ToolOutput:
    return ToolOutput(content="ok")


class ServiceGuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        self.profile_path = self.profiles / "primary.json"
        self.profile_path.write_text(
            json.dumps({"protocol": "openai-responses", "model": "test"}),
            encoding="utf-8",
        )
        self.env_path = self.root / ".env.agent_chat"
        self.service = AgentChatService(
            Config(
                agent_chat_data_dir=self.root / "data",
                agent_chat_profile_dir=self.profiles,
                agent_chat_default_profile="primary",
                agent_chat_global_concurrency=2,
                agent_chat_cleanup_interval_seconds=0,
            ),
            env_file=self.env_path,
        )
        await self.service.initialize()

    async def asyncTearDown(self) -> None:
        await self.service.close()
        self.temporary.cleanup()

    async def test_same_subject_cannot_start_two_runs(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_operation() -> RunResult:
            started.set()
            await release.wait()
            return RunResult("ok", [], Usage(), 1, 0, 0, actual_profile="primary")

        first = asyncio.create_task(
            self.service._guarded_run(
                subject_key="user:1",
                context_key="group:1",
                requested_profile="primary",
                operation=slow_operation,
                enforce_cooldown=False,
                exclusive_context=False,
            )
        )
        await started.wait()
        with self.assertRaises(BusyError):
            await self.service._guarded_run(
                subject_key="user:1",
                context_key="group:1",
                requested_profile="primary",
                operation=slow_operation,
                enforce_cooldown=False,
                exclusive_context=False,
            )
        release.set()
        result = await first
        self.assertEqual(result.text, "ok")

    async def test_deferred_input_is_guarded_before_collection(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        second_collected = False

        async def collect_first() -> CollectedInput:
            entered.set()
            await release.wait()
            return CollectedInput("question")

        async def collect_second() -> CollectedInput:
            nonlocal second_collected
            second_collected = True
            return CollectedInput("question")

        async def fake_run(service, root, chain, collected, **kwargs):
            return RunResult("answer", [], Usage(), 1, 0, 0, root.name)

        with patch.object(
            self.service,
            "_run_collected",
            new=types.MethodType(fake_run, self.service),
        ):
            first = asyncio.create_task(
                self.service.ask_deferred(
                    collect_first,
                    subject_key="deferred-user",
                    context_key="group:1",
                    enforce_cooldown=False,
                )
            )
            await entered.wait()
            with self.assertRaises(BusyError):
                await self.service.ask_deferred(
                    collect_second,
                    subject_key="deferred-user",
                    context_key="group:1",
                    enforce_cooldown=False,
                )
            self.assertFalse(second_collected)
            release.set()
            await first

    async def test_run_timeout_includes_deferred_collection(self) -> None:
        self.service.config.agent_chat_run_timeout_seconds = 0.01

        async def collect() -> CollectedInput:
            await asyncio.Event().wait()
            return CollectedInput("never")

        with self.assertRaises(ProviderError) as captured:
            await self.service.ask_deferred(
                collect,
                subject_key="timeout-user",
                context_key="group:1",
                enforce_cooldown=False,
            )

        self.assertEqual(captured.exception.error_type, "run_timeout")
        self.assertFalse(self.service._runs)

    async def test_provider_error_type_is_recorded(self) -> None:
        async def operation() -> RunResult:
            raise ProviderError("rate limited", error_type="rate_limit")

        with self.assertRaises(ProviderError):
            await self.service._guarded_run(
                subject_key="error-user",
                context_key="group:1",
                requested_profile="primary",
                operation=operation,
                enforce_cooldown=False,
                exclusive_context=False,
            )

        rows = await self.service.store.recent_runs()
        self.assertEqual(rows[0]["error_type"], "rate_limit")

    async def test_optional_daily_limit_uses_metadata_only(self) -> None:
        self.service.config.agent_chat_daily_request_limit = 1
        self.service.config.agent_chat_user_cooldown_seconds = 0

        async def fake_run(service, root, chain, collected, **kwargs):
            return RunResult("ok", [], Usage(), 1, 0, 0, actual_profile=root.name)

        with patch.object(
            self.service,
            "_run_collected",
            new=types.MethodType(fake_run, self.service),
        ):
            await self.service.ask(
                CollectedInput("first"),
                subject_key="user:limited",
                context_key="group:1",
                enforce_cooldown=False,
            )
            with self.assertRaises(BusyError):
                await self.service.ask(
                    CollectedInput("second"),
                    subject_key="user:limited",
                    context_key="group:1",
                    enforce_cooldown=False,
                )

    async def test_public_ask_enforces_cooldown_by_default(self) -> None:
        async def fake_run(service, root, chain, collected, **kwargs):
            return RunResult("ok", [], Usage(), 1, 0, 0, actual_profile=root.name)

        with patch.object(
            self.service,
            "_run_collected",
            new=types.MethodType(fake_run, self.service),
        ):
            await self.service.ask(
                CollectedInput("first"),
                subject_key="cooldown-user",
                context_key="group:1",
            )
            with self.assertRaises(BusyError):
                await self.service.ask(
                    CollectedInput("second"),
                    subject_key="cooldown-user",
                    context_key="group:1",
                )

    async def test_semantic_reload_failure_keeps_previous_snapshot(self) -> None:
        self.profile_path.write_text(
            json.dumps(
                {
                    "protocol": "openai-responses",
                    "model": "new",
                    "api_key_env": "MISSING_KEY",
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaises(ConfigurationError):
            await self.service.reload_profiles()

        self.assertEqual(
            self.service.profiles.get("primary").config.model,
            "test",
        )
        self.assertIsNotNone(self.service.status()["profile_reload_error"])

        self.profile_path.write_text(
            json.dumps({"protocol": "openai-responses", "model": "test"}),
            encoding="utf-8",
        )
        await self.service.reload_profiles()
        self.assertIsNone(self.service.status()["profile_reload_error"])

    async def test_configured_default_wins_over_stale_stored_profile(self) -> None:
        """A stored value must never override AGENT_CHAT_DEFAULT_PROFILE."""

        (self.profiles / "other.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "other"}),
            encoding="utf-8",
        )
        await self.service.store.set_setting("active_profile", "renamed-away")
        await self.service.store.close()

        restarted = AgentChatService(self.service.config)
        await restarted.initialize()
        try:
            self.assertEqual(restarted.active_profile_name, "primary")
            self.assertIsNone(restarted.startup_error)
            # The legacy column is cleared instead of being left to confuse an
            # operator inspecting the database.
            self.assertIsNone(await restarted.store.get_setting("active_profile"))
        finally:
            await restarted.close()
        # asyncTearDown closes the original service again; reopening keeps it valid.
        await self.service.store.initialize()

    async def test_manual_switch_survives_reload_but_not_restart(self) -> None:
        (self.profiles / "other.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "other"}),
            encoding="utf-8",
        )
        await self.service.use_profile("other")
        await self.service.reload_profiles()

        self.assertEqual(self.service.active_profile_name, "other")

        restarted = AgentChatService(self.service.config)
        await restarted.initialize()
        try:
            self.assertEqual(restarted.active_profile_name, "primary")
        finally:
            await restarted.close()

    async def test_renamed_active_profile_falls_back_when_unambiguous(self) -> None:
        self.service.profile_manager.forget_active()
        self.service.profiles.default_profile = "deleted-name"

        await self.service.reload_profiles()

        self.assertEqual(self.service.active_profile_name, "primary")

    async def test_missing_default_with_several_profiles_is_reported(self) -> None:
        (self.profiles / "other.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "other"}),
            encoding="utf-8",
        )
        config = self.service.config.model_copy(
            update={"agent_chat_default_profile": "deleted-name"}
        )
        service = AgentChatService(config)
        await service.initialize()
        try:
            self.assertIsNotNone(service.startup_error)
            assert service.startup_error is not None
            self.assertIn("Default profile does not exist", service.startup_error)
        finally:
            await service.close()

    async def test_use_recovers_after_a_failed_startup(self) -> None:
        """A failed startup leaves no snapshot; the command must still work."""

        (self.profiles / "other.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "other"}),
            encoding="utf-8",
        )
        config = self.service.config.model_copy(
            update={"agent_chat_default_profile": "deleted-name"}
        )
        service = AgentChatService(config)
        await service.initialize()
        try:
            self.assertIsNotNone(service.startup_error)
            await service.use_profile("primary")
            self.assertIsNone(service.startup_error)
            self.assertEqual(service.active_profile_name, "primary")
        finally:
            await service.close()

    async def test_invalid_reload_keeps_previous_snapshot(self) -> None:
        self.profile_path.write_text("{ not json", encoding="utf-8")

        with self.assertRaises(ConfigurationError):
            await self.service.reload_profiles()

        self.assertEqual(
            self.service.profiles.get("primary").config.model,
            "test",
        )

    async def test_reload_switches_away_from_a_deleted_selection(self) -> None:
        """Deleting the selected profile file must self-heal, not brick the bot."""

        (self.profiles / "other.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "other"}),
            encoding="utf-8",
        )
        await self.service.use_profile("other")
        self.assertEqual(self.service.active_profile_name, "other")

        (self.profiles / "other.json").unlink()
        await self.service.reload_profiles()

        self.assertEqual(self.service.active_profile_name, "primary")
        self.assertIsNone(self.service.status()["profile_reload_error"])

    async def test_reload_profiles_activates_valid_change(self) -> None:
        self.profile_path.write_text(
            json.dumps({"protocol": "openai-responses", "model": "new"}),
            encoding="utf-8",
        )

        await self.service.reload_profiles()

        self.assertEqual(self.service.profiles.get("primary").config.model, "new")

    async def test_marker_triggers_a_lazy_reload(self) -> None:
        """The CLI writes a marker; the next ask picks the new env up."""

        with patch.dict(os.environ, {}, clear=True):
            self.env_path.write_text("AGENT_CHAT_MAX_SEARCHES=7\n", encoding="utf-8")
            self.assertFalse(await self.service.apply_pending_reload())
            await self.service.request_reload()

            self.assertTrue(await self.service.apply_pending_reload())
            self.assertEqual(self.service.config.agent_chat_max_searches, 7)
            self.assertEqual(os.environ["AGENT_CHAT_MAX_SEARCHES"], "7")
            # No marker movement, no second reload.
            self.assertFalse(await self.service.apply_pending_reload())

    async def test_concurrent_message_waits_for_permission_reload(self) -> None:
        for entrypoint in ("apply_pending_reload", "reload_now"):
            with (
                self.subTest(entrypoint=entrypoint),
                patch.dict(os.environ, {}, clear=True),
            ):
                self.service.config.agent_chat_allowed_users = {"QQClient:123"}
                self.service._last_reload_epoch = 0.0
                self.env_path.write_text(
                    "AGENT_CHAT_ALLOWED_USERS=[]\n", encoding="utf-8"
                )
                await self.service.request_reload()
                entered = asyncio.Event()
                release = asyncio.Event()
                original_stage = self.service._stage_environment

                async def delayed_stage(
                    owned, *, entered=entered, release=release, stage=original_stage
                ):
                    entered.set()
                    await release.wait()
                    return await stage(owned)

                with patch.object(
                    self.service, "_stage_environment", side_effect=delayed_stage
                ) as stage:
                    first = asyncio.create_task(getattr(self.service, entrypoint)())
                    await entered.wait()
                    second = asyncio.create_task(self.service.apply_pending_reload())
                    try:
                        await asyncio.sleep(0)
                        self.assertFalse(second.done())
                        release.set()
                        await first
                        self.assertFalse(await second)
                        self.assertEqual(
                            self.service.config.agent_chat_allowed_users, set()
                        )
                        stage.assert_awaited_once()
                    finally:
                        release.set()
                        await asyncio.gather(first, second, return_exceptions=True)

    async def test_cancelled_reload_does_not_consume_the_marker(self) -> None:
        for entrypoint in ("apply_pending_reload", "reload_now"):
            with (
                self.subTest(entrypoint=entrypoint),
                patch.dict(os.environ, {}, clear=True),
            ):
                self.service._last_reload_epoch = 0.0
                self.env_path.write_text(
                    "AGENT_CHAT_MAX_SEARCHES=7\n", encoding="utf-8"
                )
                await self.service.request_reload()
                entered = asyncio.Event()

                async def blocked_stage(owned, *, entered=entered):
                    entered.set()
                    await asyncio.Event().wait()

                with patch.object(
                    self.service, "_stage_environment", side_effect=blocked_stage
                ):
                    task = asyncio.create_task(getattr(self.service, entrypoint)())
                    await entered.wait()
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

                self.assertEqual(self.service._last_reload_epoch, 0.0)
                self.assertTrue(await self.service.apply_pending_reload())
                self.assertEqual(self.service.config.agent_chat_max_searches, 7)
                self.assertFalse(await self.service.apply_pending_reload())

    async def test_pre_start_marker_is_already_applied(self) -> None:
        """A marker written while the bot was down must not reload on first ask."""

        marker = self.service._reload_marker_path
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("12345.0\n", encoding="utf-8")

        restarted = AgentChatService(self.service.config)
        await restarted.initialize()
        try:
            self.assertFalse(await restarted.apply_pending_reload())
        finally:
            await restarted.close()

    async def test_corrupt_marker_is_ignored(self) -> None:
        self.service._reload_marker_path.write_text("not-a-number", encoding="utf-8")

        self.assertFalse(await self.service.apply_pending_reload())

    async def test_lazy_reload_keeps_the_old_config_on_a_bad_edit(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            before = self.service.config
            self.env_path.write_text(
                "AGENT_CHAT_MAX_SEARCHES=not-a-number\n", encoding="utf-8"
            )
            await self.service.request_reload()

            with patch.object(
                self.service, "_reload_locked", wraps=self.service._reload_locked
            ) as reload:
                self.assertFalse(await self.service.apply_pending_reload())
                self.assertFalse(await self.service.apply_pending_reload())
                reload.assert_awaited_once()
            self.assertIs(self.service.config, before)
            details = await self.service.status_details()
            self.assertIn("last_reload_error", details)

    async def test_reload_now_reports_config_and_profiles(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.env_path.write_text("AGENT_CHAT_MAX_SEARCHES=5\n", encoding="utf-8")
            self.profile_path.write_text(
                json.dumps({"protocol": "openai-responses", "model": "now"}),
                encoding="utf-8",
            )

            report = await self.service.reload_now()

            self.assertIn("primary", report["profiles"])
            config = report["config"]
            self.assertEqual(config["applied"]["AGENT_CHAT_MAX_SEARCHES"], ("10", "5"))
            self.assertEqual(config["restart_required"], {})
            self.assertEqual(self.service.profiles.get("primary").config.model, "now")

    async def test_unset_line_takes_effect_on_reload(self) -> None:
        """Removing a line must fall back to the default, not keep the old value."""

        with patch.dict(os.environ, {}, clear=True):
            self.env_path.write_text("AGENT_CHAT_MAX_SEARCHES=7\n", encoding="utf-8")
            self.service.env_owned = env_file.load_into_environ(self.env_path)
            await self.service.reload_now()
            self.assertEqual(self.service.config.agent_chat_max_searches, 7)

            transaction = config_editor.ConfigEditTransaction(
                self.env_path, self.service.config, self.service.env_owned
            )
            transaction.unset("AGENT_CHAT_MAX_SEARCHES")
            transaction.commit()
            report = await self.service.reload_now()

            self.assertEqual(self.service.config.agent_chat_max_searches, 10)
            self.assertEqual(
                report["config"]["applied"]["AGENT_CHAT_MAX_SEARCHES"], ("7", "10")
            )
            self.assertNotIn("AGENT_CHAT_MAX_SEARCHES", os.environ)

    async def test_failed_reload_keeps_the_previous_credentials(self) -> None:
        """A rejected reload must never switch the runnable credentials."""

        self.profile_path.write_text(
            json.dumps(
                {
                    "protocol": "openai-responses",
                    "model": "test",
                    "api_key_env": "AGENT_CHAT_TEST_TOKEN",
                }
            ),
            encoding="utf-8",
        )
        with patch.dict(os.environ, {}, clear=True):
            self.env_path.write_text(
                "AGENT_CHAT_TEST_TOKEN=old-canary\nAGENT_CHAT_MAX_SEARCHES=3\n",
                encoding="utf-8",
            )
            self.service.env_owned = env_file.load_into_environ(self.env_path)
            await self.service.reload_now()
            self.assertEqual(
                self.service.secret_resolver("AGENT_CHAT_TEST_TOKEN"),
                "old-canary",
            )

            # New credential plus a broken profile: nothing may publish.
            self.env_path.write_text(
                "AGENT_CHAT_TEST_TOKEN=new-canary\nAGENT_CHAT_MAX_SEARCHES=9\n",
                encoding="utf-8",
            )
            self.profile_path.write_text("{bad json", encoding="utf-8")

            with self.assertRaises(ConfigurationError):
                await self.service.reload_now()

            self.assertEqual(self.service.config.agent_chat_max_searches, 3)
            self.assertEqual(os.environ["AGENT_CHAT_TEST_TOKEN"], "old-canary")
            loaded = self.service.profiles.get("primary")
            self.assertEqual(
                self.service.profiles.resolve_api_key(
                    loaded.config, self.service.secret_resolver
                ),
                "old-canary",
            )

    async def test_changed_configured_default_switches_the_active_profile(self) -> None:
        """A hot edit of AGENT_CHAT_DEFAULT_PROFILE must actually take over."""

        (self.profiles / "other.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "other"}),
            encoding="utf-8",
        )
        with patch.dict(os.environ, {}, clear=True):
            self.env_path.write_text(
                "AGENT_CHAT_DEFAULT_PROFILE=primary\n", encoding="utf-8"
            )
            self.service.env_owned = env_file.load_into_environ(self.env_path)
            await self.service.reload_now()
            self.assertEqual(self.service.active_profile_name, "primary")

            self.env_path.write_text(
                "AGENT_CHAT_DEFAULT_PROFILE=other\n", encoding="utf-8"
            )
            await self.service.reload_now()

            self.assertEqual(self.service.active_profile_name, "other")

    async def test_local_tool_limit_validates_against_the_candidate_config(
        self,
    ) -> None:
        """The reload gate must read the config being applied in both directions."""

        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "echo", "test tool", _EchoArguments, _echo_tool, ToolRisk.READ_ONLY
            )
        )
        self.profile_path.write_text(
            json.dumps(
                {
                    "protocol": "openai-responses",
                    "model": "test",
                    "capabilities": {"tools": True},
                    "enabled_tools": ["echo"],
                }
            ),
            encoding="utf-8",
        )
        service = AgentChatService(
            self.service.config.model_copy(
                update={"agent_chat_max_local_tool_calls": 0}
            ),
            tool_registry=registry,
            env_file=self.env_path,
        )
        with patch.dict(os.environ, {}, clear=True):
            self.env_path.write_text(
                "AGENT_CHAT_MAX_LOCAL_TOOL_CALLS=0\n", encoding="utf-8"
            )
            service.env_owned = env_file.load_into_environ(self.env_path)
            try:
                await service.initialize()
                self.assertIsNotNone(service.startup_error)

                # Raising the limit must be accepted without a restart.
                self.env_path.write_text(
                    "AGENT_CHAT_MAX_LOCAL_TOOL_CALLS=1\n", encoding="utf-8"
                )
                report = await service.reload_now()
                self.assertEqual(service.config.agent_chat_max_local_tool_calls, 1)
                self.assertIsNone(service.startup_error)
                self.assertIn(
                    "AGENT_CHAT_MAX_LOCAL_TOOL_CALLS",
                    report["config"]["applied"],  # type: ignore[index]
                )

                # Lowering it back to zero must be rejected, not applied.
                self.env_path.write_text(
                    "AGENT_CHAT_MAX_LOCAL_TOOL_CALLS=0\n", encoding="utf-8"
                )
                with self.assertRaises(ConfigurationError):
                    await service.reload_now()
                self.assertEqual(service.config.agent_chat_max_local_tool_calls, 1)
            finally:
                await service.close()

    async def test_failed_profile_preparation_keeps_the_previous_state(self) -> None:
        """A broken profile set must not half-apply the new settings."""

        with patch.dict(os.environ, {}, clear=True):
            self.env_path.write_text("AGENT_CHAT_MAX_SEARCHES=7\n", encoding="utf-8")
            before = self.service.config
            before_active = self.service.active_profile_name
            self.profile_path.write_text("{not json", encoding="utf-8")

            with self.assertRaises(ConfigurationError):
                await self.service.reload_now()

            self.assertIs(self.service.config, before)
            self.assertEqual(self.service.active_profile_name, before_active)
            self.assertEqual(self.service.profiles.get("primary").config.model, "test")
            details = await self.service.status_details()
            self.assertIn("last_reload_error", details)

    async def test_reload_enables_a_disabled_cleanup_loop(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(self.service._maintenance_task)
            self.env_path.write_text(
                "AGENT_CHAT_CLEANUP_INTERVAL_SECONDS=60\n", encoding="utf-8"
            )
            await self.service.reload_now()
            # The helper may start a task; the enabled interval is what matters.
            self.assertEqual(
                self.service.config.agent_chat_cleanup_interval_seconds, 60
            )
            task = self.service._maintenance_task
            self.assertTrue(task is None or not task.done())
            if task is not None:
                task.cancel()
                self.service._maintenance_task = None

    async def test_cleanup_loop_follows_interval_changes(self) -> None:
        """A changed interval restarts the loop instead of waiting out the old sleep."""

        with patch.dict(os.environ, {}, clear=True):
            self.env_path.write_text(
                "AGENT_CHAT_CLEANUP_INTERVAL_SECONDS=60\n", encoding="utf-8"
            )
            self.service.env_owned = env_file.load_into_environ(self.env_path)
            await self.service.reload_now()
            enabled = self.service._maintenance_task
            self.assertIsNotNone(enabled)

            self.env_path.write_text(
                "AGENT_CHAT_CLEANUP_INTERVAL_SECONDS=0.5\n", encoding="utf-8"
            )
            await self.service.reload_now()
            restarted = self.service._maintenance_task
            self.assertIsNotNone(restarted)
            self.assertIsNot(restarted, enabled)
            self.assertEqual(self.service._maintenance_interval, 0.5)

            self.env_path.write_text(
                "AGENT_CHAT_CLEANUP_INTERVAL_SECONDS=0\n", encoding="utf-8"
            )
            await self.service.reload_now()
            self.assertIsNone(self.service._maintenance_task)
            await asyncio.sleep(0)

    async def test_prompt_file_change_applies_on_reload(self) -> None:
        prompt_dir = self.root / "data" / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        default_prompt = prompt_dir / "default.md"
        default_prompt.write_text("first prompt\n", encoding="utf-8")
        await self.service.reload_profiles()
        self.assertEqual(
            self.service.profiles.get("primary").config.system_prompt,
            "first prompt",
        )
        default_prompt.write_text("second prompt\n", encoding="utf-8")

        await self.service.reload_profiles()

        loaded = self.service.profiles.get("primary")
        assert loaded.prompt is not None
        self.assertEqual(loaded.prompt.label, "global:default.md")
        self.assertEqual(loaded.config.system_prompt, "second prompt")

    async def test_cleanup_runs_periodically_without_restart(self) -> None:
        self.service.config.agent_chat_cleanup_interval_seconds = 0.01
        self.service.rooms.last_cleanup = 0
        cleanup_runs = AsyncMock()
        cleanup_rooms = AsyncMock(return_value=[])
        cleanup_images = AsyncMock()
        with (
            patch.object(self.service.store, "cleanup_runs", cleanup_runs),
            patch.object(
                self.service.store,
                "cleanup_closed_rooms",
                cleanup_rooms,
            ),
            patch.object(self.service.image_cache, "cleanup", cleanup_images),
        ):
            self.service._maintenance_task = asyncio.create_task(
                self.service._maintenance_loop()
            )

            async def wait_for_cleanup() -> None:
                while cleanup_runs.await_count == 0:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_cleanup(), timeout=1)
            self.service._maintenance_task.cancel()
            await asyncio.gather(
                self.service._maintenance_task,
                return_exceptions=True,
            )
            self.service._maintenance_task = None

        cleanup_runs.assert_awaited()
        cleanup_rooms.assert_awaited()
        cleanup_images.assert_awaited()

    async def test_room_retries_bound_profile_without_fallback(self) -> None:
        self.service.config.agent_chat_room_enabled = True
        room = await self.service.create_room("room-retry", "room")
        attempts: list[str] = []
        budgets: list[object] = []

        async def fake_run(service, loaded, messages, **kwargs):
            del service, messages
            attempts.append(loaded.name)
            budgets.append(kwargs["budget"])
            if len(attempts) == 1:
                raise ProviderError("temporary", retriable=True)
            return RunResult(
                "answer",
                [],
                Usage(),
                1,
                0,
                0,
                actual_profile=loaded.name,
            )

        pause = AsyncMock()
        with (
            patch.object(
                self.service,
                "_run_profile",
                new=types.MethodType(fake_run, self.service),
            ),
            patch(
                "nonebot_plugin_agent_chat.service.asyncio.sleep",
                new=pause,
            ),
        ):
            result = await self.service.ask_room(
                CollectedInput("question"),
                subject_key="room-user",
                context_key="room-retry",
            )

        self.assertEqual(result.text, "answer")
        self.assertEqual(result.actual_profile, "primary")
        self.assertEqual(attempts, ["primary", "primary"])
        self.assertEqual(len({id(budget) for budget in budgets}), 1)
        pause.assert_awaited_once_with(0.5)
        history = await self.service.store.room_history(room.id, 20, 60000)
        self.assertEqual(len(history), 2)

    async def test_room_mutations_wait_for_active_request(self) -> None:
        self.service.config.agent_chat_room_enabled = True
        old_room = await self.service.create_room("room-context", "old")
        entered = asyncio.Event()
        release = asyncio.Event()
        original_load = self.service._load_room_history

        async def delayed_load(service, room, current_images):
            entered.set()
            await release.wait()
            return await original_load(room, current_images)

        async def fake_run(service, loaded, messages, **kwargs):
            return RunResult("answer", [], Usage(), 1, 0, 0, loaded.name)

        with (
            patch.object(
                self.service,
                "_load_room_history",
                new=types.MethodType(delayed_load, self.service),
            ),
            patch.object(
                self.service,
                "_run_profile",
                new=types.MethodType(fake_run, self.service),
            ),
        ):
            ask_task = asyncio.create_task(
                self.service.ask_room(
                    CollectedInput("question"),
                    subject_key="room-user",
                    context_key="room-context",
                )
            )
            await entered.wait()
            create_task = asyncio.create_task(
                self.service.create_room("room-context", "new")
            )
            await asyncio.sleep(0)
            self.assertFalse(create_task.done())
            release.set()
            await ask_task
            new_room = await create_task

        old_history = await self.service.store.room_history(old_room.id, 20, 60000)
        new_history = await self.service.store.room_history(new_room.id, 20, 60000)
        self.assertEqual(len(old_history), 2)
        self.assertEqual(new_history, [])

    async def test_second_same_context_room_ask_is_rejected(self) -> None:
        self.service.config.agent_chat_room_enabled = True
        await self.service.create_room("room-busy", "room")
        entered = asyncio.Event()
        release = asyncio.Event()

        async def collect_first():
            entered.set()
            await release.wait()
            return CollectedInput("first")

        collect_second = AsyncMock(return_value=CollectedInput("second"))
        result = RunResult("answer", [], Usage(), 1, 0, 0, "primary")
        with patch.object(self.service, "_run_profile", return_value=result) as run:
            first = asyncio.create_task(
                self.service.ask_room_deferred(
                    collect_first,
                    subject_key="room-user-a",
                    context_key="room-busy",
                )
            )
            await entered.wait()
            try:
                with self.assertRaisesRegex(BusyError, "Agent Room"):
                    await asyncio.wait_for(
                        self.service.ask_room_deferred(
                            collect_second,
                            subject_key="room-user-b",
                            context_key="room-busy",
                        ),
                        timeout=0.5,
                    )
                collect_second.assert_not_awaited()
            finally:
                release.set()
                await first
            run.assert_awaited_once()

    async def _assert_room_lock_wait_is_cancelled(self, *, shutdown: bool) -> None:
        self.service.config.agent_chat_room_enabled = True
        self.service.config.agent_chat_user_cooldown_seconds = 0
        await self.service.create_room("room-locked", "room")
        lock = self.service.rooms.context_lock("room-locked")
        entered = asyncio.Event()
        original_admit = self.service.guard._admit

        async def admit(*args, **kwargs):
            run_id = await original_admit(*args, **kwargs)
            entered.set()
            return run_id

        result = RunResult("answer", [], Usage(), 1, 0, 0, "primary")
        with (
            patch.object(self.service.guard, "_admit", side_effect=admit),
            patch.object(self.service, "_run_profile", return_value=result) as run,
        ):
            await lock.acquire()
            task = asyncio.create_task(
                self.service.ask_room(
                    CollectedInput("cancel before collection"),
                    subject_key="room-user",
                    context_key="room-locked",
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), timeout=0.5)
                self.assertEqual(len(self.service.guard.active_runs), 1)
                if shutdown:
                    await self.service.close()
                else:
                    self.assertEqual(await self.service.cancel_runs("all"), 1)
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(self.service.guard.active_runs)
                run.assert_not_awaited()
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                lock.release()

            if not shutdown:
                response = await self.service.ask_room(
                    CollectedInput("retry after cancel"),
                    subject_key="room-user",
                    context_key="room-locked",
                )
                self.assertEqual(response.text, "answer")
                run.assert_awaited_once()

    async def test_cancel_all_includes_room_waiting_for_management_lock(self) -> None:
        await self._assert_room_lock_wait_is_cancelled(shutdown=False)

    async def test_shutdown_includes_room_waiting_for_management_lock(self) -> None:
        await self._assert_room_lock_wait_is_cancelled(shutdown=True)

    async def test_room_timeout_includes_management_lock_and_releases_admission(
        self,
    ) -> None:
        self.service.config.agent_chat_room_enabled = True
        self.service.config.agent_chat_user_cooldown_seconds = 0
        self.service.config.agent_chat_run_timeout_seconds = 0.05
        await self.service.create_room("room-timeout", "room")
        lock = self.service.rooms.context_lock("room-timeout")
        result = RunResult("answer", [], Usage(), 1, 0, 0, "primary")
        with patch.object(self.service, "_run_profile", return_value=result) as run:
            async with lock:
                with self.assertRaises(ProviderError) as raised:
                    await asyncio.wait_for(
                        self.service.ask_room(
                            CollectedInput("timeout before collection"),
                            subject_key="room-user",
                            context_key="room-timeout",
                        ),
                        timeout=0.5,
                    )
                self.assertEqual(raised.exception.error_type, "run_timeout")
                self.assertFalse(self.service.guard.active_runs)
                run.assert_not_awaited()

            self.service.config.agent_chat_run_timeout_seconds = 5
            response = await self.service.ask_room(
                CollectedInput("retry after timeout"),
                subject_key="room-user",
                context_key="room-timeout",
            )
            self.assertEqual(response.text, "answer")
            run.assert_awaited_once()

    async def test_room_profile_can_be_switched_keeping_history(self) -> None:
        self.service.config.agent_chat_room_enabled = True
        (self.profiles / "other.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "other"}),
            encoding="utf-8",
        )
        room = await self.service.create_room("room-switch", "room")
        self.assertEqual(room.profile, "primary")

        switched = await self.service.set_room_profile("room-switch", "other")

        self.assertEqual(switched.profile, "other")
        stored = await self.service.store.active_room("room-switch")
        assert stored is not None
        self.assertEqual(stored.profile, "other")
        # The switch is durable domain data: a new service instance sees it.
        restarted = AgentChatService(self.service.config)
        await restarted.initialize()
        try:
            reloaded = await restarted.store.active_room("room-switch")
            assert reloaded is not None
            self.assertEqual(reloaded.profile, "other")
        finally:
            await restarted.close()

    async def test_room_profile_switch_rejects_unknown_profile(self) -> None:
        self.service.config.agent_chat_room_enabled = True
        await self.service.create_room("room-unknown", "room")

        with self.assertRaises(ConfigurationError):
            await self.service.set_room_profile("room-unknown", "nope")

    async def test_room_with_deleted_profile_reports_how_to_fix_it(self) -> None:
        self.service.config.agent_chat_room_enabled = True
        (self.profiles / "other.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "other"}),
            encoding="utf-8",
        )
        await self.service.create_room("room-gone", "room")
        await self.service.set_room_profile("room-gone", "other")
        await self.service.reload_profiles()
        (self.profiles / "other.json").unlink()
        await self.service.reload_profiles()

        with self.assertRaises(RoomError) as raised:
            await self.service.ask_room(
                CollectedInput("question"),
                subject_key="room-user",
                context_key="room-gone",
            )

        self.assertIn("/agent_room use", str(raised.exception))

    async def test_room_history_respects_aggregate_image_limits(self) -> None:
        self.profile_path.write_text(
            json.dumps(
                {
                    "protocol": "openai-responses",
                    "model": "test",
                    "capabilities": {"vision": True},
                }
            ),
            encoding="utf-8",
        )
        await self.service.reload_profiles()
        self.service.config.agent_chat_room_enabled = True
        self.service.config.agent_chat_max_images = 2
        self.service.config.agent_chat_max_image_bytes = 4
        self.service.config.agent_chat_user_cooldown_seconds = 0
        await self.service.create_room("room-context", "room")
        histories = []

        async def fake_run(service, loaded, messages, **kwargs):
            histories.append(messages)
            return RunResult("answer", [], Usage(), 1, 0, 0, loaded.name)

        images = [
            AgentImage(media_type="image/png", data=b"aa"),
            AgentImage(media_type="image/png", data=b"bb"),
        ]
        with patch.object(
            self.service,
            "_run_profile",
            new=types.MethodType(fake_run, self.service),
        ):
            await self.service.ask_room(
                CollectedInput("first", images=images),
                subject_key="room-user",
                context_key="room-context",
            )
            await asyncio.sleep(0.01)
            await self.service.ask_room(
                CollectedInput("second", images=images),
                subject_key="room-user",
                context_key="room-context",
            )

        second_history = histories[1]
        self.assertEqual(sum(len(message.images) for message in second_history), 2)
        self.assertEqual(
            sum(
                len(image.data)
                for message in second_history
                for image in message.images
            ),
            4,
        )
        self.assertEqual(second_history[0].images, [])
        self.assertEqual(len(second_history[-1].images), 2)

    async def test_room_rejects_images_for_non_vision_profile(self) -> None:
        self.service.config.agent_chat_room_enabled = True
        await self.service.create_room("room-context", "room")

        with self.assertRaises(InputError):
            await self.service.ask_room(
                CollectedInput(
                    "question",
                    images=[AgentImage(media_type="image/png", data=b"png")],
                ),
                subject_key="room-user",
                context_key="room-context",
            )

    async def test_superuser_can_cancel_by_run_prefix(self) -> None:
        started = asyncio.Event()

        async def slow_operation() -> RunResult:
            started.set()
            await asyncio.Event().wait()
            return RunResult("never", [], Usage(), 1, 0, 0)

        task = asyncio.create_task(
            self.service._guarded_run(
                subject_key="user:1",
                context_key="group:1",
                requested_profile="primary",
                operation=slow_operation,
                enforce_cooldown=False,
                exclusive_context=False,
            )
        )
        await started.wait()
        run_id = next(iter(self.service._runs))
        count = await self.service.cancel_runs(run_id[:4])
        self.assertEqual(count, 1)
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.service._runs)


if __name__ == "__main__":
    unittest.main()
