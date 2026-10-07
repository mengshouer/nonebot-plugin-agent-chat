import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import nonebot
from nonebot.adapters import Bot, Event


class MatcherPermissionProcessTests(unittest.TestCase):
    def test_permissions_in_a_fresh_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "-v"],
                cwd=directory,
                env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])


class MatcherPermissionTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        nonebot.init(driver="~none")
        if nonebot.load_plugin("nonebot_plugin_agent_chat") is None:
            raise AssertionError("agent-chat failed to load")
        from nonebot_plugin_agent_chat import matchers

        cls.matchers = matchers

    def setUp(self) -> None:
        from nonebot_plugin_alconna.uniseg import SupportScope, Target

        self.config = nonebot.get_driver().config
        self.config.superusers = set()
        self.bot = Mock(spec=Bot)
        self.bot.config = self.config
        self.bot.self_id = "99"
        self.bot.adapter = SimpleNamespace(get_name=lambda: "Telegram")
        self.event = Mock(spec=Event)
        self.event.get_type.return_value = "message"
        self.event.get_user_id.return_value = "12345"
        self.event.get_plaintext.return_value = "/llm hello"
        # Keep real identity, ACL, trigger and permission logic; fake only the
        # transport boundary. No lifespan startup, database or live API calls.
        target = patch(
            "nonebot_plugin_alconna.get_target",
            return_value=Target("12345", private=True, scope=SupportScope.telegram),
        )
        self.target = target.start()
        self.addCleanup(target.stop)
        send = patch.object(self.matchers, "_send_plain_text", new_callable=AsyncMock)
        self.sent = send.start()
        self.addCleanup(send.stop)

    async def test_routing_uses_the_control_commands_superuser_semantics(self) -> None:
        from nonebot_plugin_alconna.uniseg import SupportScope, Target

        cases = (
            ({"telegram:12345"}, "Telegram", True),
            ({"12345"}, "Telegram", True),
            ({"telegram:99999"}, "Telegram", False),
            ({"telegram:12345"}, "OneBot V11", False),
            ({"onebot:12345"}, "OneBot V11", True),
            (set(), "Telegram", False),
        )
        for superusers, adapter_name, expected in cases:
            with self.subTest(superusers=superusers, adapter=adapter_name):
                self.config.superusers = superusers
                self.bot.adapter = SimpleNamespace(
                    get_name=lambda name=adapter_name: name
                )
                scope = (
                    SupportScope.telegram
                    if adapter_name == "Telegram"
                    else SupportScope.qq_client
                )
                self.target.return_value = Target("12345", private=True, scope=scope)
                for command in (self.matchers.agentctl, self.matchers.agent_room):
                    self.assertEqual(
                        await command.permission(self.bot, self.event), expected
                    )
                self.assertEqual(
                    await self.matchers._should_trigger(self.bot, self.event),
                    expected,
                )

    async def test_agentctl_parser_dispatches_the_registered_command(self) -> None:
        command = self.matchers._agentctl_command
        cases = (
            ("", "status", ""),
            ("status", "status", ""),
            ("profiles", "profiles", ""),
            ("use my profile", "use", "my profile"),
            ("reload", "reload", ""),
            ("cancel all", "cancel", "all"),
            ("rule list", "rule", "list"),
        )
        for suffix, expected_command, expected_value in cases:
            with self.subTest(suffix=suffix):
                parsed = command.parse(f"/agentctl {suffix}".rstrip())
                self.assertTrue(parsed.matched)
                with patch.object(
                    self.matchers, "_execute_agentctl", new_callable=AsyncMock
                ) as execute:
                    await self.matchers.handle_agentctl(self.bot, self.event, parsed)
                    execute.assert_awaited_once_with(
                        self.bot,
                        self.event,
                        expected_command,
                        expected_value,
                        self.matchers.resolve_chat_identity(self.bot, self.event),
                    )
        for removed in ("/llmctl status", "llmctl status"):
            with self.subTest(removed=removed):
                self.assertFalse(command.parse(removed).matched)

    async def test_management_commands_do_not_trigger_model_requests(self) -> None:
        self.config.superusers = {"telegram:12345"}
        self.bot.username = "mybot"
        with patch.object(
            self.matchers.plugin_config, "agent_chat_triggers", ["/agent"]
        ):
            self.event.get_plaintext.return_value = "/agent hello"
            self.assertTrue(await self.matchers._should_trigger(self.bot, self.event))
            for text in (
                "/agentctl status",
                "/agentctl@mybot status",
                "/agentctl@otherbot status",
                "/agent_room ask hello",
            ):
                with self.subTest(text=text):
                    self.event.get_plaintext.return_value = text
                    self.assertFalse(
                        await self.matchers._should_trigger(self.bot, self.event)
                    )

    async def test_scoped_admin_receives_the_same_error_details_as_bare_admin(self):
        from nonebot_plugin_agent_chat.errors import ConfigurationError, ProviderError

        for error in (
            ConfigurationError("fixture detail"),
            ProviderError("fixture detail"),
        ):
            for superusers in ({"telegram:12345"}, {"12345"}):
                with self.subTest(error=type(error).__name__, superusers=superusers):
                    self.config.superusers = superusers
                    self.sent.reset_mock()
                    await self.matchers._report_failure(self.bot, self.event, error)
                    self.assertIn("fixture detail", self.sent.await_args.args[2])

    async def test_other_adapter_and_non_admin_cannot_see_error_details(self):
        from nonebot_plugin_agent_chat.errors import ConfigurationError, ProviderError

        for error in (
            ConfigurationError("fixture detail"),
            ProviderError("fixture detail"),
        ):
            for superusers in ({"onebot:12345"}, {"telegram:99999"}, set()):
                with self.subTest(error=type(error).__name__, superusers=superusers):
                    self.config.superusers = superusers
                    self.sent.reset_mock()
                    await self.matchers._report_failure(self.bot, self.event, error)
                    self.assertNotIn("fixture detail", self.sent.await_args.args[2])

    async def test_event_without_user_id_is_not_a_superuser(self) -> None:
        from nonebot_plugin_agent_chat.errors import ConfigurationError, ProviderError

        self.config.superusers = {"telegram:12345"}
        self.event.get_user_id.side_effect = NotImplementedError
        self.assertFalse(await self.matchers._should_trigger(self.bot, self.event))
        for error in (
            ConfigurationError("fixture detail"),
            ProviderError("fixture detail"),
        ):
            with self.subTest(error=type(error).__name__):
                self.sent.reset_mock()
                await self.matchers._report_failure(self.bot, self.event, error)
                self.assertNotIn("fixture detail", self.sent.await_args.args[2])


def load_tests(loader, tests, pattern):
    # Discovery runs only the subprocess wrapper; the child executes the real
    # cases without contaminating the rest of the suite's NoneBot state.
    case = (
        MatcherPermissionTests
        if __name__ == "__main__"
        else MatcherPermissionProcessTests
    )
    return loader.loadTestsFromTestCase(case)


if __name__ == "__main__":
    unittest.main()
