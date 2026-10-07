"""Inline credentials: precedence, validation, and the no-leak guarantee.

The threat the leak checks answer: an agent (or a person) runs the CLI while
testing the plugin, and whatever it prints lands in that agent's context. A key
stored in a profile JSON must therefore never appear on any of our surfaces.
"""

import contextlib
import io
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from nonebot_plugin_agent_chat.cli import _interactive_command, build_parser, run_cli
from nonebot_plugin_agent_chat.errors import ProfileCredentialError
from nonebot_plugin_agent_chat.models import ProviderProfile
from nonebot_plugin_agent_chat.profiles import ProfileRegistry
from nonebot_plugin_agent_chat.prompts import PromptRegistry

CANARY = "sk-canary-3f9a1c"
EXA_CANARY = "exa-canary-77b2"
BASE = {"protocol": "openai-responses", "model": "test"}


class InlineCredentialTests(unittest.TestCase):
    def test_inline_key_wins_over_a_real_environment_variable(self) -> None:
        profile = ProviderProfile.model_validate(
            {**BASE, "api_key_env": "OPENAI_API_KEY", "api_key": CANARY}
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "from-the-environment"}):
            self.assertEqual(ProfileRegistry.resolve_api_key(profile), CANARY)

    def test_blank_inline_key_falls_back_to_the_environment(self) -> None:
        for blank in ("", "   ", None):
            profile = ProviderProfile.model_validate(
                {**BASE, "api_key_env": "OPENAI_API_KEY", "api_key": blank}
            )
            self.assertIsNone(profile.api_key)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "from-the-environment"}):
                self.assertEqual(
                    ProfileRegistry.resolve_api_key(profile), "from-the-environment"
                )

    def test_missing_environment_variable_still_fails_loud(self) -> None:
        profile = ProviderProfile.model_validate({**BASE, "api_key_env": "ABSENT_KEY"})

        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(ProfileCredentialError) as caught,
        ):
            ProfileRegistry.resolve_api_key(profile)

        self.assertIn("ABSENT_KEY", str(caught.exception))

    def test_a_profile_without_credentials_needs_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                ProfileRegistry.resolve_api_key(ProviderProfile.model_validate(BASE)),
                "not-required",
            )

    def test_exa_inline_key_wins_and_falls_back(self) -> None:
        inline = ProviderProfile.model_validate({**BASE, "exa_api_key": EXA_CANARY})
        self.assertEqual(ProfileRegistry.resolve_exa_api_key(inline), EXA_CANARY)

        env_only = ProviderProfile.model_validate(
            {**BASE, "exa_api_key_env": "EXA_KEY"}
        )
        with patch.dict(os.environ, {"EXA_KEY": "from-the-environment"}):
            self.assertEqual(
                ProfileRegistry.resolve_exa_api_key(env_only), "from-the-environment"
            )
            with self.assertRaises(ProfileCredentialError):
                ProfileRegistry.resolve_exa_api_key(
                    ProviderProfile.model_validate({**BASE, "exa_api_key_env": "OTHER"})
                )

    def test_pasted_whitespace_and_control_characters_are_rejected(self) -> None:
        for bad in (
            " sk-1",
            "sk-1 ",
            "sk-1\n",
            "sk\n1",
            "sk\x7f1",
            "\tsk-1",
            "sk\x851",  # C1 control
            "sk\x9f1",
            "\ufeffsk-1",  # BOM
            "sk\u200b1",  # zero-width space
        ):
            with self.assertRaises(ValidationError):
                ProviderProfile.model_validate({**BASE, "api_key": bad})

        # An interior plain space is odd but not a paste accident, so it stays
        # legal: the validator rejects edges and invisible characters, not keys.
        allowed = ProviderProfile.model_validate({**BASE, "api_key": "sk-a b"})
        self.assertEqual(allowed.api_key.get_secret_value(), "sk-a b")

    def test_an_inline_key_stays_a_string(self) -> None:
        # A numeric-looking token must not be JSON-parsed into an int on write.
        profile = ProviderProfile.model_validate({**BASE, "api_key": "12345678"})
        self.assertEqual(profile.api_key.get_secret_value(), "12345678")
        self.assertEqual(profile.model_dump(mode="json")["api_key"], "**********")

    def test_the_key_survives_a_runtime_override(self) -> None:
        """--reasoning-effort round-trips the profile: the key must survive."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = root / "profiles"
            profiles.mkdir()
            (profiles / "primary.json").write_text(
                json.dumps(
                    {**BASE, "api_key": CANARY, "exa_api_key": EXA_CANARY},
                ),
                encoding="utf-8",
            )
            registry = ProfileRegistry(
                profiles, "primary", PromptRegistry(root / "prompts", "default.md")
            )
            registry.load(required_profile="primary")

            before = registry.resolve_api_key(registry.get("primary").config)
            loaded = registry.override_for_runtime("primary", reasoning_effort="high")

            self.assertEqual(loaded.config.reasoning_effort.value, "high")
            self.assertEqual(registry.resolve_api_key(loaded.config), before)
            self.assertEqual(registry.resolve_exa_api_key(loaded.config), EXA_CANARY)

    def test_missing_credentials_ignores_inline_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = root / "profiles"
            profiles.mkdir()
            (profiles / "inline.json").write_text(
                json.dumps(
                    {
                        **BASE,
                        "api_key_env": "ABSENT_KEY",
                        "api_key": CANARY,
                        "search_mode": "exa",
                        "exa_api_key_env": "ABSENT_EXA",
                        "exa_api_key": EXA_CANARY,
                    }
                ),
                encoding="utf-8",
            )
            (profiles / "env-only.json").write_text(
                json.dumps({**BASE, "api_key_env": "ABSENT_KEY"}),
                encoding="utf-8",
            )
            registry = ProfileRegistry(
                profiles, "inline", PromptRegistry(root / "prompts", "default.md")
            )
            registry.load(required_profile="inline")

            with patch.dict(os.environ, {}, clear=True):
                missing = registry.missing_credentials(registry.profiles.values())

            self.assertEqual(missing, {"env-only": "ABSENT_KEY"})


class _LogCollector(logging.Handler):
    """Collect formatted log records so the leak checks can scan them."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class CliLeakTests(unittest.IsolatedAsyncioTestCase):
    """No CLI or session surface may print a stored credential."""

    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        self.profile_path = self.profiles / "primary.json"
        self.profile_path.write_text(
            json.dumps(
                {
                    **BASE,
                    "api_key_env": "ABSENT_KEY",
                    "api_key": CANARY,
                    "search_mode": "exa",
                    "exa_api_key": EXA_CANARY,
                }
            ),
            encoding="utf-8",
        )
        self.env_path = self.root / ".env.agent_chat"
        self.env_path.write_text(
            f"AGENT_CHAT_DATA_DIR={self.root / 'data'}\n"
            f"AGENT_CHAT_PROFILE_DIR={self.profiles}\n"
            "AGENT_CHAT_DEFAULT_PROFILE=primary\n"
            "AGENT_CHAT_CLEANUP_INTERVAL_SECONDS=0\n"
            "AGENT_CHAT_MAX_SEARCHES=3\n",
            encoding="utf-8",
        )
        self.records = _LogCollector()
        logging.getLogger().addHandler(self.records)
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        # The fixture must really carry the canaries, or the test proves nothing.
        assert CANARY in self.profile_path.read_text(encoding="utf-8")

    async def asyncTearDown(self) -> None:
        logging.getLogger().removeHandler(self.records)
        self.env_patch.stop()
        self.temporary.cleanup()

    def assert_silent(self, text: str) -> None:
        self.assertNotIn(CANARY, text)
        self.assertNotIn(EXA_CANARY, text)

    async def _run(self, *extra: str) -> tuple[int, str]:
        argv = [
            "--no-env",
            "--env-file",
            str(self.env_path),
            "--profiles-dir",
            str(self.profiles),
            *extra,
        ]
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            status = await run_cli(build_parser().parse_args(argv))
        combined = output.getvalue() + error.getvalue()
        self.assert_silent(combined)
        return status, combined

    async def test_listing_and_check_surfaces(self) -> None:
        status, out = await self._run("--profile-list")
        self.assertEqual(status, 0)
        self.assertIn("primary", out)

        status, out = await self._run("--check")
        self.assertEqual(status, 0, out)
        self.assertIn("profile check: ok", out)

        status, out = await self._run("--config-list")
        self.assertEqual(status, 0)
        self.assertIn("AGENT_CHAT_MAX_SEARCHES=10", out)

    async def test_setting_and_unsetting_the_key_is_masked_but_written(self) -> None:
        status, out = await self._run(
            "--profile-set", "primary", f"api_key={CANARY}2", "--json"
        )
        self.assertEqual(status, 0)
        self.assert_silent(out)
        self.assertNotIn(CANARY + "2", out)

        on_disk = json.loads(self.profile_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["api_key"], CANARY + "2")

        status, out = await self._run("--profile-unset", "primary", "api_key")
        self.assertEqual(status, 0)
        self.assertNotIn("api_key", json.loads(self.profile_path.read_text("utf-8")))

    async def test_session_commands_are_masked(self) -> None:
        service = await self._service()
        self.addAsyncCleanup(service.close)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            for line in (":profile list", f":profile set primary api_key={CANARY}9"):
                quit_ = await _interactive_command(
                    service,
                    line,
                    env_file=self.env_path,
                    profile_dir=self.profiles,
                    bot_data_dir=self.root / "data",
                    room=False,
                    room_name="local-debug",
                )
                self.assertFalse(quit_)
        self.assert_silent(output.getvalue())

    async def test_exa_literal_keeps_the_service_available(self) -> None:
        """The startup gate accepts an inline search key, with no env fallback."""

        service = await self._service()
        self.addAsyncCleanup(service.close)

        self.assertIsNone(service.startup_error)
        self.assertEqual(service.active_profile_name, "primary")
        self.assertIsNotNone(service.profiles.get("primary"))

    async def test_no_log_record_contains_the_key(self) -> None:
        await self._run("--profile-list")
        await self._run("--check")
        await self._run("--profile-set", "primary", "model=other")
        self.assert_silent("\n".join(self.records.messages))

    async def _service(self):
        from nonebot_plugin_agent_chat import env_file
        from nonebot_plugin_agent_chat.config import Config
        from nonebot_plugin_agent_chat.service import AgentChatService

        service = AgentChatService(
            Config(
                _env_file=None,
                agent_chat_data_dir=self.root / "session",
                agent_chat_profile_dir=self.profiles,
                agent_chat_default_profile="primary",
                agent_chat_cleanup_interval_seconds=0,
            ),
            env_file=self.env_path,
        )
        service.env_owned = env_file.load_into_environ(self.env_path)
        await service.initialize()
        return service
