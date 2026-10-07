import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from nonebot_plugin_agent_chat.config import Config
from nonebot_plugin_agent_chat.models import ImageReplyMode
from nonebot_plugin_agent_chat.platforms import SupportScope


class ConfigTests(unittest.TestCase):
    def test_upstream_scope_values_are_preserved(self) -> None:
        config = Config(
            _env_file=None,
            agent_chat_allowed_groups={"QQClient:123", "Telegram:*"},
            agent_chat_message_chunk_chars_by_platform={"Discord": 1200},
        )
        self.assertEqual(
            config.agent_chat_allowed_groups, {"QQClient:123", "Telegram:*"}
        )
        self.assertEqual(
            config.agent_chat_message_chunk_chars_by_platform, {"Discord": 1200}
        )

    def test_old_lowercase_scope_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "platform-list"):
            Config(_env_file=None, agent_chat_allowed_groups={"qqclient:123"})

    def test_all_upstream_scopes_are_valid_without_local_policy(self) -> None:
        scopes = {scope.value for scope in SupportScope}
        config = Config(
            _env_file=None,
            agent_chat_allowed_groups={f"{scope}:123" for scope in scopes},
            agent_chat_allowed_users={f"{scope}:456" for scope in scopes},
            agent_chat_message_chunk_chars_by_platform={
                scope: 1200 for scope in scopes
            },
        )
        self.assertEqual(
            config.agent_chat_allowed_groups, {f"{scope}:123" for scope in scopes}
        )
        self.assertEqual(set(config.agent_chat_message_chunk_chars_by_platform), scopes)

    def test_scope_aliases_and_unknown_values_are_rejected(self) -> None:
        for value in ("telegram", "QQCLIENT", "qq_client", "OneBot V11", "MyRPC"):
            with self.subTest(scope=value):
                with self.assertRaisesRegex(ValidationError, "platform-list"):
                    Config(_env_file=None, agent_chat_allowed_groups={f"{value}:123"})
                with self.assertRaisesRegex(ValidationError, "platform-list"):
                    Config(
                        _env_file=None,
                        agent_chat_message_chunk_chars_by_platform={value: 1200},
                    )

    def test_process_environment_populates_optional_limits(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AGENT_CHAT_MAX_MODEL_TURNS": "9",
                "AGENT_CHAT_ALLOWED_GROUPS": '["QQClient:123"]',
            },
        ):
            config = Config(_env_file=None)
        self.assertEqual(config.agent_chat_max_model_turns, 9)
        self.assertEqual(config.agent_chat_allowed_groups, {"QQClient:123"})

    def test_platform_image_modes_preserve_upstream_values(self) -> None:
        config = Config(
            _env_file=None,
            agent_chat_image_reply_mode_by_platform={"Telegram": "off"},
        )
        self.assertEqual(
            config.agent_chat_image_reply_mode_by_platform,
            {"Telegram": ImageReplyMode.OFF},
        )

    def test_platform_image_modes_accept_json_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AGENT_CHAT_IMAGE_REPLY_MODE_BY_PLATFORM": (
                    '{"Telegram": "off", "QQClient": "always"}'
                )
            },
        ):
            config = Config(_env_file=None)
        self.assertEqual(
            config.agent_chat_image_reply_mode_by_platform,
            {"Telegram": ImageReplyMode.OFF, "QQClient": ImageReplyMode.ALWAYS},
        )

    def test_invalid_platform_image_mode_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Config(
                _env_file=None,
                agent_chat_image_reply_mode_by_platform={"Telegram": "sometimes"},
            )

    def test_non_object_platform_image_modes_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Config(_env_file=None, agent_chat_image_reply_mode_by_platform=["Telegram"])

    def test_platform_chunk_chars_preserve_scope_values_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AGENT_CHAT_MESSAGE_CHUNK_CHARS_BY_PLATFORM": (
                    '{"Telegram": 3500, "QQClient": 1200}'
                )
            },
        ):
            config = Config(_env_file=None)
        self.assertEqual(
            config.agent_chat_message_chunk_chars_by_platform,
            {"Telegram": 3500, "QQClient": 1200},
        )

    def test_platform_chunk_chars_below_floor_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Config(
                _env_file=None,
                agent_chat_message_chunk_chars_by_platform={"Telegram": 100},
            )

    def test_platform_chunk_chars_zero_means_unlimited(self) -> None:
        config = Config(
            _env_file=None, agent_chat_message_chunk_chars_by_platform={"Telegram": 0}
        )
        self.assertEqual(
            config.agent_chat_message_chunk_chars_by_platform, {"Telegram": 0}
        )

    def test_global_chunk_chars_zero_means_unlimited(self) -> None:
        self.assertEqual(
            Config(
                _env_file=None, agent_chat_message_chunk_chars=0
            ).agent_chat_message_chunk_chars,
            0,
        )

    def test_global_chunk_chars_below_floor_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Config(_env_file=None, agent_chat_message_chunk_chars=100)

    def test_platform_defaults_keep_telegram_text_and_unsplit(self) -> None:
        config = Config(_env_file=None)
        self.assertEqual(
            config.agent_chat_image_reply_mode_by_platform,
            {"Telegram": ImageReplyMode.OFF},
        )
        self.assertEqual(
            config.agent_chat_message_chunk_chars_by_platform, {"Telegram": 0}
        )

    def test_acl_entries_require_a_scope(self) -> None:
        for entry in ("123", ":123", "Telegram:"):
            with self.subTest(entry=entry), self.assertRaises(ValidationError):
                Config(_env_file=None, agent_chat_allowed_groups={entry})

    def test_acl_wildcards_preserve_scope(self) -> None:
        config = Config(
            _env_file=None,
            agent_chat_allowed_groups={" Telegram:-100123 ", "Discord:*"},
        )
        self.assertEqual(
            config.agent_chat_allowed_groups, {"Telegram:-100123", "Discord:*"}
        )

    def test_platform_keys_are_not_trimmed_or_silently_dropped(self) -> None:
        for value in ("", " Telegram "):
            with (
                self.subTest(scope=value),
                self.assertRaisesRegex(ValidationError, "platform-list"),
            ):
                Config(
                    _env_file=None,
                    agent_chat_message_chunk_chars_by_platform={value: 1200},
                )

    def test_non_object_platform_chunk_chars_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Config(_env_file=None, agent_chat_message_chunk_chars_by_platform=[3500])


class ShowSourcesConfigTests(unittest.TestCase):
    def test_defaults_show_both_paths(self) -> None:
        config = Config(_env_file=None)
        self.assertTrue(config.agent_chat_show_sources_text)
        self.assertTrue(config.agent_chat_show_sources_image)
        self.assertEqual(config.agent_chat_show_sources_text_by_platform, {})
        self.assertEqual(config.agent_chat_show_sources_image_by_platform, {})

    def test_non_object_map_names_the_concrete_field(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            Config(
                _env_file=None, agent_chat_show_sources_text_by_platform=["Telegram"]
            )
        self.assertIn("AGENT_CHAT_SHOW_SOURCES_TEXT_BY_PLATFORM", str(caught.exception))

    def test_source_maps_preserve_upstream_scope_values(self) -> None:
        config = Config(
            _env_file=None,
            agent_chat_show_sources_text_by_platform={"QQClient": False},
            agent_chat_show_sources_image_by_platform={"Telegram": True},
        )
        self.assertEqual(
            config.agent_chat_show_sources_text_by_platform, {"QQClient": False}
        )
        self.assertEqual(
            config.agent_chat_show_sources_image_by_platform, {"Telegram": True}
        )

    def test_source_maps_reject_old_scope_aliases(self) -> None:
        for field in (
            "agent_chat_show_sources_text_by_platform",
            "agent_chat_show_sources_image_by_platform",
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValidationError, "platform-list"),
            ):
                Config(_env_file=None, **{field: {"telegram": False}})


if __name__ == "__main__":
    unittest.main()
