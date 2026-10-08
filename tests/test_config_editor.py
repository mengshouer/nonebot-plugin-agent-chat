import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nonebot_plugin_agent_chat import config_editor
from nonebot_plugin_agent_chat.config import Config
from nonebot_plugin_agent_chat.errors import ConfigurationError, InputError


class ValueFormattingTests(unittest.TestCase):
    def test_display_values(self) -> None:
        self.assertEqual(config_editor.display_value(None), "(未设置)")
        self.assertEqual(config_editor.display_value(True), "true")
        self.assertEqual(config_editor.display_value(Path("a/b")), "a/b")
        self.assertEqual(
            config_editor.display_value({"Telegram": "off"}),
            '{"Telegram": "off"}',
        )
        self.assertEqual(
            config_editor.display_value({"QQClient", "Telegram"}),
            '["QQClient", "Telegram"]',
        )

    def test_parse_value_respects_string_fields(self) -> None:
        # String fields stay literal even when the text looks like JSON.
        self.assertEqual(
            config_editor.parse_value("AGENT_CHAT_DEFAULT_PROFILE", "true"), "true"
        )
        self.assertEqual(config_editor.parse_value("AGENT_CHAT_MAX_SEARCHES", "3"), 3)
        self.assertEqual(
            config_editor.parse_value("AGENT_CHAT_ENABLE_AT", "true"), True
        )
        self.assertEqual(
            config_editor.parse_value("AGENT_CHAT_ALLOWED_GROUPS", '["QQClient:1"]'),
            ["QQClient:1"],
        )
        self.assertEqual(
            config_editor.parse_value("AGENT_CHAT_IMAGE_REPLY_MODE", "off"), "off"
        )

    def test_parse_value_rejects_unknown_keys(self) -> None:
        with self.assertRaises(InputError):
            config_editor.parse_value("AGENT_CHAT_NOT_A_SETTING", "1")


class ValidationTests(unittest.TestCase):
    def test_unknown_key_is_rejected(self) -> None:
        with self.assertRaises(InputError):
            config_editor.validate_change(Config(_env_file=None), "NOPE", "1")

    def test_invalid_value_is_rejected_with_detail(self) -> None:
        with self.assertRaises(InputError) as caught:
            config_editor.validate_change(
                Config(_env_file=None), "AGENT_CHAT_IMAGE_REPLY_MIN_CHARS", "5"
            )
        self.assertIn("AGENT_CHAT_IMAGE_REPLY_MIN_CHARS", str(caught.exception))

    def test_cross_field_constraints_apply(self) -> None:
        # builtin web search needs a protocol that supports it.
        with self.assertRaises(InputError):
            config_editor.validate_change(
                Config(_env_file=None), "AGENT_CHAT_SEARCH_MODE", "builtin_web_search"
            )

    def test_valid_change_returns_candidate(self) -> None:
        candidate = config_editor.validate_change(
            Config(_env_file=None), "AGENT_CHAT_MAX_SEARCHES", "5"
        )
        self.assertEqual(candidate.agent_chat_max_searches, 5)

    def test_setting_edit_writes_and_keeps_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.agent_chat"
            path.write_text(
                "# comment\nAGENT_CHAT_MAX_SEARCHES=3\nOTHER=1\n", encoding="utf-8"
            )
            transaction = config_editor.ConfigEditTransaction(
                path, Config(_env_file=None)
            )
            transaction.set("AGENT_CHAT_MAX_SEARCHES", "7")
            transaction.commit()
            text = path.read_text(encoding="utf-8")
            self.assertIn("# comment", text)
            self.assertIn("AGENT_CHAT_MAX_SEARCHES=7", text)
            self.assertIn("OTHER=1", text)
            self.assertEqual(transaction.config.agent_chat_max_searches, 7)

    def test_unset_removes_only_that_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.agent_chat"
            path.write_text("AGENT_CHAT_MAX_SEARCHES=3\nOTHER=1\n", encoding="utf-8")
            transaction = config_editor.ConfigEditTransaction(
                path, Config(_env_file=None)
            )
            self.assertTrue(transaction.unset("AGENT_CHAT_MAX_SEARCHES"))
            transaction.commit()
            self.assertFalse(transaction.unset("AGENT_CHAT_MAX_SEARCHES"))
            self.assertEqual(path.read_text(encoding="utf-8"), "OTHER=1\n")


class ListingTests(unittest.TestCase):
    def test_sources_unknown_and_unmanaged_keys(self) -> None:
        config = Config(_env_file=None)
        file_values = {
            "AGENT_CHAT_MAX_SEARCHES": "3",
            "AGENT_CHAT_TYPO_KEY": "1",
            "AGENT_CHAT_ENV_FILE": ".env.agent_chat",
            "EXA_API_KEY": "secret",
        }
        with patch.dict(os.environ, {"AGENT_CHAT_MAX_SEARCHES": "3"}, clear=True):
            listing = config_editor.build_config_list(
                config, file_values=file_values, owned={"AGENT_CHAT_MAX_SEARCHES"}
            )
        by_key = {entry.key: entry for entry in listing.entries}
        self.assertEqual(by_key["AGENT_CHAT_MAX_SEARCHES"].source, "file")
        self.assertEqual(by_key["AGENT_CHAT_MAX_MODEL_TURNS"].source, "default")
        self.assertEqual(by_key["AGENT_CHAT_DATA_DIR"].source, "default")
        self.assertTrue(by_key["AGENT_CHAT_DATA_DIR"].restart_required)
        self.assertFalse(by_key["AGENT_CHAT_MAX_SEARCHES"].restart_required)
        self.assertEqual(listing.unknown_keys, ["AGENT_CHAT_TYPO_KEY"])
        self.assertEqual(listing.cli_only_keys, ["AGENT_CHAT_ENV_FILE"])
        self.assertEqual(listing.unmanaged_keys, ["EXA_API_KEY"])
        self.assertEqual(
            sorted(entry.key for entry in listing.entries),
            sorted(config_editor.KEY_TO_FIELD),
        )

    def test_real_environment_wins_over_default(self) -> None:
        with patch.dict(os.environ, {"AGENT_CHAT_MAX_MODEL_TURNS": "9"}, clear=True):
            listing = config_editor.build_config_list(
                Config(_env_file=None), file_values={}, owned=set()
            )
        entry = next(
            item for item in listing.entries if item.key == "AGENT_CHAT_MAX_MODEL_TURNS"
        )
        self.assertEqual(entry.source, "env")

    def test_restart_marks_only_cover_real_settings(self) -> None:
        """A mark for a key no entry can carry would never reach a listing."""

        self.assertTrue(
            config_editor.RESTART_ONLY_KEYS <= set(config_editor.KEY_TO_FIELD)
        )


class RebuildTests(unittest.TestCase):
    def test_environment_wins_and_programmatic_values_survive(self) -> None:
        previous = Config(
            _env_file=None,
            agent_chat_max_searches=9,
            agent_chat_max_model_turns=4,
        )
        with patch.dict(os.environ, {"AGENT_CHAT_MAX_SEARCHES": "2"}, clear=True):
            rebuilt = config_editor.rebuild_config(previous)
        # The environment overrides the running value...
        self.assertEqual(rebuilt.agent_chat_max_searches, 2)
        # ...and a field the environment never mentioned keeps its value.
        self.assertEqual(rebuilt.agent_chat_max_model_turns, 4)

    def test_invalid_environment_value_raises_a_configuration_error(self) -> None:
        """pydantic must not leak: callers catch AgentChatError, not ValidationError."""

        # The "previous" config must be built before the bad value is in the
        # environment: Config itself reads it at construction time.
        previous = Config(_env_file=None)
        with (
            patch.dict(os.environ, {"AGENT_CHAT_MAX_SEARCHES": "nope"}, clear=True),
            self.assertRaises(ConfigurationError) as caught,
        ):
            config_editor.rebuild_config(previous)
        message = str(caught.exception)
        self.assertIn("无法重建配置", message)
        self.assertIn("agent_chat_max_searches", message)

    def test_default_values_carry_no_source_mark(self) -> None:
        """README and the legend document "no mark" as the built-in default."""

        values = {"AGENT_CHAT_MAX_SEARCHES": "3"}
        with patch.dict(os.environ, values, clear=True):
            listing = config_editor.build_config_list(
                Config(_env_file=None), file_values=values, owned=set(values)
            )
        entries = {entry.key: entry for entry in listing.entries}
        from_file = config_editor.format_entry(entries["AGENT_CHAT_MAX_SEARCHES"])
        from_default = config_editor.format_entry(entries["AGENT_CHAT_ENABLE_AT"])
        self.assertIn("[文件]", from_file)
        self.assertNotIn("[", from_default)
        self.assertNotIn("默认", from_default)

    def test_unset_keys_fall_back_to_the_default(self) -> None:
        previous = Config(_env_file=None, agent_chat_max_searches=9)
        with patch.dict(os.environ, {}, clear=True):
            kept = config_editor.rebuild_config(previous)
            dropped = config_editor.rebuild_config(
                previous, unset_keys={"AGENT_CHAT_MAX_SEARCHES"}
            )
        # Without the hint the running value survives (programmatic setup)...
        self.assertEqual(kept.agent_chat_max_searches, 9)
        # ...but a removed file line means "no override": the default applies.
        self.assertEqual(dropped.agent_chat_max_searches, 10)

    def test_sensitive_name_excludes_env_pointers(self) -> None:
        for name in ("api_key", "authorization", "BOT_TOKEN", "db_password"):
            self.assertTrue(config_editor.sensitive_name(name), name)
        for name in ("api_key_env", "exa_api_key_env", "AGENT_CHAT_MAX_SEARCHES"):
            self.assertFalse(config_editor.sensitive_name(name), name)

    def test_json_env_values_are_parsed(self) -> None:
        with patch.dict(
            os.environ, {"AGENT_CHAT_ALLOWED_GROUPS": '["QQClient:1"]'}, clear=True
        ):
            rebuilt = config_editor.rebuild_config(Config(_env_file=None))
        self.assertEqual(rebuilt.agent_chat_allowed_groups, {"QQClient:1"})


class TransactionTests(unittest.TestCase):
    def test_batch_stays_unwritten_until_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.agent_chat"
            path.write_text("# note\nAGENT_CHAT_MAX_SEARCHES=3\n", encoding="utf-8")
            transaction = config_editor.ConfigEditTransaction(
                path, Config(_env_file=None)
            )

            transaction.set("AGENT_CHAT_MAX_SEARCHES", "9")
            transaction.unset("AGENT_CHAT_MAX_SEARCHES")

            # Nothing is on disk until commit, so a later rejection is harmless.
            self.assertIn("AGENT_CHAT_MAX_SEARCHES=3", path.read_text(encoding="utf-8"))
            self.assertEqual(transaction.config.agent_chat_max_searches, 10)
            transaction.commit()
            text = path.read_text(encoding="utf-8")
            self.assertIn("# note", text)
            self.assertNotIn("AGENT_CHAT_MAX_SEARCHES", text)

    def test_failed_set_leaves_the_file_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env.agent_chat"
            path.write_text("AGENT_CHAT_MAX_SEARCHES=3\n", encoding="utf-8")
            transaction = config_editor.ConfigEditTransaction(
                path, Config(_env_file=None)
            )

            with self.assertRaises(InputError):
                transaction.set("AGENT_CHAT_MAX_SEARCHES", "nope")

            self.assertEqual(
                path.read_text(encoding="utf-8"), "AGENT_CHAT_MAX_SEARCHES=3\n"
            )

    def test_build_editing_config_drops_only_invalid_keys(self) -> None:
        config, invalid = config_editor.build_editing_config(
            {
                "AGENT_CHAT_MAX_SEARCHES": "7",
                "AGENT_CHAT_MESSAGE_CHUNK_CHARS": "100",
            }
        )

        self.assertEqual(config.agent_chat_max_searches, 7)
        self.assertEqual(config.agent_chat_message_chunk_chars, 1000)
        self.assertEqual(invalid, ["AGENT_CHAT_MESSAGE_CHUNK_CHARS"])


class EnvExampleContractTests(unittest.TestCase):
    def test_env_example_keys_are_real_settings(self) -> None:
        """The copyable example must not drift from the Config model."""

        from dotenv import dotenv_values

        repository = Path(__file__).resolve().parents[1]
        values = dotenv_values(repository / ".env.agent_chat.example")
        known = set(config_editor.KEY_TO_FIELD) | set(config_editor.CLI_ONLY_KEYS)
        unknown = sorted(
            key for key in values if key.startswith("AGENT_CHAT_") and key not in known
        )
        self.assertEqual(unknown, [])


class DiffTests(unittest.TestCase):
    def test_diff_splits_applied_and_restart_required(self) -> None:
        old = Config(_env_file=None)
        new = Config(
            _env_file=None,
            agent_chat_max_searches=5,
            agent_chat_data_dir="elsewhere",
        )
        applied, restart = config_editor.diff_configs(old, new)
        self.assertEqual(applied, {"AGENT_CHAT_MAX_SEARCHES": ("10", "5")})
        self.assertEqual(
            restart, {"AGENT_CHAT_DATA_DIR": ("data/agent_chat", "elsewhere")}
        )
        lines = config_editor.format_report(applied, restart)
        self.assertTrue(any("AGENT_CHAT_MAX_SEARCHES" in line for line in lines))
        self.assertTrue(any("需重启" in line for line in lines))

    def test_diff_json_serialisable(self) -> None:
        old = Config(_env_file=None)
        new = Config(_env_file=None, agent_chat_max_searches=5)
        applied, _ = config_editor.diff_configs(old, new)
        self.assertIn("AGENT_CHAT_MAX_SEARCHES", json.dumps({"applied": applied}))


if __name__ == "__main__":
    unittest.main()
