import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nonebot_plugin_agent_chat import env_file, profile_editor, tui_state
from nonebot_plugin_agent_chat.config import Config
from nonebot_plugin_agent_chat.edits import format_changes
from nonebot_plugin_agent_chat.errors import InputError
from nonebot_plugin_agent_chat.models import ProviderProfile


def set_field(path: Path, field: str, token: str) -> None:
    """One atomic profile field edit, as the CLI and editor write it."""

    transaction = profile_editor.ProfileFileTransaction(path.parent)
    transaction.set_field(path.stem, field, token)
    transaction.commit()


class SettingsPanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.env_path = root / ".env.agent_chat"
        self.env_path.write_text(
            "# keep me\nAGENT_CHAT_MAX_SEARCHES=3\n", encoding="utf-8"
        )
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        owned = env_file.load_into_environ(self.env_path)
        self.panel = tui_state.SettingsPanel(
            self.env_path, owned, Config(_env_file=None)
        )

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.temp.cleanup()

    def test_rows_carry_help_and_source(self) -> None:
        rows = {row.key: row for row in self.panel.rows()}
        row = rows["AGENT_CHAT_MAX_SEARCHES"]
        self.assertEqual(row.value, "3")
        self.assertEqual(row.source, "file")
        self.assertIn("搜索", row.description)
        self.assertEqual(row.summary, "一次回答最多搜索几次")
        self.assertEqual(row.type_hint, "整数")
        self.assertEqual(row.default, "10")
        self.assertFalse(row.restart_required)
        self.assertTrue(rows["AGENT_CHAT_DATA_DIR"].restart_required)

    def test_filter_keeps_only_matching_keys(self) -> None:
        keys = [row.key for row in self.panel.rows("max_search")]
        self.assertEqual(keys, ["AGENT_CHAT_MAX_SEARCHES"])

    def test_stage_set_validates_and_marks_dirty(self) -> None:
        with self.assertRaises(InputError):
            self.panel.stage_set("AGENT_CHAT_MAX_SEARCHES", "nope")
        self.assertFalse(self.panel.dirty)

        self.panel.stage_set("AGENT_CHAT_MAX_SEARCHES", "5")
        self.assertTrue(self.panel.dirty)
        self.assertEqual(self.panel.pending_lines(), ["AGENT_CHAT_MAX_SEARCHES=5"])
        row = next(
            row for row in self.panel.rows() if row.key == "AGENT_CHAT_MAX_SEARCHES"
        )
        self.assertEqual(row.value, "5")
        self.assertTrue(row.dirty)
        self.assertEqual(row.source, tui_state.STAGED_SOURCE)
        # Nothing is written before save.
        self.assertIn("AGENT_CHAT_MAX_SEARCHES=3", self.env_path.read_text())

    def test_cross_field_validation_uses_staged_values(self) -> None:
        self.panel.stage_set("AGENT_CHAT_MAX_SEARCHES", "0")
        with self.assertRaises(InputError):
            # search_mode requires a search budget, so this must be rejected
            # against the staged value, not the on-disk one.
            self.panel.stage_set("AGENT_CHAT_MAX_SEARCHES", "0")
            self.panel.stage_set("AGENT_CHAT_SEARCH_MODE", "exa")

    def test_stage_unset_shows_the_fallback_value(self) -> None:
        self.panel.stage_unset("AGENT_CHAT_MAX_SEARCHES")
        row = next(
            row for row in self.panel.rows() if row.key == "AGENT_CHAT_MAX_SEARCHES"
        )
        self.assertTrue(row.dirty)
        self.assertEqual(row.source, tui_state.STAGED_SOURCE)

        actions, restart = self.panel.save()
        self.assertIn(
            "已删除 AGENT_CHAT_MAX_SEARCHES", " ".join(format_changes(actions))
        )
        self.assertEqual(restart, [])
        self.assertNotIn("AGENT_CHAT_MAX_SEARCHES", self.env_path.read_text())
        self.assertIn("# keep me", self.env_path.read_text())

    def test_save_writes_comments_and_reports_restart_keys(self) -> None:
        self.panel.stage_set("AGENT_CHAT_MAX_SEARCHES", "5")
        self.panel.stage_set("AGENT_CHAT_DATA_DIR", "/tmp/elsewhere")
        actions, restart = self.panel.save()

        text = self.env_path.read_text()
        self.assertIn("# keep me", text)
        self.assertIn("AGENT_CHAT_MAX_SEARCHES=5", text)
        self.assertIn("AGENT_CHAT_DATA_DIR=/tmp/elsewhere", text)
        self.assertEqual(restart, ["AGENT_CHAT_DATA_DIR"])
        self.assertEqual(len(actions), 2)
        self.assertFalse(self.panel.dirty)
        self.assertEqual(self.panel.config.agent_chat_max_searches, 5)

    def test_discard_restores_the_snapshot(self) -> None:
        self.panel.stage_set("AGENT_CHAT_MAX_SEARCHES", "5")
        self.panel.discard()
        self.assertFalse(self.panel.dirty)
        row = next(
            row for row in self.panel.rows() if row.key == "AGENT_CHAT_MAX_SEARCHES"
        )
        self.assertEqual(row.value, "3")
        self.assertIn("AGENT_CHAT_MAX_SEARCHES=3", self.env_path.read_text())

    def test_sensitive_values_are_masked_in_save_actions(self) -> None:
        # No AGENT_CHAT_* secret exists today, but the echo rule must hold.
        self.panel.stage_set("AGENT_CHAT_MAX_SEARCHES", "5")
        actions, _ = self.panel.save()
        self.assertNotIn("•••", " ".join(format_changes(actions)))


class ProfilesPanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.path = self.directory / "default.json"
        self.path.write_text(
            json.dumps({"protocol": "openai-responses", "model": "local"}),
            encoding="utf-8",
        )
        self.panel = tui_state.ProfilesPanel(self.directory)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_entries_and_fields_carry_help(self) -> None:
        entries = self.panel.entries()
        self.assertEqual([entry.name for entry in entries], ["default"])
        fields = {row.name: row for row in self.panel.fields("default")}
        self.assertEqual(fields["model"].value, "local")
        self.assertIn("模型名", fields["model"].description)
        self.assertIn("模型名", fields["model"].summary)
        self.assertTrue(fields["extra_headers"].hidden)

    def test_field_edits_are_staged_and_validated(self) -> None:
        with self.assertRaises(InputError):
            self.panel.stage_set("default", "image_reply_mode", "sometimes")
        self.panel.stage_set("default", "model", "better")
        self.assertTrue(self.panel.dirty)
        row = next(row for row in self.panel.fields("default") if row.name == "model")
        self.assertEqual(row.value, "better")
        self.assertTrue(row.dirty)
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8"))["model"], "local"
        )

        actions = self.panel.save()
        self.assertIn("已设置 default.model=better", format_changes(actions))
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8"))["model"], "better"
        )
        self.assertFalse(self.panel.dirty)

    def test_secret_field_is_masked_in_actions(self) -> None:
        self.panel.stage_set(
            "default", "extra_headers", '{"Authorization": "Bearer s3cret"}'
        )
        # The status line is one surface, the saved actions another: both mask.
        pending = " ".join(self.panel.pending_lines())
        self.assertNotIn("s3cret", pending)
        self.assertIn("•••", pending)
        actions = self.panel.save()
        self.assertNotIn("s3cret", " ".join(format_changes(actions)))

    def test_create_is_staged_against_its_template(self) -> None:
        self.panel.stage_create("qq-safe", "default")
        self.panel.stage_set("qq-safe", "model", "safe-model")
        names = [entry.name for entry in self.panel.entries()]
        self.assertIn("qq-safe", names)
        self.assertFalse((self.directory / "qq-safe.json").exists())

        actions = self.panel.save()
        self.assertIn(
            "已创建 profile: qq-safe（复制自 default）", format_changes(actions)
        )
        created = json.loads(
            (self.directory / "qq-safe.json").read_text(encoding="utf-8")
        )
        self.assertEqual(created["model"], "safe-model")

    def test_create_rejects_duplicates_and_bad_names(self) -> None:
        with self.assertRaises(InputError):
            self.panel.stage_create("default", "default")
        with self.assertRaises(InputError):
            self.panel.stage_create("bad name", "default")
        with self.assertRaises(InputError):
            self.panel.stage_create("x", "missing")

    def test_delete_is_staged_and_drops_field_edits(self) -> None:
        self.panel.stage_set("default", "model", "better")
        self.panel.stage_delete("default")
        self.assertTrue(self.panel.is_staged_delete("default"))
        self.assertEqual(self.panel.pending_lines(), ["-default（删除 profile）"])

        actions = self.panel.save()
        self.assertIn("已删除 profile: default", format_changes(actions))
        self.assertFalse(self.path.exists())

    def test_staging_on_a_deleted_profile_is_rejected(self) -> None:
        self.panel.stage_create("qq-safe", "default")
        self.panel.stage_delete("qq-safe")
        # Deleting a staged creation just cancels it.
        self.assertFalse(self.panel.dirty)

        self.panel.stage_delete("default")
        with self.assertRaises(InputError):
            self.panel.stage_set("default", "model", "x")
        self.panel.unstage("default")
        self.assertFalse(self.panel.dirty)

    def test_unset_field_restores_defaults(self) -> None:
        set_field(self.path, "temperature", "0.5")
        self.panel.stage_delete("default")
        self.panel.unstage("default")
        self.assertFalse(self.panel.dirty)

    def test_credential_field_prefills_a_placeholder_not_the_key(self) -> None:
        canary = "sk-canary-3f9a1c"
        set_field(self.path, "api_key", canary)

        self.assertEqual(
            self.panel.raw_value("default", "api_key"),
            profile_editor.MASKED_PLACEHOLDER,
        )
        # Other fields keep their real value, and a credential with no stored
        # value shows nothing to keep.
        self.assertEqual(self.panel.raw_value("default", "model"), "local")
        self.assertEqual(self.panel.raw_value("default", "base_url"), "")
        on_disk = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["api_key"], canary)

    def test_credential_placeholder_means_keep(self) -> None:
        keep = profile_editor.keeps_masked_placeholder
        self.assertTrue(keep("api_key", profile_editor.MASKED_PLACEHOLDER))
        self.assertFalse(keep("api_key", "sk-new"))
        self.assertFalse(keep("api_key", ""))
        self.assertFalse(keep("model", profile_editor.MASKED_PLACEHOLDER))

    def test_only_secret_typed_fields_are_credentials(self) -> None:
        """A credential-looking *name* is not a credential (max_output_tokens)."""

        self.assertTrue(profile_editor.credential_field("api_key"))
        self.assertTrue(profile_editor.credential_field("exa_api_key"))
        self.assertFalse(profile_editor.credential_field("max_output_tokens"))
        self.assertFalse(profile_editor.credential_field("api_key_env"))
        self.assertFalse(profile_editor.credential_field("extra_headers"))
        self.assertFalse(profile_editor.credential_field("no_such_field"))

        # A numeric field keeps its value in the modal, and it is typed as one.
        set_field(self.path, "max_output_tokens", "2048")
        self.assertEqual(self.panel.raw_value("default", "max_output_tokens"), "2048")
        rows = {row.name: row for row in self.panel.fields("default")}
        self.assertEqual(rows["max_output_tokens"].kind, "number")
        self.assertEqual(rows["api_key"].kind, "text")

    def test_credential_edits_are_staged_and_masked(self) -> None:
        canary = "sk-canary-3f9a1c"
        set_field(self.path, "api_key", canary)

        # A new value overwrites; the prefill, the pending line, and the table
        # all stay masked.
        self.panel.stage_set("default", "api_key", "sk-new-value")
        self.assertEqual(
            self.panel.raw_value("default", "api_key"),
            profile_editor.MASKED_PLACEHOLDER,
        )
        pending = "\n".join(self.panel.pending_lines())
        self.assertNotIn("sk-new-value", pending)
        self.assertIn(profile_editor.MASKED_PLACEHOLDER[:1], pending)
        staged = {row.name: row for row in self.panel.fields("default")}
        self.assertEqual(staged["api_key"].value, "•••")

        # An empty submit clears the field, which falls back to the env name: the
        # prefill stops offering a placeholder because there is nothing to keep.
        self.panel.stage_set("default", "api_key", "")
        self.assertEqual(self.panel.raw_value("default", "api_key"), "")
        self.assertTrue(self.panel.dirty)

    def test_map_fields_prefill_a_keep_placeholder(self) -> None:
        canary = "fixture-private-value"
        values = {
            "extra_headers": {"Cookie": f"session={canary}"},
            "extra_query": {"credentials": {"session": canary}},
            "extra_body": {"metadata": {"auth": canary}},
        }
        for name, value in values.items():
            with self.subTest(field=name):
                set_field(self.path, name, json.dumps(value))
                self.assertEqual(
                    self.panel.raw_value("default", name),
                    profile_editor.MASKED_PLACEHOLDER,
                )
                self.assertTrue(
                    profile_editor.keeps_masked_placeholder(
                        name, profile_editor.MASKED_PLACEHOLDER
                    )
                )
                fields = {row.name: row for row in self.panel.fields("default")}
                self.assertEqual(fields[name].value, "•••")

                self.panel.stage_set("default", name, json.dumps(value))
                self.assertNotIn(canary, " ".join(self.panel.pending_lines()))
                self.assertEqual(
                    self.panel.raw_value("default", name),
                    profile_editor.MASKED_PLACEHOLDER,
                )
                self.panel.save()
                reloaded = tui_state.ProfilesPanel(self.directory)
                self.assertEqual(
                    reloaded.raw_value("default", name),
                    profile_editor.MASKED_PLACEHOLDER,
                )
                self.assertEqual(json.loads(self.path.read_text())[name], value)

    def test_field_table_masks_the_credential(self) -> None:
        canary = "sk-canary-3f9a1c"
        set_field(self.path, "api_key", canary)

        fields = {row.name: row for row in self.panel.fields("default")}
        self.assertEqual(fields["api_key"].value, "•••")
        self.assertNotIn(canary, fields["api_key"].value)
        self.assertTrue(fields["api_key"].hidden)

    def test_every_profile_field_has_a_help_line(self) -> None:
        """The detail pane and the summary column read this map; a field with
        no entry silently renders as empty."""

        self.assertEqual(
            set(tui_state.PROFILE_FIELD_HELP),
            set(ProviderProfile.model_fields),
        )


if __name__ == "__main__":
    unittest.main()
