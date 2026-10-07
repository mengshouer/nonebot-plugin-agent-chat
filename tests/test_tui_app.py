from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nonebot_plugin_agent_chat import env_file, profile_editor, tui_state
from nonebot_plugin_agent_chat.config import Config

# textual is an optional extra: skip these tests when it is missing, like the
# renderer tests do for playwright.
HAS_TEXTUAL = importlib.util.find_spec("textual") is not None


class ReloadRecorder:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


@unittest.skipUnless(HAS_TEXTUAL, "textual is not installed ([tui] extra)")
class EditorAppTests(unittest.IsolatedAsyncioTestCase):
    """Drive the real app headlessly through textual's pilot."""

    async def asyncSetUp(self) -> None:
        # textual's message pumps trip asyncio's slow-callback warnings.
        asyncio.get_running_loop().set_debug(False)
        from textual.widgets import (
            DataTable,
            Input,
            Static,
            TabbedContent,
            TextArea,
        )

        from nonebot_plugin_agent_chat import tui_app

        self.DataTable = DataTable
        self.Input = Input
        self.Static = Static
        self.TabbedContent = TabbedContent
        self.TextArea = TextArea
        self.EditValueModal = tui_app.EditValueModal
        self.ConfirmModal = tui_app.ConfirmModal
        self.ConfirmNameModal = tui_app.ConfirmNameModal
        self.NewProfileModal = tui_app.NewProfileModal
        self.EditorApp = tui_app.EditorApp
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.profiles = root / "profiles"
        self.profiles.mkdir()
        (self.profiles / "default.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "local"}),
            encoding="utf-8",
        )
        self.env_path = root / ".env.agent_chat"
        self.env_path.write_text("AGENT_CHAT_MAX_SEARCHES=3\n", encoding="utf-8")
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.recorder = ReloadRecorder()
        self.warnings: list[str] = []
        self.context = tui_state.EditorContext(
            settings=tui_state.SettingsPanel(
                self.env_path,
                env_file.load_into_environ(self.env_path),
                Config(_env_file=None, agent_chat_max_searches=3),
            ),
            profiles=tui_state.ProfilesPanel(self.profiles),
            request_reload=self.recorder,
            warnings_for_delete=self._warnings,
        )
        self.app = self.EditorApp(self.context)

    async def _warnings(self, name: str) -> list[str]:
        self.warnings.append(name)
        return [f"规则引用 {name}"]

    async def asyncTearDown(self) -> None:
        self.env_patch.stop()
        self.temp.cleanup()

    def _table(self, widget_id: str):
        return self.app.query_one(f"#{widget_id}", self.DataTable)

    async def _filter(self, pilot, needle: str) -> None:
        self.app.query_one("#filter", self.Input).value = needle
        await self._settle(pilot)

    @staticmethod
    async def _settle(pilot) -> None:
        """Let workers, messages, and modal pushes land before asserting."""

        for _ in range(4):
            await pilot.pause()

    def _footer_text(self) -> str:
        """What the footer actually renders: `str(Footer(...))` is only its repr."""

        footer = self.app.query_one("#footer")
        return " ".join(str(child.render()) for child in footer.children)

    async def test_lists_show_the_purpose_column(self) -> None:
        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            settings_table = self._table("settings-table")
            labels = [str(column.label) for column in settings_table.columns.values()]
            self.assertEqual(labels, ["键 / 说明", "当前值", "生效"])

            self.app.query_one(self.TabbedContent).active = "profiles"
            await self._settle(pilot)
            fields_table = self._table("fields-table")
            labels = [str(column.label) for column in fields_table.columns.values()]
            self.assertEqual(labels, ["字段 / 说明", "当前值"])

    def test_column_widths_follow_the_terminal(self) -> None:
        from nonebot_plugin_agent_chat.tui_app import column_widths, summary_limit

        narrow = column_widths(80)
        wide = column_widths(160)
        self.assertLess(narrow[0], wide[0])  # the label column spreads out
        self.assertLess(narrow[1], wide[1])
        self.assertEqual(narrow[2], wide[2])  # the effect mark stays fixed
        # Everything must stay inside the terminal (padding included).
        for width in (80, 100, 140, 200):
            label, value, effect = column_widths(width)
            self.assertLessEqual(label + value + effect, width)
            self.assertGreaterEqual(label, 26)
        self.assertGreater(summary_limit(wide[0]), summary_limit(narrow[0]))

    async def test_resizing_widens_the_columns(self) -> None:
        async with self.app.run_test(size=(80, 30)) as pilot:
            await self._settle(pilot)
            table = self._table("settings-table")
            narrow = next(iter(table.columns.values())).width

            await pilot.resize_terminal(150, 30)
            await self._settle(pilot)
            self.assertGreater(next(iter(table.columns.values())).width, narrow)

    async def test_modal_ctrl_s_submits(self) -> None:
        """Ctrl+S inside the editor saves the value (the app save must not eat it)."""

        async with self.app.run_test() as pilot:
            await self._filter(pilot, "max_searches")
            self._table("settings-table").focus()
            await pilot.press("enter")
            await self._settle(pilot)
            self.app.screen.query_one("#modal-value", self.TextArea).text = "9"
            await pilot.press("ctrl+s")
            await self._settle(pilot)
            self.assertEqual(self.context.settings.pending_count, 1)

    async def test_modal_ctrl_enter_submits(self) -> None:
        """Ctrl+Enter is the fallback where ^S means XOFF."""

        async with self.app.run_test() as pilot:
            await self._filter(pilot, "max_searches")
            self._table("settings-table").focus()
            await pilot.press("enter")
            await self._settle(pilot)
            self.app.screen.query_one("#modal-value", self.TextArea).text = "8"
            await pilot.press("ctrl+enter")
            await self._settle(pilot)
            self.assertEqual(self.context.settings.pending_count, 1)

    async def test_edit_modal_is_centred_and_roomy(self) -> None:
        async with self.app.run_test(size=(100, 30)) as pilot:
            await self._filter(pilot, "max_searches")
            self._table("settings-table").focus()
            await pilot.press("enter")
            await self._settle(pilot)

            box = self.app.screen.query_one("#modal-box")
            expected_left = (100 - box.region.width) // 2
            expected_top = (30 - box.region.height) // 2
            self.assertLessEqual(abs(box.region.x - expected_left), 2)
            self.assertLessEqual(abs(box.region.y - expected_top), 2)
            area = self.app.screen.query_one("#modal-value", self.TextArea)
            self.assertGreaterEqual(area.region.height, 6)
            self.assertGreaterEqual(box.region.width, 60)
            # The current value is selected, so typing replaces it.
            self.assertEqual(area.selected_text, "3")
            await pilot.press("escape")
            await self._settle(pilot)

    async def test_long_whitelist_value_round_trips(self) -> None:
        entry = (
            '["QQClient:598683145", "QQClient:625355153", '
            '"QQClient:1026647706", "Telegram:487037110"]'
        )
        async with self.app.run_test(size=(100, 30)) as pilot:
            await self._filter(pilot, "ALLOWED_GROUPS")
            self._table("settings-table").focus()
            await pilot.press("enter")
            await self._settle(pilot)
            self.app.screen.query_one("#modal-value", self.TextArea).text = entry
            await pilot.click("#modal-ok")
            await self._settle(pilot)
            await pilot.press("ctrl+s")
            await self._settle(pilot)

        # The writer quotes values that need it; what matters is the round trip.
        written = env_file.read_values(self.env_path)
        self.assertEqual(written["AGENT_CHAT_ALLOWED_GROUPS"], entry)
        self.assertIn("AGENT_CHAT_ALLOWED_GROUPS=", self.env_path.read_text("utf-8"))

    async def test_rows_are_styled_and_spaced(self) -> None:
        """Keys are bold, summaries dim, marks coloured, items separated."""

        from nonebot_plugin_agent_chat.tui_app import setting_cells

        row = next(
            item
            for item in self.context.settings.rows()
            if item.key == "AGENT_CHAT_MAX_SEARCHES"
        )
        label, value, effect = setting_cells(row, show_summary=True)
        self.assertEqual(str(label).splitlines()[0], "AGENT_CHAT_MAX_SEARCHES")
        # The base style is the key's; the summary/source are appended spans.
        self.assertIn("bold", str(label.style))
        spans = {span.start: str(span.style) for span in label.spans}
        self.assertIn("dim", spans[len(row.key) + 1])
        self.assertEqual(len(label.spans), 2)
        self.assertEqual(str(label).count("\n"), 2)  # summary line + gap
        self.assertEqual(effect.plain, "热生效")
        self.assertEqual(str(effect.style), "dim")
        self.assertEqual(value.plain, "3")

        restart = next(
            item
            for item in self.context.settings.rows()
            if item.key == "AGENT_CHAT_PRIORITY"
        )
        _, _, restart_effect = setting_cells(restart, show_summary=True)
        self.assertEqual(restart_effect.plain, "需重启")
        self.assertIn("yellow", str(restart_effect.style))

    async def test_staged_value_shows_the_unsaved_source(self) -> None:
        """A staged row's source label comes from the shared label map."""

        from nonebot_plugin_agent_chat.tui_app import setting_cells

        self.context.settings.stage_set("AGENT_CHAT_MAX_SEARCHES", "5")
        row = next(
            item
            for item in self.context.settings.rows()
            if item.key == "AGENT_CHAT_MAX_SEARCHES"
        )
        self.assertEqual(row.source, tui_state.STAGED_SOURCE)
        self.assertTrue(row.dirty)
        label, value, _ = setting_cells(row, show_summary=True)
        self.assertIn("（来源：未保存）", str(label))
        self.assertEqual(value.plain, "5")

    async def test_credential_field_never_reaches_the_editor(self) -> None:
        """The modal shows a placeholder, and the stored key stays nowhere."""

        canary = "sk-canary-3f9a1c"
        (self.profiles / "keys.json").write_text(
            json.dumps(
                {
                    "protocol": "openai-responses",
                    "model": "local",
                    "api_key": canary,
                }
            ),
            encoding="utf-8",
        )

        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            # The fields table only receives keys while its tab is showing.
            self.app.query_one(self.TabbedContent).active = "profiles"
            await self._settle(pilot)
            self.app.refresh_profiles(select_name="keys")
            await self._settle(pilot)

            fields = self._table("fields-table")
            fields.move_cursor(row=fields.get_row_index("api_key"))
            await self._settle(pilot)
            self.assertNotIn(canary, await self._rendered())

            fields.focus()
            await pilot.press("enter")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, self.EditValueModal)
            field = self.app.screen.query_one("#modal-input", self.Input)
            self.assertEqual(field.value, profile_editor.MASKED_PLACEHOLDER)
            self.assertNotIn(canary, await self._rendered())

            # Keeping the placeholder saves nothing at all.
            await pilot.click("#modal-ok")
            await self._settle(pilot)
            self.assertFalse(self.context.profiles.dirty)
            self.assertEqual(self.context.profiles.pending_lines(), [])
            self.assertNotIn(canary, await self._rendered())

    async def test_sensitive_maps_stay_hidden_and_unchanged_in_editor(self) -> None:
        canary = "fixture-private-value"
        values = {
            "extra_headers": {"Cookie": f"session={canary}"},
            "extra_query": {"credentials": {"session": canary}},
            "extra_body": {"metadata": {"auth": canary}},
        }
        path = self.profiles / "default.json"
        path.write_text(
            json.dumps({"protocol": "openai-responses", "model": "local", **values}),
            encoding="utf-8",
        )
        original = path.read_bytes()
        async with self.app.run_test() as pilot:
            self.app.query_one(self.TabbedContent).active = "profiles"
            await self._settle(pilot)
            for name in values:
                with self.subTest(field=name):
                    fields = self._table("fields-table")
                    fields.move_cursor(row=fields.get_row_index(name))
                    fields.focus()
                    await self._settle(pilot)
                    self.assertNotIn(canary, await self._rendered())
                    await pilot.press("enter")
                    await self._settle(pilot)
                    field = self.app.screen.query_one("#modal-input", self.Input)
                    self.assertEqual(field.value, profile_editor.MASKED_PLACEHOLDER)
                    self.assertNotIn(canary, await self._rendered())
                    await pilot.click("#modal-ok")
                    await self._settle(pilot)
                    self.assertFalse(self.context.profiles.dirty)
                    self.assertNotIn(canary, await self._rendered())
            await pilot.press("ctrl+s")
            await self._settle(pilot)
        self.assertEqual(path.read_bytes(), original)

    async def _rendered(self) -> str:
        """Every widget's rendered text: what a screenshot or pty would show."""

        parts = []
        for widget in self.app.query("*"):
            render = getattr(widget, "render", None)
            parts.append(str(render()) if callable(render) else str(widget))
        return "\n".join(parts)

    async def test_rows_use_three_lines_then_compact(self) -> None:
        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            table = self._table("settings-table")
            from textual.widgets.data_table import RowKey

            first_key = RowKey(next(iter(table.rows)))
            self.assertTrue(table.zebra_stripes)  # alternating backgrounds
            self.assertEqual(table.get_row_height(first_key), 3)

            await pilot.press("i")
            await self._settle(pilot)
            self.assertEqual(table.get_row_height(first_key), 1)

    async def test_summary_column_can_be_toggled(self) -> None:
        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            table = self._table("settings-table")
            labels = [str(column.label) for column in table.columns.values()]
            self.assertEqual(labels, ["键 / 说明", "当前值", "生效"])

            await pilot.press("i")
            await self._settle(pilot)
            labels = [str(column.label) for column in table.columns.values()]
            self.assertEqual(labels, ["键", "当前值", "生效"])

            await pilot.press("i")
            await self._settle(pilot)
            labels = [str(column.label) for column in table.columns.values()]
            self.assertEqual(labels, ["键 / 说明", "当前值", "生效"])

    def test_run_async_is_awaited_not_context_managed(self) -> None:
        """textual's run_async() is a coroutine; a pty smoke caught this once."""

        import inspect

        from textual.app import App

        self.assertTrue(inspect.iscoroutinefunction(App.run_async))

    async def test_settings_edit_and_save(self) -> None:
        async with self.app.run_test() as pilot:
            await self._filter(pilot, "max_searches")
            table = self._table("settings-table")
            self.assertEqual(table.row_count, 1)
            self.assertIn(
                "搜索", str(self.app.query_one("#setting-detail", self.Static).content)
            )

            table.focus()
            await pilot.press("enter")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, self.EditValueModal)
            self.app.screen.query_one("#modal-value", self.TextArea).text = "5"
            await pilot.click("#modal-ok")
            await self._settle(pilot)

            self.assertEqual(self.context.settings.pending_count, 1)
            self.assertIn(
                "未保存 1 项", str(self.app.query_one("#status", self.Static).content)
            )
            await pilot.press("ctrl+s")
            await self._settle(pilot)

        self.assertEqual(self.recorder.calls, 1)
        text = self.env_path.read_text(encoding="utf-8")
        self.assertIn("AGENT_CHAT_MAX_SEARCHES=5", text)
        self.assertFalse(self.context.settings.dirty)

    async def test_invalid_value_keeps_the_modal_and_the_text(self) -> None:
        async with self.app.run_test() as pilot:
            await self._filter(pilot, "max_searches")
            self._table("settings-table").focus()
            await pilot.press("enter")
            await self._settle(pilot)
            self.app.screen.query_one("#modal-value", self.TextArea).text = "nope"
            await pilot.click("#modal-ok")
            await self._settle(pilot)

            # Nothing staged, the file untouched, and the editor still open
            # with the operator's text plus the reason.
            self.assertEqual(self.context.settings.pending_count, 0)
            self.assertIn(
                "AGENT_CHAT_MAX_SEARCHES=3", self.env_path.read_text(encoding="utf-8")
            )
            self.assertIsInstance(self.app.screen, self.EditValueModal)
            self.assertIn(
                "设置无效",
                str(self.app.screen.query_one("#modal-error", self.Static).content),
            )
            self.assertEqual(
                self.app.screen.query_one("#modal-value", self.TextArea).text, "nope"
            )
            await pilot.press("escape")
            await self._settle(pilot)
            self.assertNotIsInstance(self.app.screen, self.EditValueModal)

    async def test_left_right_moves_between_the_profiles_panes(self) -> None:
        """The arrows belong to the two-column pane, not to the tab strip."""

        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            tabs = self.app.query_one(self.TabbedContent)
            tabs.active = "profiles"
            await self._settle(pilot)

            profiles_table = self._table("profiles-table")
            profiles_table.focus()
            await pilot.press("right")
            await self._settle(pilot)
            self.assertEqual(self.app.focused, self._table("fields-table"))
            self.assertEqual(tabs.active, "profiles")

            await pilot.press("left")
            await self._settle(pilot)
            self.assertEqual(tabs.active, "profiles")
            self.assertEqual(self.app.focused, profiles_table)
            self.assertEqual(tabs.active, "profiles")

    async def test_the_footer_only_offers_live_keys(self) -> None:
        """The settings tab has one column, so it must not advertise 左栏/右栏."""

        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            self._table("settings-table").focus()
            await self._settle(pilot)
            settings_footer = self._footer_text()
            # An empty read would make the assertion below vacuously true.
            self.assertTrue(settings_footer, "the footer rendered nothing")
            self.assertNotIn("左栏", settings_footer)

            self.app.query_one(self.TabbedContent).active = "profiles"
            await self._settle(pilot)
            self._table("profiles-table").focus()
            await self._settle(pilot)
            profiles_footer = self._footer_text()
            self.assertIn("左栏", profiles_footer)
            self.assertIn("右栏", profiles_footer)
            self.assertIn("上一页", profiles_footer)

    async def test_left_right_does_not_switch_tabs(self) -> None:
        """Switching tabs moved to Ctrl+arrows, so a list must not steal them."""

        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            tabs = self.app.query_one(self.TabbedContent)
            self._table("settings-table").focus()

            await pilot.press("right", "left")
            await self._settle(pilot)

            self.assertEqual(tabs.active, "settings")

    async def test_left_right_stay_in_a_text_field(self) -> None:
        """In the filter box the caret wins: the arrows never leave it."""

        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            tabs = self.app.query_one(self.TabbedContent)
            filter_box = self.app.query_one("#filter", self.Input)
            filter_box.focus()
            await pilot.press(*"max_search")
            await self._settle(pilot)
            self.assertEqual(filter_box.value, "max_search")

            await pilot.press("left", "left")
            await self._settle(pilot)
            self.assertEqual(tabs.active, "settings")
            self.assertEqual(filter_box.cursor_position, len("max_search") - 2)

    async def test_ctrl_left_right_switch_tabs_outside_text_fields(self) -> None:
        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            tabs = self.app.query_one(self.TabbedContent)

            self._table("settings-table").focus()
            await pilot.press("ctrl+right")
            await self._settle(pilot)
            self.assertEqual(tabs.active, "profiles")
            # Switching must leave the new list usable straight away.
            self.assertEqual(self.app.focused, self._table("profiles-table"))
            await pilot.press("ctrl+left")
            await self._settle(pilot)
            self.assertEqual(tabs.active, "settings")
            self.assertEqual(self.app.focused, self._table("settings-table"))

            # In a text field Ctrl+arrows mean word-wise caret movement.
            self.app.query_one("#filter", self.Input).focus()
            await pilot.press("ctrl+right")
            await self._settle(pilot)
            self.assertEqual(tabs.active, "settings")

    async def test_the_tab_strip_switches_tabs_with_its_own_arrows(self) -> None:
        """README: with the tab strip focused, plain ←/→ switch tabs."""

        from textual.widgets import Tabs

        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            tabs = self.app.query_one(self.TabbedContent)
            self.app.query_one(Tabs).focus()

            await pilot.press("right")
            await self._settle(pilot)
            self.assertEqual(tabs.active, "profiles")
            # Activation hands the focus to the new list, so another switch needs
            # the strip back (Tab) or the Ctrl+arrows.
            self.assertEqual(self.app.focused, self._table("profiles-table"))

            self.app.query_one(Tabs).focus()
            await pilot.press("left")
            await self._settle(pilot)
            self.assertEqual(tabs.active, "settings")

    async def test_profiles_only_mode_saves(self) -> None:
        """`--no-env` without an env file has no settings panel, but still saves."""

        context = tui_state.EditorContext(
            settings=None,
            profiles=tui_state.ProfilesPanel(self.profiles),
            request_reload=self.recorder,
            warnings_for_delete=self._warnings,
        )
        app = self.EditorApp(context)
        async with app.run_test() as pilot:
            await self._settle(pilot)
            profiles_table = app.query_one("#profiles-table", self.DataTable)
            profiles_table.move_cursor(row=0)
            await self._settle(pilot)
            fields = app.query_one("#fields-table", self.DataTable)
            fields.focus()
            index = next(
                position
                for position, key in enumerate(fields.rows)
                if getattr(key, "value", None) == "model"
            )
            fields.move_cursor(row=index)
            await self._settle(pilot)
            self.assertEqual(app._selected_field().name, "model")

            await pilot.press("enter")
            await self._settle(pilot)
            app.screen.query_one("#modal-value", self.TextArea).text = "faster"
            await pilot.click("#modal-ok")
            await self._settle(pilot)
            self.assertEqual(context.profiles.pending_count, 1)

            await pilot.press("ctrl+s")
            await self._settle(pilot)

        self.assertEqual(self.recorder.calls, 1)
        self.assertFalse(context.profiles.dirty)
        written = json.loads(
            (self.profiles / "default.json").read_text(encoding="utf-8")
        )
        self.assertEqual(written["model"], "faster")

    async def test_tab_switching_is_a_no_op_without_a_settings_panel(self) -> None:
        context = tui_state.EditorContext(
            settings=None,
            profiles=tui_state.ProfilesPanel(self.profiles),
            request_reload=self.recorder,
            warnings_for_delete=self._warnings,
        )
        app = self.EditorApp(context)
        async with app.run_test() as pilot:
            await self._settle(pilot)
            await pilot.press("right", "left", "ctrl+right")
            await self._settle(pilot)
            self.assertEqual(app.query_one(self.TabbedContent).active, "profiles")
            self.assertTrue(app.is_running)

    async def test_ctrl_enter_saves_as_well(self) -> None:
        """The app-level alias keeps saving possible where ^S means XOFF."""

        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            self.context.settings.stage_set("AGENT_CHAT_MAX_SEARCHES", "11")
            self.app.refresh_settings()
            await self._settle(pilot)
            await pilot.press("ctrl+enter")
            await self._settle(pilot)
        self.assertEqual(self.recorder.calls, 1)
        self.assertIn(
            "AGENT_CHAT_MAX_SEARCHES=11", self.env_path.read_text(encoding="utf-8")
        )

    async def test_plain_s_and_u_are_save_and_discard_aliases(self) -> None:
        """^S is XOFF and ^Z suspends in some terminals; plain keys always work."""

        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            self.context.settings.stage_set("AGENT_CHAT_MAX_SEARCHES", "12")
            self.app.refresh_settings()
            await self._settle(pilot)
            self._table("settings-table").focus()
            await pilot.press("s")
            await self._settle(pilot)
            self.assertEqual(self.recorder.calls, 1)
            self.assertIn(
                "AGENT_CHAT_MAX_SEARCHES=12", self.env_path.read_text("utf-8")
            )

            self.context.settings.stage_set("AGENT_CHAT_MAX_SEARCHES", "13")
            self.app.refresh_settings()
            await self._settle(pilot)
            await pilot.press("u")
            await self._settle(pilot)
            self.assertFalse(self.context.settings.dirty)

    async def test_plain_letters_do_not_fire_bindings_while_filtering(self) -> None:
        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            self.context.settings.stage_set("AGENT_CHAT_MAX_SEARCHES", "14")
            self.app.refresh_settings()
            await self._settle(pilot)
            filter_box = self.app.query_one("#filter", self.Input)
            filter_box.focus()
            await pilot.press(*"su")
            await self._settle(pilot)
            self.assertEqual(filter_box.value, "su")
            self.assertTrue(self.context.settings.dirty)
            self.assertEqual(self.recorder.calls, 0)

    async def test_quit_confirmation_is_keyboard_only(self) -> None:
        """Ctrl+Q with unsaved work: Enter lands on the primary button."""

        async with self.app.run_test() as pilot:
            self.context.settings.stage_set("AGENT_CHAT_MAX_SEARCHES", "7")
            self.app.refresh_settings()
            await self._settle(pilot)
            await pilot.press("ctrl+q")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, self.ConfirmModal)
            self.assertEqual(getattr(self.app.focused, "id", ""), "modal-ok")
            await pilot.press("enter")
            await self._settle(pilot)
        self.assertFalse(self.app.is_running)

    async def test_delete_profile_by_typing_its_name(self) -> None:
        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            profiles_table = self._table("profiles-table")
            profiles_table.focus()
            profiles_table.move_cursor(row=0)
            await self._settle(pilot)
            await pilot.press("ctrl+d")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, self.ConfirmNameModal)
            await pilot.press(*"default")
            await pilot.press("enter")
            await self._settle(pilot)
            self.assertEqual(
                self.context.profiles.pending_lines(), ["-default（删除 profile）"]
            )

    async def test_new_profile_is_keyboard_only(self) -> None:
        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            self._table("profiles-table").focus()
            await pilot.press("n")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, self.NewProfileModal)
            # The name field has focus; Enter submits with the offered template.
            await pilot.press(*"extra")
            await pilot.press("enter")
            await self._settle(pilot)
            self.assertEqual(
                self.context.profiles.pending_lines(), ["+extra（复制自 default）"]
            )

    async def test_home_end_and_page_keys_navigate_a_long_list(self) -> None:
        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            table = self._table("settings-table")
            table.focus()
            table.move_cursor(row=0)
            await self._settle(pilot)
            last = table.row_count - 1
            await pilot.press("end")
            await self._settle(pilot)
            self.assertEqual(table.cursor_row, last)
            await pilot.press("home")
            await self._settle(pilot)
            self.assertEqual(table.cursor_row, 0)
            await pilot.press("pagedown")
            await self._settle(pilot)
            self.assertGreater(table.cursor_row, 0)

    async def test_tab_reaches_the_fields_list(self) -> None:
        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            self.app.query_one(self.TabbedContent).active = "profiles"
            await self._settle(pilot)
            for _ in range(4):  # any of the first few stops is fine
                if getattr(self.app.focused, "id", None) == "fields-table":
                    break
                await pilot.press("tab")
                await self._settle(pilot)
                await self._settle(pilot)
            self.assertEqual(getattr(self.app.focused, "id", None), "fields-table")

    async def test_every_action_is_bound_to_a_key(self) -> None:
        """No operation may be mouse-only: every `action_*` needs a binding."""

        import inspect

        classes = [
            self.EditorApp,
            self.EditValueModal,
            self.ConfirmModal,
            self.ConfirmNameModal,
            self.NewProfileModal,
            self.app.query_one,  # placeholder replaced below
        ]
        from nonebot_plugin_agent_chat import tui_app

        classes[-1] = tui_app.EditorTable
        for cls in classes:
            bound = {binding.action.split("(")[0] for binding in cls.BINDINGS}
            own_actions = [
                name[len("action_") :]
                for name, member in vars(cls).items()
                if name.startswith("action_") and inspect.isfunction(member)
            ]
            for action in own_actions:
                with self.subTest(cls=cls.__name__, action=action):
                    self.assertIn(
                        action,
                        bound,
                        f"{cls.__name__}.action_{action} has no key binding",
                    )

    async def test_ctrl_c_quits_like_ctrl_q(self) -> None:
        """Ctrl+C must exit through the app, not raise into the CLI."""

        keys = [binding.key for binding in self.EditorApp.BINDINGS]
        self.assertIn("ctrl+c", keys)

        async with self.app.run_test() as pilot:
            await self._settle(pilot)
            await pilot.press("ctrl+c")
            await self._settle(pilot)
        self.assertFalse(self.app.is_running)

    async def test_ctrl_c_confirms_when_dirty(self) -> None:
        async with self.app.run_test() as pilot:
            self.context.settings.stage_set("AGENT_CHAT_MAX_SEARCHES", "7")
            self.app.refresh_settings()
            await pilot.press("ctrl+c")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, self.ConfirmModal)
            await pilot.press("escape")
            await self._settle(pilot)
            self.assertTrue(self.app.is_running)

    async def test_quit_confirms_when_dirty(self) -> None:
        async with self.app.run_test() as pilot:
            self.context.settings.stage_set("AGENT_CHAT_MAX_SEARCHES", "7")
            self.app.refresh_settings()
            await pilot.press("ctrl+q")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, self.ConfirmModal)

            await pilot.press("escape")
            await self._settle(pilot)
            self.assertTrue(self.app.is_running)
            self.assertTrue(self.context.settings.dirty)

            await pilot.press("ctrl+z")
            await self._settle(pilot)
            self.assertFalse(self.context.settings.dirty)
            await pilot.press("ctrl+q")
            await self._settle(pilot)
        self.assertFalse(self.app.is_running)

    async def test_profile_field_edit_and_save(self) -> None:
        async with self.app.run_test() as pilot:
            self.app.query_one(self.TabbedContent).active = "profiles"
            await self._settle(pilot)
            fields = self._table("fields-table")
            self.assertTrue(fields.row_count > 0)
            fields.focus()
            fields.move_cursor(row=fields.get_row_index("model"))
            await self._settle(pilot)
            self.assertIn(
                "模型名", str(self.app.query_one("#field-detail", self.Static).content)
            )
            await pilot.press("enter")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, self.EditValueModal)
            self.app.screen.query_one("#modal-value", self.TextArea).text = "better"
            await pilot.click("#modal-ok")
            await self._settle(pilot)
            self.assertEqual(self.context.profiles.pending_count, 1)

            await pilot.press("ctrl+s")
            await self._settle(pilot)

        self.assertEqual(self.recorder.calls, 1)
        raw = json.loads((self.profiles / "default.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["model"], "better")

    async def test_new_profile_is_staged(self) -> None:
        async with self.app.run_test() as pilot:
            self.app.query_one(self.TabbedContent).active = "profiles"
            await self._settle(pilot)
            await pilot.press("n")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, self.NewProfileModal)
            self.app.screen.query_one("#modal-name", self.Input).value = "qq-safe"
            await pilot.press("enter")
            await self._settle(pilot)
            names = [entry.name for entry in self.context.profiles.entries()]
            self.assertIn("qq-safe", names)
            self.assertFalse((self.profiles / "qq-safe.json").exists())

            await pilot.press("ctrl+s")
            await self._settle(pilot)
        self.assertTrue((self.profiles / "qq-safe.json").is_file())

    async def test_profile_delete_needs_the_exact_name(self) -> None:
        async with self.app.run_test() as pilot:
            self.app.query_one(self.TabbedContent).active = "profiles"
            await self._settle(pilot)
            self._table("profiles-table").focus()
            await pilot.press("ctrl+d")
            await self._settle(pilot)
            self.assertIsInstance(self.app.screen, self.ConfirmNameModal)
            self.assertEqual(self.warnings, ["default"])
            self.assertIn(
                "规则引用 default",
                str(self.app.screen.query_one("#modal-hint", self.Static).content),
            )

            # Wrong name: nothing is staged.
            self.app.screen.query_one("#modal-input", self.Input).value = "nope"
            await pilot.press("enter")
            await self._settle(pilot)
            self.assertFalse(self.context.profiles.dirty)

            # Right name: staged, then saved.
            await pilot.press("ctrl+d")
            await self._settle(pilot)
            self.app.screen.query_one("#modal-input", self.Input).value = "default"
            await pilot.press("enter")
            await self._settle(pilot)
            self.assertTrue(self.context.profiles.is_staged_delete("default"))
            await pilot.press("ctrl+s")
            await self._settle(pilot)
        self.assertFalse((self.profiles / "default.json").exists())

    async def test_filter_without_matches_says_so(self) -> None:
        async with self.app.run_test() as pilot:
            await self._filter(pilot, "no-such-setting")
            self.assertEqual(self._table("settings-table").row_count, 0)
            self.assertIn(
                "没有匹配",
                str(self.app.query_one("#setting-detail", self.Static).content),
            )


if __name__ == "__main__":
    unittest.main()
