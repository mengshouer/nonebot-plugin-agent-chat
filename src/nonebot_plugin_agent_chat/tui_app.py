"""Full-screen editors (textual): settings and profiles in one window.

``tui_state`` owns load/validate/stage/save; this module only binds keys,
renders, and asks for confirmation. textual is optional (``[tui]`` extra) and
is imported here alone, so the plugin and its tests never need it.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Sequence
from typing import ClassVar, Literal

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.coordinate import Coordinate
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from . import env_file, profile_editor, setting_help
from .config_editor import SOURCE_LABELS
from .edits import EFFECT_HINT, EditChange, format_changes
from .errors import InputError
from .tui_state import EditorContext

logger = logging.getLogger(__name__)

_NOTIFY_SECONDS = 6.0

# DataTable has no flex columns, so widths are computed from the terminal width
# and recomputed on resize: a wide terminal spreads out, an 80-column one still
# keeps the value and the effect mark on screen. The summary (with its source)
# lives on a second line inside the label column, so a narrow terminal keeps the
# full key; `i` drops that line for a compact list.
_TABLE_CHROME = 8  # cell padding + cursor column + slack
_EFFECT_WIDTH = 8
_MIN_VALUE_WIDTH = 18
_MAX_VALUE_WIDTH = 56


def column_widths(table_width: int) -> tuple[int, int, int]:
    """(label, value, effect) widths for one table at this terminal width."""

    usable = max(56, table_width - _TABLE_CHROME)
    value = min(_MAX_VALUE_WIDTH, max(_MIN_VALUE_WIDTH, round(usable * 0.33)))
    label = max(26, usable - value - _EFFECT_WIDTH)
    return label, value, _EFFECT_WIDTH


def summary_limit(label_width: int) -> int:
    """How much of the description fits before the `（来源：x）` suffix."""

    return max(
        setting_help.SUMMARY_CHARS_DEFAULT,
        label_width - len("（来源：环境变量）"),
    )


def source_label(row) -> str:
    """Where a setting's value comes from, in short Chinese."""

    return SOURCE_LABELS[row.source]


def setting_cells(
    row,
    *,
    show_summary: bool,
    summary_chars: int = setting_help.SUMMARY_CHARS_DEFAULT,
) -> tuple[Text, Text, Text]:
    """One settings row as styled cells.

    Forty identical-looking lines are unreadable, so the key carries the weight,
    the summary recedes, and 需重启 / unsaved changes stand out.
    """

    label = Text(row.key, style="bold yellow" if row.dirty else "bold")
    if show_summary:
        label.append("\n")
        label.append(
            setting_help.summarize(row.description, limit=summary_chars),
            style="dim",
        )
        label.append(f"（来源：{source_label(row)}）", style="dim italic")
        label.append("\n")  # breathing room between items
    value = Text(row.value, style="yellow" if row.dirty else "")
    effect = Text(
        "需重启" if row.restart_required else "热生效",
        style="yellow" if row.restart_required else "dim",
    )
    return label, value, effect


def field_cells(
    row,
    *,
    show_summary: bool,
    summary_chars: int = setting_help.SUMMARY_CHARS_DEFAULT,
) -> tuple[Text, Text]:
    """One profile-field row: field name bold, description dim."""

    label = Text(row.name, style="bold yellow" if row.dirty else "bold")
    if show_summary:
        label.append("\n")
        label.append(
            setting_help.summarize(row.description, limit=summary_chars)
            or "（没有说明）",
            style="dim",
        )
        label.append("\n")
    value = Text(
        row.value, style="yellow" if row.dirty else ("dim" if row.hidden else "")
    )
    return label, value


def require_tty() -> None:
    """A full-screen app needs a real terminal on both ends."""

    import sys

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise InputError(
            "交互编辑器需要终端；非交互场景请用对应的 --config-* / --profile-* 参数"
        )


class EditorTable(DataTable):
    """A table that shadows DataTable's cell cursor and keeps widths fresh.

    `DataTable` binds left/right to its cell cursor, which this editor never
    uses (rows are addressed by key), so they are rebound here to `focus_pane`:
    on the settings tab that is a no-op (one column only) and the binding stays
    hidden so the footer does not advertise a dead key. The Profiles pane uses
    `ProfilesTable`, which shows the same arrows because there they move between
    the two columns. Switching tabs lives on `Ctrl+left`/`Ctrl+right`, or on the
    tab strip itself, which keeps its own arrow bindings and which `Tab`
    reaches.

    textual sends `Resize` to widgets (the App itself never sees it), so the
    table is also the honest place to notice that the terminal changed.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("left", "focus_pane_left", "左栏", show=False),
        Binding("right", "focus_pane_right", "右栏", show=False),
        # In a list Home/End should mean "first/last row"; DataTable reserves
        # them for the cell cursor's columns, which this editor never uses.
        Binding("home", "scroll_top", "首行", show=False),
        Binding("end", "scroll_bottom", "末行", show=False),
    ]

    def action_focus_pane_left(self) -> None:
        self._focus_pane("left")

    def action_focus_pane_right(self) -> None:
        self._focus_pane("right")

    def _focus_pane(self, side: Literal["left", "right"]) -> None:
        app = self.app
        if isinstance(app, EditorApp):
            app.focus_pane(side)

    def on_resize(self, event) -> None:
        app = self.app
        if isinstance(app, EditorApp):
            app.refresh_widths()


class ProfilesTable(EditorTable):
    """A pane table: here the arrows really do move between the columns."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("left", "focus_pane_left", "左栏", show=True),
        Binding("right", "focus_pane_right", "右栏", show=True),
    ]


class ProfilesPane(Horizontal):
    """The profile list plus the field editor for the selected profile."""

    def compose(self) -> ComposeResult:
        # Two columns side by side: ProfilesTable lets left/right move between them.
        yield ProfilesTable(id="profiles-table")
        with VerticalScroll():
            yield ProfilesTable(id="fields-table")
            yield Static("", id="field-detail")


class EditValueModal(ModalScreen[str | None]):
    """One value, with the item's help text above it.

    Multi-line values (whitelist arrays, platform maps) need room to wrap, so
    the editor is a soft-wrapped TextArea; credential-looking fields keep a
    single-line password Input instead. A validation error re-opens this screen
    with the text preserved, so nothing typed is lost.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "取消"),
        # Ctrl+S is the familiar key, but some terminals/tmux map ^S to XOFF
        # (stop output), so Ctrl+Enter is an equally supported submit.
        Binding("ctrl+s", "submit", "保存"),
        Binding("ctrl+enter", "submit", "保存"),
    ]

    def __init__(
        self,
        title: str,
        hint: str,
        initial: str,
        *,
        hidden: bool = False,
        error: str = "",
    ) -> None:
        super().__init__()
        self._title = title
        self._hint = hint
        self._initial = initial
        self._hidden = hidden
        self._error = error

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label(self._title, id="modal-title")
            yield Static(self._hint, id="modal-hint")
            if self._error:
                yield Static(f"⚠️ {self._error}", id="modal-error")
            if self._hidden:
                yield Input(value=self._initial, password=True, id="modal-input")
            else:
                yield TextArea(self._initial, soft_wrap=True, id="modal-value")
            with Horizontal(id="modal-buttons"):
                yield Button("保存（Ctrl+S）", variant="primary", id="modal-ok")
                yield Button("取消（Esc）", id="modal-cancel")

    def on_mount(self) -> None:
        # Select the current value so typing replaces it: appending to a JSON
        # array by accident is the easiest way to produce an invalid value.
        if self._hidden:
            field = self.query_one("#modal-input", Input)
            field.focus()
            field.select_all()
        else:
            area = self.query_one("#modal-value", TextArea)
            area.focus()
            area.select_all()

    def value(self) -> str:
        if self._hidden:
            return self.query_one("#modal-input", Input).value
        return self.query_one("#modal-value", TextArea).text

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self._finish(event.button.id == "modal-ok")

    def on_input_submitted(self) -> None:
        self._finish(True)

    def action_submit(self) -> None:
        self._finish(True)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _finish(self, accepted: bool) -> None:
        self.dismiss(self.value() if accepted else None)


class ConfirmModal(ModalScreen[bool]):
    """Yes/no confirmation (quitting with unsaved changes)."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "取消")]

    def __init__(self, question: str) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static(self._question, id="modal-hint")
            with Horizontal(id="modal-buttons"):
                yield Button("确定", variant="warning", id="modal-ok")
                yield Button("取消", id="modal-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "modal-ok")

    def action_cancel(self) -> None:
        self.dismiss(False)


class ConfirmNameModal(ModalScreen[bool]):
    """Deleting a profile requires typing its exact name."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "取消")]

    def __init__(self, name: str, warnings: Sequence[str]) -> None:
        super().__init__()
        self._name = name
        self._warnings = list(warnings)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label(f"删除 profile：{self._name}", id="modal-title")
            if self._warnings:
                yield Static(
                    "\n".join(f"⚠️ {line}" for line in self._warnings), id="modal-hint"
                )
            yield Static("输入名字以确认删除：", id="modal-hint2")
            yield Input(placeholder=self._name, id="modal-input")
            with Horizontal(id="modal-buttons"):
                yield Button("删除", variant="error", id="modal-ok")
                yield Button("取消", id="modal-cancel")

    def on_mount(self) -> None:
        self.query_one("#modal-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        typed = self.query_one("#modal-input", Input).value.strip()
        self.dismiss(event.button.id == "modal-ok" and typed == self._name)

    def on_input_submitted(self) -> None:
        typed = self.query_one("#modal-input", Input).value.strip()
        self.dismiss(typed == self._name)

    def action_cancel(self) -> None:
        self.dismiss(False)


class NewProfileModal(ModalScreen[tuple[str, str] | None]):
    """Name plus the profile that acts as the template."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "取消")]

    def __init__(self, templates: Sequence[str]) -> None:
        super().__init__()
        self._templates = list(templates)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label("新建 profile（复制现有）", id="modal-title")
            yield Static("名字（字母数字._-）：", id="modal-hint")
            yield Input(id="modal-name")
            yield Static(
                "模板：" + ("、".join(self._templates) or "（没有现有 profile）"),
                id="modal-hint2",
            )
            yield Input(
                value=profile_editor.preferred_template(self._templates),
                id="modal-template",
            )
            with Horizontal(id="modal-buttons"):
                yield Button("创建", variant="primary", id="modal-ok")
                yield Button("取消", id="modal-cancel")

    def on_mount(self) -> None:
        self.query_one("#modal-name", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self._finish(event.button.id == "modal-ok")

    def on_input_submitted(self) -> None:
        self._finish(True)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _finish(self, accepted: bool) -> None:
        if not accepted:
            self.dismiss(None)
            return
        name = self.query_one("#modal-name", Input).value.strip()
        template = self.query_one("#modal-template", Input).value.strip()
        self.dismiss((name, template))


class EditorApp(App[None]):
    """Two tabs: AGENT_CHAT_* settings and profile files."""

    CSS = """
    #filter { margin: 1 1 0 1; }
    #settings-table { height: 1fr; margin: 0 1; }
    #setting-detail { height: 6; margin: 0 1 1 1; border: round $panel; padding: 0 1; }
    #profiles-table { width: 32; height: 1fr; margin: 1 1 1 1; }
    #fields-table { height: 1fr; margin: 1 1 0 0; }
    #field-detail { height: 5; margin: 0 1 1 0; border: round $panel; padding: 0 1; }
    #status { height: 1; padding: 0 1; background: $boost; }
    EditValueModal, ConfirmModal, ConfirmNameModal, NewProfileModal {
        align: center middle;
    }
    #modal-box {
        width: 80%; min-width: 40; max-width: 110;
        height: auto; max-height: 85%;
        border: round $accent; background: $surface; padding: 1 2;
    }
    #modal-title { text-style: bold; }
    #modal-hint, #modal-hint2 { color: $text-muted; }
    #modal-error { color: $error; text-style: bold; }
    #modal-value { height: 8; border: round $panel; }
    #modal-buttons { height: 3; align: right middle; }
    #modal-buttons Button { margin-left: 2; min-width: 14; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+s", "save", "保存并重载"),
        # Terminal-safe aliases: ^S is XOFF (stop output) in some terminals and
        # tmux, ^Z suspends the process unless the terminal stays in raw mode,
        # and Ctrl+Enter needs CSI-u support. Plain letters always arrive.
        Binding("ctrl+enter", "save", "保存并重载", show=False),
        Binding("s", "save", "保存并重载", show=False),
        Binding("u", "discard", "放弃改动", show=False),
        Binding("ctrl+q", "quit_app", "退出"),
        # Non-priority: inside a text field Ctrl+C still copies, as expected.
        Binding("ctrl+c", "quit_app", "退出", show=False),
        # The only tab switch, aside from the tab strip's own arrows.
        Binding("ctrl+left", "previous_tab", "上一页", show=True),
        Binding("ctrl+right", "next_tab", "下一页", show=True),
        Binding("enter", "edit", "编辑"),
        Binding("ctrl+d", "delete_item", "删除该项"),
        Binding("ctrl+z", "discard", "放弃改动"),
        # (`u` is bound with the other alias above.)
        Binding("slash", "focus_filter", "过滤"),
        Binding("i", "toggle_summary", "说明列"),
        Binding("n", "new_profile", "新建 profile"),
    ]

    def __init__(
        self,
        context: EditorContext,
        *,
        tab: str = "settings",
        focus_profile: str | None = None,
    ) -> None:
        super().__init__()
        self.context = context
        self._tab = tab if context.settings is not None else "profiles"
        self._focus_key: str | None = None
        self._focus_profile = focus_profile
        self._show_summary = True
        self._summary_chars = setting_help.SUMMARY_CHARS_DEFAULT
        self._configured_width = 0

    # --- layout ----------------------------------------------------------
    def compose(self) -> ComposeResult:
        with TabbedContent(initial=self._tab):
            if self.context.settings is None:
                with TabPane("Profiles", id="profiles"):
                    yield ProfilesPane()
                yield Static("", id="status")
                yield Footer(id="footer")
                return
            with TabPane("设置", id="settings"):
                yield Input(
                    placeholder="过滤设置（输入即筛选）；Enter 编辑，Ctrl+D 删除该行",
                    id="filter",
                )
                yield EditorTable(id="settings-table")
                yield Static("", id="setting-detail")
            with TabPane("Profiles", id="profiles"):
                yield ProfilesPane()
        yield Static("", id="status")
        yield Footer(id="footer")

    @property
    def _settings(self):
        return self.context.settings

    def _table_width(self) -> int:
        """The settings table's width (0 before it is mounted)."""

        try:
            return int(self.query_one("#settings-table", DataTable).size.width)
        except Exception:  # noqa: BLE001 - not mounted yet
            return 0

    def _configure_tables(self, table_width: int | None = None) -> None:
        """(Re)build the column layout for the current terminal width.

        `i` keeps the summary line, so the label column carries two lines; the
        widths follow the terminal instead of being pinned.
        """

        width = table_width or self._table_width()
        label_width, value_width, effect_width = column_widths(width)
        self._summary_chars = summary_limit(label_width)
        if self._settings is not None:
            table = self.query_one("#settings-table", DataTable)
            table.cursor_type = "row"
            table.zebra_stripes = True
            table.clear(columns=True)
            table.add_column(
                "键 / 说明" if self._show_summary else "键",
                width=label_width if self._show_summary else min(label_width, 30),
            )
            table.add_column("当前值", width=value_width)
            table.add_column("生效", width=effect_width)
        profiles_table = self.query_one("#profiles-table", DataTable)
        profiles_table.cursor_type = "row"
        profiles_table.zebra_stripes = True
        profiles_table.clear(columns=True)
        profiles_table.add_columns("profile", "状态")
        fields_table = self.query_one("#fields-table", DataTable)
        fields_table.cursor_type = "row"
        fields_table.zebra_stripes = True
        fields_table.clear(columns=True)
        field_width = int(fields_table.size.width) or max(40, width - 32)
        field_label, field_value, _ = column_widths(field_width)
        fields_table.add_column(
            "字段 / 说明" if self._show_summary else "字段",
            width=field_label if self._show_summary else min(field_label, 26),
        )
        fields_table.add_column("当前值", width=field_value)

    def on_mount(self) -> None:
        # The table has no size until the first layout pass, so the terminal
        # width stands in and `call_after_refresh` calibrates once it exists.
        self._configured_width = self._table_width() or int(self.size.width)
        self._configure_tables(self._configured_width)
        self.call_after_refresh(self.refresh_widths)
        if self._settings is not None:
            self.refresh_settings(select_key=None)
        self.refresh_profiles(select_name=self._focus_profile)
        self._focus_active_table()

    # --- refresh ---------------------------------------------------------
    def _filter_text(self) -> str:
        if self._settings is None:
            return ""
        return self.query_one("#filter", Input).value

    def refresh_settings(self, select_key: str | None = None) -> None:
        if self._settings is None:
            return
        rows = self._settings.rows(self._filter_text())
        table = self.query_one("#settings-table", DataTable)
        table.clear()
        for row in rows:
            label, value, effect = setting_cells(
                row,
                show_summary=self._show_summary,
                summary_chars=self._summary_chars,
            )
            table.add_row(
                label,
                value,
                effect,
                height=3 if self._show_summary else 1,
                key=row.key,
            )
        if rows:
            wanted = select_key or self._focus_key or rows[0].key
            if not self._select_row(table, wanted):
                table.move_cursor(row=0)
        self._refresh_setting_detail()
        self._update_status()

    @staticmethod
    def _row_key_at_cursor(table: DataTable) -> str | None:
        """The row key under the cursor: cells are stacked labels, not keys."""

        if table.row_count == 0:
            return None
        try:
            cell_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0))
        except Exception:  # noqa: BLE001 - no cursor yet
            return None
        value = getattr(cell_key.row_key, "value", None)
        return None if value is None else str(value)

    @staticmethod
    def _select_row(table: DataTable, key: str) -> bool:
        """Move the cursor to one row key; cells are rendered Text, not strings."""

        try:
            index = table.get_row_index(key)
        except Exception:  # noqa: BLE001 - RowDoesNotExist for a filtered-out key
            return False
        table.move_cursor(row=index)
        return True

    def _selected_setting(self):
        if self._settings is None:
            return None
        table = self.query_one("#settings-table", DataTable)
        key = self._row_key_at_cursor(table)
        if key is None:
            return None
        for row in self._settings.rows(self._filter_text()):
            if row.key == key:
                return row
        return None

    def _refresh_setting_detail(self) -> None:
        if self._settings is None:
            return
        row = self._selected_setting()
        detail = self.query_one("#setting-detail", Static)
        if row is None:
            detail.update("没有匹配的设置")
            return
        dirty = "（未保存改动）" if row.dirty else ""
        detail.update(
            f"{row.key}{dirty}\n"
            f"说明：{row.description}\n"
            f"类型：{row.type_hint}    默认：{row.default}\n"
            f"当前：{row.value}"
        )

    def refresh_profiles(self, select_name: str | None = None) -> None:
        entries = self.context.profiles.entries()
        table = self.query_one("#profiles-table", DataTable)
        table.clear()
        for entry in entries:
            status = entry.detail or ("未保存改动" if entry.dirty else "")
            table.add_row(entry.name, status, key=entry.name)
        if entries:
            wanted = select_name or self._focus_profile or entries[0].name
            if not self._select_row(table, wanted):
                table.move_cursor(row=0)
        self.refresh_fields()
        self._update_status()

    def _focus_active_table(self) -> None:
        """Row actions need the table focused; `/` jumps to the filter instead."""

        widget_id = (
            "#profiles-table" if self._profiles_tab_active() else "#settings-table"
        )
        try:
            self.query_one(widget_id, DataTable).focus()
        except Exception:  # noqa: BLE001 - widget may not exist in this mode
            logger.debug("No table to focus for %s", widget_id)

    def _profiles_tab_active(self) -> bool:
        return self.query_one(TabbedContent).active == "profiles"

    def switch_tab(self, step: int) -> None:
        """Show the next/previous tab (a no-op when only one pane exists)."""

        tabs = ["settings", "profiles"] if self._settings is not None else ["profiles"]
        active = self.query_one(TabbedContent).active
        if active not in tabs:
            return
        index = (tabs.index(active) + step) % len(tabs)
        self.query_one(TabbedContent).active = tabs[index]

    def focus_pane(self, side: Literal["left", "right"]) -> None:
        """Move the focus between the Profiles tab's two columns.

        The settings tab has a single column, so there the arrows do nothing.
        """

        if not self._profiles_tab_active():
            return
        panes = {"left": "#profiles-table", "right": "#fields-table"}
        self.query_one(panes[side], DataTable).focus()

    def _selected_profile(self) -> str | None:
        return self._row_key_at_cursor(self.query_one("#profiles-table", DataTable))

    def refresh_fields(self) -> None:
        name = self._selected_profile()
        table = self.query_one("#fields-table", DataTable)
        table.clear()
        if name is None:
            self.query_one("#field-detail", Static).update("没有 profile")
            return
        for row in self.context.profiles.fields(name):
            label, value = field_cells(
                row,
                show_summary=self._show_summary,
                summary_chars=self._summary_chars,
            )
            self.query_one("#fields-table", DataTable).add_row(
                label,
                value,
                height=3 if self._show_summary else 1,
                key=row.name,
            )
        self._refresh_field_detail()

    def _selected_field(self):
        name = self._selected_profile()
        table = self.query_one("#fields-table", DataTable)
        field_name = self._row_key_at_cursor(table)
        if name is None or field_name is None:
            return None
        for row in self.context.profiles.fields(name):
            if row.name == field_name:
                return row
        return None

    def _refresh_field_detail(self) -> None:
        row = self._selected_field()
        detail = self.query_one("#field-detail", Static)
        if row is None:
            detail.update("")
            return
        note = "（未保存改动）" if row.dirty else ""
        detail.update(
            f"{row.name}{note}\n{row.description}\n类型：{row.kind}"
            + (f"    取值：{'/'.join(row.options)}" if row.options else "")
        )

    def _update_status(self) -> None:
        settings, profiles = self._settings, self.context.profiles
        parts = []
        if settings is not None and settings.dirty:
            parts.append(f"设置未保存 {settings.pending_count} 项：")
            parts.extend(settings.pending_lines())
        if profiles.dirty:
            parts.append(f"profile 未保存 {profiles.pending_count} 项：")
            parts.extend(profiles.pending_lines())
        text = (
            " | ".join(parts)
            if parts
            else "没有未保存的改动（Ctrl+S 保存并让 bot 重载）"
        )
        self.query_one("#status", Static).update(text)

    # --- events ----------------------------------------------------------
    def on_key(self, event) -> None:
        if event.key == "escape" and getattr(self.focused, "id", None) == "filter":
            self._focus_active_table()
            event.stop()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter" and self._settings is not None:
            self._focus_key = None
            self.refresh_settings()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "settings-table":
            self._refresh_setting_detail()
        elif event.data_table.id == "fields-table":
            self._refresh_field_detail()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Enter belongs to DataTable (select_cursor), so editing hangs off this.
        if event.data_table.id == "settings-table":
            self._edit_setting()
        elif event.data_table.id == "fields-table":
            self._edit_field()
        elif event.data_table.id == "profiles-table":
            self.refresh_fields()
            self._refresh_field_detail()
            self.query_one("#fields-table", DataTable).focus()

    def refresh_widths(self) -> None:
        """Recompute both tables when their width changed (resize or mount)."""

        width = self._table_width()
        if width and width != self._configured_width:
            self._configured_width = width
            self._configure_tables(width)
            self.refresh_settings(select_key=self._selected_setting_key())
            self.refresh_profiles(select_name=self._selected_profile())

    def _selected_setting_key(self) -> str | None:
        row = self._selected_setting()
        return row.key if row is not None else None

    def on_tabbed_content_tab_activated(self) -> None:
        self._refresh_field_detail()
        self._focus_active_table()

    # --- bindings --------------------------------------------------------
    def action_previous_tab(self) -> None:
        self.switch_tab(-1)

    def action_next_tab(self) -> None:
        self.switch_tab(+1)

    def action_edit(self) -> None:
        if self._profiles_tab_active():
            self._edit_field()
        else:
            self._edit_setting()

    @work(exclusive=False)
    async def _edit_setting(self) -> None:
        if self._settings is None:
            return
        row = self._selected_setting()
        if row is None:
            return
        hint = (
            f"{row.description}\n类型：{row.type_hint}    默认：{row.default}\n"
            '新值（JSON 写法：true/false、数字、["..."]；输入即替换选中内容，'
            "Ctrl+S 或 Ctrl+Enter 保存 / Esc 取消）"
        )
        text = self._settings.raw_value(row.key)
        error = ""
        while True:
            value = await self.push_screen_wait(
                EditValueModal(row.key, hint, text, error=error)
            )
            if value is None:
                return
            try:
                self._settings.stage_set(row.key, value)
            except InputError as exc:
                # Keep the operator's text and say what is wrong, in place.
                text, error = value, str(exc)
                continue
            break
        self._focus_key = row.key
        self.refresh_settings(select_key=row.key)
        self.notify(f"{row.key} 已暂存，Ctrl+S 保存", timeout=_NOTIFY_SECONDS)

    @work(exclusive=False)
    async def _edit_field(self) -> None:
        name = self._selected_profile()
        row = self._selected_field()
        if name is None or row is None:
            return
        hint = f"{row.description}\n类型：{row.kind}"
        if row.options:
            hint += f"    取值：{'/'.join(row.options)}"
        if profile_editor.credential_field(row.name):
            hint += "\n原样保留占位符＝不改动；清空后保存＝清除（回落环境变量）"
        elif profile_editor.placeholder_field(row.name):
            hint += "\n原样保留占位符＝不改动；输入完整 JSON 替换；{}＝清空"
        text = self.context.profiles.raw_value(name, row.name)
        error = ""
        while True:
            value = await self.push_screen_wait(
                EditValueModal(
                    f"{name}.{row.name}",
                    hint,
                    text,
                    hidden=row.hidden,
                    error=error,
                )
            )
            if value is None:
                return
            if profile_editor.keeps_masked_placeholder(row.name, value):
                # The placeholder came back untouched: nothing to stage.
                return
            try:
                self.context.profiles.stage_set(name, row.name, value)
            except InputError as exc:
                text, error = value, str(exc)
                continue
            break
        self.refresh_fields()
        self.notify(f"{name}.{row.name} 已暂存，Ctrl+S 保存", timeout=_NOTIFY_SECONDS)

    @work(exclusive=False)
    async def action_delete_item(self) -> None:
        if self._profiles_tab_active() or self._settings is None:
            await self._delete_profile()
            return
        row = self._selected_setting()
        if row is None:
            return
        if self._settings.is_staged_unset(row.key):
            self._settings.unstage(row.key)
            self.notify(f"{row.key} 已恢复")
        else:
            self._settings.stage_unset(row.key)
            self.notify(f"{row.key} 已标记删除（回落默认），Ctrl+S 保存")
        self.refresh_settings(select_key=row.key)

    async def _delete_profile(self) -> None:
        name = self._selected_profile()
        if name is None:
            return
        if name not in self.context.profiles.names():
            # A staged new profile: dropping it is enough.
            self.context.profiles.unstage(name)
            self.refresh_profiles()
            self.notify(f"已取消新建 {name}")
            return
        warnings = await self.context.warnings_for_delete(name)
        confirmed = await self.push_screen_wait(ConfirmNameModal(name, warnings))
        if not confirmed:
            self.notify("名字不一致或已取消，未删除")
            return
        self.context.profiles.stage_delete(name)
        self.refresh_profiles()
        self.notify(f"{name} 已标记删除，Ctrl+S 保存", severity="warning")

    @work(exclusive=False)
    async def action_new_profile(self) -> None:
        if not self._profiles_tab_active():
            return
        result = await self.push_screen_wait(
            NewProfileModal(self.context.profiles.names())
        )
        if result is None:
            return
        name, template = result
        if not name or not template:
            self.notify("需要名字和模板", severity="error")
            return
        try:
            self.context.profiles.stage_create(name, template)
        except InputError as exc:
            self.notify(str(exc), severity="error", timeout=_NOTIFY_SECONDS)
            return
        self.refresh_profiles(select_name=name)
        self.notify(f"{name} 已暂存（复制自 {template}），Ctrl+S 保存")

    @work(exclusive=False)
    async def action_save(self) -> None:
        settings, profiles = self._settings, self.context.profiles
        # `--no-env` without `--env-file` runs profiles-only: there is no
        # settings panel, but profile edits still have to be written.
        if not profiles.dirty and (settings is None or not settings.dirty):
            self.notify("没有需要保存的改动")
            return
        settings_plan = None
        if settings is not None and settings.dirty:
            try:
                settings_plan = settings.plan()
            except InputError as exc:
                self.notify(
                    f"设置保存失败：{exc}", severity="error", timeout=_NOTIFY_SECONDS
                )
                return
        try:
            profiles_plan = profiles.plan()
        except InputError as exc:
            self.notify(
                f"profile 保存失败：{exc}（设置未写入）",
                severity="error",
                timeout=_NOTIFY_SECONDS,
            )
            return
        # Both panels are planned and validated before either file changes: one
        # batch write means a rejected edit leaves every file untouched.
        files = dict(profiles_plan.files)
        if settings_plan is not None:
            files.update(settings_plan.files)
        try:
            env_file.write_many_atomic(files)
            for path in sorted(profiles_plan.deletions):
                path.unlink(missing_ok=True)
        except OSError as exc:
            self.notify(
                f"写入失败：{type(exc).__name__}",
                severity="error",
                timeout=_NOTIFY_SECONDS,
            )
            return
        actions: list[EditChange] = []
        restart: list[str] = []
        if settings is not None and settings_plan is not None:
            actions.extend(settings_plan.changes)
            restart.extend(settings_plan.restart)
            settings.record(settings_plan)
        actions.extend(profiles_plan.changes)
        profiles.record(profiles_plan)
        try:
            await self.context.request_reload()
        except Exception as exc:  # noqa: BLE001 - report, never crash the app
            self.notify(
                f"请求重载失败：{exc}", severity="error", timeout=_NOTIFY_SECONDS
            )
        saved = "；".join(format_changes(actions))
        self.notify(f"已保存：{saved}\n{EFFECT_HINT}", timeout=8.0)
        if restart:
            self.notify(
                "需重启 bot 才生效：" + "、".join(sorted(set(restart))),
                severity="warning",
                timeout=_NOTIFY_SECONDS,
            )
        self.refresh_settings(select_key=self._focus_key)
        self.refresh_profiles(select_name=self._selected_profile())

    def action_focus_filter(self) -> None:
        """`/` jumps into the filter box (Esc/Tab leaves it)."""

        if self._settings is None:
            return
        self.query_one("#filter", Input).focus()
        self.notify("输入即筛选；Tab 回到表格", timeout=3.0)

    def action_toggle_summary(self) -> None:
        """Hide/show the summary column (narrow terminals)."""

        self._show_summary = not self._show_summary
        self._configure_tables(self._configured_width or None)
        self.refresh_settings(select_key=self._focus_key)
        self.refresh_profiles(select_name=self._selected_profile())
        self.notify("说明列：" + ("显示" if self._show_summary else "隐藏"))

    def action_discard(self) -> None:
        if self._settings is not None:
            self._settings.discard()
        self.context.profiles.discard()
        self.refresh_settings(select_key=self._focus_key)
        self.refresh_profiles()
        self.notify("已放弃未保存的改动")

    @work(exclusive=False)
    async def action_quit_app(self) -> None:
        settings, profiles = self._settings, self.context.profiles
        dirty_settings = settings.dirty if settings is not None else False
        if dirty_settings or profiles.dirty:
            confirmed = await self.push_screen_wait(
                ConfirmModal(
                    "有未保存的改动（设置 "
                    f"{settings.pending_count if settings is not None else 0} 项、"
                    f"profile {profiles.pending_count} 项）。放弃并退出？"
                )
            )
            if not confirmed:
                return
        self.exit()


async def run_editor(
    context: EditorContext, *, tab: str = "settings", focus_profile: str | None = None
) -> None:
    """Run the app inside the caller's event loop (the CLI already has one)."""

    require_tty()
    app = EditorApp(context, tab=tab, focus_profile=focus_profile)
    loop = asyncio.get_running_loop()
    handler_installed = False
    try:
        # Ctrl+C quits through the app (unsaved changes still get confirmed)
        # instead of raising KeyboardInterrupt into the CLI, which then fights
        # the interpreter's thread-pool shutdown.
        loop.add_signal_handler(signal.SIGINT, app.action_quit_app)
        handler_installed = True
    except (NotImplementedError, RuntimeError):  # pragma: no cover - not main thread
        logger.debug("Could not install a SIGINT handler for the editor")
    try:
        # run_async() is a coroutine (run_test() is the async context manager).
        await app.run_async()
    except KeyboardInterrupt:
        logger.info("Editor interrupted; exiting without saving")
    finally:
        if handler_installed:
            loop.remove_signal_handler(signal.SIGINT)
