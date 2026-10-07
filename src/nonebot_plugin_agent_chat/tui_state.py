"""Headless state for the full-screen editors.

Everything the TUI can do lives here — load, validate, stage, save — so the
logic is unit-testable and ``tui_app`` stays a thin binding/render layer. The
staged model is deliberate: edits accumulate in memory (rows shown as dirty)
and only ``Ctrl+S`` writes files, asks the bot to reload, and reloads the
current session.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from . import config_editor, env_file, profile_editor, setting_help
from .config import Config
from .edits import EditChange
from .errors import InputError
from .models import ProviderProfile

# The source label of a staged (unsaved) value; the Chinese wording lives in
# ``config_editor.SOURCE_LABELS`` like every other source label.
STAGED_SOURCE: config_editor.RowSource = "staged"

PROFILE_FIELD_HELP: dict[str, str] = {
    "protocol": "接口协议：openai-completions / openai-responses / anthropic-messages",
    "model": "模型名（必填）",
    "base_url": "自定义 API 地址；留空用官方地址",
    "api_key_env": "读取密钥的环境变量名（不是密钥本身）；内联 api_key 优先",
    "system_prompt": "内联系统提示词；非空时优先于 system_prompt_file",
    "system_prompt_file": "提示词文件（相对 <数据目录>/prompts）",
    "capabilities": "能力开关：视觉、推理等（JSON 对象）",
    "reasoning_effort": (
        "推理强度：provider_default/off/minimal/low/medium/high/xhigh/max"
    ),
    "search_mode": "搜索方式：off / builtin_web_search / exa",
    "image_reply_mode": "该 profile 的图片模式覆盖：off / auto / always；留空跟随全局",
    "show_sources_text": "文本来源页脚开关；留空跟随全局",
    "show_sources_image": "图片来源页脚开关；留空跟随全局",
    "responses_store": "是否让服务端保存响应（openai-responses 的 store 参数）",
    "fallback_profiles": "失败时依次尝试的 profile（最多 2 个，不递归）",
    "enabled_tools": "显式启用的本地只读工具名",
    "max_output_tokens": "单次回答的最大输出 token 数",
    "max_builtin_tool_calls": "内置工具（如搜索）最多调用几次",
    "request_timeout_seconds": "单次模型请求超时（秒）",
    "temperature": "采样温度 0–2；留空用服务端默认",
    "extra_headers": "附加请求头（JSON 对象；可放 Authorization，注意保密）",
    "extra_query": "附加 URL 查询参数（JSON 对象）",
    "extra_body": "附加请求体字段（JSON 对象）",
    "exa_api_key_env": "Exa 密钥的环境变量名（默认 EXA_API_KEY）",
    "api_key": "内联的密钥（明文落盘，优先于 api_key_env）",
    "exa_api_key": "内联的 Exa 密钥（明文落盘，优先于 exa_api_key_env）",
    "exa_base_url": "Exa 自定义地址",
    "exa_num_results": "Exa 每次返回的结果数（1–10）",
}


def describe_profile_field(name: str) -> str:
    return PROFILE_FIELD_HELP.get(name, "")


@dataclass
class SettingRow:
    key: str
    value: str
    source: config_editor.RowSource
    restart_required: bool
    description: str
    summary: str
    type_hint: str
    default: str
    dirty: bool = False


@dataclass
class FieldRow:
    name: str
    value: str
    kind: profile_editor.FieldKind
    options: list[str]
    hidden: bool
    description: str
    summary: str
    dirty: bool = False


@dataclass
class ProfileEntry:
    name: str
    detail: str
    dirty: bool = False


@dataclass(frozen=True)
class SettingsPlan:
    """A validated settings save, not yet written."""

    files: dict[Path, str]
    config: Config
    changes: list[EditChange]
    restart: list[str]


@dataclass(frozen=True)
class ProfilesPlan:
    """A validated profile save, not yet written."""

    files: dict[Path, str]
    deletions: set[Path]
    changes: list[EditChange]


class SettingsPanel:
    """Staged edits for the dotenv settings file."""

    def __init__(self, env_file_path: Path, owned: set[str], config: Config) -> None:
        self.env_file = env_file_path
        self.owned = {key.upper() for key in owned}
        self.config = config
        self._sets: dict[str, str] = {}
        self._unsets: set[str] = set()
        self.last_changes: list[EditChange] = []

    # --- state -----------------------------------------------------------
    @property
    def dirty(self) -> bool:
        return bool(self._sets or self._unsets)

    @property
    def pending_count(self) -> int:
        return len(self._sets) + len(self._unsets)

    def pending_lines(self) -> list[str]:
        lines = [f"{key}={raw}" for key, raw in self._sets.items()]
        lines.extend(f"-{key}（删除，回落默认）" for key in sorted(self._unsets))
        return lines

    # --- load ------------------------------------------------------------
    def _file_values(self) -> dict[str, str]:
        return env_file.read_values(self.env_file)

    def _recompute(self) -> Config:
        candidate = self.config
        for key, raw in self._sets.items():
            candidate = config_editor.validate_change(candidate, key, raw)
        for key in sorted(self._unsets):
            candidate = config_editor.without_setting(
                candidate, key, owned=key.upper() in self.owned
            )
        return candidate

    def rows(self, needle: str = "") -> list[SettingRow]:
        """A pure query: the staged candidate is computed, never cached."""

        listing = config_editor.build_config_list(
            self._recompute(), file_values=self._file_values(), owned=self.owned
        )
        lowered = needle.strip().lower()
        rows: list[SettingRow] = []
        for entry in listing.entries:
            if lowered and lowered not in entry.key.lower():
                continue
            dirty = entry.key in self._sets or entry.key in self._unsets
            # A staged unset stages no value, so the lookup already yields the
            # post-unset value the listing reported.
            value = self._sets.get(entry.key, entry.value)
            description = setting_help.describe(entry.key) or ""
            rows.append(
                SettingRow(
                    key=entry.key,
                    value=value,
                    source=STAGED_SOURCE if dirty else entry.source,
                    restart_required=entry.restart_required,
                    description=description,
                    summary=setting_help.summarize(description),
                    type_hint=setting_help.type_hint(
                        Config.model_fields[entry.key.lower()].annotation
                    ),
                    default=setting_help.default_text(entry.key),
                    dirty=dirty,
                )
            )
        return rows

    # --- stage -----------------------------------------------------------
    def stage_set(self, key: str, raw: str) -> None:
        previous = (dict(self._sets), set(self._unsets))
        self._sets[key] = raw
        self._unsets.discard(key)
        try:
            self._recompute()
        except InputError:
            self._sets, self._unsets = previous
            raise

    def stage_unset(self, key: str) -> None:
        if key not in config_editor.KEY_TO_FIELD:
            raise InputError(f"未知设置项：{key}")
        self._sets.pop(key, None)
        self._unsets.add(key)

    def raw_value(self, key: str) -> str:
        """The token an editor should prefill (staged edit, else the file)."""

        if key in self._unsets:
            return ""
        if key in self._sets:
            return self._sets[key]
        return env_file.read_values(self.env_file).get(key, "")

    def is_staged_unset(self, key: str) -> bool:
        return key in self._unsets

    def unstage(self, key: str) -> None:
        self._sets.pop(key, None)
        self._unsets.discard(key)

    def discard(self) -> None:
        self._sets.clear()
        self._unsets.clear()

    # --- save ------------------------------------------------------------
    def plan(self) -> SettingsPlan:
        """Validate every staged change; nothing is written yet."""

        transaction = config_editor.ConfigEditTransaction(
            self.env_file, self.config, self.owned
        )
        changes: list[EditChange] = []
        restart: list[str] = []
        for key, raw in self._sets.items():
            transaction.set(key, raw)
            changes.append(
                EditChange(
                    "set",
                    "setting",
                    key,
                    config_editor.display_token(key, raw),
                )
            )
            if key in config_editor.RESTART_ONLY_KEYS:
                restart.append(key)
        for key in sorted(self._unsets):
            if transaction.unset(key):
                changes.append(EditChange("unset", "setting", key))
                if key in config_editor.RESTART_ONLY_KEYS:
                    restart.append(key)
        return SettingsPlan(
            files={transaction.path: transaction.text},
            config=transaction.config,
            changes=changes,
            restart=restart,
        )

    def record(self, plan: SettingsPlan) -> None:
        """Adopt a plan that has been written to disk."""

        self.config = plan.config
        self._sets.clear()
        self._unsets.clear()
        self.last_changes = list(plan.changes)

    def save(self) -> tuple[list[EditChange], list[str]]:
        """Write every staged change; returns (changes, restart_required)."""

        plan = self.plan()
        env_file.write_many_atomic(plan.files)
        self.record(plan)
        return plan.changes, plan.restart


class ProfilesPanel:
    """Staged edits for profile files (field edits, creates, deletes)."""

    def __init__(self, profile_dir: Path) -> None:
        self.profile_dir = profile_dir
        self._sets: dict[tuple[str, str], str] = {}
        self._creates: dict[str, str] = {}  # name -> template
        self._deletes: set[str] = set()
        self.last_changes: list[EditChange] = []

    # --- state -----------------------------------------------------------
    @property
    def dirty(self) -> bool:
        return bool(self._sets or self._creates or self._deletes)

    @property
    def pending_count(self) -> int:
        return len(self._sets) + len(self._creates) + len(self._deletes)

    def pending_lines(self) -> list[str]:
        lines = [
            f"{name}.{field}={profile_editor.display_token(field, raw)}"
            for (name, field), raw in self._sets.items()
        ]
        lines.extend(
            f"+{name}（复制自 {template}）" for name, template in self._creates.items()
        )
        lines.extend(f"-{name}（删除 profile）" for name in sorted(self._deletes))
        return lines

    # --- load ------------------------------------------------------------
    def names(self) -> list[str]:
        return profile_editor.list_profiles(self.profile_dir)

    def entries(self) -> list[ProfileEntry]:
        entries = []
        for name in self.names():
            dirty = name in self._deletes or any(
                staged == name for staged, _ in self._sets
            )
            detail = "待删除" if name in self._deletes else ""
            entries.append(ProfileEntry(name=name, detail=detail, dirty=dirty))
        entries.extend(
            ProfileEntry(name=name, detail=f"新建（复制自 {template}）", dirty=True)
            for name, template in self._creates.items()
        )
        return entries

    def _source_path(self, name: str) -> Path:
        """The file a profile's content comes from (its template while staged)."""

        if name in self._creates:
            return profile_editor.profile_path(self.profile_dir, self._creates[name])
        return profile_editor.profile_path(self.profile_dir, name)

    def fields(self, name: str) -> list[FieldRow]:
        path = self._source_path(name)
        rows: list[FieldRow] = []
        for info in profile_editor.describe_fields(path):
            staged = self._sets.get((name, info.name))
            dirty = staged is not None
            description = describe_profile_field(info.name)
            value = staged if dirty else info.value
            if dirty and profile_editor.hidden_field(info.name):
                # A staged credential is masked like a stored one: the table gets
                # rendered, screenshotted, and read by whoever runs the session.
                value = profile_editor.display_token(info.name, staged or "")
            rows.append(
                FieldRow(
                    name=info.name,
                    value=value,
                    kind=info.kind,
                    options=info.options,
                    hidden=profile_editor.hidden_field(info.name),
                    description=description,
                    summary=setting_help.summarize(description),
                    dirty=dirty,
                )
            )
        return rows

    # --- stage -----------------------------------------------------------
    def _candidate_raw(self, name: str) -> dict[str, object]:
        """The profile's raw JSON with its staged field edits applied.

        A profile staged for creation uses its template as the base, so its
        edits validate before anything touches the disk.
        """

        raw = profile_editor.read_raw(self._source_path(name))
        for (staged_name, field_name), token in self._sets.items():
            if staged_name != name:
                continue
            annotation = config_editor.annotation_of(ProviderProfile, field_name)
            raw[field_name] = config_editor.parse_typed(token, annotation)
        return raw

    def stage_set(self, name: str, field_name: str, raw: str) -> None:
        if name in self._deletes:
            raise InputError(f"{name} 已标记删除，先取消再编辑")
        candidate = self._candidate_raw(name)
        annotation = ProviderProfile.model_fields[field_name].annotation
        value = config_editor.parse_typed(raw, annotation)
        profile_editor.validate_candidate(candidate, field_name, value)
        self._sets[(name, field_name)] = raw

    def stage_create(self, name: str, template: str) -> None:
        profile_editor.creation_source(self.profile_dir, name, template)
        if name in self._creates:
            raise InputError(f"{name} 已在新建列表里")
        self._creates[name] = template

    def raw_value(self, name: str, field_name: str) -> str:
        """The token an editor should prefill for one profile field.

        A credential field is never prefilled with its value: the editor shows a
        placeholder that means "keep as is", so a stored key cannot reach a
        widget, the terminal, or a screenshot.
        """

        staged = self._sets.get((name, field_name))
        if staged is not None:
            token = staged
        else:
            raw = self._candidate_raw(name)
            if field_name not in raw:
                return ""
            value = raw[field_name]
            if isinstance(value, str):
                token = value
            else:
                token = json.dumps(value, ensure_ascii=False)
        if token and profile_editor.placeholder_field(field_name):
            return profile_editor.MASKED_PLACEHOLDER
        return token

    def is_staged_delete(self, name: str) -> bool:
        return name in self._deletes

    def stage_delete(self, name: str) -> None:
        if name in self._creates:
            # Deleting a profile that was only staged for creation cancels it.
            del self._creates[name]
            self._drop_field_edits(name)
            return
        path = profile_editor.profile_path(self.profile_dir, name)
        if not path.is_file():
            raise InputError(f"profile 不存在：{name}")
        self._deletes.add(name)
        self._drop_field_edits(name)

    def _drop_field_edits(self, name: str) -> None:
        self._sets = {key: value for key, value in self._sets.items() if key[0] != name}

    def unstage(self, name: str) -> None:
        self._deletes.discard(name)
        self._creates.pop(name, None)
        self._sets = {key: value for key, value in self._sets.items() if key[0] != name}

    def discard(self) -> None:
        self._sets.clear()
        self._creates.clear()
        self._deletes.clear()

    # --- save ------------------------------------------------------------
    def plan(self) -> ProfilesPlan:
        """Validate every staged profile change; nothing is written yet."""

        transaction = profile_editor.ProfileFileTransaction(self.profile_dir)
        changes: list[EditChange] = []
        for name, template in self._creates.items():
            transaction.create(name, template)
            changes.append(EditChange("create", "profile", name, template))
        for (name, field_name), raw in self._sets.items():
            if name in self._deletes:
                continue
            transaction.set_field(name, field_name, raw)
            changes.append(
                EditChange(
                    "set",
                    "profile_field",
                    f"{name}.{field_name}",
                    profile_editor.display_token(field_name, raw),
                )
            )
        for name in sorted(self._deletes):
            transaction.delete(name)
            changes.append(EditChange("remove", "profile", name))
        return ProfilesPlan(
            files=transaction.files,
            deletions=transaction.deletions,
            changes=changes,
        )

    def record(self, plan: ProfilesPlan) -> None:
        """Adopt a plan that has been written to disk."""

        self.discard()
        self.last_changes = list(plan.changes)

    def save(self) -> list[EditChange]:
        plan = self.plan()
        env_file.write_many_atomic(plan.files)
        for path in sorted(plan.deletions):
            path.unlink(missing_ok=True)
        self.record(plan)
        return plan.changes


@dataclass
class EditorContext:
    """What the app needs from the CLI (kept as callables: no import cycle)."""

    profiles: ProfilesPanel
    request_reload: Callable[[], Awaitable[None]]
    warnings_for_delete: Callable[[str], Awaitable[list[str]]]
    # None when the editor is opened for profiles only (--profile-edit).
    settings: SettingsPanel | None = None
