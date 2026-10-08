"""Settings-level operations on the plugin's dotenv file.

The CLI editors (flags and the full-screen editor) both build on this
module: listing effective values with their source, validating a change
against the real ``Config`` model, writing it back without disturbing the
file's comments or ordering, and reporting what a reload changed.
"""

from __future__ import annotations

import json
import logging
import os
import types
import typing
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr, ValidationError

from . import env_file
from .config import Config, _default_dirs
from .errors import ConfigurationError, InputError

logger = logging.getLogger(__name__)

# Names that suggest the value itself is a credential.
SENSITIVE_MARKERS = ("key", "token", "auth", "secret", "password")

# Settings whose value is consumed at process start / matcher registration and
# therefore cannot take effect through a reload.
RESTART_ONLY_KEYS = frozenset({"AGENT_CHAT_DATA_DIR", "AGENT_CHAT_PRIORITY"})

KEY_TO_FIELD: dict[str, str] = {field.upper(): field for field in Config.model_fields}
# Read by the CLI itself (paths/bootstrap), not by the settings model.
# AGENT_CHAT_ENV_FILE is restart-only for that CLI process: it resolved the path
# at start, so a reload cannot move it.
CLI_ONLY_KEYS = frozenset({"AGENT_CHAT_DEBUG_DATA_DIR", "AGENT_CHAT_ENV_FILE"})

# Where a value comes from, and the single source of truth for its wording.
SettingSource = Literal["env", "file", "default"]
# Editor-only: a row whose value is staged in memory and not written yet. It is
# not a place a value can come from, so ``ConfigEntry.source`` stays narrower.
RowSource = SettingSource | Literal["staged"]

SOURCE_LABELS: dict[RowSource, str] = {
    "env": "环境变量",
    "file": "文件",
    "default": "默认",
    "staged": "未保存",
}
FIELD_TO_KEY: dict[str, str] = {field: key for key, field in KEY_TO_FIELD.items()}


@dataclass(frozen=True)
class ConfigEntry:
    key: str
    value: str
    source: SettingSource
    restart_required: bool


@dataclass(frozen=True)
class ConfigList:
    entries: list[ConfigEntry]
    cli_only_keys: list[str]  # AGENT_CHAT_* the CLI reads for itself
    unknown_keys: list[str]  # AGENT_CHAT_* lines with no matching setting
    unmanaged_keys: list[str]  # every other line (secrets stay masked)


def sensitive_name(name: str) -> bool:
    """True when a field/key name suggests its value is a credential.

    ``*_env`` names are excluded: they hold the *name* of an environment
    variable, not the secret itself.
    """

    lowered = name.strip().lower()
    if lowered.endswith("_env"):
        return False
    return any(marker in lowered for marker in SENSITIVE_MARKERS)


def display_token(key: str, raw: str) -> str:
    """Echo one written setting; a credential-looking key never prints."""

    return "•••" if sensitive_name(key) else raw


def display_value(value: object) -> str:
    """Render one setting for humans (JSON for containers, true/false)."""

    if value is None:
        return "(未设置)"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return json.dumps(sorted(map(str, value)), ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False)
    return str(value)


def build_config_list(
    config: Config, *, file_values: Mapping[str, str], owned: set[str]
) -> ConfigList:
    """Every known setting with its effective value and where it came from."""

    entries = [
        ConfigEntry(
            key=field.upper(),
            value=display_value(getattr(config, field)),
            source=(
                "file"
                if field.upper() in owned
                else ("env" if field.upper() in os.environ else "default")
            ),
            restart_required=field.upper() in RESTART_ONLY_KEYS,
        )
        for field in Config.model_fields
    ]
    entries.sort(key=lambda entry: entry.key)
    fields = set(KEY_TO_FIELD)
    cli_only = sorted(key for key in file_values if key in CLI_ONLY_KEYS)
    unknown = sorted(
        key
        for key in file_values
        if key.startswith("AGENT_CHAT_")
        and key not in fields
        and key not in CLI_ONLY_KEYS
    )
    unmanaged = sorted(key for key in file_values if not key.startswith("AGENT_CHAT_"))
    return ConfigList(
        entries=entries,
        cli_only_keys=cli_only,
        unknown_keys=unknown,
        unmanaged_keys=unmanaged,
    )


def unwrap_optional(annotation: object) -> tuple[object, bool]:
    """``X | None`` collapses to ``(X, True)``; anything else stays as is."""

    if (
        isinstance(annotation, types.UnionType)
        or typing.get_origin(annotation) is typing.Union
    ):
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def environment_value(name: str, default: str) -> str:
    """The real environment value for a key, else the given fallback."""

    return os.getenv(name) or default


def annotation_of(model: type[BaseModel], field_name: str) -> object:
    """The declared type of one pydantic field."""

    return model.model_fields[field_name].annotation


def _is_string_field(annotation: object) -> bool:
    inner, _ = unwrap_optional(annotation)
    if inner is str or (isinstance(inner, type) and issubclass(inner, Path)):
        return True
    # SecretStr holds a credential, so a token like 12345 must stay the string
    # the operator typed instead of being JSON-parsed into an int.
    return isinstance(inner, type) and issubclass(inner, SecretStr)


def parse_typed(raw: str, annotation: object) -> object:
    """Parse one CLI token against a field annotation; strings stay literal."""

    if _is_string_field(annotation):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Enums and other lax coercions use the raw token.
        return raw


def parse_value(key: str, raw: str) -> object:
    """Parse one CLI value against its setting; string fields stay literal."""

    field = KEY_TO_FIELD.get(key)
    if field is None:
        raise InputError(f"未知设置项：{key}")
    return parse_typed(raw, Config.model_fields[field].annotation)


def _validation_detail(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return str(exc)
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "invalid value"))
    return f"{location}: {message}" if location else message


def validate_change(config: Config, key: str, raw: str) -> Config:
    """The config that would result from setting ``KEY=raw``; raises on error."""

    field = KEY_TO_FIELD.get(key)
    if field is None:
        raise InputError(f"未知设置项：{key}")
    value = parse_value(key, raw)
    data = config.model_dump()
    data[field] = value
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise InputError(f"{key} 设置无效：{_validation_detail(exc)}") from exc


def without_setting(config: Config, key: str, *, owned: bool) -> Config:
    """The config after removing ``key``'s line from the dotenv file.

    ``owned`` says the value currently visible in the environment came from that
    very file, so removing the line falls back to the built-in default instead
    of reading the stale file-provided value back.
    """

    field = KEY_TO_FIELD.get(key.upper())
    if field is None:
        raise InputError(f"未知设置项：{key}")
    if not owned:
        return rebuild_config(config, unset_keys={key.upper()})
    values = config.model_dump()
    values[field] = Config.model_fields[field].get_default(call_default_factory=True)
    return Config.model_validate(values)


class ConfigEditTransaction:
    """Validated dotenv edits staged in memory; the file changes only on commit.

    A batch of edits (several `--config-set` flags, one editor save) validates
    every step against the intermediate config and writes a single file, so a
    rejected value can never leave earlier lines behind.
    """

    def __init__(
        self,
        path: Path,
        config: Config,
        owned: Collection[str] = (),
    ) -> None:
        self.path = path
        self.config = config
        # Keys the live environment received from this file; normalized so a
        # lookup works whatever case the file used.
        self.owned = {key.upper() for key in owned}
        self.text = path.read_text(encoding="utf-8") if path.is_file() else ""

    def set(self, key: str, raw: str) -> None:
        """Validate and stage one `KEY=raw` line."""

        self.config = validate_change(self.config, key, raw)
        self.text = env_file.set_value(self.text, key, raw)
        if env_file.parse_text(self.text).get(key) != raw:
            # A quoting bug here would silently change the operator's value.
            raise InputError(f"{key} 无法安全写入（值需要引号处理）：{raw!r}")

    def unset(self, key: str) -> bool:
        """Stage removing one line; reports whether the line was present."""

        if key.upper() not in KEY_TO_FIELD:
            raise InputError(f"未知设置项：{key}")
        self.text, present = env_file.unset_value(self.text, key)
        if present:
            # Removing a file line means "no override": the built-in default
            # applies, not the value this process happens to hold.
            self.config = without_setting(
                self.config, key, owned=key.upper() in self.owned
            )
        return present

    def commit(self) -> None:
        env_file.write_atomic(self.path, self.text)


def environment_config() -> Config:
    """The config a fresh process reads from its environment."""

    # NoneBot's BaseSettings resolves its sources in __init__, which pyright
    # cannot see through the pydantic base signature.
    return Config(_env_file=None)  # type: ignore[call-arg]


def rebuild_config(
    previous: Config,
    *,
    unset_keys: Collection[str] = (),
    environ: Mapping[str, str] | None = None,
) -> Config:
    """Apply a candidate environment on top of the running config.

    ``environ`` is the environment a staged reload would publish; when omitted,
    the live process environment is used. Values are parsed through the same
    field annotations the CLI uses, so JSON lists and booleans behave exactly as
    they do for a real environment variable.

    ``unset_keys`` are the file-owned keys whose lines were just removed: the
    file explicitly stopped providing them, so their previous value must not
    survive -- the default applies instead.
    """

    environment = os.environ if environ is None else environ
    normalized = {key.upper(): value for key, value in environment.items()}
    unset = {key.upper() for key in unset_keys}
    try:
        # Start from the running values (a programmatic setup must survive a
        # reload that never mentioned the field), overlay what the candidate
        # environment provides, and reset the file-owned keys the file just
        # stopped providing to their built-in defaults.
        values = previous.model_dump()
        for key, field in KEY_TO_FIELD.items():
            if key in normalized:
                values[field] = parse_typed(
                    normalized[key], Config.model_fields[field].annotation
                )
            elif key in unset:
                values[field] = Config.model_fields[field].get_default(
                    call_default_factory=True
                )
        return Config.model_validate(values)
    except ValidationError as exc:
        raise ConfigurationError(
            "dotenv 或环境变量里有无效取值，无法重建配置：" + _validation_detail(exc)
        ) from exc


def build_editing_config(
    environ: Mapping[str, str] | None = None,
) -> tuple[Config, list[str]]:
    """Best-effort config for edit actions, plus the keys that had to be dropped.

    A single invalid dotenv value must not lock the operator out of the very
    commands that repair it. Every key is validated on its own; a failing key is
    reported and left at its default so listing, unsetting, and editing still
    work.
    """

    environment = os.environ if environ is None else environ
    normalized = {key.upper(): value for key, value in environment.items()}
    values = Config.model_construct().model_dump()
    default_data_dir, default_profile_dir = _default_dirs()
    values["agent_chat_data_dir"] = default_data_dir
    values["agent_chat_profile_dir"] = default_profile_dir
    invalid: list[str] = []
    for key, field in KEY_TO_FIELD.items():
        if key not in normalized:
            continue
        candidate = dict(values)
        candidate[field] = parse_typed(
            normalized[key], Config.model_fields[field].annotation
        )
        try:
            values = Config.model_validate(candidate).model_dump()
        except ValidationError:
            invalid.append(key)
    return Config.model_validate(values), invalid


def diff_configs(
    old: Config, new: Config
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    """Split changed settings into (applied now, restart required)."""

    applied: dict[str, tuple[str, str]] = {}
    restart: dict[str, tuple[str, str]] = {}
    for field in Config.model_fields:
        before = getattr(old, field)
        after = getattr(new, field)
        if before == after:
            continue
        key = field.upper()
        target = restart if key in RESTART_ONLY_KEYS else applied
        target[key] = (display_value(before), display_value(after))
    return applied, restart


def format_entry(entry: ConfigEntry) -> str:
    """One human-readable line for ``--config-list`` and the editor."""

    marks = []
    if entry.restart_required:
        marks.append("需重启")
    # Only file/env get a mark: the legend documents "no mark" as "built-in
    # default", so tagging it would contradict README and setting_help.LEGEND.
    if entry.source in ("file", "env"):
        marks.append(SOURCE_LABELS[entry.source])
    suffix = f"  [{'/'.join(marks)}]" if marks else ""
    return f"{entry.key}={entry.value}{suffix}"


def format_report(
    applied: Mapping[str, tuple[str, str]], restart: Mapping[str, tuple[str, str]]
) -> list[str]:
    """Human-readable reload diff lines (private chat / CLI)."""

    lines: list[str] = []
    for key, (before, after) in sorted(applied.items()):
        lines.append(f"{key}: {before} → {after}")
    for key, (before, after) in sorted(restart.items()):
        lines.append(f"{key}: {before} → {after}  ⚠️ 需重启生效")
    return lines
