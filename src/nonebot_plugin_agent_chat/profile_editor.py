"""Profile-file editing for the CLI: typed field edits, copy, delete.

Both front-ends (flags and the full-screen editor) go through this module. It
works on the raw JSON so field order and unset fields stay exactly as the
operator wrote them, and validates every candidate through ``ProviderProfile``
before writing.
"""

from __future__ import annotations

import json
import logging
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, ValidationError

from . import env_file
from .config_editor import (
    annotation_of,
    display_value,
    parse_typed,
    sensitive_name,
    unwrap_optional,
)
from .errors import InputError
from .models import ProviderProfile
from .profiles import is_valid_profile_name

# Widget kinds the editor can offer for a field.
FieldKind = Literal["enum", "bool", "number", "text", "json"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfileField:
    name: str
    value: str  # display form; "•••" for sensitive values
    kind: FieldKind
    options: list[str]


def profile_path(directory: Path, name: str) -> Path:
    if not is_valid_profile_name(name):
        raise InputError(f"profile 名字不合法：{name!r}")
    return directory / f"{name}.json"


def list_profiles(directory: Path) -> list[str]:
    """Existing profile names, sorted; broken files included (operator sees them)."""

    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))


def preferred_template(names: Sequence[str]) -> str:
    """The template a new profile copies when the operator names none.

    The first existing profile: the editor's create modal prefills the same one,
    so the flags, the session and the editor offer one rule.
    """

    return names[0] if names else ""


def read_raw(path: Path) -> dict[str, object]:
    """The profile file as written (order and unset fields preserved)."""
    if not path.is_file():
        raise InputError(f"profile 不存在：{path.stem}")
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{path.name} 读取失败：{type(exc).__name__}") from exc
    if not isinstance(raw, dict):
        raise InputError(f"{path.name} 不是一个 JSON 对象")
    return raw


def field_kind(annotation: object) -> tuple[FieldKind, list[str]]:
    """The editor widget for one annotation, plus enum options when relevant."""

    resolved = unwrap_optional(annotation)[0]
    if isinstance(resolved, type) and issubclass(resolved, Enum):
        return "enum", [str(member.value) for member in resolved]
    if resolved is bool:
        return "bool", ["true", "false"]
    if resolved in (int, float):
        return "number", []
    if resolved is str or (
        isinstance(resolved, type) and issubclass(resolved, (Path, SecretStr))
    ):
        return "text", []
    return "json", []


def _display_field(field: str, value: object) -> str:
    if hidden_field(field) and value not in (None, ""):
        return "•••"
    return display_value(value)


def describe_fields(path: Path) -> list[ProfileField]:
    """Every schema field of one profile, with its current (raw) value."""

    raw = read_raw(path)
    fields: list[ProfileField] = []
    for name, info in ProviderProfile.model_fields.items():
        kind, options = field_kind(info.annotation)
        present = name in raw
        fields.append(
            ProfileField(
                name=name,
                value=_display_field(name, raw.get(name)) if present else "(默认)",
                kind=kind,
                options=options,
            )
        )
    return fields


def validate_candidate(raw: Mapping[str, object], field: str, value: object) -> None:
    """Raise ``InputError`` unless the whole profile stays valid after ``field``."""

    if field not in ProviderProfile.model_fields:
        raise InputError(f"未知 profile 字段：{field}")
    candidate = dict(raw)
    candidate[field] = value
    try:
        ProviderProfile.model_validate(candidate)
    except ValidationError as exc:
        errors = exc.errors()
        detail = errors[0].get("msg", str(exc)) if errors else str(exc)
        raise InputError(f"{field} 设置无效：{detail}") from exc


def display_token(field: str, token: str) -> str:
    """Echo one written value; anything the editor types with echo off is hidden.

    `hidden_field` is the editor's notion of "secret" (credential-looking names
    and the free-form maps that can carry an Authorization header), so those
    never print at all -- not even a non-secret map's contents.
    """

    if hidden_field(field):
        return "•••" if token else ""
    annotation = annotation_of(ProviderProfile, field)
    return _display_field(field, parse_typed(token, annotation))


# The token the editor prefills for a credential field. It means "keep the
# stored value": the real secret never reaches a widget, the terminal, or a
# screenshot, and an unchanged submit saves nothing.
MASKED_PLACEHOLDER = "••••••"


def credential_field(field: str) -> bool:
    """True when the field's whole value is a credential (the inline keys).

    Narrower than ``hidden_field`` by design: that one also hides free-form
    maps and every credential-looking *name*, numeric fields like
    ``max_output_tokens`` included. The declared type is the honest test:
    the inline keys are the ``SecretStr`` fields.
    """

    info = ProviderProfile.model_fields.get(field)
    if info is None:
        return False
    inner, _ = unwrap_optional(info.annotation)
    return isinstance(inner, type) and issubclass(inner, SecretStr)


def placeholder_field(field: str) -> bool:
    """Hide stored sensitive text/JSON while keeping numeric fields editable."""

    info = ProviderProfile.model_fields.get(field)
    return (
        info is not None
        and hidden_field(field)
        and field_kind(info.annotation)[0] in {"text", "json"}
    )


def keeps_masked_placeholder(field: str, text: str) -> bool:
    """True when the editor submitted the untouched credential placeholder.

    The modal prefills a credential field with the placeholder, never with the
    stored value, so an unchanged submit must save nothing at all.
    """

    return placeholder_field(field) and text == MASKED_PLACEHOLDER


def hidden_field(field: str) -> bool:
    """Fields typed with echo off: credential-looking names and free-form maps.

    ``extra_headers``/``extra_query``/``extra_body`` can carry an
    ``Authorization`` header, so they are hidden even though the field name
    itself says nothing.
    """

    info = ProviderProfile.model_fields.get(field)
    if info is None:
        return False
    if sensitive_name(field):
        return True
    inner, _ = unwrap_optional(info.annotation)
    return typing.get_origin(inner) is dict


def render_profile(raw: Mapping[str, object]) -> str:
    """Serialize one profile exactly as the editor writes it."""

    return json.dumps(raw, ensure_ascii=False, indent=2) + "\n"


class ProfileFileTransaction:
    """Validated profile-file edits staged in memory; commit applies them.

    Every change is validated against the profile's staged content, so a batch
    (several `--profile-set` flags, one editor save) either writes all of its
    files or none of them.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._raw: dict[Path, dict[str, object]] = {}
        self._deletions: set[Path] = set()

    @property
    def files(self) -> dict[Path, str]:
        return {path: render_profile(raw) for path, raw in self._raw.items()}

    @property
    def deletions(self) -> set[Path]:
        return set(self._deletions)

    def _content(self, path: Path) -> dict[str, object]:
        if path in self._raw:
            return dict(self._raw[path])
        return read_raw(path)

    def _stage(self, path: Path, raw: dict[str, object]) -> None:
        self._raw[path] = raw

    def set_field(self, name: str, field: str, token: str) -> None:
        """Validate and stage one field edit."""

        path = profile_path(self.directory, name)
        if path in self._deletions:
            raise InputError(f"{name} 已标记删除，先取消再编辑")
        if field not in ProviderProfile.model_fields:
            raise InputError(f"未知 profile 字段：{field}")
        raw = self._content(path)
        value = parse_typed(token, annotation_of(ProviderProfile, field))
        validate_candidate(raw, field, value)
        self._stage(path, {**raw, field: value})

    def unset_field(self, name: str, field: str) -> bool:
        """Stage removing one field; reports whether it was present."""

        path = profile_path(self.directory, name)
        if field not in ProviderProfile.model_fields:
            raise InputError(f"未知 profile 字段：{field}")
        raw = self._content(path)
        if field not in raw:
            return False
        candidate = {key: value for key, value in raw.items() if key != field}
        try:
            ProviderProfile.model_validate(candidate)
        except ValidationError as exc:
            raise InputError(
                f"{field} 是必填字段，不能删除：{exc.errors()[0]['msg']}"
            ) from exc
        self._stage(path, candidate)
        return True

    def create(self, name: str, template: str) -> None:
        """Stage creating a profile as a copy of ``template``."""

        source = creation_source(self.directory, name, template)
        self._stage(profile_path(self.directory, name), read_raw(source))

    def delete(self, name: str) -> None:
        """Stage deleting one profile file."""

        path = profile_path(self.directory, name)
        if path in self._raw:
            # Created in this same transaction: dropping the staged content is
            # the whole deletion, since no file exists on disk yet.
            del self._raw[path]
            self._deletions.discard(path)
            return
        if not path.is_file():
            raise InputError(f"profile 不存在：{name}")
        self._deletions.add(path)

    def commit(self) -> None:
        env_file.write_many_atomic(self.files)
        for path in sorted(self._deletions):
            path.unlink(missing_ok=True)


def creation_source(directory: Path, name: str, template: str) -> Path:
    """The template path for a new profile, or raise if the creation conflicts.

    Shared by the CLI/one-shot flow and the editor's staged create so both
    reject the same inputs with the same message.
    """

    if profile_path(directory, name).exists():
        raise InputError(f"profile 已存在：{name}")
    source = profile_path(directory, template)
    if not source.is_file():
        raise InputError(f"模板 profile 不存在：{template}")
    return source


def reference_warnings(
    name: str,
    *,
    rule_labels: Sequence[str] = (),
    room_names: Sequence[str] = (),
    configured_default: str | None = None,
) -> list[str]:
    """Operator-facing consequences of deleting ``name``."""

    warnings: list[str] = []
    if rule_labels:
        labels = "、".join(sorted(rule_labels))
        warnings.append(f"以下规则引用该 profile，删除后那些会话会直接报错：{labels}")
    if room_names:
        rooms = "、".join(sorted(room_names))
        warnings.append(f"以下 Room 绑定该 profile，删除后无法对话：{rooms}")
    if configured_default == name:
        warnings.append("该 profile 是当前配置的默认 profile，删除后需指定新的默认值")
    return warnings
