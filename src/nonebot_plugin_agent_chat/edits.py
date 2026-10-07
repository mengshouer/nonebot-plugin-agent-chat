"""One vocabulary for "what an edit did", shared by every surface.

The CLI flags, the session commands and the full-screen editor all describe the
same edits. They agree on `EditChange` (data) and read it through
`format_change` (the only place that words one), so the wording cannot drift
between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EditAction = Literal["set", "unset", "create", "remove", "none"]
EditSubject = Literal["setting", "profile", "profile_field"]

# How a saved edit tells the operator when it takes effect. Two surfaces: the
# one-shot CLI (which may run with no bot at all) and the editor/session.
EFFECT_HINT = "bot 在下一条提问时生效"
EFFECT_HINT_CLI = "bot 会在下一条提问时重载（未运行时启动即为最新配置）"


@dataclass(frozen=True)
class EditChange:
    """One edit result, as data.

    `target` and `detail` per action and subject:

    * ``setting`` + set: target = the key, detail = the value (already masked)
    * ``setting`` + unset: target = the key; ``none`` means the line was absent
    * ``profile_field`` + set: target = ``name.field``, detail = masked value
    * ``profile_field`` + unset: target = ``name.field``
    * ``profile_field`` + none: target = the profile name, detail = the field
    * ``profile`` + create: target = the new name, detail = the template
    * ``profile`` + remove: target = the name
    """

    action: EditAction
    subject: EditSubject
    target: str
    detail: str = ""


def format_change(change: EditChange) -> str:
    """The one wording for an edit outcome."""

    if change.action == "set":
        return f"已设置 {change.target}={change.detail}"
    if change.action == "create":
        return f"已创建 profile: {change.target}（复制自 {change.detail}）"
    if change.action == "remove":
        return f"已删除 profile: {change.target}"
    if change.action == "unset":
        if change.subject == "setting":
            return f"已删除 {change.target}（回落默认或环境变量）"
        return f"已删除 {change.target}"
    if change.detail:
        return f"{change.target} 没有设置 {change.detail}"
    return f"文件里没有 {change.target}，未改动"


def format_changes(changes: list[EditChange]) -> list[str]:
    return [format_change(change) for change in changes]
