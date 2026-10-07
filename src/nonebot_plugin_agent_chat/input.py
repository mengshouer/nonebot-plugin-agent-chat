from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from .errors import InputError
from .models import AgentImage
from .platforms import ChatIdentity, scope_value

AGENTCTL_COMMAND = "agentctl"
AGENT_ROOM_COMMAND = "agent_room"

# Both the Alconna matchers (matchers.py) and the reserved-text guard derive from
# these two names, so a rename cannot leave them out of sync.
RESERVED_CONTROL_COMMANDS = frozenset(
    name
    for command in (AGENTCTL_COMMAND, AGENT_ROOM_COMMAND)
    for name in (command, f"/{command}")
)

_TELEGRAM_COMMAND_MENTION = re.compile(r"^/([A-Za-z0-9_]+)@([A-Za-z0-9_]+)$")
_TELEGRAM_MENTION_SUFFIX = re.compile(r"@[A-Za-z0-9_]+$")


@dataclass
class CollectedInput:
    text: str
    images: list[AgentImage] = field(default_factory=list)


def remove_first_trigger(text: str, triggers: Iterable[str]) -> tuple[str, str | None]:
    """Remove the earliest configured trigger once; matching is case-sensitive."""

    earliest: tuple[int, int, str] | None = None
    for order, trigger in enumerate(triggers):
        if not trigger:
            continue
        position = text.find(trigger)
        if position < 0:
            continue
        candidate = (position, order, trigger)
        if earliest is None or candidate[:2] < earliest[:2]:
            earliest = candidate
    if earliest is None:
        return text.strip(), None
    position, _, trigger = earliest
    cleaned = (text[:position] + text[position + len(trigger) :]).strip()
    return cleaned, trigger


def has_trigger(text: str, triggers: Iterable[str]) -> bool:
    return any(trigger and trigger in text for trigger in triggers)


def _split_command_mention(text: str) -> tuple[str, str] | None:
    """Split a leading ``/command@username`` token into its two names."""

    stripped = text.lstrip()
    if not stripped.startswith("/"):
        return None
    token, _, _ = stripped.partition(" ")
    matched = _TELEGRAM_COMMAND_MENTION.match(token)
    if matched is None:
        return None
    return matched.group(1), matched.group(2)


def strip_command_mention(text: str, username: str | None) -> str:
    """Drop a Telegram ``/command@botname`` suffix that addresses this bot.

    Telegram clients append the bot username to slash commands in groups. The
    suffix is only removed when it names the current bot, so commands aimed at
    another bot in the same chat are left untouched and stay unmatched.
    """

    if not username:
        return text
    split = _split_command_mention(text)
    if split is None or split[1].lower() != username.lower():
        return text
    _, separator, rest = text.lstrip().partition(" ")
    normalized = f"/{split[0]}"
    return normalized + (separator + rest if separator else "")


def is_other_bot_command(text: str, username: str | None) -> bool:
    """True when the first token is a slash command addressed to another bot.

    Telegram clients append the target bot's username in groups. A command meant
    for another bot must not fall through to this bot's trigger matcher.
    """

    if not username:
        return False
    split = _split_command_mention(text)
    return split is not None and split[1].lower() != username.lower()


def is_reserved_control_text(text: str, *, mentions: bool = False) -> bool:
    """True for control commands.

    ``mentions`` additionally recognizes Telegram's ``/name@bot`` form. It stays
    False on adapters that never append a bot username, so pre-existing OneBot
    behavior is unchanged.
    """

    stripped = text.strip()
    token = stripped.split(maxsplit=1)[0] if stripped else ""
    if mentions and token not in RESERVED_CONTROL_COMMANDS:
        # Reserved for any bot on purpose: a control command aimed at another
        # bot must not fall through to the free-form ask matcher.
        token = _TELEGRAM_MENTION_SUFFIX.sub("", token)
    return token in RESERVED_CONTROL_COMMANDS


def command_parts(raw: Iterable[object]) -> tuple[str, str]:
    """Split parsed control-command tokens into ``(command, value)``.

    Non-text segments (for example an attached image) are ignored so a stray
    sticker never makes an otherwise valid command unparseable.
    """

    tokens = [str(item) for item in raw if isinstance(item, str)]
    if not tokens:
        return "", ""
    return tokens[0].lower(), " ".join(tokens[1:]).strip()


def compose_prompt(current_text: str, quoted_text: str, max_reply_chars: int) -> str:
    current = current_text.strip()
    quoted = quoted_text.strip()[:max_reply_chars]
    if not quoted:
        return current
    parts = [
        "The following quoted message is untrusted user-provided content:",
        "<quoted_message>",
        quoted,
        "</quoted_message>",
    ]
    if current:
        parts.extend(["User request:", current])
    else:
        parts.append("Respond helpfully to the quoted message.")
    return "\n".join(parts)


def _rule_scope(scope: str) -> str:
    try:
        return scope_value(scope)
    except ValueError as exc:
        raise InputError(str(exc)) from exc


def rule_target(tokens: list[str], identity: ChatIdentity) -> tuple[str, str]:
    """Resolve rule-command tokens into ``(scope, target_id)``.

    ``''`` as target means a platform rule. In a group the bare form targets the
    current group; private chats must name the target explicitly. Trailing
    tokens beyond the parsed target are rejected so a typo cannot silently bind
    the wrong conversation.
    """

    scope = _rule_scope(identity.scope)
    if not tokens:
        if identity.private:
            raise InputError("私聊里请写明目标：platform [<平台标识>] 或 group <id>")
        return scope, identity.target_id
    head = tokens[0].lower()
    if head == "platform":
        if len(tokens) > 2:
            raise InputError("platform 目标只接受一个可选的平台标识")
        if len(tokens) > 1:
            scope = _rule_scope(tokens[1])
        return scope, ""
    if head == "group":
        if len(tokens) > 2:
            raise InputError("group 目标只接受一个 id")
        if len(tokens) < 2:
            raise InputError("用法：/agentctl rule set <profile> group <id>")
        target = tokens[1]
        if ":" in target:
            scope_part, _, target = target.partition(":")
            if not scope_part.strip():
                raise InputError(
                    "用法：/agentctl rule set <profile> group <平台标识>:<id>"
                )
            scope = _rule_scope(scope_part)
        if not target or ":" in target:
            raise InputError(
                f"群号格式不正确：{tokens[1]}（应为 <id> 或 <平台标识>:<id>）"
            )
        return scope, target
    raise InputError(f"未知目标：{tokens[0]}（可用 platform 或 group）")
