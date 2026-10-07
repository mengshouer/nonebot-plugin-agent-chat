"""UniSeg platform scopes, generic identities, and delivery policies.

Platform identifiers come directly from UniSeg's ``SupportScope`` and
``Target.scope``. Local policies describe only delivery differences; a valid
upstream scope needs no local registration to use the generic behavior.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from nonebot import get_driver
from nonebot.log import logger as nonebot_logger

from .models import ImageReplyMode
from .telegram_markdown import markdown_to_telegram_html

# Alconna loads UniSeg and probes adapters on import. Standalone CLI needs neither
# a driver nor adapters; suppress only those expected warnings during this import.
# Keep unrelated warnings and restore the application's warning/logging settings.
with (
    warnings.catch_warnings(),
    nonebot_logger.contextualize(nonebot_log_level="WARNING"),
):
    try:
        get_driver()
    except ValueError:
        warnings.filterwarnings(
            "ignore",
            message=(
                r"^Failed to get nonebot adapters: "
                r"NoneBot has not been initialized\.$"
            ),
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=(
                r"^No adapters found, please make sure you have installed "
                r"at least one adapter\.$"
            ),
            category=RuntimeWarning,
        )
    from nonebot_plugin_alconna.uniseg.constraint import SupportScope

if TYPE_CHECKING:
    from nonebot.adapters import Bot, Event

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Bumped whenever the key payload changes; old keys stay readable as history.
KEY_VERSION = 1


def scope_value(value: str | SupportScope) -> str:
    """Validate an exact UniSeg scope and return its upstream value."""

    try:
        return SupportScope(value).value
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "平台标识不正确（区分大小写，例如 QQClient、Telegram）；"
            "请用 nonebot-agent-chat --platform-list 查询"
        ) from exc


def text_units(text: str) -> int:
    """Text length in UTF-16 code units, which is what Telegram's limit counts."""

    # Telegram 的 4096 上限按 UTF-16 单元计，emoji 占 2 个，不能用 len()。
    return len(text.encode("utf-16-le")) // 2


@dataclass(frozen=True)
class AnswerFormat:
    """How one platform renders answer text.

    ``transform`` converts the Markdown answer into whatever the platform
    understands and ``send_kwargs`` carries the platform-specific send options
    (for example Telegram's ``parse_mode``). ``markup`` is declared explicitly so
    the send path never has to inspect the transformation. The default is plain
    text, which is what every platform that does not render Markdown should use.
    """

    name: str = "plain"
    # ``str`` is the identity for text, so plain platforms need no helper.
    transform: Callable[[str], str] = str
    send_kwargs: Mapping[str, Any] = field(default_factory=dict)
    markup: bool = False


PLAIN_ANSWER_FORMAT = AnswerFormat()
TELEGRAM_ANSWER_FORMAT = AnswerFormat(
    name="telegram-html",
    transform=markdown_to_telegram_html,
    send_kwargs={"parse_mode": "HTML"},
    markup=True,
)


def _telegram_isolation(target: Any) -> str | None:
    """Forum topics are separate conversations inside one supergroup."""

    thread = (getattr(target, "extra", {}) or {}).get("message_thread_id")
    return str(thread) if thread not in (None, "") else None


@dataclass(frozen=True)
class Platform:
    """Everything platform-specific in one place.

    A limit and the unit it is counted in live together so a new cap cannot
    silently use the wrong measure, and every field has a conservative default
    that applies to platforms nobody has described yet.
    """

    scope: str
    isolation: Callable[[Any], str | None] | None = None
    command_mentions: bool = False
    text_limit: int = 0  # 0 = no known hard cap
    text_measure: Callable[[str], int] = len
    max_image_pages: int = 0  # 0 = no plugin-imposed cap
    answer_format: AnswerFormat = PLAIN_ANSWER_FORMAT

    @property
    def text_measure_name(self) -> str:
        return "utf-16" if self.text_measure is text_units else "characters"


_PLATFORMS: dict[str, Platform] = {
    SupportScope.telegram.value: Platform(
        scope=SupportScope.telegram.value,
        isolation=_telegram_isolation,
        command_mentions=True,
        text_limit=4096,
        text_measure=text_units,
        max_image_pages=10,
        answer_format=TELEGRAM_ANSWER_FORMAT,
    ),
    SupportScope.qq_client.value: Platform(scope=SupportScope.qq_client.value),
}

PLATFORM_TEXT_LIMITS: dict[str, int] = {
    key: platform.text_limit
    for key, platform in _PLATFORMS.items()
    if platform.text_limit
}
PLATFORM_MAX_IMAGE_PAGES: dict[str, int] = {
    key: platform.max_image_pages
    for key, platform in _PLATFORMS.items()
    if platform.max_image_pages
}


DEFAULT_PLATFORM = Platform(scope="")


def platform_for(scope: str) -> Platform | None:
    """Facts for a scope, or ``None`` when the platform is not described yet."""

    return _PLATFORMS.get(scope_value(scope))


def _platform_facts(scope: str) -> Platform:
    """Facts for a scope, falling back to the conservative defaults."""

    return platform_for(scope) or DEFAULT_PLATFORM


def platform_entries() -> tuple[Platform, ...]:
    """Every described platform, for diagnostics."""

    return tuple(_PLATFORMS.values())


def is_directed_at_bot(event: Event) -> bool:
    """``event.is_tome()`` where an adapter may not implement it at all."""

    try:
        return bool(event.is_tome())
    except Exception as exc:  # noqa: BLE001 - adapter boundary, never fatal
        logger.debug(
            "is_tome() failed for %s: %s", type(event).__name__, type(exc).__name__
        )
        return False


def default_image_mode(scope: str) -> ImageReplyMode | None:
    """Image mode for a platform without an explicit override.

    Registered platforms inherit the profile/global mode (``None``); unrecorded
    platforms stay text-only, matching the conservative generic defaults.
    """

    return None if platform_for(scope) is not None else ImageReplyMode.OFF


def platform_override(values: Mapping[str, T], scope: str) -> T | None:
    """Look up an exact upstream scope, preserving the caller's default if absent."""

    return values.get(scope_value(scope))


def scope_for_target(target: Any) -> str:
    """Read the upstream scope; never infer an identity from an adapter name."""

    declared = getattr(target, "scope", None)
    if declared is None:
        raise ValueError("消息目标未提供平台标识")
    return scope_value(declared)


def isolation_for(scope: str, target: Any) -> str | None:
    """The platform's conversation-isolation value for a target, if any."""

    extractor = _platform_facts(scope).isolation
    return extractor(target) if extractor is not None else None


def uses_command_mentions(scope: str) -> bool:
    """True when the platform appends ``@username`` to slash commands."""

    return _platform_facts(scope).command_mentions


def resolve_chunk_size(
    global_size: int,
    platform_size: int | None,
    scope: str,
) -> int:
    """Resolve the effective text chunk size for one platform.

    ``0`` means "no plugin-imposed limit": the platform's own hard limit is used
    when known, otherwise the whole answer is sent and the delivery layer splits
    it only if the transport rejects it.
    """

    size = global_size if platform_size is None else platform_size
    if size > 0:
        return size
    return _platform_facts(scope).text_limit


def text_measure_for(scope: str) -> Callable[[str], int]:
    """Length function matching how the platform counts a message."""

    return _platform_facts(scope).text_measure


def max_image_pages_for(scope: str) -> int:
    """Images per message the platform accepts; 0 means no plugin-imposed cap."""

    return _platform_facts(scope).max_image_pages


def answer_format_for(scope: str) -> AnswerFormat:
    """Answer-format strategy for one platform; plain text when unknown."""

    return _platform_facts(scope).answer_format


def _encode(fields: Mapping[str, Any]) -> str:
    """Canonical, self-describing key: stable, readable and JSON-escaped."""

    payload = {"v": KEY_VERSION, **fields}
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


@dataclass(frozen=True)
class ConversationRef:
    """The conversation facts the profile-rule layer needs.

    Private chats are excluded from rules, so the OneBot group/user id-space
    collision cannot make a group rule match a private chat (and the reverse).
    """

    scope: str
    target_id: str
    private: bool = False

    @property
    def rule_scope(self) -> str | None:
        """Canonical scope to look rules up under; ``None`` for private chats."""

        return None if self.private else scope_value(self.scope)


def applicable_rule_rows(
    rules: Sequence[Mapping[str, object]], conversation: ConversationRef
) -> list[Mapping[str, object]]:
    """The rules that apply to a conversation: its own group, then its platform."""

    scope = conversation.rule_scope
    if scope is None:
        return []
    applicable = [
        rule
        for rule in rules
        if rule.get("scope") == scope
        and rule.get("target_id") in ("", conversation.target_id)
    ]
    # The stronger tier first, mirroring the precedence ladder.
    applicable.sort(key=lambda rule: 0 if rule.get("target_id") else 1)
    return applicable


def rule_label(scope: str, target_id: str) -> str:
    """Human-readable rule key: ``QQClient`` or ``QQClient:group:598683145``."""

    key = scope_value(scope)
    return key if not target_id else f"{key}:group:{target_id}"


@dataclass(frozen=True)
class ChatIdentity:
    """Stable actor and conversation identity at the adapter boundary."""

    scope: str
    bot_id: str
    user_id: str
    target_id: str
    parent_id: str = ""
    channel: bool = False
    private: bool = False
    isolation: str | None = None

    @property
    def subject_key(self) -> str:
        """Per-user identity used for quotas, concurrency and run metadata."""

        return _encode({"bot": self.bot_id, "scope": self.scope, "user": self.user_id})

    @property
    def context_key(self) -> str:
        """Conversation identity used for rooms and concurrency contexts."""

        fields: dict[str, Any] = {
            "bot": self.bot_id,
            "scope": self.scope,
            "user": self.user_id,
            "target": {
                "channel": self.channel,
                "id": self.target_id,
                "parent_id": self.parent_id,
                "private": self.private,
            },
        }
        if self.isolation:
            fields["isolation"] = self.isolation
        return _encode(fields)

    @property
    def conversation(self) -> ConversationRef:
        """This conversation's profile-rule lookup key."""

        return ConversationRef(
            scope=self.scope,
            target_id=self.target_id,
            private=self.private,
        )

    @property
    def acl_id(self) -> str:
        return self.user_id if self.private else self.target_id

    @property
    def acl_scope(self) -> str:
        return scope_value(self.scope)

    @property
    def acl_entry(self) -> str:
        return f"{self.acl_scope}:{self.acl_id}"

    @property
    def operator_id(self) -> str:
        """The acting user, for audit columns (never the group they typed in)."""

        return f"{self.acl_scope}:{self.user_id}"


def identity_from_target(
    *,
    bot_id: str,
    user_id: str,
    target: Any,
) -> ChatIdentity | None:
    """Build an identity from a UniSeg target; ``None`` when it is unusable."""

    target_id = str(getattr(target, "id", ""))
    if not target_id or not user_id:
        return None
    try:
        scope = scope_for_target(target)
    except ValueError:
        logger.warning(
            "Ignoring message: missing or invalid platform ID; "
            "check the adapter's cross-platform message support"
        )
        return None
    return ChatIdentity(
        scope=scope,
        bot_id=str(bot_id),
        user_id=str(user_id),
        target_id=target_id,
        parent_id=str(getattr(target, "parent_id", "") or ""),
        channel=bool(getattr(target, "channel", False)),
        private=bool(getattr(target, "private", False)),
        isolation=isolation_for(scope, target),
    )


def resolve_chat_identity(bot: Bot, event: Event) -> ChatIdentity | None:
    """Resolve supported message events without importing a native adapter."""

    if event.get_type() != "message":
        return None
    try:
        user_id = event.get_user_id()
    except (NotImplementedError, ValueError):
        # Channel posts and other chatter-less events have no user to attribute.
        return None

    from nonebot_plugin_alconna import get_target
    from nonebot_plugin_alconna.uniseg.constraint import SerializeFailed

    try:
        target = get_target(event=event, bot=bot)
    except (NotImplementedError, SerializeFailed, ValueError):
        return None

    return identity_from_target(
        bot_id=bot.self_id,
        user_id=str(user_id),
        target=target,
    )


def is_identity_allowed(
    identity: ChatIdentity,
    *,
    allowed_groups: set[str],
    allowed_users: set[str],
) -> bool:
    """Match scoped ACL entries, including the ``scope:*`` wildcard."""

    allowed = allowed_users if identity.private else allowed_groups
    return identity.acl_entry in allowed or f"{identity.acl_scope}:*" in allowed
