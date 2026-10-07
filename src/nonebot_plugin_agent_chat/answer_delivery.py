"""Deliver answer text using the platform's formatting strategy.

Formatting is best-effort in one direction only: when a platform rejects the
*markup itself*, the same answer is retried as plain text so content is never
lost. Transport problems (rate limits, network failures, timeouts) are re-raised
so the delivery layer can retry with pacing and still account for the attempts.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from nonebot.exception import ActionFailed
from nonebot_plugin_alconna import UniMessage

from .platforms import answer_format_for

if TYPE_CHECKING:
    from nonebot.adapters import Bot, Event
    from nonebot_plugin_alconna import Target

logger = logging.getLogger(__name__)

# Adapters raise their ActionFailed (nonebot's core base class) with the API
# description for 4xx responses; a parse/entity/tag complaint means the markup
# was invalid rather than the transport being unavailable.
_MARKUP_ERROR_HINTS = ("parse", "entit", "tag", "markup")


def _rejection_text(exc: BaseException) -> str:
    """Collect the API description from the shapes adapters expose it in."""

    parts = [str(exc), *(str(arg) for arg in exc.args)]
    info = getattr(exc, "info", None)
    if isinstance(info, Mapping):
        parts.extend(str(info.get(key) or "") for key in ("message", "wording", "msg"))
    return " ".join(parts).lower()


def _looks_like_markup_rejection(exc: BaseException) -> bool:
    if not isinstance(exc, ActionFailed):
        return False
    description = _rejection_text(exc)
    return any(hint in description for hint in _MARKUP_ERROR_HINTS)


async def send_answer_text(
    bot: Bot,
    target: Event | Target,
    text: str,
    scope: str,
) -> None:
    """Send one answer chunk, falling back to plain text on a markup error."""

    answer_format = answer_format_for(scope)
    if answer_format.markup:
        try:
            await UniMessage.text(answer_format.transform(text)).send(
                target=target,
                bot=bot,
                **answer_format.send_kwargs,
            )
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not _looks_like_markup_rejection(exc):
                raise
            logger.warning(
                "Formatted delivery rejected (%s); retrying as plain text",
                type(exc).__name__,
            )
    await UniMessage.text(text).send(target=target, bot=bot)
