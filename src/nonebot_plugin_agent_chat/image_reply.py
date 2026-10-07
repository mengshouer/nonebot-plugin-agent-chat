"""Decide whether an answer becomes an image, and deliver it without loss.

The renderer is optional: when it is missing, fails, or is switched off, the
answer is delivered as paced text chunks exactly as before. Length alone never
causes a text fallback — a long answer becomes more pages, because text that is
too long is exactly what the image path exists to fix.

The renderer's browser is not kept either: the batch releases it when it ends,
so an idle bot holds no Chromium process and the next answer pays one cold
start.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import Config
from .delivery import DeliveryReport, send_chunks
from .errors import RenderError
from .models import ImageReplyMode
from .platforms import (
    default_image_mode,
    max_image_pages_for,
    platform_override,
    resolve_chunk_size,
    text_measure_for,
)
from .render import MIN_SLICE_HEIGHT

logger = logging.getLogger(__name__)

DEFAULT_MIN_CHARS = 1000
DEFAULT_MAX_HEIGHT = 8000
DEFAULT_BATCH_BYTES = 3_500_000
MIN_BATCH_BYTES = 400_000
# Telegram media groups accept at most ten items; other adapters keep the
# byte-budget behaviour and pass 0 here.
DEFAULT_MAX_PAGES_PER_MESSAGE = 0
_MAX_SPLITS = 8
CODE_BLOCK_MIN_LINES = 8

TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:\-|]+\|\s*$", re.MULTILINE)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)

INTERRUPTED_NOTICE = "（图片渲染中断，完整内容改为文字发送）"
IMAGE_FAILED_NOTICE = "（图片发送失败，已改为文字发送）"
PARTIAL_NOTICE = "（部分图片发送失败，已省略）"


@dataclass(frozen=True)
class ImageReplySettings:
    mode: ImageReplyMode = ImageReplyMode.OFF
    min_chars: int = DEFAULT_MIN_CHARS
    max_height: int = DEFAULT_MAX_HEIGHT
    timeout_seconds: float = 20.0
    chunk_chars: int = 1000
    delay_seconds: float = 1.0
    batch_max_bytes: int = DEFAULT_BATCH_BYTES
    # 0 means "no plugin-imposed cap" so adapters without a media-group limit
    # keep the original single-message batch behaviour.
    max_pages_per_message: int = DEFAULT_MAX_PAGES_PER_MESSAGE
    max_splits: int = _MAX_SPLITS
    # Characters by default; UTF-16 code units where the platform counts that.
    chunk_measure: Callable[[str], int] = len

    @classmethod
    def from_config(
        cls,
        config: Config,
        mode: ImageReplyMode | None = None,
        scope: str | None = None,
    ) -> ImageReplySettings:
        """Build settings, resolving the platform facts when a scope is given."""

        resolved = config.agent_chat_image_reply_mode if mode is None else mode
        # The platform override wins over the global chunk size; the chunker
        # then adapts within whatever limit the transport enforces.
        chunk_chars = config.agent_chat_message_chunk_chars
        max_pages = DEFAULT_MAX_PAGES_PER_MESSAGE
        chunk_measure: Callable[[str], int] = len
        if scope is not None:
            chunk_chars = resolve_chunk_size(
                config.agent_chat_message_chunk_chars,
                platform_override(
                    config.agent_chat_message_chunk_chars_by_platform,
                    scope,
                ),
                scope,
            )
            max_pages = max_image_pages_for(scope)
            chunk_measure = text_measure_for(scope)
        return cls(
            mode=resolved,
            min_chars=config.agent_chat_image_reply_min_chars,
            max_height=config.agent_chat_image_reply_max_height,
            timeout_seconds=config.agent_chat_image_reply_timeout_seconds,
            chunk_chars=chunk_chars,
            delay_seconds=config.agent_chat_message_send_delay_seconds,
            max_pages_per_message=max_pages,
            chunk_measure=chunk_measure,
        )


@dataclass(frozen=True)
class AnswerDeliveryReport:
    """How one answer reached the chat, and whether anything was lost."""

    delivery: str = "text"
    sent_messages: int = 0
    sent_pages: int = 0
    dropped_pages: int = 0
    send_attempts: int = 0
    render_error: str | None = None
    text_report: DeliveryReport | None = None
    notice: str | None = None

    @property
    def used_image(self) -> bool:
        return self.sent_pages > 0

    @property
    def complete(self) -> bool:
        """True when the whole answer reached the chat, by whichever path."""

        if self.dropped_pages and self.text_report is None:
            # Image pages were lost and no text fallback carried them.
            return False
        return self.text_report is None or self.text_report.complete


def resolve_mode(
    global_mode: ImageReplyMode,
    profile_mode: ImageReplyMode | None,
) -> ImageReplyMode:
    """A profile may override the global switch, but only explicitly."""

    return global_mode if profile_mode is None else profile_mode


def resolve_show_sources(
    global_default: bool,
    profile_value: bool | None,
    platform_values: Mapping[str, bool],
    scope: str,
) -> bool:
    """Whether sources show on one delivery path: platform > profile > global."""

    value = profile_value if profile_value is not None else global_default
    override = platform_override(platform_values, scope)
    if override is not None:
        return override
    return value


def resolve_platform_mode(
    global_mode: ImageReplyMode,
    profile_mode: ImageReplyMode | None,
    platform_modes: Mapping[str, ImageReplyMode],
    scope: str,
) -> ImageReplyMode:
    """Platform override > profile > global, then the platform default.

    An unrecorded platform defaults to ``off`` so an unverified adapter never
    starts rendering images just because the host enabled them globally.
    """

    mode = resolve_mode(global_mode, profile_mode)
    override = platform_override(platform_modes, scope)
    if override is not None:
        return override
    default = default_image_mode(scope)
    return mode if default is None else default


def has_table(text: str) -> bool:
    """True for a pipe table with a separator row, not for incidental pipes."""

    if TABLE_SEPARATOR.search(text) is None:
        return False
    return len(TABLE_ROW.findall(text)) >= 2


def has_long_code_block(text: str) -> bool:
    """True when a fenced block holds at least ``CODE_BLOCK_MIN_LINES`` lines."""

    inside = False
    counted = 0
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if inside and counted >= CODE_BLOCK_MIN_LINES:
                return True
            inside = not inside
            counted = 0
            continue
        if inside:
            counted += 1
    return False


def should_render(text: str, settings: ImageReplySettings) -> bool:
    """Whether image delivery applies; length, structure, or both may trigger."""

    if settings.mode is ImageReplyMode.OFF:
        return False
    if not text.strip():
        return False
    if settings.mode is ImageReplyMode.ALWAYS:
        return True
    if len(text) > settings.min_chars:
        return True
    return has_table(text) or has_long_code_block(text)


async def deliver_answer(
    *,
    text: str,
    sources: Sequence[str] | None,
    settings: ImageReplySettings,
    send_text: Callable[[str], Awaitable[None]],
    send_images: Callable[[Sequence[bytes]], Awaitable[None]],
    renderer: Any = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    on_failure: Callable[[str], None] | None = None,
    fallback_text: str | None = None,
) -> AnswerDeliveryReport:
    """Deliver one answer as image pages when applicable, else as text.

    ``send_text`` sends one text chunk and ``send_images`` sends one batch of
    pages as a single message. ``fallback_text`` is what the text path sends
    (the answer, with the sources footer appended when that switch is on); it
    defaults to ``text``. ``sources=None`` renders no image footer at all.
    ``renderer`` defaults to the process-wide renderer and must offer ``render``
    plus ``close_if_idle``; its browser is released once nothing is being
    rendered. Nothing is silently dropped: every failure path either resends the
    answer as text or reports the lost pages.
    """

    text_body = text if fallback_text is None else fallback_text
    if not should_render(text, settings):
        return AnswerDeliveryReport(
            delivery="text",
            text_report=await _send_text(
                send_text, text_body, settings, sleep=sleep, on_failure=on_failure
            ),
        )

    renderer = renderer or _shared_renderer(settings.timeout_seconds)
    try:
        document = await renderer.render(
            text, sources=None if sources is None else list(sources)
        )
    except RenderError as exc:
        logger.warning(
            "Falling back to text: image rendering failed (%s)",
            type(exc).__name__,
        )
        # Belt and braces: render() already released the browser on its own
        # failure path, including a cancellation, but the delivery path states
        # the policy at its own boundary too.
        await _close_idle_browser(renderer)
        return AnswerDeliveryReport(
            delivery="text",
            render_error=type(exc).__name__,
            text_report=await _send_text(
                send_text, text_body, settings, sleep=sleep, on_failure=on_failure
            ),
        )

    try:
        return await _send_pages(
            document,
            text=text_body,
            settings=settings,
            send_text=send_text,
            send_images=send_images,
            sleep=sleep,
            on_failure=on_failure,
        )
    finally:
        await document.close()
        await _close_idle_browser(renderer)


async def _close_idle_browser(renderer: Any) -> None:
    """Release the renderer's browser once nothing is being rendered.

    Best effort, like the other teardown in this path: the browser is an
    optimisation, so a failing close must not fail a delivery that succeeded.
    """

    try:
        await renderer.close_if_idle()
    except Exception as exc:  # noqa: BLE001 - browser teardown boundary
        logger.debug("Idle browser release failed: %s", type(exc).__name__)


async def _send_text(
    send_text: Callable[[str], Awaitable[None]],
    text: str,
    settings: ImageReplySettings,
    *,
    sleep: Callable[[float], Awaitable[None]] | None,
    on_failure: Callable[[str], None] | None,
) -> DeliveryReport:
    return await send_chunks(
        send_text,
        text,
        settings.chunk_chars,
        delay_seconds=settings.delay_seconds,
        sleep=sleep,
        on_failure=on_failure,
        measure=settings.chunk_measure,
    )


async def _send_pages(
    document: Any,
    *,
    text: str,
    settings: ImageReplySettings,
    send_text: Callable[[str], Awaitable[None]],
    send_images: Callable[[Sequence[bytes]], Awaitable[None]],
    sleep: Callable[[float], Awaitable[None]] | None,
    on_failure: Callable[[str], None] | None,
) -> AnswerDeliveryReport:
    pause = sleep or asyncio.sleep
    slice_cap = settings.max_height
    # The configured budget is honoured as-is; the floor only bounds how far the
    # adaptive shrink may go, so a small configured value is never inflated.
    batch_budget = max(1, settings.batch_max_bytes)
    shrink_floor = min(batch_budget, MIN_BATCH_BYTES)
    sent_pages = 0
    sent_messages = 0
    dropped_pages = 0
    send_attempts = 0
    top = 0
    batch: list[bytes] = []
    batch_regions: list[tuple[int, int]] = []
    batch_bytes = 0
    retry: list[tuple[int, int]] = []
    splits_left = settings.max_splits

    async def drain_retries() -> None:
        """Retry rejected regions in reading order before sending more pages."""

        nonlocal sent_pages, sent_messages, dropped_pages, send_attempts, splits_left
        while retry:
            region_top, region_height = retry.pop(0)
            try:
                page = (await document.slice(region_top, region_height)).data
            except RenderError as exc:
                logger.warning("Retried slice failed to render: %s", type(exc).__name__)
                dropped_pages += 1
                continue
            send_attempts += 1
            if await _send_batch(send_images, [page], on_failure):
                sent_pages += 1
                sent_messages += 1
                if settings.delay_seconds > 0:
                    await pause(settings.delay_seconds)
                continue
            first = document.slice_height(
                region_top, max(MIN_SLICE_HEIGHT, region_height // 2)
            )
            if (
                splits_left > 0
                and region_height > MIN_SLICE_HEIGHT
                and 0 < first < region_height
            ):
                splits_left -= 1
                # Keep the two halves ahead of the remaining regions so the
                # answer stays in reading order even after a split.
                retry[0:0] = [
                    (region_top, first),
                    (region_top + first, region_height - first),
                ]
                continue
            # Nothing further to shrink: this page is genuinely lost and reported.
            dropped_pages += 1

    while True:
        if retry:
            await drain_retries()
            continue
        height = document.slice_height(top, slice_cap)
        if height <= 0:
            break
        try:
            page = (await document.slice(top, height)).data
        except RenderError as exc:
            logger.warning(
                "Image rendering stopped midway (%s); resending as text",
                type(exc).__name__,
            )
            if batch:
                send_attempts += 1
            outcome = await _flush(
                send_images, batch, on_failure, pause, settings.delay_seconds
            )
            sent_pages += outcome[0]
            sent_messages += 1 if outcome[0] else 0
            dropped_pages += outcome[1]
            text_report = await _send_text(
                send_text, text, settings, sleep=sleep, on_failure=on_failure
            )
            return AnswerDeliveryReport(
                delivery="text_fallback",
                sent_pages=sent_pages,
                sent_messages=sent_messages,
                dropped_pages=dropped_pages,
                send_attempts=send_attempts,
                render_error=type(exc).__name__,
                text_report=text_report,
                notice=INTERRUPTED_NOTICE,
            )

        if batch and (
            (
                settings.max_pages_per_message > 0
                and len(batch) >= settings.max_pages_per_message
            )
            or batch_bytes + len(page) > batch_budget
        ):
            send_attempts += 1
            delivered = await _send_batch(send_images, batch, on_failure)
            if delivered:
                sent_pages += len(batch)
                sent_messages += 1
            else:
                # Keep the content: the rejected regions are retried as smaller
                # slices instead of being dropped, like text delivery does.
                retry.extend(batch_regions)
                batch_budget = max(shrink_floor, batch_budget // 2)
                slice_cap = max(MIN_SLICE_HEIGHT, min(slice_cap, height // 2))
                logger.warning(
                    "Image message rejected; retrying smaller slices "
                    "(budget=%s bytes, cap=%s px)",
                    batch_budget,
                    slice_cap,
                )
                # Deliver the failed regions before any later page.
                await drain_retries()
            batch = []
            batch_regions = []
            batch_bytes = 0
            if delivered and settings.delay_seconds > 0:
                await pause(settings.delay_seconds)

        batch.append(page)
        batch_regions.append((top, height))
        batch_bytes += len(page)
        top += height

    if batch:
        send_attempts += 1
        if await _send_batch(send_images, batch, on_failure):
            sent_pages += len(batch)
            sent_messages += 1
        else:
            retry.extend(batch_regions)

    await drain_retries()

    if dropped_pages and not sent_pages:
        # Nothing arrived at all, so this is not partial loss: resend the answer
        # as text instead of leaving the user with a notice and no answer.
        logger.warning(
            "Every image message was rejected (%s page(s)); sending text instead",
            dropped_pages,
        )
        text_report = await _send_text(
            send_text, text, settings, sleep=sleep, on_failure=on_failure
        )
        return AnswerDeliveryReport(
            delivery="text_fallback",
            dropped_pages=dropped_pages,
            send_attempts=send_attempts,
            text_report=text_report,
            notice=IMAGE_FAILED_NOTICE,
        )

    notice = PARTIAL_NOTICE if dropped_pages else None
    return AnswerDeliveryReport(
        delivery="image",
        sent_pages=sent_pages,
        sent_messages=sent_messages,
        dropped_pages=dropped_pages,
        send_attempts=send_attempts,
        notice=notice,
    )


async def _flush(
    send_images: Callable[[Sequence[bytes]], Awaitable[None]],
    batch: list[bytes],
    on_failure: Callable[[str], None] | None,
    pause: Callable[[float], Awaitable[None]],
    delay_seconds: float,
) -> tuple[int, int]:
    """Send the pending batch, returning ``(sent_pages, dropped_pages)``."""

    if not batch:
        return 0, 0
    if await _send_batch(send_images, batch, on_failure):
        if delay_seconds > 0:
            await pause(delay_seconds)
        return len(batch), 0
    return 0, len(batch)


async def _send_batch(
    send_images: Callable[[Sequence[bytes]], Awaitable[None]],
    batch: Sequence[bytes],
    on_failure: Callable[[str], None] | None,
) -> bool:
    try:
        await send_images(list(batch))
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - transport boundary
        if on_failure is not None:
            on_failure(type(exc).__name__)
        return False
    return True


def _shared_renderer(timeout_seconds: float) -> Any:
    # Imported lazily so the plugin keeps working without the render-image extra.
    from .render import shared_renderer

    return shared_renderer(timeout_seconds)
