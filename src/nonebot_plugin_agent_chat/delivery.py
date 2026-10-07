from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

SendChunk = Callable[[str], Awaitable[None]]
Sleep = Callable[[float], Awaitable[None]]

DEFAULT_MIN_CHARS = 200
DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 1.5
DEFAULT_MAX_SPLITS = 8

PARTIAL_TEXT_NOTICE = "（部分回复内容发送失败，已省略）"
TOTAL_TEXT_NOTICE = "（回复发送失败，请稍后重试）"


@dataclass(frozen=True)
class DeliveryReport:
    """Outcome of one multi-chunk delivery."""

    sent_chunks: int = 0
    dropped_chunks: int = 0
    dropped_chars: int = 0
    send_attempts: int = 0

    @property
    def complete(self) -> bool:
        return self.dropped_chunks == 0

    @property
    def delivered_anything(self) -> bool:
        return self.sent_chunks > 0


def loss_notice(report: DeliveryReport) -> str | None:
    """User-facing notice for a delivery that lost content.

    A fully dropped delivery still needs a notice: with a large chunk size the
    whole answer can be one chunk, so silence would look like a lost request.
    """

    if report.complete:
        return None
    return PARTIAL_TEXT_NOTICE if report.delivered_anything else TOTAL_TEXT_NOTICE


def _limit_index(text: str, size: int, measure: Callable[[str], int]) -> int:
    """Index at which ``text`` first exceeds ``size`` measured units."""

    # 逐字符累加单位而不是按 len 切，保证不会把代理对（emoji）切成半个。

    used = 0
    for index, char in enumerate(text):
        used += measure(char)
        if used > size:
            return index
    return len(text)


def chunk_text(
    text: str,
    size: int,
    measure: Callable[[str], int] = len,
) -> list[str]:
    """Split text into chat-sized pieces, preferring newline boundaries.

    ``size`` is measured with ``measure`` (characters by default, UTF-16 code
    units for Telegram). ``size == 0`` sends the whole text as one piece;
    negative sizes are invalid.
    """

    if size < 0:
        raise ValueError("size must be >= 0")
    if not text.strip():
        # A whitespace-only payload is rejected by chat transports anyway.
        return []
    if size == 0 or measure(text) <= size:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if measure(remaining) <= size:
            chunks.append(remaining)
            break
        # A single character may exceed the budget (for example an emoji over a
        # 1-unit limit), so always make progress.
        limit = max(1, _limit_index(remaining, size, measure))
        # Split after a late newline when one exists, otherwise at the limit, and
        # keep every character: only the cut position is chosen here.
        newline_at = remaining.rfind("\n", 0, limit)
        split_at = (
            newline_at + 1
            if newline_at >= 0 and measure(remaining[: newline_at + 1]) >= size // 2
            else limit
        )
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    return chunks


def _describe(exc: BaseException) -> str:
    """Summarize a send failure without echoing message content."""

    parts = [type(exc).__name__]
    info = getattr(exc, "info", None)
    if isinstance(info, dict):
        retcode = info.get("retcode")
        if retcode is not None:
            parts.append(f"retcode={retcode}")
        wording = str(info.get("wording") or info.get("message") or "").strip()
        if wording:
            parts.append(wording[:120])
    return " ".join(parts)


async def send_chunks(
    send: SendChunk,
    text: str,
    size: int,
    *,
    delay_seconds: float = 0.0,
    attempts: int = DEFAULT_ATTEMPTS,
    min_chars: int = DEFAULT_MIN_CHARS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    max_splits: int = DEFAULT_MAX_SPLITS,
    sleep: Sleep | None = None,
    on_failure: Callable[[str], None] | None = None,
    measure: Callable[[str], int] = len,
) -> DeliveryReport:
    """Send ``text`` in chunks, adapting to transport limits.

    Long chunks that the transport keeps rejecting are split in half instead of
    being dropped, short chunks are retried with backoff, and one failing chunk
    never aborts the remaining ones. Callers get a report so they can tell the
    user that content was lost without misreporting a fully failed request.
    """

    pause = sleep or asyncio.sleep
    pending: deque[str] = deque(chunk_text(text, size, measure))
    sent_chunks = 0
    dropped_chunks = 0
    dropped_chars = 0
    send_attempts = 0
    splits_left = max_splits

    while pending:
        chunk = pending.popleft()
        delivered = False
        for attempt in range(1, attempts + 1):
            send_attempts += 1
            try:
                await send(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - transport boundary
                if on_failure is not None:
                    on_failure(_describe(exc))
                if len(chunk) > min_chars and splits_left > 0:
                    # Length limits are the likelier cause than a transient
                    # fault, so shrink instead of repeating the same payload.
                    break
                if attempt < attempts:
                    await pause(backoff_seconds * attempt)
                continue
            delivered = True
            break

        if delivered:
            sent_chunks += 1
        elif len(chunk) > min_chars and splits_left > 0:
            splits_left -= 1
            half = len(chunk) // 2
            pending.appendleft(chunk[half:])
            pending.appendleft(chunk[:half])
        else:
            dropped_chunks += 1
            dropped_chars += len(chunk)

        if pending and delay_seconds > 0:
            await pause(delay_seconds)

    return DeliveryReport(
        sent_chunks=sent_chunks,
        dropped_chunks=dropped_chunks,
        dropped_chars=dropped_chars,
        send_attempts=send_attempts,
    )
