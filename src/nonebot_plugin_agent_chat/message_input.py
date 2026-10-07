from __future__ import annotations

from collections.abc import Iterable, Sequence

from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna import Image, Reply, UniMessage, image_fetch

from .errors import InputError
from .images import ImageSource, load_image_sources
from .input import (
    CollectedInput,
    compose_prompt,
    remove_first_trigger,
    strip_command_mention,
)


def reply_of(message: UniMessage) -> Reply | None:
    for segment in message:
        if isinstance(segment, Reply):
            return segment
    return None


def _images_of(message: UniMessage | None) -> list[Image]:
    if message is None:
        return []
    return [segment for segment in message if isinstance(segment, Image)]


async def _quoted_message(reply: Reply | None, bot: Bot) -> UniMessage | None:
    if reply is None or reply.msg is None:
        return None
    if isinstance(reply.msg, str):
        return UniMessage.text(reply.msg)
    try:
        return UniMessage.of(reply.msg, bot=bot)
    except Exception:  # noqa: BLE001 - unsupported quoted adapter message
        return None


def _sources(images: Sequence[Image]) -> list[ImageSource]:
    """Map universal images to inline bytes, public URLs, or adapter handles."""

    sources: list[ImageSource] = []
    for image in images:
        if image.raw is not None:
            try:
                sources.append(ImageSource(data=image.raw_bytes))
            except ValueError:
                continue
        elif image.url and image.url.startswith(("http://", "https://", "base64://")):
            sources.append(ImageSource(url=image.url))
        elif image.id:
            sources.append(ImageSource(media=image))
    return sources


def _media_fetcher(bot: Bot, event: Event):
    async def fetch(image: Image) -> bytes | None:
        return await image_fetch(event, bot, {}, image)

    return fetch


async def collect_message_input(
    message: UniMessage,
    *,
    bot: Bot,
    event: Event,
    current_text_override: str | None = None,
    current_message: UniMessage | None = None,
    remove_triggers: Iterable[str] = (),
    bot_username: str | None = None,
    max_reply_chars: int = 8000,
    max_images: int = 4,
    max_image_bytes: int = 10 * 1024 * 1024,
) -> CollectedInput:
    """Collect adapter-neutral text, quoted text, and bounded images.

    ``message`` is the current message with its reply attached, as produced by
    ``UniMessage.of(...).attach_reply(...)``. ``current_message`` optionally
    narrows which images count as current (used by control-command arguments).
    """

    current_text = (
        current_text_override
        if current_text_override is not None
        else message.extract_plain_text()
    )
    current_text = strip_command_mention(current_text, bot_username)
    if remove_triggers:
        current_text, _ = remove_first_trigger(current_text, remove_triggers)

    reply = reply_of(message)
    quoted = await _quoted_message(reply, bot)
    quoted_text = quoted.extract_plain_text() if quoted is not None else ""
    text = compose_prompt(current_text, quoted_text, max_reply_chars)

    image_scope = message if current_message is None else current_message
    sources = _sources(_images_of(image_scope)) + _sources(_images_of(quoted))
    images = await load_image_sources(
        sources,
        max_images=max_images,
        max_image_bytes=max_image_bytes,
        fetch_media=_media_fetcher(bot, event),
    )

    if not text and not images:
        raise InputError("请在触发词后提供问题，或回复一条消息")
    return CollectedInput(text=text, images=images)
