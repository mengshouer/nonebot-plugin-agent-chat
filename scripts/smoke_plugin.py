from __future__ import annotations

import asyncio
import os
from typing import Any

import nonebot
from nonebot.adapters.onebot.v11 import Adapter, Bot
from nonebot.adapters.onebot.v11.event import GroupMessageEvent
from nonebot.adapters.onebot.v11.utils import handle_api_result


class RecordingAdapter(Adapter):
    """Register a real OneBot Bot without a live connection.

    API calls are recorded, and the recorded policy injects the same transport
    failures the delivery layer must survive: one oversized text message, one
    rejected image message, and any text longer than the fake server limit.
    """

    def __init__(self, driver: Any, **kwargs: Any) -> None:
        super().__init__(driver, **kwargs)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.failures: list[tuple[str, dict[str, Any]]] = []
        self.fail_text_once = True
        self.fail_image_once = True
        self.text_limit = 400

    async def _call_api(self, bot: Bot, api: str, **data: Any) -> Any:
        message = data.get("message")
        segments = list(message) if message is not None else []
        is_image = bool(segments) and all(
            getattr(segment, "type", None) == "image" for segment in segments
        )
        failed = False
        if is_image:
            failed = self.fail_image_once
            self.fail_image_once = False
        else:
            text = str(message) if message is not None else ""
            failed = self.fail_text_once or len(text) > self.text_limit
            self.fail_text_once = False
        if failed:
            self.failures.append((api, data))
            return handle_api_result(
                {"status": "failed", "retcode": 100, "wording": "fake transport limit"}
            )
        self.calls.append((api, data))
        return handle_api_result(
            {"status": "ok", "retcode": 0, "data": {"message_id": len(self.calls)}}
        )

    def text_messages(self) -> list[str]:
        return [
            str(data["message"])
            for _, data in self.calls
            if data.get("message") is not None
            and all(
                getattr(segment, "type", None) == "text" for segment in data["message"]
            )
        ]

    def image_pages(self) -> int:
        return sum(
            len(data["message"])
            for _, data in self.calls
            if data.get("message") is not None
            and any(
                getattr(segment, "type", None) == "image" for segment in data["message"]
            )
        )


async def main() -> None:
    nonebot.init(driver="~none", superusers={"1"})
    driver = nonebot.get_driver()
    driver.register_adapter(RecordingAdapter)
    plugin = nonebot.load_plugin("nonebot_plugin_agent_chat")
    if plugin is None:
        raise RuntimeError("installed plugin did not load")

    await driver._lifespan.startup()
    try:
        from nonebot_plugin_agent_chat import matchers
        from nonebot_plugin_agent_chat.matchers import service

        if service.startup_error is not None:
            raise RuntimeError(service.startup_error)
        if service.active_profile_name != "smoke":
            raise RuntimeError("unexpected active profile")
        for name in ("llm", "agentctl", "agent_room"):
            if not hasattr(matchers, name):
                raise RuntimeError(f"matcher {name} was not registered")
        for name, command in (
            ("agentctl", matchers._agentctl_command),
            ("agent_room", matchers._agent_room_command),
        ):
            parsed = command.parse(f"/{name} use my profile")
            if not parsed.matched:
                raise RuntimeError(
                    f"registered {name} command does not match its command line"
                )
            parts = tuple(parsed.all_matched_args.get("parts") or ())
            if parts != ("use", "my", "profile"):
                raise RuntimeError(f"unexpected {name} parse: {parts}")
            if command.parse(f"/{name}@otherbot status").matched:
                raise RuntimeError(f"{name} matched a command addressed to another bot")
        await _check_text_delivery(matchers)
        await _check_image_delivery(matchers)
        await _check_generic_platform(matchers)
    finally:
        await driver._lifespan.shutdown()
    print("installed NoneBot lifecycle smoke: ok")


def _recording_adapter() -> RecordingAdapter:
    from nonebot import get_driver

    return RecordingAdapter(get_driver())


def _bot(adapter: RecordingAdapter, self_id: str = "1") -> Bot:
    return Bot(adapter, self_id)


def _fresh_scene(
    *,
    fail_text_once: bool = False,
    fail_image_once: bool = False,
) -> tuple[RecordingAdapter, Bot, GroupMessageEvent]:
    """A cleared recording adapter with a real OneBot group event."""

    adapter = _recording_adapter()
    adapter.calls.clear()
    adapter.failures.clear()
    adapter.fail_text_once = fail_text_once
    adapter.fail_image_once = fail_image_once
    return adapter, _bot(adapter), _group_event()


def _group_event() -> GroupMessageEvent:
    return GroupMessageEvent.model_validate(
        {
            "time": 1,
            "self_id": 1,
            "post_type": "message",
            "message_type": "group",
            "sub_type": "normal",
            "message_id": 1,
            "user_id": 2,
            "group_id": 3,
            "message": [{"type": "text", "data": {"text": "hi"}}],
            "raw_message": "hi",
            "font": 0,
            "sender": {"user_id": 2, "nickname": "u"},
        }
    )


async def _check_text_delivery(matchers: object) -> None:
    """Prove the wired text path retries and splits instead of raising."""

    from nonebot_plugin_agent_chat.platforms import resolve_chat_identity

    adapter, bot, event = _fresh_scene(fail_text_once=True)
    identity = resolve_chat_identity(bot, event)
    if identity is None:
        raise RuntimeError("OneBot group identity was not resolved")
    body = "z" * 1200
    await matchers._deliver_result(bot, event, _result(body, sources=[]), identity)
    if "".join(adapter.text_messages()) != body:
        raise RuntimeError("chunked delivery lost content")
    if len(adapter.text_messages()) < 3:
        raise RuntimeError(
            f"expected adaptive splitting, got {len(adapter.text_messages())} chunks"
        )


async def _check_image_delivery(matchers: object) -> None:
    """Prove an image-mode answer is rendered and sent without a request."""

    from nonebot_plugin_agent_chat.matchers import plugin_config
    from nonebot_plugin_agent_chat.models import ImageReplyMode
    from nonebot_plugin_agent_chat.render import probe_renderer

    available, detail = probe_renderer()
    if not available:
        if os.environ.get("REQUIRE_RENDERER") == "1":
            raise RuntimeError(f"required image renderer unavailable: {detail}")
        print(f"image delivery smoke skipped: {detail}")
        return
    from nonebot_plugin_agent_chat.platforms import resolve_chat_identity

    adapter, bot, event = _fresh_scene(fail_image_once=True)
    identity = resolve_chat_identity(bot, event)
    if identity is None:
        raise RuntimeError("OneBot group identity was not resolved")
    previous = plugin_config.agent_chat_image_reply_mode
    plugin_config.agent_chat_image_reply_mode = ImageReplyMode.ALWAYS
    try:
        await matchers._deliver_result(
            bot,
            event,
            _result("# 标题\n\n中文段落。", sources=[]),
            identity,
        )
    finally:
        plugin_config.agent_chat_image_reply_mode = previous  # type: ignore[misc]
    if adapter.image_pages() == 0:
        raise RuntimeError("image delivery sent no image segments")
    if adapter.text_messages():
        raise RuntimeError("image delivery also sent the text body")


async def _check_generic_platform(matchers: object) -> None:
    """An adapter outside the registry still resolves and sends plain text."""

    import nonebot_plugin_alconna

    from nonebot_plugin_agent_chat import platforms

    adapter, bot, event = _fresh_scene()
    # A real upstream scope without an agent-chat-specific delivery policy.
    target = nonebot_plugin_alconna.Target(
        id="room-3",
        channel=True,
        scope=nonebot_plugin_alconna.SupportScope.discord,
    )
    original = nonebot_plugin_alconna.get_target
    nonebot_plugin_alconna.get_target = lambda event, bot: target
    try:
        identity = platforms.resolve_chat_identity(bot, event)
    finally:
        nonebot_plugin_alconna.get_target = original
    if identity is None or identity.scope != "Discord":
        raise RuntimeError(f"generic adapter identity was not resolved: {identity}")
    if platforms.answer_format_for(identity.scope).name != "plain":
        raise RuntimeError("generic adapter should use plain answer text")
    await matchers._deliver_result(
        bot, event, _result("generic answer", sources=[]), identity
    )
    if adapter.text_messages() != ["generic answer"]:
        raise RuntimeError("generic adapter delivery did not send plain text")
    if adapter.image_pages() != 0:
        raise RuntimeError("generic adapter must not render images by default")


def _result(text: str, sources: list) -> object:
    from nonebot_plugin_agent_chat.models import RunResult, Usage

    return RunResult(
        text=text,
        sources=sources,
        usage=Usage(),
        model_turns=1,
        local_tool_calls=0,
        searches=0,
        actual_profile="smoke",
    )


asyncio.run(main())
