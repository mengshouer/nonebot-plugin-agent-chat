import base64
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nonebot.adapters.onebot.v11.event import (
    GroupMessageEvent,
    PrivateMessageEvent,
    Reply,
)
from nonebot.adapters.telegram.event import Event as TelegramEvent
from nonebot_plugin_alconna import SupportAdapter, UniMessage

from nonebot_plugin_agent_chat.alconna_ext import CommandMentionExtension
from nonebot_plugin_agent_chat.errors import InputError
from nonebot_plugin_agent_chat.message_input import collect_message_input
from nonebot_plugin_agent_chat.platforms import (
    is_identity_allowed,
    resolve_chat_identity,
)

# Adapter display name: the framework's UniSeg loader is registered under it.
ONEBOT_V11 = "OneBot V11"

PNG = b"\x89PNG\r\n\x1a\n" + b"payload"


def fake_bot(adapter_name: str, **attrs: object) -> SimpleNamespace:
    adapter = SimpleNamespace(get_name=lambda: adapter_name)
    return SimpleNamespace(adapter=adapter, self_id="10000", **attrs)


def onebot_group_event(
    text: str,
    *,
    message: list[dict] | None = None,
    reply: bool = True,
) -> GroupMessageEvent:
    event = GroupMessageEvent.model_validate(
        {
            "time": 1,
            "self_id": 10000,
            "post_type": "message",
            "message_type": "group",
            "sub_type": "normal",
            "message_id": 1,
            "user_id": 20000,
            "group_id": 30000,
            "message": message or [{"type": "text", "data": {"text": text}}],
            "raw_message": text,
            "font": 0,
            "sender": {"user_id": 20000, "nickname": "u"},
        }
    )
    event.reply = (
        Reply.model_validate(
            {
                "time": 1,
                "message_type": "group",
                "message_id": 2,
                "real_id": 3,
                "sender": {"user_id": 20000, "nickname": "u"},
                "message": [{"type": "text", "data": {"text": "quoted text"}}],
            }
        )
        if reply
        else None
    )
    return event


def telegram_event(message: dict) -> TelegramEvent:
    return TelegramEvent.parse_event({"update_id": 1, "message": message})


def telegram_group_message(**extra: object) -> dict:
    payload = {
        "message_id": 10,
        "date": 1,
        "chat": {"id": -100123, "type": "supergroup", "title": "g"},
        "from": {"id": 42, "is_bot": False, "first_name": "u"},
    }
    payload.update(extra)
    return payload


PHOTO = [
    {"file_id": "small", "file_unique_id": "u1", "width": 10, "height": 10},
    {"file_id": "big", "file_unique_id": "u2", "width": 100, "height": 100},
]


class OneBotInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_trigger_and_quoted_text(self) -> None:
        bot = fake_bot(ONEBOT_V11)
        event = onebot_group_event("/llm hello")
        message = await UniMessage.of(event.get_message(), bot=bot).attach_reply(
            event=event,
            bot=bot,
        )

        collected = await collect_message_input(
            message,
            bot=bot,
            event=event,
            remove_triggers=["/llm"],
        )

        self.assertIn(
            "<quoted_message>\nquoted text\n</quoted_message>", collected.text
        )
        self.assertIn("User request:\nhello", collected.text)
        self.assertEqual(collected.images, [])

    async def test_base64_image_is_loaded(self) -> None:
        bot = fake_bot(ONEBOT_V11)
        source = "base64://" + base64.b64encode(PNG).decode("ascii")
        event = onebot_group_event(
            "/llm look",
            message=[
                {"type": "text", "data": {"text": "/llm look"}},
                {"type": "image", "data": {"file": source}},
            ],
            reply=False,
        )
        message = await UniMessage.of(event.get_message(), bot=bot).attach_reply(
            event=event,
            bot=bot,
        )

        collected = await collect_message_input(
            message,
            bot=bot,
            event=event,
            remove_triggers=["/llm"],
        )

        self.assertEqual(collected.text, "look")
        self.assertEqual(len(collected.images), 1)
        self.assertEqual(collected.images[0].data, PNG)


class TelegramInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_command_mention_is_normalized(self) -> None:
        bot = fake_bot(SupportAdapter.telegram.value, username="mybot")
        event = telegram_event(
            telegram_group_message(
                text="/llm@mybot hello",
                entities=[
                    {
                        "type": "bot_command",
                        "offset": 0,
                        "length": len("/llm@mybot"),
                    }
                ],
            )
        )
        message = await UniMessage.of(event.get_message(), bot=bot).attach_reply(
            event=event,
            bot=bot,
        )

        collected = await collect_message_input(
            message,
            bot=bot,
            event=event,
            remove_triggers=["/llm"],
            bot_username="mybot",
        )

        self.assertEqual(collected.text, "hello")

    async def test_photo_is_resolved_through_adapter_fetch(self) -> None:
        bot = fake_bot(SupportAdapter.telegram.value, username="mybot")
        event = telegram_event(
            telegram_group_message(caption="/llm describe", photo=PHOTO)
        )
        message = await UniMessage.of(event.get_message(), bot=bot).attach_reply(
            event=event,
            bot=bot,
        )
        fetched: list[str] = []

        async def fake_fetch(event, bot, state, img, **kwargs):
            fetched.append(img.id)
            return PNG

        with patch(
            "nonebot_plugin_agent_chat.message_input.image_fetch",
            fake_fetch,
        ):
            collected = await collect_message_input(
                message,
                bot=bot,
                event=event,
                remove_triggers=["/llm"],
                bot_username="mybot",
            )

        self.assertEqual(fetched, ["big"])
        self.assertEqual(collected.text, "describe")
        self.assertEqual(len(collected.images), 1)
        self.assertEqual(collected.images[0].data, PNG)

    async def test_quoted_photo_without_text_uses_reply(self) -> None:
        bot = fake_bot(SupportAdapter.telegram.value, username="mybot")
        event = telegram_event(
            telegram_group_message(
                text="/llm",
                reply_to_message={
                    "message_id": 9,
                    "date": 1,
                    "chat": {"id": -100123, "type": "supergroup", "title": "g"},
                    "from": {"id": 7, "is_bot": False, "first_name": "other"},
                    "caption": "quoted caption",
                    "photo": PHOTO,
                },
            )
        )
        message = await UniMessage.of(event.get_message(), bot=bot).attach_reply(
            event=event,
            bot=bot,
        )

        async def fake_fetch(event, bot, state, img, **kwargs):
            return PNG

        with patch(
            "nonebot_plugin_agent_chat.message_input.image_fetch",
            fake_fetch,
        ):
            collected = await collect_message_input(
                message,
                bot=bot,
                event=event,
                remove_triggers=["/llm"],
                bot_username="mybot",
            )

        self.assertIn("quoted caption", collected.text)
        self.assertEqual(len(collected.images), 1)

    async def test_adapter_failures_are_sanitized(self) -> None:
        bot = fake_bot(SupportAdapter.telegram.value, username="mybot")
        event = telegram_event(telegram_group_message(caption="/llm", photo=PHOTO))
        message = await UniMessage.of(event.get_message(), bot=bot).attach_reply(
            event=event,
            bot=bot,
        )
        secret = "https://api.telegram.org/file/bot123:SECRET/file.jpg"

        async def failing_fetch(event, bot, state, img, **kwargs):
            raise RuntimeError(f"request failed for {secret}")

        with (
            patch(
                "nonebot_plugin_agent_chat.message_input.image_fetch",
                failing_fetch,
            ),
            self.assertRaises(InputError) as raised,
        ):
            await collect_message_input(
                message,
                bot=bot,
                event=event,
                remove_triggers=["/llm"],
                bot_username="mybot",
            )

        self.assertEqual(str(raised.exception), "下载图片失败")
        self.assertNotIn("SECRET", str(raised.exception))

    async def test_image_total_budget_is_enforced(self) -> None:
        bot = fake_bot(SupportAdapter.telegram.value, username="mybot")
        event = telegram_event(telegram_group_message(caption="/llm", photo=PHOTO))
        message = await UniMessage.of(event.get_message(), bot=bot).attach_reply(
            event=event,
            bot=bot,
        )

        async def fake_fetch(event, bot, state, img, **kwargs):
            return PNG

        with (
            patch(
                "nonebot_plugin_agent_chat.message_input.image_fetch",
                fake_fetch,
            ),
            self.assertRaises(InputError),
        ):
            await collect_message_input(
                message,
                bot=bot,
                event=event,
                remove_triggers=["/llm"],
                bot_username="mybot",
                max_image_bytes=4,
            )


class TelegramMentionExtensionTests(unittest.IsolatedAsyncioTestCase):
    async def test_extension_normalizes_own_username(self) -> None:
        bot = fake_bot(SupportAdapter.telegram.value, username="mybot")
        event = telegram_event(
            telegram_group_message(
                text="/agentctl@mybot status",
                entities=[
                    {
                        "type": "bot_command",
                        "offset": 0,
                        "length": len("/agentctl@mybot"),
                    }
                ],
            )
        )
        receive = UniMessage.of(event.get_message(), bot=bot)

        normalized = await CommandMentionExtension().receive_wrapper(
            bot,
            event,
            None,  # type: ignore[arg-type] - command is unused by the wrapper
            receive,
        )

        self.assertEqual(normalized.extract_plain_text(), "/agentctl status")
        # The cached original message must stay untouched.
        self.assertEqual(receive.extract_plain_text(), "/agentctl@mybot status")

    async def test_extension_ignores_other_bot_mentions(self) -> None:
        bot = fake_bot(SupportAdapter.telegram.value, username="mybot")
        event = telegram_event(
            telegram_group_message(
                text="/agentctl@otherbot status",
                entities=[
                    {
                        "type": "bot_command",
                        "offset": 0,
                        "length": len("/agentctl@otherbot"),
                    }
                ],
            )
        )
        receive = UniMessage.of(event.get_message(), bot=bot)

        normalized = await CommandMentionExtension().receive_wrapper(
            bot,
            event,
            None,  # type: ignore[arg-type]
            receive,
        )

        self.assertEqual(normalized.extract_plain_text(), "/agentctl@otherbot status")


class IdentityResolutionTests(unittest.TestCase):
    def test_onebot_group_event_uses_the_generic_scoped_keys(self) -> None:
        bot = fake_bot(ONEBOT_V11)
        event = onebot_group_event("hi")

        identity = resolve_chat_identity(bot, event)

        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertEqual(identity.scope, "QQClient")
        self.assertEqual(
            json.loads(identity.subject_key),
            {"bot": "10000", "scope": "QQClient", "user": "20000", "v": 1},
        )
        self.assertEqual(
            json.loads(identity.context_key),
            {
                "bot": "10000",
                "scope": "QQClient",
                "target": {
                    "channel": False,
                    "id": "30000",
                    "parent_id": "",
                    "private": False,
                },
                "user": "20000",
                "v": 1,
            },
        )

    def test_onebot_private_event_uses_the_generic_scoped_keys(self) -> None:
        bot = fake_bot(ONEBOT_V11)
        event = PrivateMessageEvent.model_validate(
            {
                "time": 1,
                "self_id": 10000,
                "post_type": "message",
                "message_type": "private",
                "sub_type": "friend",
                "message_id": 1,
                "user_id": 20000,
                "message": [{"type": "text", "data": {"text": "hi"}}],
                "raw_message": "hi",
                "font": 0,
                "sender": {"user_id": 20000, "nickname": "u"},
            }
        )

        identity = resolve_chat_identity(bot, event)

        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertEqual(identity.scope, "QQClient")
        self.assertEqual(
            json.loads(identity.context_key),
            {
                "bot": "10000",
                "scope": "QQClient",
                "target": {
                    "channel": False,
                    "id": "20000",
                    "parent_id": "",
                    "private": True,
                },
                "user": "20000",
                "v": 1,
            },
        )

    def test_telegram_forum_thread_isolated(self) -> None:
        bot = fake_bot(SupportAdapter.telegram.value, username="mybot")
        event = telegram_event(
            telegram_group_message(
                text="hi",
                is_topic_message=True,
                message_thread_id=7,
            )
        )

        identity = resolve_chat_identity(bot, event)

        self.assertIsNotNone(identity)
        assert identity is not None
        self.assertEqual(identity.scope, "Telegram")
        self.assertEqual(identity.isolation, "7")
        self.assertEqual(
            json.loads(identity.context_key),
            {
                "bot": "10000",
                "isolation": "7",
                "scope": "Telegram",
                "target": {
                    "channel": False,
                    "id": "-100123",
                    "parent_id": "",
                    "private": False,
                },
                "user": "42",
                "v": 1,
            },
        )

    def test_telegram_channel_post_is_rejected(self) -> None:
        bot = fake_bot(SupportAdapter.telegram.value)
        event = TelegramEvent.parse_event(
            {
                "update_id": 2,
                "channel_post": {
                    "message_id": 11,
                    "date": 1,
                    "chat": {"id": -1009, "type": "channel", "title": "c"},
                    "text": "hi",
                },
            }
        )

        self.assertIsNone(resolve_chat_identity(bot, event))

    def test_scoped_acl_required_for_telegram(self) -> None:
        bot = fake_bot(SupportAdapter.telegram.value)
        event = telegram_event(telegram_group_message(text="hi"))
        identity = resolve_chat_identity(bot, event)
        assert identity is not None

        self.assertFalse(
            is_identity_allowed(
                identity,
                allowed_groups={"-100123"},
                allowed_users=set(),
            )
        )
        self.assertTrue(
            is_identity_allowed(
                identity,
                allowed_groups={"Telegram:-100123"},
                allowed_users=set(),
            )
        )


if __name__ == "__main__":
    unittest.main()
