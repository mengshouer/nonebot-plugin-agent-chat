"""Identity and delivery checks for upstream scopes without a local policy."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nonebot_plugin_alconna import Target, UniMessage
from nonebot_plugin_alconna.uniseg.adapters.onebot11.exporter import (
    Onebot11MessageExporter,
)
from nonebot_plugin_alconna.uniseg.adapters.telegram.exporter import (
    TelegramMessageExporter,
)

from nonebot_plugin_agent_chat import platforms
from nonebot_plugin_agent_chat.answer_delivery import send_answer_text
from nonebot_plugin_agent_chat.delivery import send_chunks
from nonebot_plugin_agent_chat.image_reply import ImageReplySettings
from nonebot_plugin_agent_chat.models import ImageReplyMode
from nonebot_plugin_agent_chat.platforms import (
    SupportScope,
    is_identity_allowed,
    resolve_chat_identity,
)


class FakeEvent:
    def __init__(self, user_id: str = "u1") -> None:
        self._user_id = user_id

    def get_type(self) -> str:
        return "message"

    def get_user_id(self) -> str:
        return self._user_id


def fake_bot(adapter_name: str = "MyRPC", self_id: str = "1") -> SimpleNamespace:
    adapter = SimpleNamespace(get_name=lambda: adapter_name)
    return SimpleNamespace(adapter=adapter, self_id=self_id)


class GenericIdentityTests(unittest.TestCase):
    def test_upstream_scope_without_a_policy_gets_generic_identity(self) -> None:
        target = Target(
            "room-3", channel=True, adapter="MyRPC", scope=SupportScope.discord
        )
        with patch("nonebot_plugin_alconna.get_target", return_value=target):
            identity = resolve_chat_identity(fake_bot(), FakeEvent())
        assert identity is not None
        self.assertEqual(identity.scope, "Discord")
        self.assertEqual(identity.acl_entry, "Discord:room-3")
        self.assertEqual(platforms.max_image_pages_for(identity.scope), 0)
        self.assertEqual(platforms.resolve_chunk_size(1000, None, identity.scope), 1000)

    def test_onebot_scope_comes_from_the_upstream_exporter(self) -> None:
        target = Onebot11MessageExporter().get_target(SimpleNamespace(group_id=12345))
        with patch("nonebot_plugin_alconna.get_target", return_value=target):
            identity = resolve_chat_identity(fake_bot("OneBot V11"), FakeEvent("42"))
        assert identity is not None
        self.assertEqual(identity.scope, SupportScope.qq_client.value)
        self.assertEqual(identity.acl_entry, "QQClient:12345")

    def test_telegram_scope_comes_from_the_upstream_exporter(self) -> None:
        event = SimpleNamespace(
            chat=SimpleNamespace(id=-100123, type="supergroup"), message_thread_id=7
        )
        target = TelegramMessageExporter().get_target(event)
        with patch("nonebot_plugin_alconna.get_target", return_value=target):
            identity = resolve_chat_identity(fake_bot("Telegram"), FakeEvent("42"))
        assert identity is not None
        self.assertEqual(identity.scope, SupportScope.telegram.value)
        self.assertEqual(identity.isolation, "7")

    def test_adapter_name_is_not_a_substitute_for_missing_scope(self) -> None:
        target = Target("-100123", adapter="Telegram")
        with (
            patch("nonebot_plugin_alconna.get_target", return_value=target),
            self.assertLogs("nonebot_plugin_agent_chat.platforms", level="WARNING"),
        ):
            self.assertIsNone(resolve_chat_identity(fake_bot("Telegram"), FakeEvent()))

    def test_invalid_declared_scope_is_rejected(self) -> None:
        target = Target("room-3", adapter="MyRPC", extra={"scope": "discord"})
        with (
            patch("nonebot_plugin_alconna.get_target", return_value=target),
            self.assertLogs("nonebot_plugin_agent_chat.platforms", level="WARNING"),
        ):
            self.assertIsNone(resolve_chat_identity(fake_bot(), FakeEvent()))

    def test_generic_acl_requires_the_exact_scope_prefix(self) -> None:
        target = Target(
            "room-3", channel=True, adapter="MyRPC", scope=SupportScope.discord
        )
        with patch("nonebot_plugin_alconna.get_target", return_value=target):
            identity = resolve_chat_identity(fake_bot(), FakeEvent())
        assert identity is not None
        self.assertTrue(
            is_identity_allowed(
                identity, allowed_groups={"Discord:room-3"}, allowed_users=set()
            )
        )
        for entry in ("room-3", "discord:room-3", "MyRPC:room-3"):
            with self.subTest(entry=entry):
                self.assertFalse(
                    is_identity_allowed(
                        identity, allowed_groups={entry}, allowed_users=set()
                    )
                )


class GenericDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_generic_platform_sends_plain_text_without_split(self) -> None:
        attempts: list[tuple[UniMessage, dict]] = []

        async def fake_send(message, *args, **kwargs):
            attempts.append((message, kwargs))

        with patch.object(UniMessage, "send", new=fake_send):
            await send_answer_text(
                fake_bot("MyRPC"), object(), "# title **x**", "Discord"
            )
        self.assertEqual(len(attempts), 1)
        message, kwargs = attempts[0]
        self.assertNotIn("parse_mode", kwargs)
        self.assertEqual(message.extract_plain_text(), "# title **x**")

    async def test_generic_platform_keeps_a_long_answer_in_one_chunk(self) -> None:
        sent: list[str] = []

        async def send(text: str) -> None:
            sent.append(text)

        settings = ImageReplySettings(mode=ImageReplyMode.OFF, chunk_chars=0)
        report = await send_chunks(send, "z" * 9000, settings.chunk_chars)
        self.assertTrue(report.complete)
        self.assertEqual(sent, ["z" * 9000])


if __name__ == "__main__":
    unittest.main()
