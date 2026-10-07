import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nonebot.exception import ActionFailed
from nonebot_plugin_alconna import UniMessage

from nonebot_plugin_agent_chat.answer_delivery import send_answer_text


class VendorActionFailed(ActionFailed):
    """A third-party adapter subclass: the core base class is what matters."""


class NetworkError(Exception):
    pass


def fake_bot(adapter_name: str) -> SimpleNamespace:
    adapter = SimpleNamespace(get_name=lambda: adapter_name)
    return SimpleNamespace(adapter=adapter)


class SendAnswerTextTests(unittest.IsolatedAsyncioTestCase):
    async def _send(
        self,
        scope: str,
        text: str,
        failures: list[Exception | None],
    ) -> list[tuple[UniMessage, dict, Exception | None]]:
        """Run one delivery, recording every attempt (including the failures)."""

        bot = fake_bot(scope)
        attempts: list[tuple[UniMessage, dict, Exception | None]] = []

        async def fake_send(self, *args, **kwargs):
            failure = failures.pop(0) if failures else None
            attempts.append((self, kwargs, failure))
            if failure is not None:
                raise failure

        with patch.object(UniMessage, "send", new=fake_send):
            await send_answer_text(bot, object(), text, scope)
        return attempts

    async def test_plain_adapter_sends_once(self) -> None:
        attempts = await self._send("QQClient", "**x**", [])

        self.assertEqual(len(attempts), 1)
        message, kwargs, failure = attempts[0]
        self.assertIsNone(failure)
        self.assertEqual(message.extract_plain_text(), "**x**")
        self.assertNotIn("parse_mode", kwargs)

    async def test_telegram_sends_html(self) -> None:
        attempts = await self._send("Telegram", "**x**", [])

        self.assertEqual(len(attempts), 1)
        message, kwargs, failure = attempts[0]
        self.assertIsNone(failure)
        self.assertEqual(message.extract_plain_text(), "<b>x</b>")
        self.assertEqual(kwargs["parse_mode"], "HTML")

    async def test_markup_rejection_falls_back_to_plain_text(self) -> None:
        attempts = await self._send(
            "Telegram",
            "**x**",
            [
                VendorActionFailed(
                    "vendor",
                    "Bad Request: can't parse entities: unexpected end tag",
                )
            ],
        )

        self.assertEqual(len(attempts), 2)
        first, second = attempts
        self.assertIsInstance(first[2], ActionFailed)
        self.assertEqual(first[1]["parse_mode"], "HTML")
        self.assertIsNone(second[2])
        self.assertEqual(second[0].extract_plain_text(), "**x**")
        self.assertNotIn("parse_mode", second[1])

    async def test_transport_error_is_raised_without_plain_retry(self) -> None:
        with self.assertRaises(NetworkError):
            await self._send("Telegram", "**x**", [NetworkError("network down")])

    async def test_non_markup_api_error_is_raised_without_plain_retry(self) -> None:
        # "message is too long" must reach the delivery layer, which splits and
        # retries with pacing instead of posting the same chunk twice.
        with self.assertRaises(ActionFailed):
            await self._send(
                "Telegram",
                "x" * 5000,
                [ActionFailed("vendor", "Bad Request: message is too long")],
            )


if __name__ == "__main__":
    unittest.main()
