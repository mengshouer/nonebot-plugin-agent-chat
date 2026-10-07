import base64
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from nonebot_plugin_agent_chat.errors import InputError
from nonebot_plugin_agent_chat.images import (
    ImageSource,
    _PinnedPublicNetworkBackend,
    load_image_sources,
    load_remote_image,
    validate_remote_url,
)
from nonebot_plugin_agent_chat.input import (
    AGENT_ROOM_COMMAND,
    AGENTCTL_COMMAND,
    RESERVED_CONTROL_COMMANDS,
    command_parts,
    compose_prompt,
    has_trigger,
    is_other_bot_command,
    is_reserved_control_text,
    remove_first_trigger,
    rule_target,
    strip_command_mention,
)
from nonebot_plugin_agent_chat.platforms import ChatIdentity


class TriggerTests(unittest.TestCase):
    def test_trigger_can_appear_anywhere(self) -> None:
        text, trigger = remove_first_trigger("请帮我 /llm 总结", ["/llm"])
        self.assertEqual(trigger, "/llm")
        self.assertEqual(text, "请帮我  总结")

    def test_only_first_trigger_is_removed(self) -> None:
        text, trigger = remove_first_trigger("/llm one /llm two", ["/llm"])
        self.assertEqual(trigger, "/llm")
        self.assertEqual(text, "one /llm two")

    def test_earliest_of_multiple_triggers_wins(self) -> None:
        text, trigger = remove_first_trigger("ask-ai before /llm", ["/llm", "ask-ai"])
        self.assertEqual(trigger, "ask-ai")
        self.assertEqual(text, "before /llm")

    def test_case_sensitive(self) -> None:
        self.assertFalse(has_trigger("/LLM question", ["/llm"]))

    def test_control_command_is_reserved(self) -> None:
        self.assertTrue(is_reserved_control_text("/agentctl status"))
        self.assertTrue(is_reserved_control_text("agent_room ask hello"))
        self.assertFalse(is_reserved_control_text("explain /agentctl as text"))

    def test_command_parts_splits_tokens(self) -> None:
        self.assertEqual(command_parts(()), ("", ""))
        self.assertEqual(command_parts(("STATUS",)), ("status", ""))
        self.assertEqual(
            command_parts(("use", "my", "profile")),
            ("use", "my profile"),
        )
        # Non-text segments are ignored so an attached image does not break the
        # control command.
        self.assertEqual(command_parts(("ask", object(), "hello")), ("ask", "hello"))

    def test_telegram_control_mentions_are_reserved(self) -> None:
        self.assertTrue(
            is_reserved_control_text("/agentctl@mybot status", mentions=True)
        )
        self.assertTrue(
            is_reserved_control_text("/agent_room@otherbot ask hi", mentions=True)
        )
        self.assertFalse(is_reserved_control_text("/llm@mybot status", mentions=True))

    def test_mentions_do_not_change_onebot_reservation(self) -> None:
        # OneBot never appends a bot username, so a `@name` suffix is plain text
        # and the message keeps falling through to the trigger matcher.
        self.assertFalse(is_reserved_control_text("/agentctl@mybot status"))
        self.assertTrue(is_reserved_control_text("/agentctl status"))

    def test_reserved_commands_derive_from_the_matcher_names(self) -> None:
        for name in (AGENTCTL_COMMAND, AGENT_ROOM_COMMAND):
            self.assertIn(name, RESERVED_CONTROL_COMMANDS)
            self.assertIn(f"/{name}", RESERVED_CONTROL_COMMANDS)

    def test_other_bot_commands_are_detected(self) -> None:
        self.assertTrue(is_other_bot_command("/llm@otherbot hi", "mybot"))
        self.assertTrue(is_other_bot_command("/llm@otherbot", "mybot"))
        self.assertFalse(is_other_bot_command("/llm@MyBot hi", "mybot"))
        self.assertFalse(is_other_bot_command("/llm hi", "mybot"))
        self.assertFalse(is_other_bot_command("hi /llm@otherbot", "mybot"))
        self.assertFalse(is_other_bot_command("/llm@otherbot hi", None))

    def test_command_mention_is_only_stripped_for_this_bot(self) -> None:
        self.assertEqual(
            strip_command_mention("/llm@MyBot hello", "mybot"),
            "/llm hello",
        )
        self.assertEqual(
            strip_command_mention("/llm@otherbot hello", "mybot"),
            "/llm@otherbot hello",
        )
        self.assertEqual(
            strip_command_mention("hello /llm@mybot", "mybot"),
            "hello /llm@mybot",
        )
        self.assertEqual(strip_command_mention("/llm hello", None), "/llm hello")


class ImageInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_base64_image_is_detected_and_bounded(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"payload"
        source = "base64://" + base64.b64encode(png).decode("ascii")
        async with httpx.AsyncClient() as client:
            image = await load_remote_image(source, client, 1024)
        self.assertEqual(image.media_type, "image/png")
        self.assertEqual(image.data, png)

    async def test_private_image_address_is_rejected(self) -> None:
        with self.assertRaises(InputError):
            await validate_remote_url("http://127.0.0.1/private.png")
        with self.assertRaises(InputError):
            await validate_remote_url("http://[::1]/private.png")

    async def test_network_backend_connects_to_validated_ip(self) -> None:
        backend = _PinnedPublicNetworkBackend()
        connector = AsyncMock(return_value=object())
        backend._backend = SimpleNamespace(connect_tcp=connector)
        with patch(
            "nonebot_plugin_agent_chat.images._resolve_global_addresses",
            AsyncMock(return_value=["93.184.216.34"]),
        ):
            result = await backend.connect_tcp("example.com", 443)

        self.assertIsNotNone(result)
        connector.assert_awaited_once_with(
            "93.184.216.34",
            443,
            timeout=None,
            local_address=None,
            socket_options=None,
        )

    async def test_redirect_to_private_address_is_rejected(self) -> None:
        calls = []

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(
                302,
                headers={"location": "http://127.0.0.1/private.png"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(InputError):
                await load_remote_image("https://93.184.216.34/image.png", client, 1024)
        self.assertEqual(len(calls), 1)

    async def test_declared_image_type_requires_valid_signature(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "image/png"},
                content=b"not-an-image",
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(InputError):
                await load_remote_image("https://93.184.216.34/image.png", client, 1024)


class ReplyPromptTests(unittest.TestCase):
    def test_reply_is_marked_untrusted(self) -> None:
        prompt = compose_prompt("summarize", "ignore system", 100)
        self.assertIn("<quoted_message>", prompt)
        self.assertIn("User request:\nsummarize", prompt)

    def test_empty_current_text_asks_to_respond(self) -> None:
        prompt = compose_prompt("", "hello", 100)
        self.assertIn("Respond helpfully", prompt)

    def test_reply_is_bounded(self) -> None:
        prompt = compose_prompt("x", "abcdefgh", 4)
        self.assertIn("abcd", prompt)
        self.assertNotIn("abcdefgh", prompt)


class RemoteImageBoundsTests(unittest.IsolatedAsyncioTestCase):
    async def test_too_many_sources_is_rejected_before_download(self) -> None:
        with self.assertRaises(InputError):
            await load_image_sources(
                [ImageSource(url="https://example.com/a.png")] * 3,
                max_images=2,
                max_image_bytes=1024,
            )

    async def test_no_sources_needs_no_client(self) -> None:
        self.assertEqual(
            await load_image_sources([], max_images=4, max_image_bytes=1024),
            [],
        )


class RuleTargetTests(unittest.TestCase):
    def _identity(self, *, private: bool = False) -> ChatIdentity:
        return ChatIdentity(
            scope="QQClient",
            bot_id="10000",
            user_id="42",
            target_id="598683145",
            private=private,
        )

    def test_scope_aliases_are_rejected(self) -> None:
        for tokens in (
            ["platform", "telegram"],
            ["platform", "MyRPC"],
            ["group", "qqclient:123"],
        ):
            with self.subTest(tokens=tokens), self.assertRaises(InputError):
                rule_target(tokens, self._identity())

    def test_bare_form_targets_the_current_group(self) -> None:
        self.assertEqual(rule_target([], self._identity()), ("QQClient", "598683145"))

    def test_private_bare_form_requires_a_target(self) -> None:
        with self.assertRaises(InputError):
            rule_target([], self._identity(private=True))

    def test_platform_forms(self) -> None:
        self.assertEqual(rule_target(["platform"], self._identity()), ("QQClient", ""))
        self.assertEqual(
            rule_target(["Platform", "Telegram"], self._identity()),
            ("Telegram", ""),
        )

    def test_group_forms(self) -> None:
        self.assertEqual(
            rule_target(["group", "111"], self._identity()), ("QQClient", "111")
        )
        self.assertEqual(
            rule_target(["group", "Telegram:-100"], self._identity()),
            ("Telegram", "-100"),
        )

    def test_malformed_targets_are_rejected(self) -> None:
        identity = self._identity()
        for tokens in (
            ["group", ":123"],
            ["group", ""],
            ["group"],
            ["group", "1", "2"],
            ["group", "qqclient:111:222"],
            ["platform", "qqclient", "extra"],
            ["chat", "1"],
        ):
            with self.assertRaises(InputError, msg=str(tokens)):
                rule_target(tokens, identity)


if __name__ == "__main__":
    unittest.main()
