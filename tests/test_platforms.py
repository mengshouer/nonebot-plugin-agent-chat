import json
import unittest
from types import SimpleNamespace

from nonebot_plugin_agent_chat.models import ImageReplyMode
from nonebot_plugin_agent_chat.platforms import (
    PLAIN_ANSWER_FORMAT,
    ChatIdentity,
    ConversationRef,
    SupportScope,
    answer_format_for,
    applicable_rule_rows,
    default_image_mode,
    identity_from_target,
    is_directed_at_bot,
    is_identity_allowed,
    isolation_for,
    max_image_pages_for,
    platform_entries,
    platform_for,
    platform_override,
    resolve_chunk_size,
    scope_for_target,
    scope_value,
    text_measure_for,
    text_units,
    uses_command_mentions,
)


def target(
    target_id: str,
    *,
    scope: str | SupportScope | None = None,
    private: bool = False,
    channel: bool = False,
    parent_id: str = "",
    extra: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=target_id,
        scope=scope,
        parent_id=parent_id,
        channel=channel,
        private=private,
        extra=dict(extra or {}),
    )


def identity_for(
    *, bot_id: str = "10000", user_id: str = "20000", **kwargs
) -> ChatIdentity:
    built = identity_from_target(
        bot_id=bot_id, user_id=user_id, target=target(**kwargs)
    )
    assert built is not None
    return built


class IdentityKeyTests(unittest.TestCase):
    def test_scope_is_taken_from_the_upstream_target(self) -> None:
        identity = identity_for(scope=SupportScope.telegram, target_id="-100")
        self.assertEqual(identity.scope, "Telegram")
        self.assertEqual(identity.acl_scope, "Telegram")

    def test_missing_or_invalid_scope_is_rejected_with_diagnostic(self) -> None:
        for scope in (None, "", "telegram", "MyRPC"):
            with (
                self.subTest(scope=scope),
                self.assertLogs(
                    "nonebot_plugin_agent_chat.platforms", level="WARNING"
                ) as logs,
            ):
                self.assertIsNone(
                    identity_from_target(
                        bot_id="1", user_id="2", target=target("3", scope=scope)
                    )
                )
            self.assertIn("missing or invalid platform ID", logs.output[0])

    def test_all_upstream_scopes_produce_identity_without_policy_registration(
        self,
    ) -> None:
        for scope in SupportScope:
            with self.subTest(scope=scope.value):
                identity = identity_for(scope=scope, target_id="room")
                self.assertEqual(identity.acl_entry, f"{scope.value}:room")

    def test_subject_key_is_per_user_and_stable(self) -> None:
        group = identity_for(target_id="30000", scope="QQClient")
        private = identity_for(target_id="20000", scope="QQClient", private=True)
        other_bot = identity_for(bot_id="99999", target_id="30000", scope="QQClient")
        self.assertEqual(group.subject_key, private.subject_key)
        self.assertNotEqual(group.subject_key, other_bot.subject_key)
        self.assertEqual(
            json.loads(group.subject_key),
            {"bot": "10000", "scope": "QQClient", "user": "20000", "v": 1},
        )

    def test_context_key_carries_the_target_fields(self) -> None:
        identity = identity_for(target_id="30000", scope="QQClient")
        self.assertEqual(
            json.loads(identity.context_key),
            {
                "bot": "10000",
                "scope": "QQClient",
                "user": "20000",
                "v": 1,
                "target": {
                    "channel": False,
                    "id": "30000",
                    "parent_id": "",
                    "private": False,
                },
            },
        )

    def test_isolation_separates_conversations(self) -> None:
        topic = identity_for(
            scope="Telegram", target_id="-100123", extra={"message_thread_id": 7}
        )
        other_topic = identity_for(
            scope="Telegram", target_id="-100123", extra={"message_thread_id": 8}
        )
        plain = identity_for(scope="Telegram", target_id="-100123")
        self.assertEqual(topic.isolation, "7")
        self.assertNotEqual(topic.context_key, other_topic.context_key)
        self.assertNotEqual(topic.context_key, plain.context_key)
        self.assertIsNone(plain.isolation)

    def test_channel_targets_are_kept_as_identity(self) -> None:
        identity = identity_for(
            scope="Discord", target_id="444", parent_id="777", channel=True
        )
        self.assertTrue(identity.channel)
        self.assertEqual(identity.parent_id, "777")
        self.assertEqual(identity.acl_entry, "Discord:444")

    def test_missing_target_or_user_is_rejected(self) -> None:
        for target_id, user_id in (("444", ""), ("", "333")):
            with self.subTest(target=target_id, user=user_id):
                self.assertIsNone(
                    identity_from_target(
                        bot_id="1",
                        user_id=user_id,
                        target=target(target_id, scope="Discord"),
                    )
                )

    def test_applicable_rule_rows_match_group_then_platform(self) -> None:
        rules = [
            {"scope": "QQClient", "target_id": "", "profile": "platform"},
            {"scope": "QQClient", "target_id": "111", "profile": "other-group"},
            {"scope": "QQClient", "target_id": "598683145", "profile": "group"},
            {"scope": "Telegram", "target_id": "", "profile": "tg"},
        ]
        group = ConversationRef(scope="QQClient", target_id="598683145")
        other = ConversationRef(scope="QQClient", target_id="222")
        private = ConversationRef(scope="QQClient", target_id="42", private=True)
        self.assertEqual(
            [rule["profile"] for rule in applicable_rule_rows(rules, group)],
            ["group", "platform"],
        )
        self.assertEqual(
            [rule["profile"] for rule in applicable_rule_rows(rules, other)],
            ["platform"],
        )
        self.assertEqual(applicable_rule_rows(rules, private), [])

    def test_operator_id_is_the_user_not_the_group(self) -> None:
        identity = identity_for(target_id="598683145", user_id="42", scope="QQClient")
        self.assertEqual(identity.operator_id, "QQClient:42")
        self.assertEqual(identity.acl_entry, "QQClient:598683145")


class AclTests(unittest.TestCase):
    def test_entries_match_on_scope_and_id(self) -> None:
        identity = identity_for(scope="QQClient", target_id="30000")
        self.assertTrue(
            is_identity_allowed(
                identity, allowed_groups={"QQClient:30000"}, allowed_users=set()
            )
        )
        for entry in ("QQClient:30001", "qqclient:30000", "Telegram:30000", "30000"):
            with self.subTest(entry=entry):
                self.assertFalse(
                    is_identity_allowed(
                        identity, allowed_groups={entry}, allowed_users=set()
                    )
                )

    def test_scope_wildcard_allows_every_conversation(self) -> None:
        identity = identity_for(scope="Telegram", target_id="-100123")
        self.assertTrue(
            is_identity_allowed(
                identity, allowed_groups={"Telegram:*"}, allowed_users=set()
            )
        )

    def test_private_users_use_the_user_list(self) -> None:
        identity = identity_for(
            scope="Telegram", user_id="487037110", target_id="487037110", private=True
        )
        self.assertTrue(
            is_identity_allowed(
                identity, allowed_groups=set(), allowed_users={"Telegram:487037110"}
            )
        )
        self.assertFalse(
            is_identity_allowed(
                identity, allowed_groups={"Telegram:487037110"}, allowed_users=set()
            )
        )

    def test_isolation_is_ignored_for_acl(self) -> None:
        topic = identity_for(
            scope="Telegram", target_id="-100123", extra={"message_thread_id": 7}
        )
        self.assertEqual(topic.acl_entry, "Telegram:-100123")


class PlatformFactsTests(unittest.TestCase):
    def test_scope_value_preserves_all_upstream_values(self) -> None:
        for scope in SupportScope:
            with self.subTest(scope=scope.name):
                self.assertEqual(scope_value(scope), scope.value)
                self.assertEqual(scope_value(scope.value), scope.value)

    def test_scope_value_rejects_aliases_and_unknown_values(self) -> None:
        for value in ("qqclient", "qq_client", "TELEGRAM", " Telegram ", "MyRPC", ""):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "platform-list"),
            ):
                scope_value(value)

    def test_invalid_platform_message_explains_how_to_find_the_id(self) -> None:
        with self.assertRaises(ValueError) as caught:
            scope_value("qqclient")
        text = str(caught.exception)
        for term in ("平台标识", "区分大小写", "QQClient", "--platform-list"):
            self.assertIn(term, text)
        for jargon in ("UniSeg", "SupportScope", "scope"):
            self.assertNotIn(jargon, text)

    def test_scope_for_target_uses_scope_property(self) -> None:
        declared = target("1", scope=SupportScope.discord, extra={"scope": "Telegram"})
        self.assertEqual(scope_for_target(declared), "Discord")

    def test_chunk_resolution_uses_scope_facts(self) -> None:
        self.assertEqual(resolve_chunk_size(1000, None, "Telegram"), 1000)
        self.assertEqual(resolve_chunk_size(1000, 0, "Telegram"), 4096)
        self.assertEqual(resolve_chunk_size(0, None, "Telegram"), 4096)
        self.assertEqual(resolve_chunk_size(1000, 0, "Discord"), 0)
        self.assertEqual(resolve_chunk_size(1000, 500, "Discord"), 500)

    def test_measure_and_pages_follow_the_scope(self) -> None:
        self.assertIs(text_measure_for("Telegram"), text_units)
        self.assertIs(text_measure_for("QQClient"), len)
        self.assertIs(text_measure_for("Discord"), len)
        self.assertEqual(max_image_pages_for("Telegram"), 10)
        self.assertEqual(max_image_pages_for("Discord"), 0)

    def test_answer_format_follows_the_scope(self) -> None:
        self.assertEqual(answer_format_for("Telegram").name, "telegram-html")
        self.assertIs(answer_format_for("QQClient"), PLAIN_ANSWER_FORMAT)
        self.assertIs(answer_format_for("Discord"), PLAIN_ANSWER_FORMAT)

    def test_command_mentions_are_telegram_only(self) -> None:
        self.assertTrue(uses_command_mentions("Telegram"))
        self.assertFalse(uses_command_mentions("QQClient"))
        self.assertFalse(uses_command_mentions("Discord"))

    def test_isolation_is_declared_per_platform(self) -> None:
        event_target = target(
            "-100123", scope="Telegram", extra={"message_thread_id": 7}
        )
        self.assertEqual(isolation_for("Telegram", event_target), "7")
        self.assertIsNone(isolation_for("QQClient", event_target))

    def test_platform_override_uses_exact_scope_values(self) -> None:
        values = {"Telegram": 4096}
        self.assertEqual(platform_override(values, "Telegram"), 4096)
        self.assertIsNone(platform_override(values, "Discord"))
        with self.assertRaises(ValueError):
            platform_override(values, "telegram")

    def test_all_scopes_without_a_policy_use_generic_defaults(self) -> None:
        for scope in SupportScope:
            if platform_for(scope.value) is not None:
                continue
            with self.subTest(scope=scope.value):
                self.assertIs(default_image_mode(scope.value), ImageReplyMode.OFF)
                self.assertIs(answer_format_for(scope.value), PLAIN_ANSWER_FORMAT)
                self.assertIs(text_measure_for(scope.value), len)
                self.assertEqual(max_image_pages_for(scope.value), 0)
        self.assertIsNone(default_image_mode("QQClient"))
        self.assertIsNone(default_image_mode("Telegram"))

    def test_text_measure_name_is_declared_by_the_platform(self) -> None:
        self.assertEqual(platform_for("Telegram").text_measure_name, "utf-16")
        self.assertEqual(platform_for("QQClient").text_measure_name, "characters")

    def test_is_directed_at_bot_survives_unimplemented_adapters(self) -> None:
        class SilentEvent:
            def is_tome(self) -> bool:
                raise NotImplementedError

        class ChattyEvent:
            def is_tome(self) -> bool:
                return True

        self.assertFalse(is_directed_at_bot(SilentEvent()))
        self.assertTrue(is_directed_at_bot(ChattyEvent()))

    def test_registry_contains_only_specialized_upstream_policies(self) -> None:
        self.assertEqual(
            {platform.scope for platform in platform_entries()},
            {"Telegram", "QQClient"},
        )


if __name__ == "__main__":
    unittest.main()
