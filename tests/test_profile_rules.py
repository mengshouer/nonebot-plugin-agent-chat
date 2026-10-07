"""Profile rules: platform/group defaults, validation and failure semantics."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nonebot_plugin_agent_chat.config import Config
from nonebot_plugin_agent_chat.errors import (
    ConfigurationError,
    InputError,
    ProfileRuleError,
)
from nonebot_plugin_agent_chat.input import CollectedInput
from nonebot_plugin_agent_chat.models import RunResult, Usage
from nonebot_plugin_agent_chat.platforms import ConversationRef
from nonebot_plugin_agent_chat.service import AgentChatService

GROUP = ConversationRef(scope="QQClient", target_id="598683145")


class RuleScopeTests(unittest.TestCase):
    def test_rule_key_preserves_upstream_scope(self) -> None:
        self.assertEqual(
            AgentChatService._validated_rule_key("QQClient", "123"), "QQClient"
        )
        self.assertEqual(
            AgentChatService._validated_rule_key("Discord", "123"), "Discord"
        )

    def test_rule_key_rejects_old_aliases_and_unknown_scopes(self) -> None:
        for scope in ("qqclient", "telegram", " TELEGRAM ", "MyRPC"):
            with self.subTest(scope=scope), self.assertRaises(InputError):
                AgentChatService._validated_rule_key(scope, "123")


class ProfileRuleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        for name in ("primary", "safe"):
            (self.profiles / f"{name}.json").write_text(
                json.dumps({"protocol": "openai-responses", "model": "test"}),
                encoding="utf-8",
            )
        self.service = AgentChatService(
            Config(
                agent_chat_data_dir=self.root / "data",
                agent_chat_profile_dir=self.profiles,
                agent_chat_default_profile="primary",
                agent_chat_allowed_groups={"QQClient:598683145"},
                agent_chat_cleanup_interval_seconds=0,
            )
        )
        await self.service.initialize()

    async def asyncTearDown(self) -> None:
        await self.service.close()
        self.temporary.cleanup()

    async def _root_name(self, conversation: ConversationRef | None) -> str:
        return (await self.service._root_for(conversation)).name

    async def test_default_is_used_without_rules(self) -> None:
        self.assertEqual(await self._root_name(GROUP), "primary")
        self.assertEqual(
            await self.service.effective_profile(GROUP), ("primary", "default")
        )

    async def test_platform_rule_applies_to_every_group(self) -> None:
        await self.service.set_profile_rule(
            scope="QQClient", target_id="", profile="safe", updated_by="cli"
        )

        self.assertEqual(
            await self._root_name(ConversationRef("QQClient", "111")), "safe"
        )
        self.assertEqual(
            await self.service.effective_profile(GROUP), ("safe", "platform-rule")
        )

    async def test_group_rule_beats_the_platform_rule(self) -> None:
        await self.service.set_profile_rule(
            scope="QQClient", target_id="", profile="primary", updated_by="cli"
        )
        await self.service.set_profile_rule(
            scope="QQClient", target_id="598683145", profile="safe", updated_by="cli"
        )

        self.assertEqual(await self._root_name(GROUP), "safe")
        self.assertEqual(
            await self.service.effective_profile(GROUP), ("safe", "group-rule")
        )
        self.assertEqual(
            await self._root_name(ConversationRef("QQClient", "222")), "primary"
        )

    async def test_private_chats_ignore_rules(self) -> None:
        await self.service.set_profile_rule(
            scope="QQClient", target_id="", profile="safe", updated_by="cli"
        )
        private = ConversationRef(scope="QQClient", target_id="487037110", private=True)

        self.assertEqual(await self._root_name(private), "primary")
        self.assertIsNone(await self.service.applicable_rule(private))

    async def test_explicit_use_overrides_rules_until_undo(self) -> None:
        await self.service.set_profile_rule(
            scope="QQClient", target_id="598683145", profile="safe", updated_by="cli"
        )
        # A deliberate switch to a non-default profile shadows the rule.
        await self.service.use_profile("safe")
        self.assertEqual(await self._root_name(GROUP), "safe")
        self.assertEqual(await self.service.effective_profile(GROUP), ("safe", "use"))

        # Switching back to the configured default is an undo, not an override.
        await self.service.use_profile("primary")
        self.assertEqual(await self._root_name(GROUP), "safe")
        self.assertEqual(
            await self.service.effective_profile(GROUP), ("safe", "group-rule")
        )

    async def test_replaced_explicit_choice_stops_overriding(self) -> None:
        await self.service.set_profile_rule(
            scope="QQClient", target_id="598683145", profile="safe", updated_by="cli"
        )
        await self.service.use_profile("safe")
        # "safe" disappears; the reload substitutes another profile, which must
        # not keep shadowing the group rule.
        (self.profiles / "safe.json").unlink()
        await self.service.reload_profiles()

        self.assertNotEqual(self.service.active_profile_name, "safe")
        self.assertFalse(self.service.profile_manager.explicit_override)
        self.assertEqual(
            await self.service.effective_profile(GROUP), ("safe", "group-rule")
        )

    async def test_reload_uses_the_new_default_when_the_selected_file_is_gone(
        self,
    ) -> None:
        """Rename plus repoint of the default in one edit: reload must not stick."""

        (self.profiles / "alt.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "test"}),
            encoding="utf-8",
        )
        (self.profiles / "primary.json").unlink()
        with patch.dict(os.environ, {"AGENT_CHAT_DEFAULT_PROFILE": "safe"}):
            report = await self.service.reload_everything()

        self.assertEqual(self.service.active_profile_name, "safe")
        self.assertEqual(
            report["config"]["applied"]["AGENT_CHAT_DEFAULT_PROFILE"],
            ("primary", "safe"),
        )

    async def test_reload_fails_loud_when_the_new_default_is_missing(self) -> None:
        (self.profiles / "alt.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "test"}),
            encoding="utf-8",
        )
        (self.profiles / "primary.json").unlink()
        with (
            patch.dict(os.environ, {"AGENT_CHAT_DEFAULT_PROFILE": "ghost"}),
            self.assertRaises(ConfigurationError) as caught,
        ):
            await self.service.reload_everything()

        message = str(caught.exception)
        self.assertIn("ghost", message)
        self.assertIn("alt", message)
        self.assertIn("safe", message)
        # A failed reload keeps the previous config live.
        self.assertEqual(self.service.config.agent_chat_default_profile, "primary")

    async def test_ask_runs_the_rule_profile(self) -> None:
        await self.service.set_profile_rule(
            scope="QQClient", target_id="598683145", profile="safe", updated_by="cli"
        )
        captured: dict[str, str] = {}

        async def fake_run_collected(root, chain, collected, **kwargs):
            captured["root"] = root.name
            return RunResult("ok", [], Usage(), 1, 0, 0, actual_profile=root.name)

        self.service._run_collected = fake_run_collected  # type: ignore[method-assign]
        await self.service.ask(
            CollectedInput(text="hi", images=[]),
            subject_key="subject",
            context_key="context",
            conversation=GROUP,
            enforce_cooldown=False,
            enforce_daily_quota=False,
        )

        self.assertEqual(captured["root"], "safe")

    async def test_unknown_profile_is_rejected_with_the_available_names(self) -> None:
        with self.assertRaises(InputError) as caught:
            await self.service.set_profile_rule(
                scope="QQClient", target_id="", profile="ghost", updated_by="cli"
            )

        self.assertIn("ghost", str(caught.exception))
        self.assertIn("safe", str(caught.exception))

    async def test_write_warns_about_acl_but_accepts_generic_upstream_scope(
        self,
    ) -> None:
        _, warnings = await self.service.set_profile_rule(
            scope="QQClient", target_id="999", profile="safe", updated_by="cli"
        )
        self.assertTrue(any("ALLOWED_GROUPS" in warning for warning in warnings))

        row, generic = await self.service.set_profile_rule(
            scope="Discord", target_id="", profile="safe", updated_by="cli"
        )
        self.assertEqual(row["scope"], "Discord")
        self.assertEqual(generic, [])
        self.assertEqual(
            await self._root_name(ConversationRef("Discord", "123")), "safe"
        )

        _, clean = await self.service.set_profile_rule(
            scope="QQClient", target_id="598683145", profile="safe", updated_by="cli"
        )
        self.assertEqual(clean, [])

    async def test_unknown_rule_scope_is_rejected_before_storage(self) -> None:
        for scope in ("My RPC", "qqclient", "telegram"):
            with (
                self.subTest(scope=scope),
                self.assertRaisesRegex(InputError, "--platform-list"),
            ):
                await self.service.set_profile_rule(
                    scope=scope, target_id="", profile="safe", updated_by="cli"
                )
        self.assertEqual(await self.service.store.list_profile_rules(), [])

    async def test_wildcard_allowlist_does_not_warn(self) -> None:
        config = Config(
            agent_chat_data_dir=self.root / "wild",
            agent_chat_profile_dir=self.profiles,
            agent_chat_default_profile="primary",
            agent_chat_allowed_groups={"QQClient:*"},
            agent_chat_cleanup_interval_seconds=0,
        )
        wildcard = AgentChatService(config)
        await wildcard.initialize()
        self.addAsyncCleanup(wildcard.close)

        _, warnings = await wildcard.set_profile_rule(
            scope="QQClient", target_id="999", profile="safe", updated_by="cli"
        )

        self.assertEqual(warnings, [])

    async def test_reveal_names_false_hides_the_profile_list(self) -> None:
        with self.assertRaises(InputError) as caught:
            await self.service.set_profile_rule(
                scope="QQClient",
                target_id="",
                profile="ghost",
                updated_by="cli",
                reveal_names=False,
            )

        message = str(caught.exception)
        self.assertIn("ghost", message)
        # No available-name list may leak into a non-private chat.
        self.assertNotIn("primary", message)
        self.assertNotIn("safe", message)

    async def test_malformed_targets_are_rejected(self) -> None:
        for scope, target_id in (("", ""), ("QQClient", ":123"), ("QQClient", "1 2")):
            with self.assertRaises(InputError):
                await self.service.set_profile_rule(
                    scope=scope,
                    target_id=target_id,
                    profile="safe",
                    updated_by="cli",
                )

    async def test_stale_rule_fails_loud(self) -> None:
        await self.service.store.set_profile_rule(
            scope="QQClient", target_id="598683145", profile="ghost", updated_by="cli"
        )

        with self.assertRaises(ProfileRuleError) as caught:
            await self._root_name(GROUP)

        self.assertIn("ghost", str(caught.exception))
        self.assertIn("rule unset", str(caught.exception))

    async def test_rule_status_marks_stale_rules(self) -> None:
        await self.service.set_profile_rule(
            scope="QQClient", target_id="598683145", profile="safe", updated_by="cli"
        )
        await self.service.store.set_profile_rule(
            scope="QQClient", target_id="111", profile="ghost", updated_by="cli"
        )

        rules = {rule["label"]: rule for rule in await self.service.rule_status()}

        self.assertFalse(rules["QQClient:group:598683145"]["stale"])
        self.assertTrue(rules["QQClient:group:111"]["stale"])

    async def test_effective_profile_survives_a_broken_startup(self) -> None:
        broken = AgentChatService(
            Config(
                agent_chat_data_dir=self.root / "broken",
                agent_chat_profile_dir=self.profiles,
                agent_chat_default_profile="missing",
                agent_chat_cleanup_interval_seconds=0,
            )
        )
        await broken.initialize()
        self.addAsyncCleanup(broken.close)

        self.assertTrue(broken.startup_error)
        self.assertEqual(await broken.effective_profile(GROUP), ("-", "unavailable"))

    async def test_conversation_status_shape(self) -> None:
        """The group-visible summary: what applies here, and nothing else."""

        healthy = await self.service.conversation_status(GROUP)
        self.assertEqual(healthy["conversation"], "QQClient:group:598683145")
        self.assertEqual(healthy["effective_profile"], "primary")
        self.assertEqual(healthy["profile_source"], "default")
        self.assertEqual(healthy["rules"], [])
        self.assertTrue(healthy["profile_ok"])
        # Exactly these keys: no active_profile/recent_runs/profiles leaks.
        self.assertEqual(
            set(healthy),
            {
                "conversation",
                "effective_profile",
                "profile_source",
                "rules",
                "profile_ok",
            },
        )

        await self.service.set_profile_rule(
            scope="QQClient", target_id="598683145", profile="safe", updated_by="cli"
        )
        ruled = await self.service.conversation_status(GROUP)
        self.assertEqual(ruled["effective_profile"], "safe")
        self.assertEqual(ruled["profile_source"], "group-rule")
        self.assertEqual(
            ruled["rules"],
            [{"label": "QQClient:group:598683145", "profile": "safe"}],
        )
        self.assertTrue(ruled["profile_ok"])

        # A stale rule must flip profile_ok without leaking a stale marker.
        await self.service.store.set_profile_rule(
            scope="QQClient", target_id="598683145", profile="ghost", updated_by="cli"
        )
        stale = await self.service.conversation_status(GROUP)
        self.assertEqual(stale["profile_source"], "group-rule")
        self.assertFalse(stale["profile_ok"])

    async def test_unset_rejects_malformed_keys_like_set(self) -> None:
        for scope, target_id in (("", ""), ("QQClient", ":123"), ("QQClient", "1 2")):
            with self.assertRaises(InputError):
                await self.service.unset_profile_rule(scope=scope, target_id=target_id)

    async def test_unset_removes_the_rule(self) -> None:
        await self.service.set_profile_rule(
            scope="QQClient", target_id="598683145", profile="safe", updated_by="cli"
        )

        self.assertTrue(
            await self.service.unset_profile_rule(
                scope="QQClient", target_id="598683145"
            )
        )
        self.assertFalse(
            await self.service.unset_profile_rule(
                scope="QQClient", target_id="598683145"
            )
        )
        self.assertEqual(await self._root_name(GROUP), "primary")


if __name__ == "__main__":
    unittest.main()
