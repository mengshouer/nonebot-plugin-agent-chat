import json
import tempfile
import unittest
from pathlib import Path

from dotenv import dotenv_values
from pydantic import ValidationError

from nonebot_plugin_agent_chat.errors import ConfigurationError
from nonebot_plugin_agent_chat.models import ProviderProfile
from nonebot_plugin_agent_chat.profiles import ProfileRegistry


class ProfileErrorPrivacyTests(unittest.TestCase):
    def test_validation_error_does_not_echo_the_inline_prompt(self) -> None:
        import tempfile
        from pathlib import Path as PathType

        from nonebot_plugin_agent_chat.errors import ConfigurationError
        from nonebot_plugin_agent_chat.profiles import ProfileRegistry

        secret_prompt = "SYSTEM-PROMPT-SECRET-0123456789"
        with tempfile.TemporaryDirectory() as temporary:
            root = PathType(temporary)
            (root / "broken.json").write_text(
                json.dumps(
                    {
                        "protocol": "openai-responses",
                        "model": "test",
                        # A wrong-typed prompt is the case where pydantic echoes
                        # the offending value (input_value=...) in str(exc).
                        "system_prompt": {"inline": secret_prompt},
                    }
                ),
                encoding="utf-8",
            )
            registry = ProfileRegistry(root)
            with self.assertRaises(ConfigurationError) as raised:
                registry.stage()
            message = str(raised.exception)
            self.assertNotIn(secret_prompt, message)
            self.assertIn("broken.json", message)
            # Prove we exercised the leaking path: the field itself is named.
            self.assertIn("system_prompt", message)


class ProviderProfileTests(unittest.TestCase):
    def test_only_protocol_and_model_are_required(self) -> None:
        profile = ProviderProfile.model_validate(
            {"protocol": "openai-completions", "model": "test"}
        )
        self.assertEqual(profile.search_mode.value, "off")
        self.assertEqual(profile.reasoning_effort.value, "provider_default")
        self.assertFalse(profile.capabilities.vision)
        self.assertFalse(profile.capabilities.tools)
        self.assertFalse(profile.capabilities.reasoning)

    def test_builtin_search_requires_a_hosted_search_protocol(self) -> None:
        with self.assertRaises(ValidationError):
            ProviderProfile.model_validate(
                {
                    "protocol": "openai-completions",
                    "model": "test",
                    "capabilities": {"tools": True},
                    "search_mode": "builtin_web_search",
                }
            )

    def test_optional_capabilities_are_inferred(self) -> None:
        reasoning = ProviderProfile.model_validate(
            {
                "protocol": "openai-responses",
                "model": "test",
                "reasoning_effort": "max",
            }
        )
        search = ProviderProfile.model_validate(
            {
                "protocol": "openai-responses",
                "model": "test",
                "search_mode": "builtin_web_search",
                "max_builtin_tool_calls": 8,
            }
        )
        anthropic = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "reasoning_effort": "xhigh",
            }
        )
        anthropic_search = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "search_mode": "builtin_web_search",
            }
        )
        self.assertTrue(reasoning.capabilities.reasoning)
        self.assertTrue(search.capabilities.tools)
        self.assertEqual(search.max_builtin_tool_calls, 8)
        self.assertEqual(anthropic.reasoning_effort.value, "xhigh")
        self.assertTrue(anthropic_search.capabilities.tools)

    def test_explicit_false_capability_rejects_conflicting_feature(self) -> None:
        with self.assertRaises(ValidationError):
            ProviderProfile.model_validate(
                {
                    "protocol": "openai-responses",
                    "model": "test",
                    "capabilities": {"reasoning": False},
                    "reasoning_effort": "high",
                }
            )

    def test_reserved_request_fields_cannot_be_overridden(self) -> None:
        for field in ("input", "thinking", "output_config", "stream_options"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ProviderProfile.model_validate(
                    {
                        "protocol": "openai-responses",
                        "model": "test",
                        "extra_body": {field: "override"},
                    }
                )


class ProfileRegistryTests(unittest.TestCase):
    def test_committed_template_matches_committed_profiles(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        values = dotenv_values(repository / ".env.agent_chat.example")
        profile_dir = repository / "src/nonebot_plugin_agent_chat/profiles.example"
        default = values.get("AGENT_CHAT_DEFAULT_PROFILE")
        registry = ProfileRegistry(profile_dir, default)
        names = registry.load(required_profile=default)
        required_envs = {
            loaded.config.api_key_env
            for loaded in registry.profiles.values()
            if loaded.config.api_key_env
        }
        required_envs.update(
            loaded.config.exa_api_key_env
            for loaded in registry.profiles.values()
            if loaded.config.search_mode.value == "exa"
        )
        self.assertIn(default, names)
        self.assertTrue(required_envs.issubset(values))

    def test_non_recursive_chain_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = {
                "primary": {
                    "protocol": "openai-responses",
                    "model": "one",
                    "fallback_profiles": ["fallback-one", "fallback-two"],
                },
                "fallback-one": {
                    "protocol": "openai-completions",
                    "model": "two",
                    "fallback_profiles": ["unused"],
                },
                "fallback-two": {
                    "protocol": "anthropic-messages",
                    "model": "three",
                },
                "unused": {
                    "protocol": "openai-completions",
                    "model": "four",
                },
            }
            for name, value in profiles.items():
                (root / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")

            registry = ProfileRegistry(root, "primary")
            registry.load()
            chain = registry.root_chain("primary", limit=3)
            self.assertEqual(
                [item.name for item in chain],
                ["primary", "fallback-one", "fallback-two"],
            )

    def test_runtime_reasoning_override_does_not_write_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "primary.json"
            original = {
                "protocol": "openai-completions",
                "model": "test",
            }
            path.write_text(json.dumps(original), encoding="utf-8")
            registry = ProfileRegistry(directory, "primary")
            registry.load()

            loaded = registry.override_for_runtime("primary", reasoning_effort="high")

            self.assertEqual(loaded.config.reasoning_effort.value, "high")
            self.assertTrue(loaded.config.capabilities.reasoning)
            self.assertEqual(json.loads(path.read_text()), original)

    def test_api_key_can_use_injected_nonebot_resolver(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-responses",
                "model": "test",
                "api_key_env": "CUSTOM_KEY",
            }
        )
        value = ProfileRegistry.resolve_api_key(
            profile,
            resolver=lambda name: "from-nonebot" if name == "CUSTOM_KEY" else None,
        )
        self.assertEqual(value, "from-nonebot")

    def test_failed_reload_keeps_previous_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "primary.json"
            path.write_text(
                json.dumps({"protocol": "openai-responses", "model": "one"}),
                encoding="utf-8",
            )
            registry = ProfileRegistry(root, "primary")
            registry.load()
            path.write_text("not-json", encoding="utf-8")

            with self.assertRaises(ConfigurationError):
                registry.load(required_profile="primary")
            self.assertEqual(registry.get("primary").config.model, "one")


if __name__ == "__main__":
    unittest.main()
