import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from nonebot_plugin_agent_chat.errors import ConfigurationError
from nonebot_plugin_agent_chat.models import ProviderProfile
from nonebot_plugin_agent_chat.profiles import ProfileRegistry
from nonebot_plugin_agent_chat.prompts import (
    BUILTIN_DEFAULT_PROMPT,
    SOURCE_BUILTIN,
    SOURCE_GLOBAL,
    SOURCE_PROFILE_FILE,
    SOURCE_PROFILE_INLINE,
    PromptRegistry,
    normalize_prompt_text,
)


def write_profile(root: Path, name: str, **fields: object) -> Path:
    payload = {"protocol": "openai-completions", "model": "test", **fields}
    path = root / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class NormalizePromptTests(unittest.TestCase):
    def test_bom_crlf_and_blank_edges_are_normalized(self) -> None:
        raw = "\ufeff\r\nline one\r\n\r\n  indented\r\n\r\n".encode()
        self.assertEqual(
            normalize_prompt_text(raw, "test"),
            "line one\n\n  indented",
        )

    def test_blank_only_file_becomes_empty(self) -> None:
        self.assertEqual(normalize_prompt_text(b"  \n\n\t\n", "test"), "")

    def test_invalid_utf8_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            normalize_prompt_text(b"\xff\xfe\x00bad", "test")


class PromptPathSafetyTests(unittest.TestCase):
    def test_relative_names_are_confined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = PromptRegistry(Path(temporary))
            resolved = registry.confined_path("roles/reviewer.md", "test")
            self.assertEqual(resolved, Path(temporary).resolve() / "roles/reviewer.md")

    def test_parent_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = PromptRegistry(Path(temporary))
            with self.assertRaises(ConfigurationError):
                registry.confined_path("../outside.md", "test")

    def test_absolute_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = PromptRegistry(Path(temporary))
            with self.assertRaises(ConfigurationError):
                registry.confined_path("/etc/passwd", "test")

    def test_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside.md"
            outside.write_text("secret", encoding="utf-8")
            prompts = root / "prompts"
            prompts.mkdir()
            (prompts / "link.md").symlink_to(outside)
            registry = PromptRegistry(prompts)
            with self.assertRaises(ConfigurationError):
                registry.confined_path("link.md", "test")

    def test_model_rejects_absolute_and_parent_names(self) -> None:
        for value in ("/abs.md", "../up.md", "a/../../b.md"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ProviderProfile.model_validate(
                    {
                        "protocol": "openai-completions",
                        "model": "test",
                        "system_prompt_file": value,
                    }
                )

    def test_invalid_shadowed_file_name_is_ignored(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-completions",
                "model": "test",
                "system_prompt": "INLINE",
                "system_prompt_file": "../../ignored.md",
            }
        )
        self.assertEqual(profile.system_prompt_file, "../../ignored.md")

    def test_blank_prompt_file_name_becomes_none(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-completions",
                "model": "test",
                "system_prompt_file": "   ",
            }
        )
        self.assertIsNone(profile.system_prompt_file)


class PromptPrecedenceTests(unittest.TestCase):
    def make_registry(self, root: Path, *, default_file: str = "default.md"):
        prompts = PromptRegistry(root / "prompts", default_file)
        return ProfileRegistry(root / "profiles", "primary", prompts), prompts

    def test_inline_prompt_wins_over_file_and_global(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts").mkdir()
            (root / "prompts" / "default.md").write_text("GLOBAL", encoding="utf-8")
            (root / "prompts" / "role.md").write_text("FILE", encoding="utf-8")
            (root / "profiles").mkdir()
            write_profile(
                root / "profiles",
                "primary",
                system_prompt="INLINE",
                system_prompt_file="role.md",
            )
            registry, _ = self.make_registry(root)
            loaded = registry.stage().profiles["primary"]
            self.assertEqual(loaded.config.system_prompt, "INLINE")
            assert loaded.prompt is not None
            self.assertEqual(loaded.prompt.source, SOURCE_PROFILE_INLINE)

    def test_file_prompt_wins_over_global(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts").mkdir()
            (root / "prompts" / "default.md").write_text("GLOBAL", encoding="utf-8")
            (root / "prompts" / "role.md").write_text("FILE", encoding="utf-8")
            (root / "profiles").mkdir()
            write_profile(root / "profiles", "primary", system_prompt_file="role.md")
            registry, _ = self.make_registry(root)
            loaded = registry.stage().profiles["primary"]
            self.assertEqual(loaded.config.system_prompt, "FILE")
            assert loaded.prompt is not None
            self.assertEqual(loaded.prompt.source, SOURCE_PROFILE_FILE)

    def test_unset_prompt_inherits_global(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts").mkdir()
            (root / "prompts" / "default.md").write_text("GLOBAL", encoding="utf-8")
            (root / "profiles").mkdir()
            write_profile(root / "profiles", "primary")
            registry, _ = self.make_registry(root)
            snapshot = registry.stage()
            loaded = snapshot.profiles["primary"]
            self.assertEqual(loaded.config.system_prompt, "GLOBAL")
            assert loaded.prompt is not None
            self.assertEqual(loaded.prompt.source, SOURCE_GLOBAL)
            self.assertEqual(snapshot.global_prompt.text, "GLOBAL")

    def test_whitespace_only_inline_falls_through_to_global(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts").mkdir()
            (root / "prompts" / "default.md").write_text("GLOBAL", encoding="utf-8")
            (root / "profiles").mkdir()
            write_profile(root / "profiles", "primary", system_prompt="   \n  ")
            registry, _ = self.make_registry(root)
            loaded = registry.stage().profiles["primary"]
            self.assertEqual(loaded.config.system_prompt, "GLOBAL")

    def test_missing_global_file_uses_builtin_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "profiles").mkdir()
            write_profile(root / "profiles", "primary")
            registry, _ = self.make_registry(root)
            snapshot = registry.stage()
            self.assertEqual(snapshot.global_prompt.source, SOURCE_BUILTIN)
            self.assertEqual(
                snapshot.profiles["primary"].config.system_prompt,
                BUILTIN_DEFAULT_PROMPT,
            )

    def test_blank_global_file_means_no_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts").mkdir()
            (root / "prompts" / "default.md").write_text("\n  \n", encoding="utf-8")
            (root / "profiles").mkdir()
            write_profile(root / "profiles", "primary")
            registry, _ = self.make_registry(root)
            snapshot = registry.stage()
            self.assertEqual(snapshot.global_prompt.text, "")
            self.assertEqual(snapshot.global_prompt.source, SOURCE_GLOBAL)
            self.assertEqual(snapshot.profiles["primary"].config.system_prompt, "")

    def test_shadowed_profile_file_is_not_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "profiles").mkdir()
            write_profile(
                root / "profiles",
                "primary",
                system_prompt="INLINE",
                system_prompt_file="../../does-not-exist.md",
            )
            registry, _ = self.make_registry(root)
            loaded = registry.stage().profiles["primary"]
            self.assertEqual(loaded.config.system_prompt, "INLINE")

    def test_missing_profile_file_fails_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "profiles").mkdir()
            write_profile(root / "profiles", "primary", system_prompt_file="gone.md")
            registry, _ = self.make_registry(root)
            with self.assertRaises(ConfigurationError) as captured:
                registry.stage()
            self.assertIn("gone.md", str(captured.exception))

    def test_custom_default_file_name_and_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts" / "personal").mkdir(parents=True)
            (root / "prompts" / "personal" / "zh.md").write_text(
                "中文", encoding="utf-8"
            )
            (root / "profiles").mkdir()
            write_profile(root / "profiles", "primary")
            registry, _ = self.make_registry(root, default_file="personal/zh.md")
            snapshot = registry.stage()
            self.assertEqual(snapshot.global_prompt.text, "中文")
            self.assertEqual(snapshot.global_prompt.label, "global:personal/zh.md")

    def test_explicit_profile_path_field_is_cleared_after_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts").mkdir()
            (root / "prompts" / "role.md").write_text("FILE", encoding="utf-8")
            (root / "profiles").mkdir()
            write_profile(root / "profiles", "primary", system_prompt_file="role.md")
            registry, _ = self.make_registry(root)
            loaded = registry.stage().profiles["primary"]
            self.assertIsNone(loaded.config.system_prompt_file)

    def test_runtime_override_keeps_resolved_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts").mkdir()
            (root / "prompts" / "default.md").write_text("GLOBAL", encoding="utf-8")
            (root / "profiles").mkdir()
            write_profile(root / "profiles", "primary")
            registry, _ = self.make_registry(root)
            registry.activate(registry.stage())

            loaded = registry.override_for_runtime("primary", reasoning_effort="high")

            self.assertEqual(loaded.config.system_prompt, "GLOBAL")
            self.assertEqual(loaded.config.reasoning_effort.value, "high")


class PromptFingerprintTests(unittest.TestCase):
    def build(self, root: Path):
        (root / "prompts").mkdir(exist_ok=True)
        (root / "profiles").mkdir(exist_ok=True)
        prompts = PromptRegistry(root / "prompts", "default.md")
        return ProfileRegistry(root / "profiles", "primary", prompts)

    def test_stage_picks_up_a_changed_global_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self.build(root)
            write_profile(root / "profiles", "primary")
            prompt_file = root / "prompts" / "default.md"
            prompt_file.write_text("one", encoding="utf-8")
            self.assertEqual(registry.stage().global_prompt.text, "one")
            prompt_file.write_text("two", encoding="utf-8")
            # No watcher any more: the next stage() simply re-reads the file.
            self.assertEqual(registry.stage().global_prompt.text, "two")

    def test_stage_picks_up_a_newly_created_global_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self.build(root)
            write_profile(root / "profiles", "primary")
            self.assertEqual(registry.stage().global_prompt.source, SOURCE_BUILTIN)
            (root / "prompts" / "default.md").write_text("now here", encoding="utf-8")
            staged = registry.stage().global_prompt
            self.assertEqual(staged.source, SOURCE_GLOBAL)
            self.assertEqual(staged.text, "now here")

    def test_stage_picks_up_a_changed_profile_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self.build(root)
            write_profile(root / "profiles", "primary", system_prompt_file="role.md")
            prompt_file = root / "prompts" / "role.md"
            prompt_file.write_text("one", encoding="utf-8")
            self.assertEqual(
                registry.stage().profiles["primary"].prompt.text,  # type: ignore[union-attr]
                "one",
            )
            prompt_file.write_text("two", encoding="utf-8")
            self.assertEqual(
                registry.stage().profiles["primary"].prompt.text,  # type: ignore[union-attr]
                "two",
            )

    def test_a_shadowed_prompt_file_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self.build(root)
            write_profile(
                root / "profiles",
                "primary",
                system_prompt="INLINE",
                system_prompt_file="role.md",
            )
            (root / "prompts" / "role.md").write_text("one", encoding="utf-8")
            first = registry.stage().profiles["primary"].prompt
            (root / "prompts" / "role.md").write_text("two", encoding="utf-8")
            after = registry.stage().profiles["primary"].prompt
            self.assertEqual(first.text, "INLINE")  # type: ignore[union-attr]
            self.assertEqual(after.text, "INLINE")  # type: ignore[union-attr]

    def test_failed_stage_keeps_the_previous_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self.build(root)
            write_profile(root / "profiles", "primary", system_prompt="ORIGINAL")
            registry.activate(registry.stage())

            (root / "profiles" / "primary.json").write_text(
                json.dumps(
                    {
                        "protocol": "openai-completions",
                        "model": "test",
                        "system_prompt_file": "missing.md",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                registry.stage()

            # The activated snapshot survives the failed stage untouched.
            self.assertEqual(registry.get("primary").config.system_prompt, "ORIGINAL")


if __name__ == "__main__":
    unittest.main()
