import unittest
from pathlib import Path

from nonebot_plugin_agent_chat.fallback import is_compatible
from nonebot_plugin_agent_chat.models import ProviderProfile
from nonebot_plugin_agent_chat.profiles import LoadedProfile


def loaded(name: str, **fields: object) -> LoadedProfile:
    payload = {"protocol": "openai-responses", "model": "test", **fields}
    return LoadedProfile(
        name=name,
        config=ProviderProfile.model_validate(payload),
        path=Path(f"{name}.json"),
    )


class FallbackCompatibilityTests(unittest.TestCase):
    def test_plain_profiles_are_compatible(self) -> None:
        self.assertTrue(is_compatible(loaded("a"), loaded("b"), has_images=False))

    def test_images_require_vision(self) -> None:
        root = loaded("a")
        self.assertFalse(is_compatible(root, loaded("b"), has_images=True))
        vision = loaded("c", capabilities={"vision": True})
        self.assertTrue(is_compatible(root, vision, has_images=True))

    def test_search_capability_must_not_be_dropped(self) -> None:
        root = loaded("a", search_mode="exa")
        self.assertFalse(is_compatible(root, loaded("b"), has_images=False))
        self.assertTrue(
            is_compatible(
                root,
                loaded("c", search_mode="builtin_web_search"),
                has_images=False,
            )
        )

    def test_offline_root_rejects_online_fallback(self) -> None:
        root = loaded("a")
        online = loaded("b", search_mode="exa")
        self.assertFalse(is_compatible(root, online, has_images=False))

    def test_reasoning_root_requires_reasoning_candidate(self) -> None:
        root = loaded("a", reasoning_effort="high")
        self.assertFalse(is_compatible(root, loaded("b"), has_images=False))
        candidate = loaded("c", reasoning_effort="low")
        self.assertTrue(is_compatible(root, candidate, has_images=False))

    def test_candidate_must_keep_enabled_tools(self) -> None:
        root = loaded("a", enabled_tools=["lookup"])
        self.assertFalse(is_compatible(root, loaded("b"), has_images=False))
        candidate = loaded("c", enabled_tools=["lookup", "extra"])
        self.assertTrue(is_compatible(root, candidate, has_images=False))


if __name__ == "__main__":
    unittest.main()
