import unittest
from types import SimpleNamespace

from nonebot_plugin_alconna.uniseg import SupportScope

from nonebot_plugin_agent_chat import answer_options
from nonebot_plugin_agent_chat.config import Config
from nonebot_plugin_agent_chat.errors import ProfileNotFoundError
from nonebot_plugin_agent_chat.models import ImageReplyMode


class _StubProfiles:
    """The registry surface ``profile_field`` needs, without a profile dir."""

    def __init__(self, overrides: dict[str, dict[str, object]]) -> None:
        self._overrides = overrides

    def get(self, name: str) -> object:
        if name not in self._overrides:
            raise ProfileNotFoundError(f"Unknown profile: {name}")
        return SimpleNamespace(config=SimpleNamespace(**self._overrides[name]))


class ProfileFieldTests(unittest.TestCase):
    def test_missing_name_or_profile_yields_none(self) -> None:
        profiles = _StubProfiles({"p": {"image_reply_mode": ImageReplyMode.AUTO}})

        self.assertIsNone(answer_options.profile_field(profiles, None, "x"))
        self.assertIsNone(answer_options.profile_field(profiles, "missing", "x"))
        self.assertEqual(
            answer_options.profile_field(profiles, "p", "image_reply_mode"),
            ImageReplyMode.AUTO,
        )


class ImageReplyModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(
            _env_file=None,
            agent_chat_image_reply_mode=ImageReplyMode.AUTO,
            agent_chat_image_reply_mode_by_platform={
                "Telegram": ImageReplyMode.OFF,
            },
        )

    def test_ladder_is_platform_then_profile_then_global(self) -> None:
        profiles = _StubProfiles({"p": {"image_reply_mode": ImageReplyMode.ALWAYS}})

        global_mode = answer_options.image_reply_mode(
            self.config, profiles, profile_name=None, scope="QQClient"
        )
        profile_mode = answer_options.image_reply_mode(
            self.config, profiles, profile_name="p", scope="QQClient"
        )
        platform_mode = answer_options.image_reply_mode(
            self.config, profiles, profile_name="p", scope="Telegram"
        )

        self.assertEqual(global_mode, ImageReplyMode.AUTO)
        self.assertEqual(profile_mode, ImageReplyMode.ALWAYS)
        self.assertEqual(platform_mode, ImageReplyMode.OFF)

    def test_unrecorded_platform_stays_text_only(self) -> None:
        """An unknown adapter must not start rendering just because a profile does."""

        profiles = _StubProfiles({"p": {"image_reply_mode": ImageReplyMode.ALWAYS}})

        mode = answer_options.image_reply_mode(
            self.config, profiles, profile_name="p", scope="Discord"
        )

        self.assertEqual(mode, ImageReplyMode.OFF)


class ShowSourcesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(
            _env_file=None,
            agent_chat_show_sources_text=True,
            agent_chat_show_sources_text_by_platform={"Telegram": True},
            agent_chat_show_sources_image=True,
            agent_chat_show_sources_image_by_platform={"Telegram": False},
        )

    def test_text_ladder(self) -> None:
        profiles = _StubProfiles({"p": {"show_sources_text": False}})

        self.assertTrue(
            answer_options.show_sources(
                self.config,
                profiles,
                profile_name=None,
                scope="QQClient",
                kind="text",
            )
        )
        self.assertFalse(
            answer_options.show_sources(
                self.config,
                profiles,
                profile_name="p",
                scope="QQClient",
                kind="text",
            )
        )
        # Platform override outranks the profile.
        self.assertTrue(
            answer_options.show_sources(
                self.config,
                profiles,
                profile_name="p",
                scope="Telegram",
                kind="text",
            )
        )

    def test_image_switch_is_independent(self) -> None:
        profiles = _StubProfiles({"p": {"show_sources_image": True}})

        self.assertFalse(
            answer_options.show_sources(
                self.config,
                profiles,
                profile_name="p",
                scope="Telegram",
                kind="image",
            )
        )
        self.assertTrue(
            answer_options.show_sources(
                self.config,
                profiles,
                profile_name="p",
                scope="QQClient",
                kind="image",
            )
        )


class StatusTests(unittest.TestCase):
    def test_platform_status_lists_builtins_and_generic_defaults(self) -> None:
        status = answer_options.platform_status(["OneBot V11", "Telegram"])

        self.assertEqual(status["installed_adapters"], ["OneBot V11", "Telegram"])
        builtin = status["builtin"]
        self.assertEqual(builtin["Telegram"]["text_limit"], 4096)
        self.assertEqual(builtin["Telegram"]["text_measure"], "utf-16")
        defaults = status["generic_defaults"]
        self.assertEqual(defaults["answer_format"], "plain")
        self.assertIs(defaults["isolation"], False)
        self.assertEqual(status["scopes"], [scope.value for scope in SupportScope])
        self.assertNotIn("ambiguous_scopes", status)

    def test_image_reply_status_reports_the_renderer_probe(self) -> None:
        config = Config(
            _env_file=None,
            agent_chat_image_reply_mode=ImageReplyMode.AUTO,
            agent_chat_show_sources_text_by_platform={"Telegram": False},
        )

        status = answer_options.image_reply_status(config, (False, "no chromium"))

        self.assertEqual(status["mode"], "auto")
        self.assertEqual(status["platform_modes"]["Telegram"], "off")
        self.assertEqual(status["show_sources_text_by_platform"]["Telegram"], False)
        self.assertIs(status["renderer_available"], False)
        self.assertEqual(status["detail"], "no chromium")


if __name__ == "__main__":
    unittest.main()
