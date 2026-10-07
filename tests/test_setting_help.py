import unittest

from nonebot_plugin_agent_chat import config_editor, setting_help
from nonebot_plugin_agent_chat.config import Config


class CatalogContractTests(unittest.TestCase):
    """Every setting must be explained; no explanation may outlive its setting."""

    def test_catalog_covers_exactly_the_config_fields(self) -> None:
        fields = {name.upper() for name in Config.model_fields}
        documented = set(setting_help.SETTING_HELP)
        self.assertEqual(sorted(fields - documented), [])
        self.assertEqual(sorted(documented - fields), [])

    def test_every_description_is_useful(self) -> None:
        for key, description in sorted(setting_help.SETTING_HELP.items()):
            with self.subTest(key=key):
                self.assertGreater(len(description), 8)
                self.assertNotIn("TODO", description)

    def test_legend_distinguishes_reload_from_cli_reload_requests(self) -> None:
        legend = "\n".join(setting_help.LEGEND)
        self.assertIn("/agentctl reload 执行时重载", legend)
        self.assertIn("Bot 收到下一条消息时生效", legend)
        self.assertNotIn("编辑动作之后立即生效", legend)

    def test_legend_explains_every_marker_the_listing_prints(self) -> None:
        listing = config_editor.build_config_list(
            Config(_env_file=None),
            file_values={"AGENT_CHAT_X_TYPO": "1", "EXA_API_KEY": "secret"},
            owned={"AGENT_CHAT_MAX_SEARCHES"},
        )
        printed = " ".join(
            config_editor.format_entry(entry) for entry in listing.entries
        )
        legend = "\n".join(setting_help.LEGEND)
        for marker in ("需重启", "文件", "环境变量"):
            if marker in printed:
                self.assertIn(marker, legend)
        for explanation in ("CLI 专用", "未管理", "未知键"):
            self.assertIn(explanation, legend)


class ExplanationTests(unittest.TestCase):
    def test_platform_help_uses_configuration_examples(self) -> None:
        text = "\n".join(setting_help.explain_lines("AGENT_CHAT_ALLOWED_GROUPS"))
        for term in ("平台标识", "QQClient:123", "区分大小写", "--platform-list"):
            self.assertIn(term, text)
        for jargon in ("UniSeg", "SupportScope", "<scope>"):
            self.assertNotIn(jargon, text)

    def test_explain_lines_include_description_type_default_and_effect(self) -> None:
        lines = setting_help.explain_lines(
            "AGENT_CHAT_MAX_SEARCHES", current="5", source="file"
        )
        text = "\n".join(lines)
        self.assertIn("一次回答最多搜索几次", text)
        self.assertIn("类型：整数", text)
        self.assertIn("默认：10", text)
        self.assertIn("当前：5（来自 dotenv 文件）", text)
        self.assertIn("热生效", text)

    def test_restart_only_settings_say_so(self) -> None:
        text = "\n".join(setting_help.explain_lines("AGENT_CHAT_PRIORITY"))
        self.assertIn("需重启 bot", text)
        self.assertNotIn("热生效", text)

    def test_summarize_keeps_the_first_clause_and_bounds_length(self) -> None:
        self.assertEqual(
            setting_help.summarize("一次回答最多搜索几次；0 = 禁用搜索"),
            "一次回答最多搜索几次",
        )
        self.assertEqual(setting_help.summarize(""), "")
        # The default fits the fixed summary column.
        self.assertLessEqual(len(setting_help.summarize("x" * 200)), 17)
        long_text = "很长的说明" * 20
        summary = setting_help.summarize(long_text, limit=10)
        self.assertEqual(len(summary), 10)
        self.assertTrue(summary.endswith("…"))

    def test_unknown_key_is_named_as_unknown(self) -> None:
        text = "\n".join(setting_help.explain_lines("AGENT_CHAT_NOPE"))
        self.assertIn("未知设置项", text)

    def test_type_hints_read_naturally(self) -> None:
        cases = {
            "AGENT_CHAT_MAX_SEARCHES": "整数",
            "AGENT_CHAT_ENABLE_AT": "true / false",
            "AGENT_CHAT_DEFAULT_PROFILE": "可留空；字符串",
            "AGENT_CHAT_ALLOWED_GROUPS": "JSON 数组（不重复）",
            "AGENT_CHAT_TRIGGERS": "JSON 数组",
            "AGENT_CHAT_IMAGE_REPLY_MODE": "取值：off / auto / always",
            "AGENT_CHAT_SHOW_SOURCES_TEXT_BY_PLATFORM": (
                "JSON 映射（平台 → true/false）"
            ),
            "AGENT_CHAT_MESSAGE_CHUNK_CHARS_BY_PLATFORM": "JSON 映射（平台 → 整数）",
            "AGENT_CHAT_IMAGE_REPLY_MODE_BY_PLATFORM": (
                "JSON 映射（平台 → off/auto/always）"
            ),
            "AGENT_CHAT_DATA_DIR": "路径",
        }
        for key, expected in cases.items():
            with self.subTest(key=key):
                annotation = Config.model_fields[key.lower()].annotation
                self.assertEqual(setting_help.type_hint(annotation), expected)


if __name__ == "__main__":
    unittest.main()
