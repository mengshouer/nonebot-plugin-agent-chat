import unittest

from nonebot_plugin_agent_chat.edits import EditChange, format_change, format_changes


class FormatChangeTests(unittest.TestCase):
    def test_every_action_and_subject_reads_the_same_way(self) -> None:
        cases = {
            EditChange("set", "setting", "AGENT_CHAT_MAX_SEARCHES", "9"): (
                "已设置 AGENT_CHAT_MAX_SEARCHES=9"
            ),
            EditChange("unset", "setting", "AGENT_CHAT_MAX_SEARCHES"): (
                "已删除 AGENT_CHAT_MAX_SEARCHES（回落默认或环境变量）"
            ),
            EditChange("set", "profile_field", "default.model", "better"): (
                "已设置 default.model=better"
            ),
            EditChange(
                "unset", "profile_field", "default.model"
            ): "已删除 default.model",
            EditChange("none", "profile_field", "default", "model"): (
                "default 没有设置 model"
            ),
            EditChange("none", "setting", "AGENT_CHAT_MAX_SEARCHES"): (
                "文件里没有 AGENT_CHAT_MAX_SEARCHES，未改动"
            ),
            EditChange("create", "profile", "extra", "default"): (
                "已创建 profile: extra（复制自 default）"
            ),
            EditChange("remove", "profile", "extra"): "已删除 profile: extra",
        }
        for change, expected in cases.items():
            with self.subTest(action=change.action, subject=change.subject):
                self.assertEqual(format_change(change), expected)

    def test_format_changes_keeps_the_order(self) -> None:
        changes = [
            EditChange("set", "setting", "A", "1"),
            EditChange("remove", "profile", "b"),
        ]
        self.assertEqual(
            format_changes(changes),
            ["已设置 A=1", "已删除 profile: b"],
        )


if __name__ == "__main__":
    unittest.main()
