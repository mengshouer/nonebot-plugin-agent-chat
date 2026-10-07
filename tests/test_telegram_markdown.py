import unittest
from html.parser import HTMLParser
from types import SimpleNamespace
from unittest.mock import AsyncMock

from nonebot_plugin_agent_chat.telegram_markdown import markdown_to_telegram_html


class _Balance(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack.pop() != tag:
            self.errors.append(tag)


def assert_balanced(test: unittest.TestCase, html: str) -> None:
    parser = _Balance()
    parser.feed(html)
    parser.close()
    test.assertEqual(parser.errors, [])
    test.assertEqual(parser.stack, [])


class InlineTests(unittest.TestCase):
    def test_html_is_escaped(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("a < b & c > d"),
            "a &lt; b &amp; c &gt; d",
        )

    def test_bold_and_italic(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("**bold** and *italic*"),
            "<b>bold</b> and <i>italic</i>",
        )

    def test_underscore_variants(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("__bold__ and _italic_"),
            "<b>bold</b> and <i>italic</i>",
        )

    def test_strikethrough_and_spoiler(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("~~gone~~ and ||hidden||"),
            '<s>gone</s> and <span class="tg-spoiler">hidden</span>',
        )

    def test_inline_code_keeps_markup_literal(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("`**not bold**`"),
            "<code>**not bold**</code>",
        )

    def test_snake_case_is_not_italicised(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("agent_chat_platform"),
            "agent_chat_platform",
        )

    def test_safe_link_becomes_anchor(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("[docs](https://example.com/a?b=1&c=2)"),
            '<a href="https://example.com/a?b=1&amp;c=2">docs</a>',
        )

    def test_unsafe_link_is_left_as_text(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("[x](javascript:alert(1))"),
            "[x](javascript:alert(1))",
        )

    def test_link_with_quote_stays_literal(self) -> None:
        # A quote inside the URL would produce a malformed href attribute.
        self.assertEqual(
            markdown_to_telegram_html('[x](https://example.com/a"b)'),
            '[x](https://example.com/a"b)',
        )

    def test_empty_text(self) -> None:
        self.assertEqual(markdown_to_telegram_html(""), "")


class BlockTests(unittest.TestCase):
    def test_heading_becomes_bold(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("## Title *x*"), "<b>Title <i>x</i></b>"
        )

    def test_heading_strips_edge_bold_markers(self) -> None:
        self.assertEqual(markdown_to_telegram_html("## **Title**"), "<b>Title</b>")

    def test_heading_keeps_dunder_identifiers(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("## Use __init__"),
            "<b>Use __init__</b>",
        )
        self.assertEqual(
            markdown_to_telegram_html("## __init__ 方法"),
            "<b>__init__ 方法</b>",
        )

    def test_heading_does_not_nest_bold(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("## Use **very** important"),
            "<b>Use very important</b>",
        )

    def test_heading_matches_body_underscore_italics(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("## a _b_ c"),
            "<b>a <i>b</i> c</b>",
        )

    def test_bullets_become_dots(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("- one\n  * two\n1. three"),
            "• one\n  • two\n1. three",
        )

    def test_blockquote(self) -> None:
        self.assertEqual(
            markdown_to_telegram_html("> quoted **text**"),
            "<blockquote>quoted <b>text</b></blockquote>",
        )

    def test_closed_fence_becomes_pre(self) -> None:
        html = markdown_to_telegram_html("```python\nx = 1 < 2\n```")
        self.assertEqual(html, "<pre>x = 1 &lt; 2</pre>")
        assert_balanced(self, html)

    def test_unclosed_fence_keeps_each_line_balanced(self) -> None:
        html = markdown_to_telegram_html("```\nx = 1\ny = 2")
        self.assertEqual(html, "<code>x = 1</code>\n<code>y = 2</code>")
        assert_balanced(self, html)

    def test_table_becomes_pre(self) -> None:
        html = markdown_to_telegram_html("| a | b |\n|---|---|\n| 1 | 2 |")
        self.assertEqual(html, "<pre>| a | b |\n|---|---|\n| 1 | 2 |</pre>")
        assert_balanced(self, html)

    def test_pipes_without_separator_stay_inline(self) -> None:
        self.assertEqual(markdown_to_telegram_html("a | b"), "a | b")

    def test_complex_answer_is_balanced(self) -> None:
        html = markdown_to_telegram_html(
            "# 标题 **粗**\n"
            "- 列表 `code`\n"
            "> 引用 [链接](https://example.com)\n"
            "```python\nx = 1\n```\n"
            "| a | b |\n|---|---|\n| 1 | 2 |\n"
            "结尾 <tag> & ||spoiler||"
        )
        assert_balanced(self, html)
        self.assertIn("<b>标题 粗</b>", html)


class TelegramSendTests(unittest.IsolatedAsyncioTestCase):
    async def test_parse_mode_html_is_forwarded(self) -> None:
        from nonebot.adapters.telegram.bot import Bot as TelegramBot
        from nonebot.adapters.telegram.config import BotConfig
        from nonebot_plugin_alconna import Target, UniMessage

        adapter = SimpleNamespace(get_name=lambda: "Telegram")
        bot = TelegramBot(adapter, "1", config=BotConfig(token="1:abc"))
        bot.call_api = AsyncMock()

        await UniMessage.text(markdown_to_telegram_html("**bold**")).send(
            target=Target.user("42", adapter="Telegram"),
            bot=bot,
            parse_mode="HTML",
        )

        call = bot.call_api.await_args
        self.assertEqual(call.args[0], "send_message")
        self.assertEqual(call.kwargs["text"], "<b>bold</b>")
        self.assertEqual(call.kwargs["parse_mode"], "HTML")
        self.assertEqual(call.kwargs["chat_id"], "42")


if __name__ == "__main__":
    unittest.main()
