import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from typing import NoReturn
from unittest.mock import patch

from nonebot_plugin_agent_chat import render
from nonebot_plugin_agent_chat.render import (
    PAGE_WIDTH,
    _expected_browser_dirs,
    probe_renderer,
)

SAMPLE = """# 标题

中文段落与 English text 混排，检查换行与字体回落。

| 列 A | 列 B |
|---|---|
| 1 | 2 |

```python
value = 1
```

<script>alert(1)</script>
"""


def _skip_optional_renderer(reason: str) -> NoReturn:
    if os.environ.get("REQUIRE_RENDERER") == "1":
        raise AssertionError(reason)
    raise unittest.SkipTest(reason)


class RendererRequirementTests(unittest.TestCase):
    def test_optional_renderer_can_be_skipped(self) -> None:
        with (
            patch.dict(os.environ, {"REQUIRE_RENDERER": "0"}),
            self.assertRaisesRegex(unittest.SkipTest, "browser unavailable"),
        ):
            _skip_optional_renderer("browser unavailable")

    def test_required_renderer_cannot_be_skipped(self) -> None:
        with (
            patch.dict(os.environ, {"REQUIRE_RENDERER": "1"}),
            self.assertRaisesRegex(AssertionError, "browser unavailable"),
        ):
            _skip_optional_renderer("browser unavailable")

    def test_missing_browser_fails_required_live_test(self) -> None:
        live_test = LiveRenderTests("test_render_produces_bounded_aligned_pages")
        with (
            patch.dict(os.environ, {"REQUIRE_RENDERER": "1"}),
            patch(f"{__name__}.probe_renderer", return_value=(False, "not installed")),
            self.assertRaisesRegex(AssertionError, "not installed"),
        ):
            asyncio.run(live_test.test_render_produces_bounded_aligned_pages())


class RendererProbeTests(unittest.TestCase):
    def test_probe_returns_reason(self) -> None:
        available, detail = probe_renderer()
        self.assertIsInstance(available, bool)
        self.assertTrue(detail)

    def test_expected_dirs_come_from_the_installed_manifest(self) -> None:
        from importlib import util

        if util.find_spec("playwright") is None:
            _skip_optional_renderer("playwright is not installed")
        names = _expected_browser_dirs()

        self.assertTrue(names)
        self.assertTrue(
            all(
                name.startswith(("chromium-", "chromium_headless_shell-"))
                for name in names
            )
        )

    def test_stale_revision_is_not_reported_as_ready(self) -> None:
        # The regression this guards: an old chromium-* directory in the cache
        # used to make probe_renderer() claim the renderer was ready.
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "chromium_headless_shell-1").mkdir()
            with (
                patch.dict(os.environ, {"PLAYWRIGHT_BROWSERS_PATH": temporary}),
                patch.object(
                    render,
                    "_expected_browser_dirs",
                    return_value=["chromium_headless_shell-2"],
                ),
            ):
                self.assertFalse(render._browsers_present())

    def test_empty_expected_directory_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "chromium_headless_shell-2").mkdir()
            with (
                patch.dict(os.environ, {"PLAYWRIGHT_BROWSERS_PATH": temporary}),
                patch.object(
                    render,
                    "_expected_browser_dirs",
                    return_value=["chromium_headless_shell-2"],
                ),
            ):
                self.assertFalse(render._browsers_present())

    def test_matching_revision_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "chromium_headless_shell-2"
            binary.mkdir()
            (binary / "chrome-headless-shell").write_text("#!/bin/sh\n")
            with (
                patch.dict(os.environ, {"PLAYWRIGHT_BROWSERS_PATH": temporary}),
                patch.object(
                    render,
                    "_expected_browser_dirs",
                    return_value=["chromium_headless_shell-2"],
                ),
            ):
                self.assertTrue(render._browsers_present())

    def test_markdown_parser_escapes_raw_html(self) -> None:
        import importlib.util

        if importlib.util.find_spec("markdown_it") is None:
            _skip_optional_renderer("markdown-it-py is not installed")
        from nonebot_plugin_agent_chat.render import build_document_html

        html = build_document_html(SAMPLE, [])
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>", html)
        self.assertIn("<table>", html)
        self.assertIn("highlight", html)


class LiveRenderTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_renderer_follows_the_live_timeout(self) -> None:
        """A reload updates the reused renderer, not just its first creation."""

        from nonebot_plugin_agent_chat.render import (
            close_shared_renderer,
            shared_renderer,
        )

        first = shared_renderer(20.0)
        second = shared_renderer(60.0)

        self.assertIs(first, second)
        self.assertEqual(second.timeout_seconds, 60.0)
        await close_shared_renderer()

    async def test_render_produces_bounded_aligned_pages(self) -> None:
        available, detail = probe_renderer()
        if not available:
            _skip_optional_renderer(f"renderer unavailable: {detail}")
        from nonebot_plugin_agent_chat.render import MarkdownRenderer

        renderer = MarkdownRenderer(timeout_seconds=30)
        try:
            document = await renderer.render(
                SAMPLE,
                sources=["https://example.com/source"],
            )
        except Exception as exc:  # noqa: BLE001 - optional browser dependency
            _skip_optional_renderer(f"browser launch failed: {type(exc).__name__}")
        try:
            self.assertGreater(document.total_height, 0)
            cap = 400
            top = 0
            pages = 0
            while True:
                height = document.slice_height(top, cap)
                if height <= 0:
                    break
                self.assertLessEqual(height, cap)
                page = await document.slice(top, height)
                self.assertEqual(page.width, PAGE_WIDTH)
                self.assertEqual(page.height, height)
                self.assertTrue(page.data.startswith(b"\x89PNG"))
                top += height
                pages += 1
            self.assertGreater(pages, 1)
        finally:
            await document.close()
            await renderer.close()

    async def test_live_render_reuses_one_browser(self) -> None:
        available, detail = probe_renderer()
        if not available:
            _skip_optional_renderer(f"renderer unavailable: {detail}")
        from nonebot_plugin_agent_chat.render import MarkdownRenderer

        renderer = MarkdownRenderer(timeout_seconds=30)
        try:
            first = await renderer.render("hello")
            await first.close()
            context = renderer._context
            second = await renderer.render("again")
            await second.close()
            # Reuse lasts until the owner asks for the release (see
            # test_idle_close_waits_for_the_last_document); closing a document
            # alone must not relaunch.
            self.assertIs(context, renderer._context)
        except Exception as exc:  # noqa: BLE001 - optional browser dependency
            _skip_optional_renderer(f"browser launch failed: {type(exc).__name__}")
        finally:
            await renderer.close()

    async def test_idle_close_waits_for_the_last_document(self) -> None:
        """The browser lives exactly as long as a render is in flight."""

        available, detail = probe_renderer()
        if not available:
            _skip_optional_renderer(f"renderer unavailable: {detail}")
        from nonebot_plugin_agent_chat.render import MarkdownRenderer

        renderer = MarkdownRenderer(timeout_seconds=30)
        try:
            first = await renderer.render("hello")
            second = await renderer.render("again")
        except Exception as exc:  # noqa: BLE001 - optional browser dependency
            _skip_optional_renderer(f"browser launch failed: {type(exc).__name__}")
        try:
            warm = renderer._context
            # Two open documents: no release may take the browser away.
            self.assertFalse(await renderer.close_if_idle())
            await first.close()
            self.assertFalse(await renderer.close_if_idle())
            # Closing the same document twice frees no second lease.
            await first.close()
            self.assertFalse(await renderer.close_if_idle())
            self.assertIs(renderer._context, warm)
            await second.close()
            self.assertTrue(await renderer.close_if_idle())
            self.assertIsNone(renderer._context)
            # The next document relaunches instead of failing.
            third = await renderer.render("third")
            try:
                self.assertIsNot(renderer._context, warm)
                height = third.slice_height(0, 400)
                self.assertGreater(height, 0)
                page = await third.slice(0, height)
                self.assertTrue(page.data.startswith(b"\x89PNG"))
            finally:
                await third.close()
        finally:
            await renderer.close()

    async def test_failed_render_leaves_no_browser(self) -> None:
        """A render that fails after the launch must not leave a browser up."""

        available, detail = probe_renderer()
        if not available:
            _skip_optional_renderer(f"renderer unavailable: {detail}")
        from nonebot_plugin_agent_chat.errors import RenderError
        from nonebot_plugin_agent_chat.render import MarkdownRenderer

        renderer = MarkdownRenderer(timeout_seconds=30)
        try:
            # Fails with the browser already launched and a page already open.
            with (
                patch.object(
                    MarkdownRenderer,
                    "_document_height",
                    side_effect=RenderError("measuring failed"),
                ),
                self.assertRaises(RenderError),
            ):
                await renderer.render("hello")
            self.assertEqual(renderer._leases, 0)
            # The failed render released the browser itself: nothing to close.
            self.assertIsNone(renderer._context)
            self.assertFalse(await renderer.close_if_idle())
        finally:
            await renderer.close()

    async def test_cancelled_render_leaves_no_browser(self) -> None:
        """Cancellation is a live path (the senders re-raise it), not a leak.

        The cancel lands once the browser is up and a page is open: cancelling
        *inside* a Playwright call is a launch-time case the driver handles on
        process exit, so it is not what this test pins down.
        """

        available, detail = probe_renderer()
        if not available:
            _skip_optional_renderer(f"renderer unavailable: {detail}")
        from nonebot_plugin_agent_chat.render import MarkdownRenderer

        renderer = MarkdownRenderer(timeout_seconds=30)
        reached = asyncio.Event()
        gate = asyncio.Event()

        async def blocked(*_args: object) -> list[int]:
            # Holds the render in flight until the test cancels it.
            reached.set()
            await gate.wait()
            return []

        try:
            with patch.object(MarkdownRenderer, "_block_boundaries", blocked):
                task = asyncio.create_task(renderer.render("hello"))
                await asyncio.wait_for(reached.wait(), 30)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertEqual(renderer._leases, 0)
            self.assertIsNone(renderer._context)
            self.assertFalse(await renderer.close_if_idle())
        finally:
            gate.set()
            await renderer.close()

    async def test_cancelled_launch_leaves_no_browser(self) -> None:
        """A half-started browser has no context to find, so it must be closed."""

        from nonebot_plugin_agent_chat.render import MarkdownRenderer

        class FakeBrowser:
            def __init__(self) -> None:
                self.closed = False

            async def close(self) -> None:
                self.closed = True

        renderer = MarkdownRenderer()
        browser = FakeBrowser()

        async def cancelled_launch() -> None:
            # The process starts, then the launch is cancelled before the
            # context exists. No browser is needed for this one.
            renderer._browser = browser
            raise asyncio.CancelledError()

        renderer._launch_context = cancelled_launch  # type: ignore[method-assign]
        with self.assertRaises(asyncio.CancelledError):
            await renderer.render("hello")
        self.assertTrue(browser.closed)
        self.assertEqual(renderer._leases, 0)
        await renderer.close()


class FooterHtmlTests(unittest.TestCase):
    def test_footer_has_sources_without_the_non_clickable_note(self) -> None:
        from nonebot_plugin_agent_chat.render import build_document_html

        html = build_document_html("hello", ["https://example.com/a"])
        self.assertIn('<section class="sources">', html)
        self.assertIn("https://example.com/a", html)
        self.assertNotIn("不可点击", html)

    def test_footer_is_absent_without_sources(self) -> None:
        from nonebot_plugin_agent_chat.render import build_document_html

        html = build_document_html("hello", [])
        self.assertNotIn('<section class="sources">', html)

    def test_none_suppresses_even_inline_link_urls(self) -> None:
        """The switch off removes the whole footer, answer's own links too."""

        from nonebot_plugin_agent_chat.render import build_document_html

        html = build_document_html("see [docs](https://example.com/doc)", None)
        self.assertNotIn('<section class="sources">', html)
        # The href survives inside the body anchor, but the URL is not listed
        # as a visible footer item.
        self.assertNotIn('<li><a href="https://example.com/doc">', html)

    def test_empty_sources_still_lists_inline_links(self) -> None:
        from nonebot_plugin_agent_chat.render import build_document_html

        html = build_document_html("see [docs](https://example.com/doc)", [])
        self.assertIn('<section class="sources">', html)
        self.assertIn('<li><a href="https://example.com/doc">', html)


if __name__ == "__main__":
    unittest.main()
