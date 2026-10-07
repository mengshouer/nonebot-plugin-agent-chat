import unittest

from nonebot_plugin_agent_chat.config import Config
from nonebot_plugin_agent_chat.image_reply import (
    IMAGE_FAILED_NOTICE,
    INTERRUPTED_NOTICE,
    PARTIAL_NOTICE,
    ImageReplySettings,
    deliver_answer,
    has_long_code_block,
    has_table,
    resolve_mode,
    resolve_platform_mode,
    resolve_show_sources,
    should_render,
)
from nonebot_plugin_agent_chat.models import ImageReplyMode
from nonebot_plugin_agent_chat.platforms import (
    platform_override,
    text_units,
)

CODE_BLOCK = "```python\n" + "\n".join(f"x{i} = 1" for i in range(9)) + "\n```"
SHORT_BLOCK = "```python\nx = 1\n```"
TABLE = "| a | b |\n|---|---|\n| 1 | 2 |"


class TriggerTests(unittest.TestCase):
    def test_off_mode_never_renders(self) -> None:
        settings = ImageReplySettings(mode=ImageReplyMode.OFF)
        self.assertFalse(should_render("x" * 5000, settings))

    def test_auto_renders_long_text(self) -> None:
        settings = ImageReplySettings(mode=ImageReplyMode.AUTO, min_chars=1000)
        self.assertTrue(should_render("x" * 1001, settings))
        self.assertFalse(should_render("x" * 999, settings))

    def test_always_renders_even_short_text(self) -> None:
        settings = ImageReplySettings(mode=ImageReplyMode.ALWAYS)
        self.assertTrue(should_render("hi", settings))

    def test_blank_text_is_never_rendered(self) -> None:
        settings = ImageReplySettings(mode=ImageReplyMode.ALWAYS)
        self.assertFalse(should_render("   \n  ", settings))

    def test_auto_detects_tables_and_long_code(self) -> None:
        settings = ImageReplySettings(mode=ImageReplyMode.AUTO)
        self.assertTrue(should_render(TABLE, settings))
        self.assertTrue(should_render(CODE_BLOCK, settings))
        self.assertFalse(should_render(SHORT_BLOCK, settings))

    def test_has_table_ignores_incidental_pipes(self) -> None:
        self.assertFalse(has_table("a | b"))
        self.assertFalse(has_table("| only one row |"))
        self.assertTrue(has_table(TABLE))

    def test_unclosed_code_fence_does_not_count(self) -> None:
        self.assertFalse(has_long_code_block("```\n" + "x\n" * 20))

    def test_profile_override_wins_over_global(self) -> None:
        self.assertIs(
            resolve_mode(ImageReplyMode.AUTO, ImageReplyMode.OFF),
            ImageReplyMode.OFF,
        )
        self.assertIs(
            resolve_mode(ImageReplyMode.AUTO, None),
            ImageReplyMode.AUTO,
        )


class ShowSourcesTests(unittest.TestCase):
    """platform > profile > global, per delivery path, defaulting to shown."""

    def test_global_default_when_nothing_overrides(self) -> None:
        self.assertTrue(resolve_show_sources(True, None, {}, "QQClient"))
        self.assertFalse(resolve_show_sources(False, None, {}, "Telegram"))

    def test_profile_wins_over_global(self) -> None:
        self.assertFalse(resolve_show_sources(True, False, {}, "Discord"))
        self.assertTrue(resolve_show_sources(False, True, {}, "Discord"))

    def test_platform_wins_over_profile_and_global(self) -> None:
        values = {"QQClient": False}
        self.assertFalse(resolve_show_sources(True, True, values, "QQClient"))
        self.assertTrue(
            resolve_show_sources(False, False, {"Telegram": True}, "Telegram")
        )

    def test_scope_lookup_uses_the_canonical_key(self) -> None:
        # Scope values are preserved exactly; no local alias translation.
        self.assertFalse(
            resolve_show_sources(True, None, {"Telegram": False}, "Telegram")
        )
        with self.assertRaises(ValueError):
            resolve_show_sources(True, None, {"Telegram": False}, "TG")

    def test_unrecorded_scope_falls_back_to_profile_then_global(self) -> None:
        self.assertFalse(resolve_show_sources(True, False, {}, "Discord"))
        self.assertTrue(resolve_show_sources(True, None, {}, "Discord"))


class PlatformModeTests(unittest.TestCase):
    def test_lookup_requires_exact_upstream_scope(self) -> None:
        modes = {"Telegram": ImageReplyMode.OFF}
        self.assertIs(platform_override(modes, "Telegram"), ImageReplyMode.OFF)
        with self.assertRaises(ValueError):
            platform_override(modes, " TELEGRAM ")

    def test_absent_platform_has_no_override(self) -> None:
        modes = {"Telegram": ImageReplyMode.OFF}
        self.assertIsNone(platform_override(modes, "QQClient"))
        self.assertIsNone(platform_override({}, "Telegram"))

    def test_platform_wins_over_profile_and_global(self) -> None:
        # Telegram must stay text even when the profile asks for images.
        mode = resolve_mode(
            resolve_mode(ImageReplyMode.AUTO, ImageReplyMode.ALWAYS),
            platform_override({"Telegram": ImageReplyMode.OFF}, "Telegram"),
        )
        self.assertIs(mode, ImageReplyMode.OFF)

    def test_unrecorded_platform_never_inherits_an_image_mode(self) -> None:
        from nonebot_plugin_agent_chat.models import ImageReplyMode

        self.assertIs(
            resolve_platform_mode(ImageReplyMode.AUTO, None, {}, "Discord"),
            ImageReplyMode.OFF,
        )
        self.assertIs(
            resolve_platform_mode(ImageReplyMode.ALWAYS, None, {}, "QQClient"),
            ImageReplyMode.ALWAYS,
        )
        self.assertIs(
            resolve_platform_mode(
                ImageReplyMode.OFF, None, {"Discord": ImageReplyMode.ALWAYS}, "Discord"
            ),
            ImageReplyMode.ALWAYS,
        )

    def test_without_platform_override_profile_still_wins(self) -> None:
        mode = resolve_mode(
            resolve_mode(ImageReplyMode.AUTO, ImageReplyMode.ALWAYS),
            platform_override({}, "Telegram"),
        )
        self.assertIs(mode, ImageReplyMode.ALWAYS)

    def test_from_config_uses_the_global_chunk_size_without_a_scope(self) -> None:
        config = Config(_env_file=None)

        settings = ImageReplySettings.from_config(config, ImageReplyMode.OFF)

        self.assertEqual(settings.chunk_chars, config.agent_chat_message_chunk_chars)
        self.assertEqual(settings.max_pages_per_message, 0)
        self.assertIs(settings.chunk_measure, len)

    def test_from_config_resolves_the_scope_facts(self) -> None:
        config = Config(_env_file=None)

        telegram = ImageReplySettings.from_config(
            config, ImageReplyMode.OFF, scope="Telegram"
        )

        self.assertEqual(telegram.chunk_chars, 4096)
        self.assertEqual(telegram.max_pages_per_message, 10)
        self.assertIs(telegram.chunk_measure, text_units)

    def test_from_config_keeps_the_global_size_for_unrecorded_scopes(self) -> None:
        config = Config(_env_file=None)

        generic = ImageReplySettings.from_config(
            config, ImageReplyMode.OFF, scope="Discord"
        )

        self.assertEqual(generic.chunk_chars, config.agent_chat_message_chunk_chars)
        self.assertEqual(generic.max_pages_per_message, 0)
        self.assertIs(generic.chunk_measure, len)


class FakePage:
    def __init__(self, data: bytes) -> None:
        self.data = data


class FakeDocument:
    def __init__(
        self,
        pages: list[int],
        failing: set[int] | None = None,
        cap_shrink_after: int | None = None,
    ) -> None:
        self.total_height = sum(pages)
        self.pages = pages
        self.failing = failing or set()
        self.cap_shrink_after = cap_shrink_after
        self.calls: list[tuple[int, int]] = []
        self.closed = False

    def slice_height(self, top: int, max_height: int, min_height: int = 200) -> int:
        if top >= self.total_height:
            return 0
        index = 0
        offset = 0
        for size in self.pages:
            if offset + size > top:
                break
            offset += size
            index += 1
        remaining = self.pages[index] - (top - offset)
        return min(remaining, max_height)

    async def slice(self, top: int, height: int) -> FakePage:
        self.calls.append((top, height))
        if top in self.failing:
            from nonebot_plugin_agent_chat.errors import RenderError

            raise RenderError("slice failed")
        return FakePage(b"P" * height)

    async def close(self) -> None:
        self.closed = True


class FakeRenderer:
    def __init__(self, document: FakeDocument | None, error: Exception | None = None):
        self.document = document
        self.error = error
        self.seen_sources: tuple | None = None
        self.idle_closes = 0

    async def render(self, text: str, *, sources=()):
        if self.error is not None:
            raise self.error
        # Record the raw argument: None (footer off) must stay distinguishable
        # from an empty list.
        self.seen_sources = sources if sources is None else tuple(sources)
        assert self.document is not None
        return self.document

    async def close_if_idle(self) -> bool:
        """Record the browser release the delivery path performs per batch."""

        self.idle_closes += 1
        return True


class Sender:
    def __init__(self, fail_images: int = 0) -> None:
        self.text: list[str] = []
        self.image_messages: list[list[bytes]] = []
        self.fail_images = fail_images

    async def send_text(self, chunk: str) -> None:
        self.text.append(chunk)

    async def send_images(self, pages: list[bytes]) -> None:
        if self.fail_images > 0:
            self.fail_images -= 1
            raise RuntimeError("too large")
        self.image_messages.append(pages)


async def no_sleep(_seconds: float) -> None:
    return None


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    def settings(self, **overrides) -> ImageReplySettings:
        values = {
            "mode": ImageReplyMode.ALWAYS,
            "max_height": 500,
            "delay_seconds": 0.0,
            "batch_max_bytes": 10_000,
            "chunk_chars": 400,
        }
        values.update(overrides)
        return ImageReplySettings(**values)

    async def test_text_path_appends_sources_when_rendering_is_off(self) -> None:
        sender = Sender()
        report = await deliver_answer(
            text="answer",
            sources=["https://example.com"],
            settings=self.settings(mode=ImageReplyMode.OFF),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(FakeDocument([100])),
            fallback_text="answer\n\n来源：\n[1] example - https://example.com",
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "text")
        self.assertEqual(sender.image_messages, [])
        self.assertIn("来源", "".join(sender.text))

    async def test_image_sources_reach_the_renderer(self) -> None:
        sender = Sender()
        renderer = FakeRenderer(FakeDocument([100]))
        report = await deliver_answer(
            text="long answer",
            sources=["https://example.com/a", "https://example.com/b"],
            settings=self.settings(),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=renderer,
            fallback_text="long answer",
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "image")
        self.assertEqual(
            renderer.seen_sources, ("https://example.com/a", "https://example.com/b")
        )

    async def test_none_sources_suppresses_the_footer(self) -> None:
        """The image switch off is None, not []: no footer at all."""

        sender = Sender()
        renderer = FakeRenderer(FakeDocument([100]))
        report = await deliver_answer(
            text="long answer",
            sources=None,
            settings=self.settings(),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=renderer,
            fallback_text="long answer",
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "image")
        self.assertIsNone(renderer.seen_sources)

    async def test_render_failure_falls_back_to_full_text(self) -> None:
        from nonebot_plugin_agent_chat.errors import RenderError

        sender = Sender()
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(None, RenderError("no browser")),
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "text")
        self.assertEqual(report.render_error, "RenderError")
        self.assertEqual(sender.text, ["answer"])
        self.assertTrue(report.complete)

    async def test_rendered_batch_releases_the_browser(self) -> None:
        """A delivered answer must not leave the renderer's browser resident."""

        sender = Sender()
        document = FakeDocument([400])
        renderer = FakeRenderer(document)
        report = await deliver_answer(
            text="long answer",
            sources=[],
            settings=self.settings(),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=renderer,
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "image")
        self.assertTrue(document.closed)
        self.assertEqual(renderer.idle_closes, 1)

    async def test_mid_render_failure_releases_the_browser(self) -> None:
        """A failure after the launch leaves a running browser behind."""

        from nonebot_plugin_agent_chat.errors import RenderError

        sender = Sender()
        renderer = FakeRenderer(None, RenderError("measuring failed"))
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=renderer,
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "text")
        self.assertEqual(sender.text, ["answer"])
        self.assertEqual(renderer.idle_closes, 1)

    async def test_text_path_never_releases_a_browser(self) -> None:
        """No browser is launched, so there is none to release."""

        sender = Sender()
        renderer = FakeRenderer(FakeDocument([400]))
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(mode=ImageReplyMode.OFF),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=renderer,
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "text")
        self.assertEqual(renderer.idle_closes, 0)

    async def test_pages_are_batched_into_one_message(self) -> None:
        sender = Sender()
        document = FakeDocument([400, 400])
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(document),
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "image")
        self.assertEqual(report.sent_pages, 2)
        self.assertEqual(report.sent_messages, 1)
        self.assertEqual(len(sender.image_messages[0]), 2)
        self.assertEqual(sender.text, [])
        self.assertTrue(document.closed)
        self.assertTrue(report.complete)

    async def test_page_cap_splits_batches(self) -> None:
        """Telegram media groups cap at ten items, so long answers chunk earlier."""

        sender = Sender()
        document = FakeDocument([100] * 12)
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(max_pages_per_message=5),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(document),
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "image")
        self.assertEqual(report.sent_pages, 12)
        self.assertEqual(report.sent_messages, 3)
        self.assertEqual([len(batch) for batch in sender.image_messages], [5, 5, 2])
        self.assertTrue(report.complete)

    async def test_zero_page_cap_keeps_one_batch(self) -> None:
        """Adapters without a media-group limit keep the byte-budget batching."""

        sender = Sender()
        document = FakeDocument([100] * 12)
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(max_pages_per_message=0),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(document),
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "image")
        self.assertEqual(report.sent_pages, 12)
        self.assertEqual(report.sent_messages, 1)
        self.assertEqual(len(sender.image_messages[0]), 12)

    async def test_utf16_measure_splits_emoji_within_units(self) -> None:
        sender = Sender()
        text = "😀" * 6  # 12 UTF-16 units, budget 4 per message
        report = await deliver_answer(
            text=text,
            sources=[],
            settings=self.settings(
                mode=ImageReplyMode.OFF,
                chunk_chars=4,
                chunk_measure=text_units,
            ),
            send_text=sender.send_text,
            send_images=sender.send_images,
            sleep=no_sleep,
        )
        self.assertTrue(report.complete)
        self.assertEqual("".join(sender.text), text)
        for chunk in sender.text:
            self.assertLessEqual(text_units(chunk), 4)

    async def test_batch_budget_splits_into_separate_messages(self) -> None:
        sender = Sender()
        document = FakeDocument([400, 400, 400])
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(batch_max_bytes=500),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(document),
            sleep=no_sleep,
        )
        self.assertEqual(report.sent_pages, 3)
        self.assertEqual(report.sent_messages, 3)

    async def test_rejected_payload_is_resplit_and_delivered(self) -> None:
        """A rejection must shrink the slice, not drop the content."""

        sender = Sender(fail_images=1)
        document = FakeDocument([400, 400])
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(batch_max_bytes=300),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(document),
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "image")
        self.assertEqual(report.dropped_pages, 0)
        self.assertIsNone(report.notice)
        self.assertTrue(report.complete)
        self.assertGreaterEqual(report.sent_pages, 2)
        self.assertGreaterEqual(report.send_attempts, 2)
        self.assertEqual(sender.text, [])

    async def test_rejected_pages_are_retried_in_reading_order(self) -> None:
        """A rejected batch must be retried before later pages are sent."""

        class EncodedDocument(FakeDocument):
            async def slice(self, top: int, height: int) -> FakePage:
                return FakePage(str(top).encode())

        sender = Sender(fail_images=1)
        document = EncodedDocument([100, 100, 100])
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(max_pages_per_message=1),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(document),
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "image")
        self.assertTrue(report.complete)
        self.assertEqual(
            [message[0] for message in sender.image_messages],
            [b"0", b"100", b"200"],
        )

    async def test_region_lost_in_a_failure_burst_keeps_the_rest_ordered(self) -> None:
        """When the transport keeps failing, only the failed region is lost."""

        class EncodedDocument(FakeDocument):
            async def slice(self, top: int, height: int) -> FakePage:
                return FakePage(str(top).encode())

        sender = Sender(fail_images=3)
        document = EncodedDocument([100, 100, 100])
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(max_pages_per_message=1),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(document),
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "image")
        self.assertEqual(report.sent_pages, 2)
        self.assertEqual(report.dropped_pages, 1)
        self.assertEqual(report.notice, PARTIAL_NOTICE)
        self.assertEqual(
            [message[0] for message in sender.image_messages],
            [b"100", b"200"],
        )

    async def test_permanent_rejection_reports_lost_pages(self) -> None:
        """When shrinking cannot help, loss is reported instead of hidden."""

        sender = Sender(fail_images=1)
        document = FakeDocument([400, 400])

        async def always_reject(pages: list[bytes]) -> None:
            raise RuntimeError("too large")

        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(batch_max_bytes=300, max_splits=0),
            send_text=sender.send_text,
            send_images=always_reject,
            renderer=FakeRenderer(document),
            sleep=no_sleep,
        )
        self.assertEqual(report.dropped_pages, len(document.pages))
        # Everything failed, so the answer still reaches the user as text and
        # nothing is actually lost.
        self.assertEqual(report.delivery, "text_fallback")
        self.assertTrue(report.complete)
        self.assertEqual(sender.text, ["answer"])
        self.assertEqual(report.notice, IMAGE_FAILED_NOTICE)

    async def test_total_image_failure_falls_back_to_text(self) -> None:
        """All batches rejected must not leave the user without an answer."""

        sender = Sender(fail_images=99)
        document = FakeDocument([400, 400])
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(chunk_chars=1000),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(document),
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "text_fallback")
        self.assertEqual(report.sent_pages, 0)
        self.assertEqual(sender.text, ["answer"])
        self.assertEqual(report.notice, IMAGE_FAILED_NOTICE)
        self.assertTrue(report.complete)

    async def test_midway_page_failure_resends_whole_answer_as_text(self) -> None:
        from nonebot_plugin_agent_chat.errors import RenderError

        class Exploding(FakeDocument):
            async def slice(self, top: int, height: int):
                if top > 0:
                    raise RenderError("page failed")
                return await super().slice(top, height)

        sender = Sender()
        document = Exploding([400, 400])
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(chunk_chars=1000),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(document),
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "text_fallback")
        self.assertEqual(report.notice, INTERRUPTED_NOTICE)
        # The whole answer is resent so nothing is lost.
        self.assertEqual(sender.text, ["answer"])
        self.assertGreaterEqual(report.sent_pages, 1)
        self.assertTrue(report.complete)

    async def test_length_alone_never_triggers_text_fallback(self) -> None:
        sender = Sender()
        document = FakeDocument([400] * 12)
        report = await deliver_answer(
            text="x" * 40_000,
            sources=[],
            settings=self.settings(max_height=400, batch_max_bytes=400),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(document),
            sleep=no_sleep,
        )
        self.assertEqual(report.delivery, "image")
        self.assertEqual(report.sent_pages, 12)
        self.assertEqual(sender.text, [])

    async def test_delay_is_applied_between_messages(self) -> None:
        pauses: list[float] = []

        async def record(seconds: float) -> None:
            pauses.append(seconds)

        sender = Sender()
        report = await deliver_answer(
            text="answer",
            sources=[],
            settings=self.settings(batch_max_bytes=300, delay_seconds=1.0),
            send_text=sender.send_text,
            send_images=sender.send_images,
            renderer=FakeRenderer(FakeDocument([200] * 4)),
            sleep=record,
        )
        self.assertGreaterEqual(report.sent_messages, 2)
        self.assertTrue(pauses)
        self.assertTrue(all(pause == 1.0 for pause in pauses))


class SharedRendererTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_renderer_is_reused_and_closed(self) -> None:
        from nonebot_plugin_agent_chat.render import (
            close_shared_renderer,
            shared_renderer,
        )

        first = shared_renderer(5.0)
        self.assertIs(first, shared_renderer(5.0))
        # Nothing was rendered, so there is no browser to release or close yet.
        self.assertFalse(await first.close_if_idle())
        await close_shared_renderer()
        self.assertIsNot(first, shared_renderer(5.0))
        await close_shared_renderer()


if __name__ == "__main__":
    unittest.main()
