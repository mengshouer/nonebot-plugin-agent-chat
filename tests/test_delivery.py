from __future__ import annotations

import unittest

from nonebot_plugin_agent_chat.delivery import (
    PARTIAL_TEXT_NOTICE,
    TOTAL_TEXT_NOTICE,
    DeliveryReport,
    chunk_text,
    loss_notice,
    send_chunks,
)


class FakeSender:
    """Records sends and rejects anything longer than a fake transport limit."""

    def __init__(
        self,
        *,
        limit: int | None = None,
        fail_times: int = 0,
        fail_every: bool = False,
    ) -> None:
        self.limit = limit
        self.fail_times = fail_times
        self.fail_every = fail_every
        self.sent: list[str] = []
        self.attempts = 0

    async def __call__(self, chunk: str) -> None:
        self.attempts += 1
        if self.limit is not None and len(chunk) > self.limit:
            raise FakeActionFailed()
        if self.fail_every:
            raise FakeActionFailed()
        if self.fail_times > 0:
            self.fail_times -= 1
            raise FakeActionFailed()
        self.sent.append(chunk)


class FakeActionFailed(Exception):
    def __init__(self) -> None:
        super().__init__("msg too long")
        self.info = {"retcode": 100, "wording": "消息过长"}


def no_sleep(_seconds: float):
    async def _inner() -> None:
        return None

    return _inner()


class ChunkTextTests(unittest.TestCase):
    def test_empty_text_produces_no_chunk(self) -> None:
        self.assertEqual(chunk_text("", 100), [])

    def test_short_text_is_untouched(self) -> None:
        self.assertEqual(chunk_text("hello", 100), ["hello"])

    def test_prefers_newline_boundaries_without_losing_characters(self) -> None:
        text = "a" * 60 + "\n" + "b" * 60
        chunks = chunk_text(text, 100)
        self.assertEqual(chunks, ["a" * 60 + "\n", "b" * 60])
        self.assertEqual("".join(chunks), text)

    def test_splits_without_newlines(self) -> None:
        chunks = chunk_text("x" * 250, 100)
        self.assertEqual([len(chunk) for chunk in chunks], [100, 100, 50])

    def test_round_trip_preserves_boundary_whitespace(self) -> None:
        text = "a" * 99 + " " + "\n" + "   indented\n" + "b" * 110
        chunks = chunk_text(text, 100)
        self.assertEqual("".join(chunks), text)
        self.assertEqual(chunks[0][-1], " ")

    def test_blank_only_text_produces_no_chunk(self) -> None:
        self.assertEqual(chunk_text("   \n\n  ", 10), [])

    def test_zero_size_sends_the_whole_text(self) -> None:
        self.assertEqual(chunk_text("x" * 9000, 0), ["x" * 9000])
        self.assertEqual(chunk_text("   ", 0), [])

    def test_negative_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("text", -1)

    def test_utf16_measure_respects_the_unit_budget(self) -> None:
        from nonebot_plugin_agent_chat.platforms import text_units

        text = "😀" * 5
        chunks = chunk_text(text, 4, measure=text_units)

        self.assertEqual(chunks, ["😀😀", "😀😀", "😀"])
        self.assertEqual("".join(chunks), text)
        for chunk in chunks:
            self.assertLessEqual(text_units(chunk), 4)

    def test_measure_never_splits_a_character(self) -> None:
        from nonebot_plugin_agent_chat.platforms import text_units

        text = "a😀b😀"
        chunks = chunk_text(text, 3, measure=text_units)

        self.assertEqual("".join(chunks), text)
        for chunk in chunks:
            self.assertLessEqual(text_units(chunk), 3)

    def test_zero_size_keeps_the_whole_text_with_a_measure(self) -> None:
        from nonebot_plugin_agent_chat.platforms import text_units

        text = "😀" * 100
        self.assertEqual(chunk_text(text, 0, measure=text_units), [text])


class LossNoticeTests(unittest.TestCase):
    def test_complete_delivery_has_no_notice(self) -> None:
        self.assertIsNone(loss_notice(DeliveryReport(sent_chunks=2)))

    def test_partial_loss_reports_omitted_content(self) -> None:
        report = DeliveryReport(sent_chunks=1, dropped_chunks=1, dropped_chars=5)
        self.assertEqual(loss_notice(report), PARTIAL_TEXT_NOTICE)

    def test_total_loss_reports_failure(self) -> None:
        report = DeliveryReport(dropped_chunks=1, dropped_chars=5)
        self.assertEqual(loss_notice(report), TOTAL_TEXT_NOTICE)


class SendChunksTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_send_when_text_fits(self) -> None:
        sender = FakeSender()
        report = await send_chunks(sender, "answer", 1000, sleep=no_sleep)
        self.assertTrue(report.complete)
        self.assertEqual(report.sent_chunks, 1)
        self.assertEqual(sender.sent, ["answer"])

    async def test_empty_text_sends_nothing(self) -> None:
        sender = FakeSender()
        report = await send_chunks(sender, "", 1000, sleep=no_sleep)
        self.assertTrue(report.complete)
        self.assertEqual(sender.sent, [])
        self.assertEqual(report.send_attempts, 0)

    async def test_transient_failure_is_retried(self) -> None:
        sender = FakeSender(fail_times=1)
        report = await send_chunks(sender, "answer", 1000, sleep=no_sleep)
        self.assertTrue(report.complete)
        self.assertEqual(report.sent_chunks, 1)
        self.assertEqual(report.send_attempts, 2)

    async def test_oversized_chunk_is_split_instead_of_dropped(self) -> None:
        body = "y" * 900
        sender = FakeSender(limit=300)
        report = await send_chunks(
            sender,
            body,
            1000,
            min_chars=100,
            sleep=no_sleep,
        )
        self.assertTrue(report.complete)
        self.assertEqual("".join(sender.sent), body)
        self.assertTrue(all(len(chunk) <= 300 for chunk in sender.sent))

    async def test_one_bad_chunk_does_not_abort_the_rest(self) -> None:
        sent: list[str] = []

        async def send(chunk: str) -> None:
            if set(chunk) == {"B"}:
                raise FakeActionFailed()
            sent.append(chunk)

        body = "A" * 50 + "B" * 50 + "C" * 50
        report = await send_chunks(
            send,
            body,
            50,
            min_chars=10_000,
            max_splits=0,
            sleep=no_sleep,
        )
        self.assertFalse(report.complete)
        self.assertEqual(report.dropped_chunks, 1)
        self.assertEqual(report.dropped_chars, 50)
        self.assertEqual(report.sent_chunks, 2)
        self.assertEqual([len(chunk) for chunk in sent], [50, 50])
        self.assertTrue(sent[0].startswith("A"))
        self.assertTrue(sent[1].startswith("C"))
        # The bad chunk is retried before being dropped.
        self.assertEqual(report.send_attempts, 5)

    async def test_failure_summary_omits_message_content(self) -> None:
        failures: list[str] = []
        sender = FakeSender(fail_every=True)
        report = await send_chunks(
            sender,
            "secret-prompt-text",
            1000,
            min_chars=10_000,
            sleep=no_sleep,
            on_failure=failures.append,
        )
        self.assertFalse(report.complete)
        self.assertTrue(failures)
        self.assertIn("FakeActionFailed", failures[0])
        self.assertIn("retcode=100", failures[0])
        self.assertFalse(any("secret-prompt-text" in item for item in failures))

    async def test_delay_is_applied_between_chunks(self) -> None:
        waits: list[float] = []

        async def sleep(seconds: float) -> None:
            waits.append(seconds)

        sender = FakeSender()
        await send_chunks(sender, "z" * 250, 100, delay_seconds=0.5, sleep=sleep)
        self.assertEqual(len(sender.sent), 3)
        self.assertEqual(waits, [0.5, 0.5])

    async def test_report_dataclass_defaults(self) -> None:
        report = DeliveryReport()
        self.assertTrue(report.complete)
        self.assertFalse(report.delivered_anything)


if __name__ == "__main__":
    unittest.main()
