import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from nonebot_plugin_agent_chat.image_cache import RoomImageCache
from nonebot_plugin_agent_chat.models import AgentImage


class RoomImageCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_paths_are_confined_to_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "images"
            cache = RoomImageCache(root, retention_days=7)
            cache.initialize()
            self.assertIsNone(cache.safe_path(str(Path(temporary) / "outside.png")))

    async def test_cancelled_persist_waits_for_write_then_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = RoomImageCache(Path(temporary) / "images", retention_days=7)
            cache.initialize()
            entered = threading.Event()
            release = threading.Event()
            original = Path.write_bytes

            def delayed_write(path: Path, data: bytes) -> int:
                entered.set()
                release.wait(timeout=2)
                return original(path, data)

            with patch.object(Path, "write_bytes", delayed_write):
                task = asyncio.create_task(
                    cache.persist("room", [AgentImage("image/png", b"image")])
                )
                await asyncio.to_thread(entered.wait, 2)
                task.cancel()
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            self.assertEqual(list(cache.root.rglob("*.png")), [])

    async def test_partial_persist_failure_removes_written_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = RoomImageCache(Path(temporary) / "images", retention_days=7)
            cache.initialize()
            original = Path.write_bytes
            calls = 0

            def fail_second(path: Path, data: bytes) -> int:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("disk failure")
                return original(path, data)

            images = [
                AgentImage("image/png", b"first"),
                AgentImage("image/png", b"second"),
            ]
            with (
                patch.object(Path, "write_bytes", fail_second),
                self.assertRaises(OSError),
            ):
                await cache.persist("room", images)

            self.assertEqual(list(cache.root.rglob("*.png")), [])


if __name__ == "__main__":
    unittest.main()
