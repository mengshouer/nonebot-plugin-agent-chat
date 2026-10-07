from __future__ import annotations

import asyncio
import logging
import mimetypes
import time
import uuid
from pathlib import Path

from .models import AgentImage
from .storage import StoredImage

logger = logging.getLogger(__name__)


class RoomImageCache:
    def __init__(self, root: Path, retention_days: int) -> None:
        self.root = root.resolve()
        self.retention_days = retention_days

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass

    def safe_path(self, raw: str) -> Path | None:
        try:
            path = Path(raw).resolve()
            path.relative_to(self.root)
        except (OSError, ValueError):
            logger.warning("Rejected image path outside cache: %s", raw)
            return None
        return path

    @staticmethod
    def _read_bounded(path: Path, max_bytes: int) -> bytes | None:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
        return data if len(data) <= max_bytes else None

    async def load_if_fits(
        self,
        images: list[StoredImage],
        *,
        max_images: int,
        max_bytes: int,
    ) -> tuple[list[AgentImage], int]:
        if len(images) > max_images or max_bytes < 1:
            return [], 0
        loaded: list[AgentImage] = []
        total_bytes = 0
        for image in images:
            path = self.safe_path(image.path)
            if path is None:
                continue
            try:
                data = await asyncio.to_thread(
                    self._read_bounded,
                    path,
                    max_bytes - total_bytes,
                )
            except OSError:
                continue
            if data is None:
                return [], 0
            loaded.append(AgentImage(media_type=image.media_type, data=data))
            total_bytes += len(data)
        return loaded, total_bytes

    async def persist(
        self,
        room_id: str,
        images: list[AgentImage],
    ) -> list[StoredImage]:
        directory = self.root / room_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            directory.chmod(0o700)
        except OSError:
            pass
        stored: list[StoredImage] = []
        attempted_paths: list[str] = []
        try:
            for image in images:
                suffix = mimetypes.guess_extension(image.media_type) or ".bin"
                path = directory / f"{uuid.uuid4().hex}{suffix}"
                attempted_paths.append(str(path))
                write_task = asyncio.create_task(
                    asyncio.to_thread(path.write_bytes, image.data)
                )
                try:
                    await asyncio.shield(write_task)
                except BaseException:
                    await asyncio.gather(write_task, return_exceptions=True)
                    raise
                try:
                    path.chmod(0o600)
                except OSError:
                    pass
                stored.append(StoredImage(path=str(path), media_type=image.media_type))
        except BaseException:
            await self.delete(attempted_paths)
            raise
        return stored

    async def delete(self, paths: list[str]) -> None:
        for raw in paths:
            path = self.safe_path(raw)
            if path is None:
                continue
            try:
                await asyncio.to_thread(path.unlink, True)
            except OSError:
                logger.warning("Could not remove cached image %s", path)

    def _expired(self, cutoff: float) -> list[str]:
        paths: list[str] = []
        for path in self.root.rglob("*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    paths.append(str(path))
            except OSError:
                continue
        return paths

    async def cleanup(self) -> None:
        cutoff = time.time() - self.retention_days * 86400
        # Walking the cache is filesystem work; keep it off the event loop.
        paths = await asyncio.to_thread(self._expired, cutoff)
        await self.delete(paths)
