from __future__ import annotations

import asyncio
import time
import weakref
from collections.abc import Sequence

from .compaction import estimate_room_tokens, summary_messages
from .config import Config
from .errors import RoomError
from .image_cache import RoomImageCache
from .images import ImageBudget
from .models import AgentImage, AgentMessage, MessageRole
from .storage import AgentRoom, AgentStore, StoredImage, StoredMessage


class RoomState:
    """Agent Room persistence: creation, history, images, compaction, retention.

    These are lock-free primitives; the service serializes them per context with
    ``context_lock`` so a context never runs two room mutations at once.
    """

    def __init__(
        self,
        config: Config,
        store: AgentStore,
        image_cache: RoomImageCache,
    ) -> None:
        self.config = config
        self.store = store
        self.image_cache = image_cache
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self.last_cleanup = 0.0

    def ensure_enabled(self) -> None:
        if not self.config.agent_chat_room_enabled:
            raise RoomError("Agent Room 未启用")

    def context_lock(self, context_key: str) -> asyncio.Lock:
        lock = self._locks.get(context_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[context_key] = lock
        return lock

    async def new_room(
        self,
        context_key: str,
        name: str,
        profile_name: str,
    ) -> AgentRoom:
        return await self.store.new_room(context_key, name or "room", profile_name)

    async def status(self, context_key: str) -> AgentRoom | None:
        return await self.store.active_room(context_key)

    async def clear(self, context_key: str) -> None:
        room = await self.require(context_key)
        paths = await self.store.clear_room(room.id)
        await self.image_cache.delete(paths)

    async def set_profile(self, room: AgentRoom, name: str) -> None:
        await self.store.set_room_profile(room.id, name)

    async def close(self, context_key: str) -> None:
        room = await self.require(context_key)
        await self.store.close_room(room.id)

    async def require(self, context_key: str) -> AgentRoom:
        room = await self.store.active_room(context_key)
        if room is None:
            raise RoomError("当前会话没有活跃的 Agent Room")
        return room

    async def stored_messages(self, room_id: str) -> list[StoredMessage]:
        return await self.store.room_history(room_id)

    async def history(
        self,
        room: AgentRoom,
        current_images: list[AgentImage],
    ) -> list[AgentMessage]:
        """Load summary and stored turns, spending image budget newest-first."""

        stored = await self.store.room_history(room.id)
        budget = ImageBudget(
            max_images=self.config.agent_chat_max_images,
            max_image_bytes=self.config.agent_chat_max_image_bytes,
        )
        remaining_images = budget.max_images - len(current_images)
        remaining_bytes = budget.remaining_bytes(
            sum(len(image.data) for image in current_images)
        )
        loaded_images: dict[int, list[AgentImage]] = {}
        for index in range(len(stored) - 1, -1, -1):
            if not stored[index].images:
                continue
            images, image_bytes = await self.image_cache.load_if_fits(
                stored[index].images,
                max_images=remaining_images,
                max_bytes=remaining_bytes,
            )
            if not images:
                continue
            loaded_images[index] = images
            remaining_images -= len(images)
            remaining_bytes -= image_bytes

        messages = summary_messages(room.summary)
        messages.extend(
            AgentMessage(
                role=MessageRole(message.role),
                text=message.text,
                images=loaded_images.get(index, []),
            )
            for index, message in enumerate(stored)
        )
        return messages

    async def append_turn(
        self,
        room: AgentRoom,
        user_text: str,
        images: list[AgentImage],
        answer: str,
        *,
        context_tokens: int = 0,
    ) -> None:
        """Persist one exchange, removing any files written by a failed save."""

        stored_images: list[StoredImage] = []
        try:
            stored_images = await self.image_cache.persist(room.id, images)
            await self.store.append_exchange(
                room.id,
                StoredMessage(role="user", text=user_text, images=stored_images),
                StoredMessage(role="assistant", text=answer),
                context_tokens=context_tokens,
            )
        except BaseException:
            await self.image_cache.delete([image.path for image in stored_images])
            raise

    async def apply_compaction(
        self,
        room_id: str,
        *,
        summary: str,
        remove_ids: Sequence[int],
        kept_messages: Sequence[StoredMessage],
        count_compaction: bool = True,
    ) -> int:
        tokens_after = estimate_room_tokens(summary, kept_messages)
        paths = await self.store.apply_room_compaction(
            room_id,
            summary=summary,
            remove_ids=remove_ids,
            context_tokens=tokens_after,
            count_compaction=count_compaction,
        )
        await self.image_cache.delete(paths)
        return tokens_after

    async def cleanup_retention(self) -> None:
        await self.store.cleanup_runs(
            self.config.agent_chat_run_metadata_retention_days
        )
        paths = await self.store.cleanup_closed_rooms(
            self.config.agent_chat_room_retention_days
        )
        await self.image_cache.delete(paths)
        await self.image_cache.cleanup()
        self.last_cleanup = time.monotonic()
