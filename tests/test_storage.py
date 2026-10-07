import tempfile
import unittest
from pathlib import Path

from nonebot_plugin_agent_chat.models import RunResult, Usage
from nonebot_plugin_agent_chat.storage import AgentStore, StoredMessage


class StorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_database_is_migrated_and_version_stamped(self) -> None:
        """A pre-versioned store gets the new columns and the schema version."""

        import aiosqlite

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.db"
            legacy = await aiosqlite.connect(str(path))
            try:
                await legacy.executescript(
                    """
                    CREATE TABLE runs (
                        id TEXT PRIMARY KEY,
                        subject_hash TEXT NOT NULL,
                        context_hash TEXT NOT NULL,
                        requested_profile TEXT NOT NULL,
                        actual_profile TEXT,
                        status TEXT NOT NULL,
                        started_at INTEGER NOT NULL,
                        finished_at INTEGER,
                        input_tokens INTEGER NOT NULL DEFAULT 0,
                        output_tokens INTEGER NOT NULL DEFAULT 0,
                        total_tokens INTEGER NOT NULL DEFAULT 0,
                        model_turns INTEGER NOT NULL DEFAULT 0,
                        tool_calls INTEGER NOT NULL DEFAULT 0,
                        searches INTEGER NOT NULL DEFAULT 0,
                        error_type TEXT
                    );
                    PRAGMA user_version=3;
                    """
                )
                await legacy.commit()
            finally:
                await legacy.close()

            store = AgentStore(path)
            await store.initialize(interrupt_running=False)
            try:
                cursor = await store._db().execute("PRAGMA table_info(runs)")
                columns = {str(row[1]) for row in await cursor.fetchall()}
                await cursor.close()
                cursor = await store._db().execute("PRAGMA user_version")
                row = await cursor.fetchone()
                await cursor.close()
            finally:
                await store.close()

            self.assertIn("duration_ms", columns)
            self.assertIn("reasoning_tokens", columns)
            self.assertEqual(int(row[0]), AgentStore.SCHEMA_VERSION)

    async def test_settings_and_room_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = AgentStore(Path(temporary) / "agent.db")
            await store.initialize()
            try:
                await store.set_setting("log_salt", "pepper")
                await store.remove_setting("log_salt")
                self.assertIsNone(await store.get_setting("log_salt"))

                room = await store.new_room("group:1", "test", "primary")
                for index in range(3):
                    await store.append_exchange(
                        room.id,
                        StoredMessage(role="user", text=f"u{index}"),
                        StoredMessage(role="assistant", text=f"a{index}"),
                    )
                await store.prune_room(room.id, max_turns=2, max_chars=100)
                history = await store.room_history(room.id, 10, 100)
                self.assertEqual(
                    [(message.role, message.text) for message in history],
                    [
                        ("user", "u1"),
                        ("assistant", "a1"),
                        ("user", "u2"),
                        ("assistant", "a2"),
                    ],
                )
            finally:
                await store.close()

    async def test_pruning_keeps_a_contiguous_newest_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = AgentStore(Path(temporary) / "agent.db")
            await store.initialize()
            try:
                room = await store.new_room("group:1", "test", "primary")
                for user, assistant in (
                    ("u0", "a0"),
                    ("U" * 10, "A" * 10),
                    ("u2", "a2"),
                ):
                    await store.append_exchange(
                        room.id,
                        StoredMessage(role="user", text=user),
                        StoredMessage(role="assistant", text=assistant),
                    )
                await store.prune_room(room.id, max_turns=10, max_chars=8)
                history = await store.room_history(room.id, 10, 100)
                self.assertEqual(
                    [message.text for message in history],
                    ["u2", "a2"],
                )
            finally:
                await store.close()

    async def test_run_metadata_keeps_reasoning_token_count_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = AgentStore(Path(temporary) / "agent.db")
            await store.initialize()
            try:
                await store.start_run("run-1", "subject-hash", "context-hash", "p")
                await store.finish_run(
                    "run-1",
                    status="completed",
                    result=RunResult(
                        text="answer",
                        sources=[],
                        usage=Usage(
                            input_tokens=3,
                            output_tokens=5,
                            total_tokens=8,
                            reasoning_tokens=4,
                        ),
                        model_turns=1,
                        local_tool_calls=0,
                        searches=0,
                        actual_profile="p",
                    ),
                )
                rows = await store.recent_runs()
                self.assertEqual(rows[0]["reasoning_tokens"], 4)
                self.assertNotIn("answer", rows[0])
            finally:
                await store.close()

    async def test_new_room_archives_previous_room(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = AgentStore(Path(temporary) / "agent.db")
            await store.initialize()
            try:
                first = await store.new_room("group:1", "first", "primary")
                second = await store.new_room("group:1", "second", "primary")
                active = await store.active_room("group:1")
                self.assertIsNotNone(active)
                assert active is not None
                self.assertEqual(active.id, second.id)
                self.assertNotEqual(first.id, second.id)
            finally:
                await store.close()


if __name__ == "__main__":
    unittest.main()
