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
                    CREATE TABLE rooms (
                        id TEXT PRIMARY KEY,
                        context_key TEXT NOT NULL,
                        name TEXT NOT NULL,
                        profile TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        closed_at INTEGER
                    );
                    CREATE TABLE room_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                        sequence INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        text TEXT NOT NULL,
                        images_json TEXT NOT NULL DEFAULT '[]',
                        created_at INTEGER NOT NULL,
                        UNIQUE(room_id, sequence)
                    );
                    INSERT INTO rooms(
                        id, context_key, name, profile, status, created_at, updated_at
                    ) VALUES(
                        'legacy-room', 'group:legacy', 'old', 'primary', 'active', 1, 2
                    );
                    INSERT INTO room_messages(
                        room_id, sequence, role, text, images_json, created_at
                    ) VALUES
                        ('legacy-room', 1, 'user', 'legacy question', '[]', 1),
                        ('legacy-room', 2, 'assistant', 'legacy answer', '[]', 2);
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
                cursor = await store._db().execute("PRAGMA table_info(rooms)")
                room_columns = {str(row[1]) for row in await cursor.fetchall()}
                await cursor.close()
                cursor = await store._db().execute("PRAGMA user_version")
                row = await cursor.fetchone()
                await cursor.close()
                migrated_room = await store.active_room("group:legacy")
                migrated_messages = await store.room_history("legacy-room")
            finally:
                await store.close()

            self.assertIn("duration_ms", columns)
            self.assertIn("reasoning_tokens", columns)
            self.assertIn("summary", room_columns)
            self.assertIn("context_tokens", room_columns)
            self.assertIn("compactions", room_columns)
            self.assertEqual(int(row[0]), AgentStore.SCHEMA_VERSION)
            assert migrated_room is not None
            self.assertEqual(migrated_room.id, "legacy-room")
            self.assertEqual(migrated_room.summary, "")
            self.assertEqual(migrated_room.context_tokens, 0)
            self.assertEqual(migrated_room.compactions, 0)
            self.assertEqual(
                [(m.role, m.text) for m in migrated_messages],
                [("user", "legacy question"), ("assistant", "legacy answer")],
            )

    async def test_settings_and_room_compaction_storage(self) -> None:
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
                        context_tokens=(index + 1) * 100,
                    )
                history = await store.room_history(room.id)
                await store.apply_room_compaction(
                    room.id,
                    summary="condensed u0/a0",
                    remove_ids=[history[0].id, history[1].id],
                    context_tokens=150,
                )
                remaining = await store.room_history(room.id)
                active = await store.active_room("group:1")
                assert active is not None
                self.assertEqual(active.summary, "condensed u0/a0")
                self.assertEqual(active.context_tokens, 150)
                self.assertEqual(active.compactions, 1)
                self.assertEqual(
                    [(message.role, message.text) for message in remaining],
                    [
                        ("user", "u1"),
                        ("assistant", "a1"),
                        ("user", "u2"),
                        ("assistant", "a2"),
                    ],
                )
                await store.clear_room(room.id)
                cleared = await store.active_room("group:1")
                assert cleared is not None
                self.assertEqual(cleared.summary, "")
                self.assertEqual(cleared.context_tokens, 0)
                self.assertEqual(cleared.compactions, 0)
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
