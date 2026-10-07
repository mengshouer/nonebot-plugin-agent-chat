from __future__ import annotations

import asyncio
import json
import secrets
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite

from .models import RunResult


@dataclass(frozen=True)
class StoredImage:
    path: str
    media_type: str


@dataclass(frozen=True)
class StoredMessage:
    role: str
    text: str
    images: list[StoredImage] = field(default_factory=list)


def _decode_stored_images(value: object) -> list[StoredImage]:
    """Decode one ``images_json`` column; corrupt or legacy rows yield no images."""

    try:
        raw = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [
        StoredImage(
            path=str(item.get("path") or ""),
            media_type=str(item.get("media_type") or "application/octet-stream"),
        )
        for item in raw
        if isinstance(item, dict) and item.get("path")
    ]


def _stored_image_paths(value: object) -> list[str]:
    return [image.path for image in _decode_stored_images(value)]


@dataclass(frozen=True)
class AgentRoom:
    id: str
    context_key: str
    name: str
    profile: str
    status: str
    created_at: int
    updated_at: int


class AgentStore:
    """SQLite-backed runtime metadata: runs, rooms, settings, profile rules.

    The schema is versioned with ``PRAGMA user_version``: the version is read
    first, the idempotent DDL and column migrations run, and the new version is
    stamped only after they succeed.
    """

    SCHEMA_VERSION = 4

    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    async def initialize(self, *, interrupt_running: bool = True) -> None:
        """Open the store; ``interrupt_running`` marks leftover runs interrupted.

        A side process (the CLI managing profile rules) must not touch the
        runs of a bot that is still alive.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(str(self.path))
        self.connection = connection
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA foreign_keys=ON")
        cursor = await connection.execute("PRAGMA user_version")
        version_row = await cursor.fetchone()
        await cursor.close()
        version = int(version_row[0]) if version_row is not None else 0
        await connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                subject_hash TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                requested_profile TEXT NOT NULL,
                actual_profile TEXT,
                status TEXT NOT NULL,
                started_at INTEGER NOT NULL,
                finished_at INTEGER,
                duration_ms INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                model_turns INTEGER NOT NULL DEFAULT 0,
                tool_calls INTEGER NOT NULL DEFAULT 0,
                searches INTEGER NOT NULL DEFAULT 0,
                error_type TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_runs_subject_started
            ON runs(subject_hash, started_at);

            CREATE TABLE IF NOT EXISTS rooms (
                id TEXT PRIMARY KEY,
                context_key TEXT NOT NULL,
                name TEXT NOT NULL,
                profile TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                closed_at INTEGER
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_rooms_active_context
            ON rooms(context_key) WHERE status = 'active';

            CREATE TABLE IF NOT EXISTS room_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id TEXT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                images_json TEXT NOT NULL DEFAULT '[]',
                created_at INTEGER NOT NULL,
                UNIQUE(room_id, sequence)
            );

            CREATE TABLE IF NOT EXISTS profile_rules (
                scope TEXT NOT NULL,
                target_id TEXT NOT NULL DEFAULT '',
                profile TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                updated_by TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (scope, target_id)
            );
            """
        )
        cursor = await connection.execute("PRAGMA table_info(runs)")
        columns = {str(row[1]) for row in await cursor.fetchall()}
        await cursor.close()
        if "duration_ms" not in columns:
            await connection.execute(
                "ALTER TABLE runs ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0"
            )
        if "reasoning_tokens" not in columns:
            await connection.execute(
                "ALTER TABLE runs ADD COLUMN reasoning_tokens "
                "INTEGER NOT NULL DEFAULT 0"
            )
        if version < self.SCHEMA_VERSION:
            # Stamped only once the idempotent migrations above have run.
            await connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            await connection.commit()
        if interrupt_running:
            now = int(time.time())
            await connection.execute(
                """
                UPDATE runs SET status = 'interrupted', finished_at = ?,
                    error_type = 'process_interrupted'
                WHERE status = 'running'
                """,
                (now,),
            )
            await connection.commit()
        if await self.get_setting("log_salt") is None:
            await self.set_setting("log_salt", secrets.token_hex(32))

    def _db(self) -> aiosqlite.Connection:
        if self.connection is None:
            raise RuntimeError("AgentStore is not initialized")
        return self.connection

    async def close(self) -> None:
        if self.connection is not None:
            await self.connection.close()
            self.connection = None

    async def _execute_write(self, sql: str, parameters: Any = ()) -> None:
        await self._execute_mutating(sql, parameters)

    async def _execute_mutating(self, sql: str, parameters: Any = ()) -> int:
        """Run one write, returning the affected-row count."""

        async with self._write_lock:
            try:
                cursor = await self._db().execute(sql, parameters)
                count = max(cursor.rowcount, 0)
                await cursor.close()
                await self._db().commit()
                return count
            except BaseException:
                await self._db().rollback()
                raise

    async def get_setting(self, key: str) -> str | None:
        cursor = await self._db().execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        return str(row["value"]) if row is not None else None

    async def set_setting(self, key: str, value: str) -> None:
        await self._execute_write(
            """
            INSERT INTO settings(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    async def remove_setting(self, key: str) -> None:
        await self._execute_write("DELETE FROM settings WHERE key = ?", (key,))

    async def set_profile_rule(
        self,
        *,
        scope: str,
        target_id: str,
        profile: str,
        updated_by: str,
    ) -> None:
        await self._execute_write(
            """
            INSERT INTO profile_rules(
                scope, target_id, profile, updated_at, updated_by
            )
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(scope, target_id) DO UPDATE SET
                profile = excluded.profile,
                updated_at = excluded.updated_at,
                updated_by = excluded.updated_by
            """,
            (scope, target_id, profile, int(time.time()), updated_by),
        )

    async def get_profile_rule(
        self, scope: str, target_id: str
    ) -> dict[str, Any] | None:
        cursor = await self._db().execute(
            "SELECT scope, target_id, profile, updated_at, updated_by "
            "FROM profile_rules WHERE scope = ? AND target_id = ?",
            (scope, target_id),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return dict(row) if row is not None else None

    async def rooms_using_profile(self, profile: str) -> list[str]:
        """Names of open rooms bound to one profile (delete warnings)."""

        cursor = await self._db().execute(
            "SELECT name FROM rooms WHERE closed_at IS NULL AND profile = ?",
            (profile,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return sorted(str(row[0]) for row in rows)

    async def list_profile_rules(self) -> list[dict[str, Any]]:
        cursor = await self._db().execute(
            "SELECT scope, target_id, profile, updated_at, updated_by "
            "FROM profile_rules ORDER BY scope, target_id"
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [dict(row) for row in rows]

    async def delete_profile_rule(self, scope: str, target_id: str) -> bool:
        # One statement with a row count: no read-then-delete race with a
        # second process (the CLI) deleting the same rule.
        return (
            await self._execute_mutating(
                "DELETE FROM profile_rules WHERE scope = ? AND target_id = ?",
                (scope, target_id),
            )
            > 0
        )

    async def start_run(
        self,
        run_id: str,
        subject_hash: str,
        context_hash: str,
        requested_profile: str,
    ) -> None:
        await self._execute_write(
            """
            INSERT INTO runs(
                id, subject_hash, context_hash, requested_profile, status, started_at
            ) VALUES(?, ?, ?, ?, 'running', ?)
            """,
            (
                run_id,
                subject_hash,
                context_hash,
                requested_profile,
                int(time.time()),
            ),
        )

    async def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        result: RunResult | None = None,
        actual_profile: str | None = None,
        error_type: str | None = None,
        duration_ms: int = 0,
    ) -> None:
        values: dict[str, Any] = {
            "actual_profile": actual_profile,
            "status": status,
            "finished_at": int(time.time()),
            "duration_ms": max(0, duration_ms),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "model_turns": 0,
            "tool_calls": 0,
            "searches": 0,
            "error_type": error_type,
        }
        if result is not None:
            values.update(
                {
                    "actual_profile": result.actual_profile,
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "total_tokens": result.usage.total_tokens,
                    "reasoning_tokens": result.usage.reasoning_tokens,
                    "model_turns": result.model_turns,
                    "tool_calls": result.local_tool_calls,
                    "searches": result.searches,
                }
            )
        await self._execute_write(
            """
            UPDATE runs SET
                actual_profile = :actual_profile,
                status = :status,
                finished_at = :finished_at,
                duration_ms = :duration_ms,
                input_tokens = :input_tokens,
                output_tokens = :output_tokens,
                total_tokens = :total_tokens,
                reasoning_tokens = :reasoning_tokens,
                model_turns = :model_turns,
                tool_calls = :tool_calls,
                searches = :searches,
                error_type = :error_type
            WHERE id = :run_id
            """,
            {"run_id": run_id, **values},
        )

    async def recent_runs(self, limit: int = 5) -> list[dict[str, Any]]:
        cursor = await self._db().execute(
            """
            SELECT id, requested_profile, actual_profile, status, started_at,
                finished_at, duration_ms, total_tokens, reasoning_tokens,
                model_turns, tool_calls, searches, error_type
            FROM runs ORDER BY started_at DESC LIMIT ?
            """,
            (max(1, min(limit, 20)),),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [dict(row) for row in rows]

    async def count_runs_since(self, subject_hash: str, since: int) -> int:
        cursor = await self._db().execute(
            "SELECT COUNT(*) AS count FROM runs "
            "WHERE subject_hash = ? AND started_at >= ?",
            (subject_hash, since),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return int(row["count"] if row is not None else 0)

    async def cleanup_runs(self, retention_days: int) -> None:
        cutoff = int(time.time()) - retention_days * 86400
        await self._execute_write(
            "DELETE FROM runs WHERE started_at < ?",
            (cutoff,),
        )

    async def new_room(self, context_key: str, name: str, profile: str) -> AgentRoom:
        now = int(time.time())
        room_id = uuid.uuid4().hex
        db = self._db()
        async with self._write_lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                await db.execute(
                    """
                    UPDATE rooms SET status = 'closed', closed_at = ?, updated_at = ?
                    WHERE context_key = ? AND status = 'active'
                    """,
                    (now, now, context_key),
                )
                await db.execute(
                    """
                    INSERT INTO rooms(
                        id, context_key, name, profile, status, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (room_id, context_key, name, profile, now, now),
                )
            except BaseException:
                await db.rollback()
                raise
            else:
                await db.commit()
        return AgentRoom(room_id, context_key, name, profile, "active", now, now)

    async def active_room(self, context_key: str) -> AgentRoom | None:
        cursor = await self._db().execute(
            """
            SELECT id, context_key, name, profile, status, created_at, updated_at
            FROM rooms WHERE context_key = ? AND status = 'active'
            """,
            (context_key,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return AgentRoom(
            id=str(row["id"]),
            context_key=str(row["context_key"]),
            name=str(row["name"]),
            profile=str(row["profile"]),
            status=str(row["status"]),
            created_at=int(row["created_at"]),
            updated_at=int(row["updated_at"]),
        )

    async def append_exchange(
        self,
        room_id: str,
        user: StoredMessage,
        assistant: StoredMessage,
    ) -> None:
        db = self._db()
        async with self._write_lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) AS sequence
                    FROM room_messages WHERE room_id = ?
                    """,
                    (room_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                sequence = int(row["sequence"] if row is not None else 0)
                now = int(time.time())
                for message in (user, assistant):
                    sequence += 1
                    images = [
                        {"path": image.path, "media_type": image.media_type}
                        for image in message.images
                    ]
                    await db.execute(
                        """
                        INSERT INTO room_messages(
                            room_id, sequence, role, text, images_json, created_at
                        ) VALUES(?, ?, ?, ?, ?, ?)
                        """,
                        (
                            room_id,
                            sequence,
                            message.role,
                            message.text,
                            json.dumps(images, ensure_ascii=False),
                            now,
                        ),
                    )
                await db.execute(
                    "UPDATE rooms SET updated_at = ? WHERE id = ?",
                    (now, room_id),
                )
            except BaseException:
                await db.rollback()
                raise
            else:
                await db.commit()

    async def room_history(
        self,
        room_id: str,
        max_turns: int,
        max_chars: int,
    ) -> list[StoredMessage]:
        cursor = await self._db().execute(
            """
            SELECT role, text, images_json FROM room_messages
            WHERE room_id = ? ORDER BY sequence DESC LIMIT ?
            """,
            (room_id, max_turns * 2),
        )
        rows = list(await cursor.fetchall())
        await cursor.close()
        rows.reverse()

        messages: list[StoredMessage] = []
        for row in rows:
            images = _decode_stored_images(row["images_json"])
            messages.append(
                StoredMessage(
                    role=str(row["role"]), text=str(row["text"]), images=images
                )
            )

        while (
            len(messages) > 2
            and sum(len(message.text) for message in messages) > max_chars
        ):
            messages = messages[2:]
        return messages

    async def prune_room(
        self,
        room_id: str,
        max_turns: int,
        max_chars: int,
    ) -> list[str]:
        cursor = await self._db().execute(
            """
            SELECT id, sequence, text, images_json FROM room_messages
            WHERE room_id = ? ORDER BY sequence DESC
            """,
            (room_id,),
        )
        rows = list(await cursor.fetchall())
        await cursor.close()

        keep_ids = set()
        chars = 0
        kept_turns = 0
        # Exchanges are committed as adjacent user/assistant pairs. Select whole
        # pairs from newest to oldest so pruning can never leave half a turn.
        for index in range(0, len(rows), 2):
            exchange = rows[index : index + 2]
            if len(exchange) != 2 or kept_turns >= max_turns:
                continue
            exchange_chars = sum(len(str(row["text"])) for row in exchange)
            if keep_ids and chars + exchange_chars > max_chars:
                break
            keep_ids.update(int(row["id"]) for row in exchange)
            chars += exchange_chars
            kept_turns += 1

        removed = [row for row in rows if int(row["id"]) not in keep_ids]
        paths = [
            path for row in removed for path in _stored_image_paths(row["images_json"])
        ]
        if removed:
            ids = [int(row["id"]) for row in removed]
            placeholders = ",".join("?" for _ in ids)
            await self._execute_write(
                f"DELETE FROM room_messages WHERE id IN ({placeholders})",
                ids,
            )
        return paths

    async def clear_room(self, room_id: str) -> list[str]:
        paths = await self._room_image_paths(room_id)
        db = self._db()
        async with self._write_lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                await db.execute(
                    "DELETE FROM room_messages WHERE room_id = ?", (room_id,)
                )
                await db.execute(
                    "UPDATE rooms SET updated_at = ? WHERE id = ?",
                    (int(time.time()), room_id),
                )
            except BaseException:
                await db.rollback()
                raise
            else:
                await db.commit()
        return paths

    async def set_room_profile(self, room_id: str, name: str) -> None:
        await self._execute_write(
            "UPDATE rooms SET profile = ?, updated_at = ? WHERE id = ?",
            (name, int(time.time()), room_id),
        )

    async def close_room(self, room_id: str) -> None:
        now = int(time.time())
        await self._execute_write(
            """
            UPDATE rooms SET status = 'closed', closed_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (now, now, room_id),
        )

    async def cleanup_closed_rooms(self, retention_days: int) -> list[str]:
        cutoff = int(time.time()) - retention_days * 86400
        cursor = await self._db().execute(
            "SELECT id FROM rooms WHERE status = 'closed' AND closed_at < ?",
            (cutoff,),
        )
        room_ids = [str(row["id"]) for row in await cursor.fetchall()]
        await cursor.close()
        paths: list[str] = []
        for room_id in room_ids:
            paths.extend(await self._room_image_paths(room_id))
        if room_ids:
            placeholders = ",".join("?" for _ in room_ids)
            await self._execute_write(
                f"DELETE FROM rooms WHERE id IN ({placeholders})",
                room_ids,
            )
        return paths

    async def _room_image_paths(self, room_id: str) -> list[str]:
        cursor = await self._db().execute(
            "SELECT images_json FROM room_messages WHERE room_id = ?", (room_id,)
        )
        paths = [
            path
            for row in await cursor.fetchall()
            for path in _stored_image_paths(row["images_json"])
        ]
        await cursor.close()
        return paths
