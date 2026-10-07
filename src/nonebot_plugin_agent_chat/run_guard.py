from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .config import Config
from .errors import BusyError, InputError, ProviderError
from .models import RunResult
from .storage import AgentStore


def hash_identifier(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()


@dataclass(frozen=True)
class ActiveRun:
    id: str
    profile: str
    started_at: float
    task: asyncio.Task


@dataclass(frozen=True)
class RunRequest:
    subject_key: str
    context_key: str
    requested_profile: str
    enforce_cooldown: bool
    exclusive_context: bool
    enforce_daily_quota: bool = True


class RunGuard:
    """Admission control and run bookkeeping: concurrency, cooldown, quota, cancel."""

    def __init__(self, config: Config, store: AgentStore) -> None:
        self.config = config
        self.store = store
        self._runs: dict[str, ActiveRun] = {}
        self._subjects: dict[str, str] = {}
        self._exclusive_contexts: dict[str, str] = {}
        self._cooldowns: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._concurrency = config.agent_chat_global_concurrency
        self._semaphore = asyncio.Semaphore(self._concurrency)

    def set_concurrency(self, value: int) -> None:
        """Rebuild the admission semaphore after a config reload.

        Runs already waiting keep the old semaphore; new admissions use this
        one, which is the closest semantics a live change can offer.
        """

        if value == self._concurrency:
            return
        self._concurrency = value
        self._semaphore = asyncio.Semaphore(value)

    @property
    def active_runs(self) -> list[ActiveRun]:
        return list(self._runs.values())

    async def execute(
        self,
        request: RunRequest,
        operation: Callable[[], Awaitable[RunResult]],
    ) -> RunResult:
        now = time.monotonic()
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("No current asyncio task")

        run_id = await self._admit(request, now=now, task=task)

        run_recorded = False
        try:

            async def execute_run() -> RunResult:
                nonlocal run_recorded
                salt = await self.store.get_setting("log_salt") or ""
                subject_hash = hash_identifier(salt, request.subject_key)
                context_hash = hash_identifier(salt, request.context_key)
                if (
                    request.enforce_daily_quota
                    and self.config.agent_chat_daily_request_limit
                ):
                    utc_day = int(time.time()) // 86400 * 86400
                    used = await self.store.count_runs_since(subject_hash, utc_day)
                    if used >= self.config.agent_chat_daily_request_limit:
                        raise BusyError("今日请求次数已达到上限")
                await self.store.start_run(
                    run_id,
                    subject_hash,
                    context_hash,
                    request.requested_profile,
                )
                run_recorded = True
                async with self._semaphore:
                    return await operation()

            result = await asyncio.wait_for(
                execute_run(),
                timeout=self.config.agent_chat_run_timeout_seconds,
            )
            result.run_id = run_id
            await self.store.finish_run(
                run_id,
                status="completed",
                result=result,
                duration_ms=int((time.monotonic() - now) * 1000),
            )
            return result
        except asyncio.CancelledError:
            if run_recorded:
                await self.store.finish_run(
                    run_id,
                    status="cancelled",
                    error_type="cancelled",
                    duration_ms=int((time.monotonic() - now) * 1000),
                )
            raise
        except asyncio.TimeoutError:
            if run_recorded:
                await self.store.finish_run(
                    run_id,
                    status="failed",
                    error_type="run_timeout",
                    duration_ms=int((time.monotonic() - now) * 1000),
                )
            raise ProviderError(
                "Run timeout",
                retriable=False,
                error_type="run_timeout",
            )
        except Exception as exc:
            if run_recorded:
                await self.store.finish_run(
                    run_id,
                    status="failed",
                    actual_profile=getattr(exc, "profile_name", None),
                    error_type=getattr(exc, "error_type", type(exc).__name__),
                    duration_ms=int((time.monotonic() - now) * 1000),
                )
            raise
        finally:
            await self._release(run_id, request)

    async def cancel(self, target: str) -> int:
        async with self._lock:
            if target == "all":
                selected = list(self._runs.values())
            else:
                selected = [
                    item
                    for run_id, item in self._runs.items()
                    if run_id.startswith(target)
                ]
                if len(selected) > 1:
                    raise InputError("run id 前缀不唯一")
        for item in selected:
            item.task.cancel()
        return len(selected)

    async def cancel_and_wait(self) -> None:
        """Cancel every active run and wait for the tasks to unwind."""

        tasks = [item.task for item in self.active_runs]
        await self.cancel("all")
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _admit(
        self,
        request: RunRequest,
        *,
        now: float,
        task: asyncio.Task,
    ) -> str:
        async with self._lock:
            cooldown_cutoff = now - max(
                self.config.agent_chat_user_cooldown_seconds,
                60.0,
            )
            self._cooldowns = {
                key: value
                for key, value in self._cooldowns.items()
                if value >= cooldown_cutoff or key in self._subjects
            }
            if request.subject_key in self._subjects:
                raise BusyError("你已有一个请求正在处理中")
            if (
                request.exclusive_context
                and request.context_key in self._exclusive_contexts
            ):
                raise BusyError("当前 Agent Room 正在处理中")
            if len(self._runs) >= self.config.agent_chat_global_concurrency:
                raise BusyError("当前请求较多，请稍后再试")
            last = self._cooldowns.get(request.subject_key, 0.0)
            if (
                request.enforce_cooldown
                and now - last < self.config.agent_chat_user_cooldown_seconds
            ):
                raise BusyError("请求过于频繁，请稍后再试")

            run_id = uuid.uuid4().hex[:12]
            self._runs[run_id] = ActiveRun(
                run_id,
                request.requested_profile,
                now,
                task,
            )
            self._subjects[request.subject_key] = run_id
            if request.exclusive_context:
                self._exclusive_contexts[request.context_key] = run_id
            self._cooldowns[request.subject_key] = now
            return run_id

    async def _release(self, run_id: str, request: RunRequest) -> None:
        async with self._lock:
            self._runs.pop(run_id, None)
            self._subjects.pop(request.subject_key, None)
            if self._exclusive_contexts.get(request.context_key) == run_id:
                self._exclusive_contexts.pop(request.context_key, None)
