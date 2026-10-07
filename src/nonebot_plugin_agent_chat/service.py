from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from . import config_editor, env_file
from .adapters import create_adapter
from .config import Config
from .errors import (
    ConfigurationError,
    InputError,
    ProfileNotFoundError,
    ProfileRuleError,
    ProviderError,
    RoomError,
)
from .fallback import MAX_FALLBACK_PROFILES, is_compatible
from .hooks import HookRegistry, RunContext
from .hooks import registry as hook_registry
from .image_cache import RoomImageCache
from .images import ImageBudget
from .input import CollectedInput
from .models import AgentImage, AgentMessage, MessageRole, RunResult, SearchMode
from .platforms import (
    ConversationRef,
    applicable_rule_rows,
    rule_label,
    scope_value,
)
from .profile_manager import ProfileManager
from .profiles import LoadedProfile, ProfileRegistry
from .room_state import RoomState
from .run_guard import ActiveRun, RunGuard, RunRequest
from .runner import AgentRunner, RunnerBudget, RunnerLimits
from .storage import AgentRoom, AgentStore
from .tools import ToolContext, ToolRegistry, registry

logger = logging.getLogger(__name__)

# The marker file the CLI writes to ask a running bot for a reload. A file keeps
# the per-message check to one small local read instead of a SQLite round trip.
RELOAD_MARKER_FILE = "reload.request"
MAX_PROFILE_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 0.5


def reload_marker() -> str:
    """The marker value; monotonic enough for "newer than my last seen"."""

    return f"{time.time():.6f}"


def _parse_epoch(raw: str | None) -> float | None:
    """Read the reload marker; unknown or corrupt values mean "no request"."""

    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class AgentChatService:
    """Coordinates profiles, admission control, rooms, and the bounded runner."""

    def __init__(
        self,
        config: Config,
        tool_registry: ToolRegistry = registry,
        hooks: HookRegistry = hook_registry,
        secret_resolver: Callable[[str], str | None] = os.getenv,
        env_file: Path | None = None,
        env_owned: set[str] | None = None,
    ) -> None:
        self.config = config
        self.data_dir = config.agent_chat_data_dir.resolve()
        self.store = AgentStore(self.data_dir / "agent_chat.db")
        self.image_cache = RoomImageCache(
            self.data_dir / "images",
            config.agent_chat_room_image_retention_days,
        )
        self.tool_registry = tool_registry
        self.hooks = hooks
        self.secret_resolver = secret_resolver
        self.env_file = env_file
        # Keys the dotenv file itself provided; a real env var always wins.
        self.env_owned: set[str] = set(env_owned or ())
        self.profile_manager = ProfileManager(
            config,
            self.store,
            tool_registry,
            secret_resolver,
            self.data_dir,
        )
        self.rooms = RoomState(config, self.store, self.image_cache)
        self.guard = RunGuard(config, self.store)
        self._maintenance_task: asyncio.Task | None = None
        self._maintenance_interval: float | None = None
        self._last_reload_epoch = 0.0
        self._last_reload_report: dict[str, object] | None = None
        self._last_reload_error: str | None = None
        self._reload_lock = asyncio.Lock()
        self._reload_marker_path = self.data_dir / RELOAD_MARKER_FILE

    @property
    def profiles(self) -> ProfileRegistry:
        return self.profile_manager.profiles

    @property
    def startup_error(self) -> str | None:
        return self.profile_manager.startup_error

    @property
    def active_profile_name(self) -> str | None:
        return self.profile_manager.active_profile_name

    @property
    def _runs(self) -> dict[str, ActiveRun]:
        return {run.id: run for run in self.guard.active_runs}

    async def initialize(self, *, interrupt_running: bool = True) -> None:
        """Prepare the service; see ``AgentStore.initialize`` for the flag."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.data_dir.chmod(0o700)
        except OSError:
            pass
        self.image_cache.initialize()
        self.profile_manager.prepare_prompt_dir()
        await self.store.initialize(interrupt_running=interrupt_running)
        await self.profile_manager.initialize()
        # Startup already read the files, so a marker left by an earlier CLI
        # edit must not trigger another reload on the first ask.
        self._last_reload_epoch = self._read_marker() or 0.0
        await self._run_maintenance()
        self._ensure_maintenance_task()

    async def close(self) -> None:
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)
            self._maintenance_task = None
        self._maintenance_interval = None
        await self.guard.cancel_and_wait()
        await self.store.close()

    async def reload_profiles(self) -> list[str]:
        async with self._reload_lock:
            return await self.profile_manager.reload()

    async def request_reload(self) -> None:
        """Ask the running bot to reload (the marker the CLI writes)."""

        await asyncio.to_thread(
            env_file.write_atomic, self._reload_marker_path, reload_marker()
        )

    def _read_marker(self) -> float | None:
        """The marker's epoch; a missing or unreadable file means "no request"."""

        try:
            raw = self._reload_marker_path.read_text(encoding="utf-8")
        except OSError:
            return None
        return _parse_epoch(raw.strip())

    async def apply_pending_reload(self) -> bool:
        """Wait for any in-flight reload before checking the current marker."""

        async with self._reload_lock:
            epoch = self._read_marker()
            if epoch is None or epoch <= self._last_reload_epoch:
                return False
            try:
                await self._reload_locked()
            except Exception as exc:  # noqa: BLE001 - keep serving the last config
                # Consume rejected edits, but not cancellations: an interrupted
                # reload must be retried by the next message with the same marker.
                self._last_reload_epoch = epoch
                self._last_reload_error = str(exc)
                logger.error("Agent chat reload failed: %s", exc)
                return False
            self._last_reload_epoch = epoch
            self._last_reload_error = None
            return True

    async def reload_now(self) -> dict[str, object]:
        """Chat-triggered reload; adopts the current marker after completion."""

        async with self._reload_lock:
            epoch = self._read_marker() or 0.0
            try:
                report = await self._reload_locked()
            except Exception as exc:
                self._last_reload_epoch = epoch
                self._last_reload_error = str(exc)
                raise
            self._last_reload_epoch = epoch
            self._last_reload_error = None
            return report

    async def reload_everything(self) -> dict[str, object]:
        """Re-read the env file, rebuild Config, and reload profiles atomically.

        Everything that can fail -- dotenv parsing, Config validation, profile
        staging, credential resolution -- happens before any live state moves.
        The commit itself cannot fail: environment, config, and profiles publish
        together, so a rejected edit can never leave the running bot using new
        credentials with the old configuration.
        """

        async with self._reload_lock:
            return await self._reload_locked()

    async def _reload_locked(self) -> dict[str, object]:
        previous = self.config
        previous_owned = set(self.env_owned)
        staged = await self._stage_environment(previous_owned)
        candidate = dict(os.environ) if staged is None else staged.values
        new_config = await asyncio.to_thread(
            config_editor.rebuild_config,
            previous,
            unset_keys=(
                set() if staged is None else previous_owned - set(staged.owned)
            ),
            environ=candidate,
        )
        applied, restart = config_editor.diff_configs(previous, new_config)
        # Stage the profiles against the staged credentials: a broken profile set
        # or a missing credential keeps environment, config, and profiles as one
        # consistent snapshot.
        prepared = await self.profile_manager.prepare_reload(
            new_config,
            secret_resolver=self._staged_secret_resolver(staged, previous_owned),
        )
        if staged is not None:
            env_file.commit_environ(staged, previous_owned)
            self.env_owned = set(staged.owned)
        self.apply_config(new_config)
        names = await self.profile_manager.commit_reload(prepared)
        self._ensure_maintenance_task()
        report: dict[str, object] = {
            "config": {"applied": applied, "restart_required": restart},
            "profiles": names,
            "at": time.time(),
        }
        self._last_reload_report = report
        logger.info(
            "Agent chat reloaded: %d profiles, %d settings changed",
            len(names),
            len(applied),
        )
        for key, (before, after) in sorted(applied.items()):
            logger.info("Agent chat setting reloaded: %s: %s -> %s", key, before, after)
        if restart:
            logger.warning(
                "Agent chat settings need a restart: %s",
                ", ".join(sorted(restart)),
            )
        return report

    async def _stage_environment(
        self,
        owned: set[str],
    ) -> env_file.EnvironmentSnapshot | None:
        """Read the dotenv file into a candidate environment, touching nothing."""

        if self.env_file is None:
            return None
        return await asyncio.to_thread(env_file.stage_environ, self.env_file, owned)

    def _staged_secret_resolver(
        self,
        staged: env_file.EnvironmentSnapshot | None,
        previous_owned: set[str],
    ) -> Callable[[str], str | None]:
        """Resolve credentials from the staged environment, not the live one."""

        def resolve(name: str) -> str | None:
            if staged is not None and name in staged.values:
                return staged.values[name]
            if name in previous_owned:
                # The file owned this key and its line is gone; never fall back
                # to the stale process value while staging.
                return None
            return self.secret_resolver(name)

        return resolve

    def apply_config(self, new_config: Config) -> None:
        """Swap a rebuilt config into every live reference."""

        self.config = new_config
        self.rooms.config = new_config
        self.guard.config = new_config
        self.guard.set_concurrency(new_config.agent_chat_global_concurrency)
        self.image_cache.retention_days = (
            new_config.agent_chat_room_image_retention_days
        )
        self.profile_manager.adopt_config(new_config)

    async def use_profile(self, name: str) -> None:
        # Serialize with reloads: a reload publishes the active profile it
        # resolved before committing, so an interleaved switch would be lost.
        async with self._reload_lock:
            await self.profile_manager.use(name)

    def status(self) -> dict[str, object]:
        details: dict[str, object] = dict(self.profile_manager.status())
        details["active_runs"] = [
            {
                "id": item.id,
                "profile": item.profile,
                "seconds": round(time.monotonic() - item.started_at, 1),
            }
            for item in self.guard.active_runs
        ]
        return details

    async def status_details(self) -> dict[str, object]:
        details = self.status()
        details["recent_runs"] = await self.store.recent_runs()
        details["profile_rules"] = await self.rule_status()
        if self._last_reload_report is not None:
            details["last_reload"] = self._last_reload_report
        if self._last_reload_error is not None:
            details["last_reload_error"] = self._last_reload_error
        return details

    async def conversation_status(
        self, conversation: ConversationRef
    ) -> dict[str, object]:
        """The group-visible summary: what applies here, nothing else."""

        profile, source = await self.effective_profile(conversation)
        rows = applicable_rule_rows(await self.rule_status(), conversation)
        return {
            "conversation": rule_label(conversation.scope, conversation.target_id),
            "effective_profile": profile,
            "profile_source": source,
            # Stale markers belong to the operator views (private chat / CLI);
            # a group only learns what currently applies to it.
            "rules": [
                {"label": row["label"], "profile": row["profile"]} for row in rows
            ],
            # profile_ok must reflect a stale rule too: asks here would fail.
            "profile_ok": not (
                any(bool(row.get("stale")) for row in rows)
                or self.profile_manager.startup_error
                or self.profile_manager.reload_error
            ),
        }

    async def rule_status(self) -> list[dict[str, object]]:
        """Rules with display labels and the state the operator must see."""

        names = set(self.profiles.names())
        rules: list[dict[str, object]] = []
        for row in await self.store.list_profile_rules():
            scope = str(row["scope"])
            target_id = str(row["target_id"])
            rules.append(
                {
                    **row,
                    "label": rule_label(scope, target_id),
                    "stale": str(row["profile"]) not in names,
                }
            )
        return rules

    @staticmethod
    def format_rule_action(
        action: str,
        scope: str,
        target_id: str,
        *,
        profile: str = "",
        warnings: Sequence[str] = (),
        removed: bool = False,
    ) -> list[str]:
        """Confirmation lines for a rule command, shared by chat and CLI."""

        label = rule_label(scope, target_id)
        if action == "set":
            lines = [f"已设置规则：{label} → {profile}"]
            lines.extend(f"⚠️ {warning}" for warning in warnings)
            return lines
        if removed:
            return [f"已删除规则：{label}"]
        return [f"没有规则：{label}"]

    @staticmethod
    def format_rule_line(rule: dict[str, object], *, with_audit: bool) -> str:
        """One rule as a line; audit info only where disclosure allows it."""

        stale = "（失效）" if rule.get("stale") else ""
        line = f"{rule['label']} → {rule['profile']}{stale}"
        if with_audit:
            updated_at = rule.get("updated_at")
            stamp = time.strftime(
                "%Y-%m-%d %H:%M",
                time.localtime(updated_at if isinstance(updated_at, int) else 0),
            )
            line += f"  [{stamp} {rule.get('updated_by') or ''}]"
        return line

    async def set_profile_rule(
        self,
        *,
        scope: str,
        target_id: str,
        profile: str,
        updated_by: str,
        reveal_names: bool = True,
    ) -> tuple[dict[str, object] | None, list[str]]:
        """Store one rule; returns the row plus warnings for the operator.

        ``reveal_names`` is false in non-private chats, where the list of
        profile names counts as operator detail.
        """

        key = self._validated_rule_key(scope, target_id)
        try:
            loaded = await self.profile_manager.ensure_loaded(profile)
        except ConfigurationError as exc:
            # `ensure_loaded` stages the profile from disk, so this also covers
            # a profile file that exists but is invalid.
            detail = f"可用：{', '.join(self.profiles.names())}" if reveal_names else ""
            separator = "；" if detail else ""
            raise InputError(
                f"profile 不存在或不可用：{profile}{separator}{detail}"
            ) from exc
        warnings: list[str] = []
        allowed = self.config.agent_chat_allowed_groups
        # `scope:*` allows every conversation on the platform, including this
        # group, so it must not trigger the spelling warning.
        if (
            target_id
            and f"{key}:{target_id}" not in allowed
            and f"{key}:*" not in allowed
        ):
            warnings.append(
                "该群不在 AGENT_CHAT_ALLOWED_GROUPS 中，规则不会生效；"
                "请检查群号是否写错"
            )
        await self.store.set_profile_rule(
            scope=key,
            target_id=target_id,
            profile=loaded.name,
            updated_by=updated_by,
        )
        return await self.store.get_profile_rule(key, target_id), warnings

    async def unset_profile_rule(self, *, scope: str, target_id: str) -> bool:
        # Same shape checks as writes, so a typo cannot "succeed" as a no-op.
        key = self._validated_rule_key(scope, target_id)
        return await self.store.delete_profile_rule(key, target_id)

    @staticmethod
    def _validated_rule_key(scope: str, target_id: str) -> str:
        """Reject malformed rule keys before they reach the store."""

        try:
            key = scope_value(scope)
        except ValueError as exc:
            raise InputError(str(exc)) from exc
        if target_id and (target_id.strip() != target_id or set(target_id) & set(": ")):
            raise InputError(f"群号格式不正确：{target_id}")
        return key

    async def applicable_rule(
        self, conversation: ConversationRef
    ) -> dict[str, object] | None:
        """The rule the ladder would apply here: group rule, then platform rule."""

        scope = conversation.rule_scope
        if scope is None:
            return None
        for target_id, source in (
            (conversation.target_id, "group-rule"),
            ("", "platform-rule"),
        ):
            row = await self.store.get_profile_rule(scope, target_id)
            if row is not None:
                return {
                    **row,
                    "label": rule_label(scope, target_id),
                    "source": source,
                }
        return None

    async def _rule_profile(
        self, conversation: ConversationRef
    ) -> LoadedProfile | None:
        rule = await self.applicable_rule(conversation)
        if rule is None:
            return None
        name = str(rule["profile"])
        try:
            loaded = self.profiles.get(name)
        except ProfileNotFoundError as exc:
            raise ProfileRuleError(
                f"本会话的 profile 规则已失效：{rule['label']} → {name}；"
                f"请用 /agentctl rule unset 删除该规则，或恢复该 profile"
            ) from exc
        self.profile_manager.validate_root_profile(loaded)
        return loaded

    async def effective_profile(
        self, conversation: ConversationRef | None
    ) -> tuple[str, str]:
        """Effective profile name and the level it came from, for display."""

        if conversation is not None and not self.profile_manager.explicit_override:
            rule = await self.applicable_rule(conversation)
            if rule is not None:
                return str(rule["profile"]), str(rule["source"])
        return self._active_choice()

    def _active_choice(self) -> tuple[str, str]:
        """The non-rule tier: a deliberate switch, else the configured default."""

        try:
            name = self._active_root().name
        except ConfigurationError:
            # A broken startup is reported by status itself; keep it readable.
            return "-", "unavailable"
        source = "use" if self.profile_manager.explicit_override else "default"
        return name, source

    async def _root_for(self, conversation: ConversationRef | None) -> LoadedProfile:
        """Resolve the profile for one run: an explicit switch outranks rules."""

        if conversation is not None and not self.profile_manager.explicit_override:
            found = await self._rule_profile(conversation)
            if found is not None:
                return found
        return self._active_root()

    def _ensure_maintenance_task(self) -> None:
        """Keep the cleanup loop in step with the live interval setting.

        A reload can enable the loop, disable it, or change the interval. The
        loop sleeps the whole interval, so a changed interval must restart it;
        otherwise an interval of hours would delay the new setting until the
        next day.
        """

        interval = self.config.agent_chat_cleanup_interval_seconds
        if interval <= 0:
            self._stop_maintenance_task()
            return
        if (
            self._maintenance_task is not None
            and not self._maintenance_task.done()
            and self._maintenance_interval == interval
        ):
            return
        self._stop_maintenance_task()
        self._maintenance_interval = interval
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    def _stop_maintenance_task(self) -> None:
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            self._maintenance_task = None
        self._maintenance_interval = None

    async def _maintenance_loop(self) -> None:
        while True:
            # Re-read the interval every tick so a reload can change it.
            interval = self.config.agent_chat_cleanup_interval_seconds
            await asyncio.sleep(interval if interval > 0 else 3600.0)
            if interval <= 0:
                continue
            try:
                await self._run_maintenance()
            except Exception as exc:  # noqa: BLE001 - background boundary
                logger.error("Agent chat maintenance failed: %s", type(exc).__name__)

    async def _run_maintenance(self) -> None:
        await self.rooms.cleanup_retention()

    def _active_root(self) -> LoadedProfile:
        return self.profile_manager.active_root()

    def _limits(self) -> RunnerLimits:
        return RunnerLimits(
            max_model_turns=self.config.agent_chat_max_model_turns,
            max_local_tool_calls=self.config.agent_chat_max_local_tool_calls,
            max_searches=self.config.agent_chat_max_searches,
            tool_timeout_seconds=self.config.agent_chat_tool_timeout_seconds,
        )

    async def _run_profile(
        self,
        loaded: LoadedProfile,
        messages: list[AgentMessage],
        *,
        subject_key: str = "",
        context_key: str = "",
        budget: RunnerBudget | None = None,
    ) -> RunResult:
        profile = loaded.config
        if profile.search_mode == SearchMode.BUILTIN_WEB_SEARCH:
            profile = profile.model_copy(
                update={
                    "max_builtin_tool_calls": min(
                        profile.max_builtin_tool_calls,
                        self.config.agent_chat_max_searches,
                    )
                }
            )
        api_key = self.profiles.resolve_api_key(profile, self.secret_resolver)
        tool_specs = self.tool_registry.for_profile(profile)
        context = RunContext(
            profile_name=loaded.name,
            profile=profile,
            subject_key=subject_key,
            context_key=context_key,
        )
        await self.hooks.run_before_run(messages, context)
        adapter = create_adapter(profile, api_key)
        try:
            runner = AgentRunner(
                adapter=adapter,
                tools=self.tool_registry,
                tool_specs=tool_specs,
                tool_context=ToolContext(
                    profile_name=loaded.name,
                    profile=profile,
                    subject_key=subject_key,
                    context_key=context_key,
                    secret_resolver=self.secret_resolver,
                ),
                limits=self._limits(),
                hooks=self.hooks,
                run_context=context,
                budget=budget,
            )
            try:
                result = await runner.run(messages)
            except ProviderError as exc:
                exc.profile_name = loaded.name
                raise
            result.actual_profile = loaded.name
            await self.hooks.run_after_response(result, context)
            return result
        finally:
            await adapter.close()

    @staticmethod
    def _can_retry(exc: ProviderError) -> bool:
        return exc.retriable and not exc.emitted_text

    async def _run_profile_with_retry(
        self,
        loaded: LoadedProfile,
        messages: list[AgentMessage],
        *,
        subject_key: str = "",
        context_key: str = "",
        budget: RunnerBudget | None = None,
    ) -> RunResult:
        for attempt in range(MAX_PROFILE_ATTEMPTS):
            try:
                return await self._run_profile(
                    loaded,
                    messages,
                    subject_key=subject_key,
                    context_key=context_key,
                    budget=budget,
                )
            except ProviderError as exc:
                if not self._can_retry(exc) or attempt + 1 >= MAX_PROFILE_ATTEMPTS:
                    raise
                logger.warning(
                    "Profile %s failed on attempt %s; retrying: %s",
                    loaded.name,
                    attempt + 1,
                    exc,
                )
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
        raise AssertionError("profile retry loop exhausted without a result")

    async def _run_with_fallback(
        self,
        root: LoadedProfile,
        messages: list[AgentMessage],
        *,
        subject_key: str = "",
        context_key: str = "",
        chain: list[LoadedProfile] | None = None,
    ) -> RunResult:
        budget = RunnerBudget()
        has_images = any(message.images for message in messages)
        last_error: BaseException | None = None
        candidates = chain or self.profiles.root_chain(
            root.name,
            limit=MAX_FALLBACK_PROFILES,
        )

        for candidate in candidates:
            if not is_compatible(root, candidate, has_images=has_images):
                continue
            try:
                self.profile_manager.validate_root_profile(candidate)
            except ConfigurationError as exc:
                if candidate.name == root.name:
                    raise
                logger.warning("Skipping unusable fallback %s: %s", candidate.name, exc)
                if last_error is None:
                    last_error = exc
                continue

            try:
                return await self._run_profile_with_retry(
                    candidate,
                    messages,
                    subject_key=subject_key,
                    context_key=context_key,
                    budget=budget,
                )
            except ProviderError as exc:
                last_error = exc
                if not self._can_retry(exc):
                    raise
                logger.warning(
                    "Profile %s exhausted retries: %s",
                    candidate.name,
                    exc,
                )

        if last_error is not None:
            raise last_error
        raise ConfigurationError("No compatible profile in fallback chain")

    def _image_budget(self) -> ImageBudget:
        return ImageBudget(
            max_images=self.config.agent_chat_max_images,
            max_image_bytes=self.config.agent_chat_max_image_bytes,
        )

    def _validate_collected_images(self, collected: CollectedInput) -> None:
        budget = self._image_budget()
        budget.check_count(len(collected.images))
        budget.check_total(sum(len(image.data) for image in collected.images))

    async def _run_collected(
        self,
        root: LoadedProfile,
        chain: list[LoadedProfile],
        collected: CollectedInput,
        *,
        subject_key: str,
        context_key: str,
    ) -> RunResult:
        self._validate_collected_images(collected)
        messages = [
            AgentMessage(
                role=MessageRole.USER, text=collected.text, images=collected.images
            )
        ]
        return await self._run_with_fallback(
            root,
            messages,
            subject_key=subject_key,
            context_key=context_key,
            chain=chain,
        )

    async def ask(
        self,
        collected: CollectedInput,
        *,
        subject_key: str,
        context_key: str,
        conversation: ConversationRef | None = None,
        enforce_cooldown: bool = True,
        enforce_daily_quota: bool = True,
    ) -> RunResult:
        async def collect_once() -> CollectedInput:
            return collected

        return await self.ask_deferred(
            collect_once,
            subject_key=subject_key,
            context_key=context_key,
            conversation=conversation,
            enforce_cooldown=enforce_cooldown,
            enforce_daily_quota=enforce_daily_quota,
        )

    async def ask_deferred(
        self,
        collect: Callable[[], Awaitable[CollectedInput]],
        *,
        subject_key: str,
        context_key: str,
        conversation: ConversationRef | None = None,
        enforce_cooldown: bool = True,
        enforce_daily_quota: bool = True,
    ) -> RunResult:
        root = await self._root_for(conversation)
        chain = self.profiles.root_chain(root.name, limit=MAX_FALLBACK_PROFILES)

        async def operation() -> RunResult:
            collected = await collect()
            return await self._run_collected(
                root,
                chain,
                collected,
                subject_key=subject_key,
                context_key=context_key,
            )

        return await self._guarded_run(
            subject_key=subject_key,
            context_key=context_key,
            requested_profile=root.name,
            operation=operation,
            enforce_cooldown=enforce_cooldown,
            exclusive_context=False,
            enforce_daily_quota=enforce_daily_quota,
        )

    async def _guarded_run(
        self,
        *,
        subject_key: str,
        context_key: str,
        requested_profile: str,
        operation: Callable[[], Awaitable[RunResult]],
        enforce_cooldown: bool,
        exclusive_context: bool,
        enforce_daily_quota: bool = True,
    ) -> RunResult:
        request = RunRequest(
            subject_key=subject_key,
            context_key=context_key,
            requested_profile=requested_profile,
            enforce_cooldown=enforce_cooldown,
            exclusive_context=exclusive_context,
            enforce_daily_quota=enforce_daily_quota,
        )
        return await self.guard.execute(request, operation)

    async def cancel_runs(self, target: str) -> int:
        return await self.guard.cancel(target)

    async def create_room(self, context_key: str, name: str) -> AgentRoom:
        self.rooms.ensure_enabled()
        # Management operations wait for the active turn so history updates stay
        # ordered. New asks are admitted separately and reject a busy context.
        async with self.rooms.context_lock(context_key):
            root = self._active_root()
            return await self.rooms.new_room(context_key, name, root.name)

    async def room_status(self, context_key: str) -> AgentRoom | None:
        self.rooms.ensure_enabled()
        return await self.rooms.status(context_key)

    async def clear_room(self, context_key: str) -> None:
        self.rooms.ensure_enabled()
        async with self.rooms.context_lock(context_key):
            await self.rooms.clear(context_key)

    async def close_room(self, context_key: str) -> None:
        self.rooms.ensure_enabled()
        async with self.rooms.context_lock(context_key):
            await self.rooms.close(context_key)

    async def set_room_profile(self, context_key: str, name: str) -> AgentRoom:
        """Rebind an active Room to another profile, keeping its history."""

        self.rooms.ensure_enabled()
        async with self.rooms.context_lock(context_key):
            room = await self.rooms.require(context_key)
            loaded = await self.profile_manager.ensure_loaded(name)
            await self.rooms.set_profile(room, loaded.name)
            # Re-read so callers see the persisted binding, not the prior one.
            return await self.rooms.require(context_key)

    async def ask_room(
        self,
        collected: CollectedInput,
        *,
        subject_key: str,
        context_key: str,
    ) -> RunResult:
        async def collect_once() -> CollectedInput:
            return collected

        return await self.ask_room_deferred(
            collect_once,
            subject_key=subject_key,
            context_key=context_key,
        )

    async def ask_room_deferred(
        self,
        collect: Callable[[], Awaitable[CollectedInput]],
        *,
        subject_key: str,
        context_key: str,
    ) -> RunResult:
        self.rooms.ensure_enabled()
        requested_room = await self.rooms.require(context_key)

        async def operation() -> RunResult:
            # Admission precedes the lock: a management operation can hold it,
            # but the waiting ask still belongs to cancellation and timeout.
            async with self.rooms.context_lock(context_key):
                self.rooms.ensure_enabled()
                room = await self.rooms.require(context_key)
                try:
                    # Re-read after waiting: management may have changed the Room
                    # or its binding. Never substitute a missing bound profile.
                    loaded = self.profiles.get(room.profile)
                except ProfileNotFoundError as exc:
                    raise RoomError(
                        f"Agent Room 绑定的 profile 已不存在：{room.profile}；"
                        f"请用 /agent_room use <profile> 切换"
                    ) from exc
                self.profile_manager.validate_root_profile(loaded)
                collected = await collect()
                self._validate_collected_images(collected)
                history = await self._load_room_history(room, collected.images)
                if not loaded.config.capabilities.vision and (
                    collected.images or any(message.images for message in history)
                ):
                    raise InputError("当前 Agent Room 的 profile 不支持图片")
                history.append(
                    AgentMessage(
                        role=MessageRole.USER,
                        text=collected.text,
                        images=collected.images,
                    )
                )
                budget = RunnerBudget()
                result = await self._run_profile_with_retry(
                    loaded,
                    history,
                    subject_key=subject_key,
                    context_key=context_key,
                    budget=budget,
                )
                await self.rooms.append_turn(
                    room,
                    collected.text,
                    collected.images,
                    result.text,
                )
                await self.rooms.prune(room)
                return result

        return await self._guarded_run(
            subject_key=subject_key,
            context_key=context_key,
            requested_profile=requested_room.profile,
            operation=operation,
            enforce_cooldown=True,
            exclusive_context=True,
            enforce_daily_quota=True,
        )

    async def _load_room_history(
        self,
        room: AgentRoom,
        current_images: list[AgentImage],
    ) -> list[AgentMessage]:
        return await self.rooms.history(room, current_images)
