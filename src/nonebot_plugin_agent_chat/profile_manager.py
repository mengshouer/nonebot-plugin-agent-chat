from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .errors import ConfigurationError
from .models import SearchMode
from .profiles import LoadedProfile, ProfileRegistry, ProfileSnapshot
from .prompts import PromptRegistry
from .storage import AgentStore
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedProfiles:
    """A staged snapshot plus the registry it belongs to, not yet live."""

    registry: ProfileRegistry
    snapshot: ProfileSnapshot
    active: str


class ProfileManager:
    """Owns profile/prompt snapshots, validation, activation, and watching."""

    def __init__(
        self,
        config: Config,
        store: AgentStore,
        tool_registry: ToolRegistry,
        secret_resolver: Callable[[str], str | None] = os.getenv,
        data_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.tool_registry = tool_registry
        self.secret_resolver = secret_resolver
        data_dir = data_dir or config.agent_chat_data_dir.resolve()
        self.prompt_dir = data_dir / "prompts"
        self.profiles = ProfileRegistry(
            config.agent_chat_profile_dir.resolve(),
            config.agent_chat_default_profile,
            PromptRegistry(
                self.prompt_dir,
                config.agent_chat_default_system_prompt_file,
            ),
        )
        self._active_profile: str | None = None
        self._explicit_override = False
        self._startup_error: str | None = None
        self._lock = asyncio.Lock()
        self._reload_error: str | None = None

    @property
    def startup_error(self) -> str | None:
        return self._startup_error

    @property
    def active_profile_name(self) -> str | None:
        return self._active_profile

    @property
    def explicit_override(self) -> bool:
        """True when ``/agentctl use`` selected a profile for this process only."""

        return self._explicit_override

    @property
    def reload_error(self) -> str | None:
        return self._reload_error

    def prepare_prompt_dir(self) -> None:
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.prompt_dir.chmod(0o700)
        except OSError:
            pass

    async def initialize(self) -> None:
        """Stage the first snapshot; the configured default profile wins."""

        # Earlier versions stored the active profile here, which silently
        # overrode AGENT_CHAT_DEFAULT_PROFILE after a rename. The selection is
        # now config plus in-memory only, so drop the stale row.
        await self.store.remove_setting("active_profile")
        try:
            snapshot = await asyncio.to_thread(self.profiles.stage)
            active = self._resolve_active(
                self.profiles.default_profile,
                snapshot.profiles,
                configured_default=self.config.agent_chat_default_profile,
            )
            self._validate_snapshot(snapshot.profiles, active, self.config)
            self.profiles.activate(snapshot)
            self._active_profile = active
            self._startup_error = None
        except ConfigurationError as exc:
            self._startup_error = str(exc)
            logger.error("Agent chat is unavailable: %s", exc)

    async def reload(self) -> list[str]:
        """Atomically activate a new snapshot, keeping the prior one on failure.

        A reload keeps the in-memory selection, so an explicit ``/agentctl use``
        survives profile edits until the next restart.
        """

        async with self._lock:
            prepared = await self.prepare_reload(self.config)
            return await self.commit_reload(prepared)

    def registry_for(self, config: Config) -> ProfileRegistry:
        """The registry a config describes; the live one when its paths match."""

        if (
            config.agent_chat_profile_dir == self.config.agent_chat_profile_dir
            and config.agent_chat_default_system_prompt_file
            == self.config.agent_chat_default_system_prompt_file
        ):
            return self.profiles
        return ProfileRegistry(
            config.agent_chat_profile_dir.resolve(),
            config.agent_chat_default_profile,
            PromptRegistry(
                self.prompt_dir,
                config.agent_chat_default_system_prompt_file,
            ),
        )

    async def prepare_reload(
        self,
        config: Config,
        *,
        secret_resolver: Callable[[str], str | None] | None = None,
    ) -> PreparedProfiles:
        """Stage the profile set a config describes; nothing live is touched.

        Everything that can fail happens here, so a caller can keep the previous
        config and profiles when a reload fails. ``secret_resolver`` lets a
        caller validate against a staged environment instead of the live one.
        """

        registry = self.registry_for(config)
        resolver = secret_resolver or self.secret_resolver
        try:
            # The active name is not required: when its file was renamed or
            # deleted, a reload must resolve a valid profile instead of failing
            # forever and keeping the stale snapshot alive.
            snapshot = await asyncio.to_thread(registry.stage)
            # An explicit /agentctl use survives reloads; otherwise a changed
            # configured default takes over at the next reload.
            candidate = self._active_profile
            if not self._explicit_override and config.agent_chat_default_profile:
                candidate = config.agent_chat_default_profile
            active = self._resolve_active(
                candidate,
                snapshot.profiles,
                configured_default=config.agent_chat_default_profile,
            )
            self._validate_snapshot(snapshot.profiles, active, config, resolver)
        except ConfigurationError as exc:
            self._reload_error = str(exc)
            raise
        return PreparedProfiles(registry=registry, snapshot=snapshot, active=active)

    async def commit_reload(self, prepared: PreparedProfiles) -> list[str]:
        """Publish a prepared snapshot; no failure is expected at this point."""

        self.profiles = prepared.registry
        names = self.profiles.activate(prepared.snapshot)
        # An override is dropped when it was substituted, and also when the
        # configured default now names the same profile: switching back to the
        # default is the documented way to undo `/agentctl use`.
        self._explicit_override = bool(
            self._explicit_override
            and prepared.active == self._active_profile
            and prepared.active != self.config.agent_chat_default_profile
        )
        self._active_profile = prepared.active
        self._startup_error = None
        self._reload_error = None
        return names

    def adopt_config(self, config: Config) -> None:
        """Adopt a reloaded config; the registry swap belongs to commit_reload."""

        self.config = config
        self.profiles.default_profile = config.agent_chat_default_profile

    async def use(self, name: str) -> None:
        """Select a profile for this process only; a restart reverts to config.

        Switching back to the configured default is an undo: profile rules
        apply again instead of staying shadowed.
        """

        async with self._lock:
            await self._load_if_unknown(name)
            self._active_profile = name
            # A deliberate global switch outranks profile rules until restart.
            self._explicit_override = name != self.profiles.default_profile
            self._startup_error = None

    async def ensure_loaded(self, name: str) -> LoadedProfile:
        """Return a usable profile, loading it from disk when still unknown."""

        async with self._lock:
            return await self._load_if_unknown(name)

    async def _load_if_unknown(self, name: str) -> LoadedProfile:
        if name not in self.profiles.profiles:
            # The file may be newer than the active snapshot (or a failed startup
            # left no snapshot at all), so look it up on disk now instead of
            # answering with a stale "unknown profile". Staging reads and hashes
            # the whole profile directory, so it must not run on the event loop.
            snapshot = await asyncio.to_thread(self.profiles.stage, name)
            self._validate_snapshot(snapshot.profiles, name, self.config)
            self.profiles.activate(snapshot)
        loaded = self.profiles.get(name)
        self.validate_root_profile(loaded)
        return loaded

    def active_root(self) -> LoadedProfile:
        if self._startup_error:
            raise ConfigurationError(self._startup_error)
        if self._active_profile is None:
            raise ConfigurationError("No active profile")
        loaded = self.profiles.get(self._active_profile)
        self.validate_root_profile(loaded)
        return loaded

    def status(self) -> dict[str, object]:
        missing = self.profiles.missing_credentials(
            self.profiles.profiles.values(), self.secret_resolver
        )
        return {
            "active_profile": self._active_profile,
            "profiles": self.profiles.names(),
            "missing_credentials": missing,
            "startup_error": self._startup_error,
            "profile_reload_error": self._reload_error,
            "system_prompt": {
                "global": self.profiles.global_prompt.summary,
                "profiles": {
                    name: loaded.prompt.summary if loaded.prompt else "unknown"
                    for name, loaded in sorted(self.profiles.profiles.items())
                },
            },
        }

    def _resolve_active(
        self,
        candidate: str | None,
        profiles: dict[str, LoadedProfile],
        *,
        configured_default: str | None,
    ) -> str:
        """Resolve the active profile, surviving a rename of the selected file.

        The selected name can outlive the file it pointed at. When that happens,
        fall back to the configured default and finally to the only available
        profile. An ambiguous situation fails with an actionable error instead
        of guessing.

        The default comes from the config being loaded, not from a live
        registry: a reload may have just changed it, and a stale value here
        would make the reload fail until the process restarts.
        """

        if candidate and candidate in profiles:
            return candidate
        fallback = self._choose_snapshot_default(
            profiles, configured_default=configured_default
        )
        if candidate:
            logger.warning(
                "Selected profile %s no longer exists; using %s instead",
                candidate,
                fallback,
            )
        return fallback

    def _choose_snapshot_default(
        self,
        profiles: dict[str, LoadedProfile],
        *,
        configured_default: str | None,
    ) -> str:
        configured = configured_default
        if configured and configured in profiles:
            return configured
        names = sorted(profiles)
        if len(names) == 1:
            if configured:
                logger.warning(
                    "Configured default profile %s does not exist; using %s",
                    configured,
                    names[0],
                )
            return names[0]
        if configured:
            available = ", ".join(names) or "none"
            raise ConfigurationError(
                f"Default profile does not exist: {configured}. "
                f"Available profiles: {available}"
            )
        raise ConfigurationError(
            "Multiple profiles exist; set AGENT_CHAT_DEFAULT_PROFILE"
        )

    def _validate_snapshot(
        self,
        profiles: dict[str, LoadedProfile],
        active: str,
        config: Config,
        secret_resolver: Callable[[str], str | None] | None = None,
    ) -> None:
        if active not in profiles:
            raise ConfigurationError(f"Active profile does not exist: {active}")
        resolver = secret_resolver or self.secret_resolver
        for loaded in profiles.values():
            self.tool_registry.for_profile(loaded.config)
        self.validate_root_profile(profiles[active], config, resolver)

    def forget_active(self) -> None:
        """Drop the in-memory selection so the next reload resolves it from config."""

        self._active_profile = None

    def validate_root_profile(
        self,
        loaded: LoadedProfile,
        config: Config | None = None,
        secret_resolver: Callable[[str], str | None] | None = None,
    ) -> None:
        """Fail fast when a profile cannot run with the given configuration."""

        config = config or self.config
        resolver = secret_resolver or self.secret_resolver
        self.profiles.resolve_api_key(loaded.config, resolver)
        self.tool_registry.for_profile(loaded.config)
        if (
            loaded.config.search_mode != SearchMode.OFF
            and config.agent_chat_max_searches < 1
        ):
            raise ConfigurationError(
                "search_mode requires AGENT_CHAT_MAX_SEARCHES >= 1"
            )
        if (
            loaded.config.search_mode == SearchMode.EXA or loaded.config.enabled_tools
        ) and config.agent_chat_max_local_tool_calls < 1:
            raise ConfigurationError(
                "local tools require AGENT_CHAT_MAX_LOCAL_TOOL_CALLS >= 1"
            )
        if loaded.config.search_mode == SearchMode.EXA:
            # One resolver decides: an inline key is a credential too, so the
            # environment variable is only the fallback.
            self.profiles.resolve_exa_api_key(loaded.config, resolver)
