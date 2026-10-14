"""Where the plugin keeps its files.

Directories come from ``nonebot-plugin-localstore``; the plugin exposes no path
settings of its own.  Standalone callers can resolve the same directories by
initializing a minimal NoneBot driver first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATABASE_NAME = "agent_chat.db"
PROFILES_NAME = "profiles"
PROMPTS_NAME = "prompts"
IMAGES_NAME = "images"
DEBUG_NAME = "debug"

# The id nonebot-plugin-localstore files this plugin under.
PLUGIN_ID = "nonebot_plugin_agent_chat"


@dataclass(frozen=True, init=False)
class StoragePaths:
    """The data and profile directories this process reads and writes."""

    data_dir: Path
    profile_dir: Path

    def __init__(
        self,
        data_dir: Path,
        profile_dir: Path | None = None,
    ) -> None:
        resolved_data = Path(data_dir).resolve()
        resolved_profiles = (
            Path(profile_dir).resolve()
            if profile_dir is not None
            else resolved_data / PROFILES_NAME
        )
        object.__setattr__(self, "data_dir", resolved_data)
        object.__setattr__(self, "profile_dir", resolved_profiles)

    @property
    def database_path(self) -> Path:
        return self.data_dir / DATABASE_NAME

    @property
    def prompt_dir(self) -> Path:
        return self.data_dir / PROMPTS_NAME

    @property
    def image_dir(self) -> Path:
        return self.data_dir / IMAGES_NAME

    @property
    def debug_dir(self) -> Path:
        return self.data_dir / DEBUG_NAME

    @classmethod
    def resolve(
        cls,
        *,
        data_dir: Path | None = None,
        profile_dir: Path | None = None,
        initialize_driver: bool = False,
    ) -> StoragePaths:
        """Resolve directories from ``nonebot-plugin-localstore`` unless overridden.

        The standalone CLI has no loaded plugin from which localstore can infer
        a caller, so ``initialize_driver`` makes it initialize a no-op NoneBot
        driver and asks localstore by this plugin's id.  An explicit
        ``data_dir`` skips querying (and creating) the store directory.  There
        is deliberately no legacy-directory fallback: localstore is the single
        source of truth.
        """

        if data_dir is not None:
            return cls(data_dir, profile_dir)

        resolved_data = _localstore_data_dir()
        if resolved_data is None and initialize_driver:
            resolved_data = _store_data_dir_for_plugin()
        if resolved_data is None:
            raise RuntimeError(
                "nonebot-plugin-localstore is unavailable or NoneBot is not "
                "initialized; initialize NoneBot first or pass explicit paths"
            )
        return cls(resolved_data, profile_dir)


def _localstore_data_dir() -> Path | None:
    """Data directory from nonebot-plugin-localstore, or ``None`` without a driver."""

    try:
        import nonebot_plugin_localstore as store

        # Importing the store already reads the driver config, so an
        # uninitialized NoneBot fails here rather than at the call below.
        return store.get_plugin_data_dir()
    except (ImportError, ValueError, RuntimeError):
        return None


def _store_data_dir_for_plugin() -> Path:
    """Resolve the localstore data directory by plugin id for the standalone CLI."""

    import nonebot
    from nonebot.log import logger as nonebot_logger

    # NoneBot logs to stdout from the moment it is imported; CLI output,
    # including --json, must stay machine-readable.
    nonebot_logger.configure(extra={"nonebot_log_level": "ERROR"})

    try:
        nonebot.get_driver()
    except ValueError:
        nonebot.init(driver="~none", log_level="ERROR")

    import nonebot_plugin_localstore as store

    custom = store.plugin_config.localstore_plugin_data_dir.get(PLUGIN_ID)
    if custom is not None:
        custom.mkdir(parents=True, exist_ok=True)
        return custom
    return store.get_data_dir(PLUGIN_ID)
