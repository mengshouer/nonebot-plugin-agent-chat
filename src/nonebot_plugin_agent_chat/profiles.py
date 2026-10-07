from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .errors import (
    ConfigurationError,
    ProfileCredentialError,
    ProfileNotFoundError,
)
from .models import ProviderProfile
from .prompts import (
    BUILTIN_DEFAULT_PROMPT,
    SOURCE_BUILTIN,
    PromptRegistry,
    ResolvedPrompt,
)

_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def is_valid_profile_name(name: str) -> bool:
    """Profile names are the file stems; the loader enforces the same shape."""

    return bool(_PROFILE_ID.fullmatch(name))


@dataclass(frozen=True)
class LoadedProfile:
    name: str
    config: ProviderProfile
    path: Path
    prompt: ResolvedPrompt | None = None


@dataclass(frozen=True)
class ProfileSnapshot:
    profiles: dict[str, LoadedProfile]
    global_prompt: ResolvedPrompt


class ProfileRegistry:
    """Atomically stages and activates validated profile snapshots."""

    def __init__(
        self,
        directory: Path,
        default_profile: str | None = None,
        prompts: PromptRegistry | None = None,
    ) -> None:
        self.directory = directory
        self.default_profile = default_profile
        self.prompts = prompts or PromptRegistry(directory.parent / "prompts")
        self._profiles: dict[str, LoadedProfile] = {}
        self._global_prompt = ResolvedPrompt.create(
            BUILTIN_DEFAULT_PROMPT,
            SOURCE_BUILTIN,
            "builtin",
        )

    @property
    def global_prompt(self) -> ResolvedPrompt:
        return self._global_prompt

    @property
    def profiles(self) -> dict[str, LoadedProfile]:
        return dict(self._profiles)

    def names(self) -> list[str]:
        return sorted(self._profiles)

    def override_for_runtime(self, name: str, **updates: object) -> LoadedProfile:
        current = self.get(name)
        # python mode keeps the SecretStr objects intact; the json mode would
        # round-trip them through their "**********" repr and lose the key.
        values = current.config.model_dump(mode="python", exclude_unset=True)
        values.update(updates)
        # The staged snapshot already resolved this profile's prompt; keep the
        # resolved text instead of re-resolving (or losing) it on override.
        if current.prompt is not None:
            values["system_prompt"] = current.prompt.text
            values["system_prompt_file"] = None
        try:
            profile = ProviderProfile.model_validate(values)
        except ValidationError as exc:
            raise ConfigurationError(
                f"Invalid runtime override for {name}: {exc}"
            ) from exc
        loaded = LoadedProfile(
            name=name,
            path=current.path,
            config=profile,
            prompt=current.prompt,
        )
        self._profiles = {**self._profiles, name: loaded}
        return loaded

    def get(self, name: str) -> LoadedProfile:
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise ProfileNotFoundError(f"Unknown profile: {name}") from exc

    @staticmethod
    def _safe_error_detail(exc: BaseException) -> str:
        """Describe a failure without echoing profile contents.

        A pydantic ``ValidationError`` renders the offending input, which can be
        an inline ``system_prompt``, so only field paths and messages are kept.
        """

        if not isinstance(exc, ValidationError):
            return str(exc)
        parts: list[str] = []
        for error in exc.errors()[:5]:
            location = ".".join(str(item) for item in error.get("loc", ()))
            parts.append(
                f"{location or 'profile'}: {error.get('msg', 'invalid value')}"
            )
        return "; ".join(parts) or "invalid profile"

    def stage(self, required_profile: str | None = None) -> ProfileSnapshot:
        self.directory.mkdir(parents=True, exist_ok=True)
        loaded: dict[str, LoadedProfile] = {}
        errors: list[str] = []
        global_prompt = self.prompts.resolve_global()

        for path in sorted(self.directory.glob("*.json")):
            name = path.stem
            if not _PROFILE_ID.fullmatch(name):
                errors.append(f"{path.name}: invalid profile filename")
                continue
            try:
                raw = json.loads(path.read_bytes())
                profile = ProviderProfile.model_validate(raw)
                prompt = self.prompts.resolve_profile(profile, global_prompt)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValidationError,
                ConfigurationError,
            ) as exc:
                errors.append(f"{path.name}: {self._safe_error_detail(exc)}")
                continue
            frozen = profile.model_copy(
                update={"system_prompt": prompt.text, "system_prompt_file": None}
            )
            loaded[name] = LoadedProfile(
                name=name,
                config=frozen,
                path=path,
                prompt=prompt,
            )

        for name, item in loaded.items():
            seen = set()
            for fallback in item.config.fallback_profiles:
                if fallback == name:
                    errors.append(f"{name}: cannot fallback to itself")
                elif fallback in seen:
                    errors.append(f"{name}: duplicate fallback {fallback}")
                elif fallback not in loaded:
                    errors.append(f"{name}: fallback profile not found: {fallback}")
                seen.add(fallback)

        if required_profile and required_profile not in loaded:
            errors.append(f"active profile not found: {required_profile}")
        if not loaded:
            errors.append(f"no profile JSON files found in {self.directory}")

        if errors:
            raise ConfigurationError(
                "Invalid profile configuration:\n" + "\n".join(errors)
            )
        return ProfileSnapshot(loaded, global_prompt)

    def activate(self, snapshot: ProfileSnapshot) -> list[str]:
        self._profiles = dict(snapshot.profiles)
        self._global_prompt = snapshot.global_prompt
        return self.names()

    def load(self, required_profile: str | None = None) -> list[str]:
        return self.activate(self.stage(required_profile))

    def root_chain(self, name: str, limit: int = 3) -> list[LoadedProfile]:
        root = self.get(name)
        names = [name] + root.config.fallback_profiles
        return [self.get(candidate) for candidate in names[:limit]]

    @staticmethod
    def resolve_api_key(
        profile: ProviderProfile,
        resolver: Callable[[str], str | None] = os.getenv,
    ) -> str:
        if profile.api_key is not None:
            # The inline credential is the most specific source, so it wins even
            # over a real environment variable (documented in the README).
            return profile.api_key.get_secret_value()
        if not profile.api_key_env:
            return "not-required"
        value = resolver(profile.api_key_env)
        if not value:
            raise ProfileCredentialError(
                f"Environment variable {profile.api_key_env} is not set"
            )
        return value

    @staticmethod
    def resolve_exa_api_key(
        profile: ProviderProfile,
        resolver: Callable[[str], str | None] = os.getenv,
    ) -> str:
        """The search credential: the inline value wins over the env variable."""

        if profile.exa_api_key is not None:
            return profile.exa_api_key.get_secret_value()
        value = resolver(profile.exa_api_key_env)
        if not value:
            raise ProfileCredentialError(
                f"Environment variable {profile.exa_api_key_env} is not set"
            )
        return value

    @staticmethod
    def missing_credentials(
        profiles: Iterable[LoadedProfile],
        resolver: Callable[[str], str | None] = os.getenv,
    ) -> dict[str, str]:
        missing: dict[str, str] = {}
        for loaded in profiles:
            env_name = loaded.config.api_key_env
            if loaded.config.api_key is None and env_name and not resolver(env_name):
                missing[loaded.name] = env_name
            if loaded.config.search_mode.value == "exa":
                exa_env = loaded.config.exa_api_key_env
                if loaded.config.exa_api_key is None and not resolver(exa_env):
                    missing[f"{loaded.name}:exa"] = exa_env
        return missing
