"""Answer policy: profile/platform overrides and superuser diagnostics.

``matchers.py`` owns the NoneBot wiring (rules, logging, sending); the
resolution rules and status payloads live here as functions of a ``Config``,
the profile registry, and a scope. That keeps them free of import-time NoneBot
state and directly unit-testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nonebot_plugin_alconna.uniseg import SupportScope

from .config import Config
from .errors import ConfigurationError
from .image_reply import resolve_platform_mode, resolve_show_sources
from .models import ImageReplyMode
from .platforms import (
    DEFAULT_PLATFORM,
    PLATFORM_MAX_IMAGE_PAGES,
    PLATFORM_TEXT_LIMITS,
    platform_entries,
    scope_value,
)
from .profiles import ProfileRegistry


def profile_field(
    profiles: ProfileRegistry, profile_name: str | None, field: str
) -> Any:
    """One optional profile override field, or None when absent/unresolvable."""

    if not profile_name:
        return None
    try:
        return getattr(profiles.get(profile_name).config, field)
    except ConfigurationError:
        return None


def image_reply_mode(
    config: Config,
    profiles: ProfileRegistry,
    *,
    profile_name: str | None,
    scope: str,
) -> ImageReplyMode:
    """Resolve the effective mode: platform > profile > global."""

    return resolve_platform_mode(
        config.agent_chat_image_reply_mode,
        profile_field(profiles, profile_name, "image_reply_mode"),
        config.agent_chat_image_reply_mode_by_platform,
        scope,
    )


def show_sources(
    config: Config,
    profiles: ProfileRegistry,
    *,
    profile_name: str | None,
    scope: str,
    kind: str,
) -> bool:
    """Whether one delivery path shows sources: platform > profile > global."""

    if kind == "text":
        return resolve_show_sources(
            config.agent_chat_show_sources_text,
            profile_field(profiles, profile_name, "show_sources_text"),
            config.agent_chat_show_sources_text_by_platform,
            scope,
        )
    return resolve_show_sources(
        config.agent_chat_show_sources_image,
        profile_field(profiles, profile_name, "show_sources_image"),
        config.agent_chat_show_sources_image_by_platform,
        scope,
    )


def platform_status(installed_adapters: Sequence[str]) -> dict[str, object]:
    """Superuser diagnostics: built-in platform facts and generic defaults."""

    builtin = {
        scope_value(platform.scope): {
            "scope": platform.scope,
            "source": "builtin",
            "text_limit": platform.text_limit,
            "text_measure": platform.text_measure_name,
            "max_image_pages": platform.max_image_pages,
            "answer_format": platform.answer_format.name,
            "command_mentions": platform.command_mentions,
            "isolation": platform.isolation is not None,
        }
        for platform in platform_entries()
    }
    return {
        "builtin": builtin,
        "generic_defaults": {
            "text_limit": DEFAULT_PLATFORM.text_limit,
            "text_measure": DEFAULT_PLATFORM.text_measure_name,
            "max_image_pages": DEFAULT_PLATFORM.max_image_pages,
            "answer_format": DEFAULT_PLATFORM.answer_format.name,
            "command_mentions": DEFAULT_PLATFORM.command_mentions,
            "isolation": DEFAULT_PLATFORM.isolation is not None,
        },
        "scopes": [scope.value for scope in SupportScope],
        "installed_adapters": sorted(installed_adapters),
    }


def image_reply_status(
    config: Config,
    renderer: tuple[bool, str],
) -> dict[str, object]:
    """Superuser diagnostics: mode plus whether images can actually render."""

    available, detail = renderer
    return {
        "mode": config.agent_chat_image_reply_mode.value,
        "platform_modes": {
            name: mode.value
            for name, mode in config.agent_chat_image_reply_mode_by_platform.items()
        },
        "show_sources_text": config.agent_chat_show_sources_text,
        "show_sources_text_by_platform": dict(
            config.agent_chat_show_sources_text_by_platform
        ),
        "show_sources_image": config.agent_chat_show_sources_image,
        "show_sources_image_by_platform": dict(
            config.agent_chat_show_sources_image_by_platform
        ),
        "message_chunk_chars": config.agent_chat_message_chunk_chars,
        "message_chunk_chars_by_platform": dict(
            config.agent_chat_message_chunk_chars_by_platform
        ),
        "platform_text_limits": dict(PLATFORM_TEXT_LIMITS),
        "platform_max_image_pages": dict(PLATFORM_MAX_IMAGE_PAGES),
        "renderer_available": available,
        "detail": detail,
    }
