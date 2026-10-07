from __future__ import annotations

from pathlib import Path

from nonebot.config import BaseSettings
from pydantic import ConfigDict, Field, ValidationInfo, field_validator

from .models import ImageReplyMode
from .platforms import SupportScope, scope_value

# A chunk below this length would only add noise; 0 means "no plugin limit".
MIN_CHUNK_CHARS = 200


def _check_chunk_size(value: int, label: str) -> int:
    if value != 0 and value < MIN_CHUNK_CHARS:
        raise ValueError(f"{label} 必须为 0（不主动分段）或 >= {MIN_CHUNK_CHARS}")
    return value


class Config(BaseSettings):
    model_config = ConfigDict(extra="ignore")

    agent_chat_data_dir: Path = Path("data/agent_chat")
    agent_chat_profile_dir: Path = Path("data/agent_chat/profiles")
    agent_chat_default_profile: str | None = None
    agent_chat_default_system_prompt_file: str = "default.md"
    agent_chat_cleanup_interval_seconds: float = Field(default=3600.0, ge=0)

    agent_chat_triggers: list[str] = Field(default_factory=lambda: ["/llm"])
    agent_chat_enable_at: bool = False
    agent_chat_enable_private_auto_reply: bool = False
    agent_chat_allowed_groups: set[str] = Field(default_factory=set)
    agent_chat_allowed_users: set[str] = Field(default_factory=set)
    agent_chat_priority: int = Field(default=20, ge=1)

    agent_chat_room_enabled: bool = False
    agent_chat_room_max_turns: int = Field(default=20, ge=1)
    agent_chat_room_max_chars: int = Field(default=60000, ge=1000)
    agent_chat_room_retention_days: int = Field(default=30, ge=1)
    agent_chat_room_image_retention_days: int = Field(default=7, ge=1)
    agent_chat_run_metadata_retention_days: int = Field(default=30, ge=1)

    agent_chat_global_concurrency: int = Field(default=4, ge=1)
    agent_chat_user_cooldown_seconds: float = Field(default=5.0, ge=0)
    agent_chat_daily_request_limit: int = Field(default=0, ge=0)
    agent_chat_max_model_turns: int = Field(default=6, ge=1)
    agent_chat_max_local_tool_calls: int = Field(default=8, ge=0)
    agent_chat_max_searches: int = Field(default=10, ge=0)
    agent_chat_tool_timeout_seconds: float = Field(default=30.0, gt=0)
    agent_chat_run_timeout_seconds: float = Field(default=180.0, gt=0)

    agent_chat_message_chunk_chars: int = Field(default=1000, ge=0)
    # 0 = no plugin-imposed limit; Telegram defaults to that so it uses its own
    # 4096-unit cap instead of the global 1000-character chunk size.
    agent_chat_message_chunk_chars_by_platform: dict[str, int] = Field(
        default_factory=lambda: {SupportScope.telegram.value: 0}
    )
    agent_chat_message_send_delay_seconds: float = Field(default=1.0, ge=0)

    agent_chat_image_reply_mode: ImageReplyMode = ImageReplyMode.OFF
    # Telegram renders Markdown natively, so it defaults to text while other
    # adapters keep the global mode; override with {"Telegram": "auto"} to opt in.
    agent_chat_image_reply_mode_by_platform: dict[str, ImageReplyMode] = Field(
        default_factory=lambda: {SupportScope.telegram.value: ImageReplyMode.OFF}
    )
    agent_chat_show_sources_text: bool = True
    agent_chat_show_sources_text_by_platform: dict[str, bool] = Field(
        default_factory=dict
    )
    agent_chat_show_sources_image: bool = True
    agent_chat_show_sources_image_by_platform: dict[str, bool] = Field(
        default_factory=dict
    )
    agent_chat_image_reply_min_chars: int = Field(default=1000, ge=100)
    agent_chat_image_reply_max_height: int = Field(default=8000, ge=400)
    agent_chat_image_reply_timeout_seconds: float = Field(default=20.0, gt=0)
    agent_chat_max_reply_chars: int = Field(default=8000, ge=100)
    agent_chat_max_images: int = Field(default=4, ge=0)
    agent_chat_max_image_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)

    @field_validator("agent_chat_allowed_groups", "agent_chat_allowed_users")
    @classmethod
    def clean_acl_entries(cls, value: set[str]) -> set[str]:
        """Require scoped ``<scope>:<id>`` entries; bare IDs are ambiguous."""

        cleaned: set[str] = set()
        for entry in value:
            text = entry.strip()
            scope, separator, target = text.partition(":")
            if not separator or not scope or not target:
                raise ValueError(
                    "白名单条目必须是 <平台标识>:<群号或用户ID> 形式"
                    f"（不支持裸 ID）：{text!r}"
                )
            cleaned.add(f"{scope_value(scope)}:{target}")
        return cleaned

    @field_validator("agent_chat_triggers")
    @classmethod
    def clean_triggers(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for trigger in value:
            trigger = trigger.strip()
            if trigger and trigger not in cleaned:
                cleaned.append(trigger)
        return cleaned

    @field_validator("agent_chat_image_reply_mode_by_platform", mode="before")
    @classmethod
    def clean_platform_modes(cls, value: object) -> dict[str, object]:
        """Validate exact upstream scope values without aliases."""

        return _clean_platform_keys(value, "AGENT_CHAT_IMAGE_REPLY_MODE_BY_PLATFORM")

    @field_validator(
        "agent_chat_show_sources_text_by_platform",
        "agent_chat_show_sources_image_by_platform",
        mode="before",
    )
    @classmethod
    def clean_show_sources_platforms(
        cls, value: object, info: ValidationInfo
    ) -> dict[str, object]:
        """Validate exact upstream scope values without aliases."""

        label = f"AGENT_CHAT_{(info.field_name or '').upper()}"
        return _clean_platform_keys(value, label, example='{"Telegram": false}')

    @field_validator("agent_chat_message_chunk_chars_by_platform", mode="before")
    @classmethod
    def clean_platform_chunk_chars(cls, value: object) -> dict[str, object]:
        return _clean_platform_keys(value, "AGENT_CHAT_MESSAGE_CHUNK_CHARS_BY_PLATFORM")

    @field_validator("agent_chat_message_chunk_chars")
    @classmethod
    def check_message_chunk_chars(cls, value: int) -> int:
        return _check_chunk_size(value, "AGENT_CHAT_MESSAGE_CHUNK_CHARS")

    @field_validator("agent_chat_message_chunk_chars_by_platform")
    @classmethod
    def check_platform_chunk_chars(cls, value: dict[str, int]) -> dict[str, int]:
        for name, size in value.items():
            _check_chunk_size(
                size, f"AGENT_CHAT_MESSAGE_CHUNK_CHARS_BY_PLATFORM[{name!r}]"
            )
        return value


def _clean_platform_keys(
    value: object, label: str, example: str = '{"Telegram": "off"}'
) -> dict[str, object]:
    """Validate upstream scope keys while preserving their original values."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        # pydantic wraps ValueError into ValidationError; TypeError would
        # escape as a raw exception, so keep ValueError here.
        raise ValueError(  # noqa: TRY004
            f"{label} 必须是 JSON 对象，例如 {example}"
        )
    return {scope_value(key): item for key, item in value.items()}
