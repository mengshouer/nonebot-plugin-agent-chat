from __future__ import annotations

import base64
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePath
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)


class Protocol(str, Enum):
    OPENAI_COMPLETIONS = "openai-completions"
    OPENAI_RESPONSES = "openai-responses"
    ANTHROPIC_MESSAGES = "anthropic-messages"


class SearchMode(str, Enum):
    OFF = "off"
    BUILTIN_WEB_SEARCH = "builtin_web_search"
    EXA = "exa"


class ImageReplyMode(str, Enum):
    """Whether answers are rendered to images before delivery."""

    OFF = "off"
    AUTO = "auto"
    ALWAYS = "always"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    def __str__(self) -> str:
        # Keep f-strings and SQL bindings on the wire value in every version.
        return self.value


class ReasoningEffort(str, Enum):
    PROVIDER_DEFAULT = "provider_default"
    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ProfileCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    vision: bool = False
    tools: bool = False
    reasoning: bool = False


class ProviderProfile(BaseModel):
    """A single immutable provider/model configuration loaded from JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Protocol
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key_env: str | None = None
    # An inline credential wins over the environment variable; SecretStr keeps
    # it out of every repr/model_dump by default.
    api_key: SecretStr | None = None
    system_prompt: str = ""
    system_prompt_file: str | None = None
    capabilities: ProfileCapabilities = Field(default_factory=ProfileCapabilities)
    reasoning_effort: ReasoningEffort = ReasoningEffort.PROVIDER_DEFAULT
    search_mode: SearchMode = SearchMode.OFF
    image_reply_mode: ImageReplyMode | None = None
    show_sources_text: bool | None = None
    show_sources_image: bool | None = None
    responses_store: bool = False
    fallback_profiles: list[str] = Field(default_factory=list, max_length=2)
    enabled_tools: list[str] = Field(default_factory=list)
    max_output_tokens: int = Field(default=4096, ge=1)
    max_builtin_tool_calls: int = Field(default=10, ge=0)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_query: dict[str, Any] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    exa_api_key_env: str = "EXA_API_KEY"
    exa_api_key: SecretStr | None = None
    exa_base_url: str | None = None
    exa_num_results: int = Field(default=5, ge=1, le=10)

    @field_validator("base_url", "exa_base_url")
    @classmethod
    def validate_optional_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value

    @field_validator("fallback_profiles", "enabled_tools")
    @classmethod
    def unique_names(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            item = item.strip()
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned

    @field_validator("api_key", "exa_api_key", mode="before")
    @classmethod
    def clean_inline_credential(cls, value: object) -> object:
        """Blank means "not set"; a key must not carry stray characters.

        A leading/trailing blank or any control character is rejected instead of
        trimmed: the key is sent byte for byte, so silently rewriting it would
        turn a paste accident into an unexplained 401.
        """

        if not isinstance(value, str):
            return value
        if not value.strip():
            return None
        if value != value.strip():
            raise ValueError("密钥首尾不能有空白（疑似粘贴污染）")
        # Every Unicode "other" category is rejected, not just C0/DEL: C1
        # controls and invisible formatters (BOM, zero-width space) arrive the
        # same way a stray space does, and they fail as an unexplained 401.
        if any(unicodedata.category(character).startswith("C") for character in value):
            raise ValueError("密钥不能包含控制字符（疑似粘贴污染）")
        return value

    @field_validator("system_prompt_file")
    @classmethod
    def clean_system_prompt_file(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @model_validator(mode="after")
    def validate_effective_system_prompt_file(self) -> ProviderProfile:
        """Validate the shape of the file that actually wins by precedence.

        A non-empty inline ``system_prompt`` shadows ``system_prompt_file``, so a
        shadowed name must never block loading the profile.
        """

        if self.system_prompt.strip() or self.system_prompt_file is None:
            return self
        path = PurePath(self.system_prompt_file)
        if path.is_absolute():
            raise ValueError(
                "system_prompt_file must be relative to the prompt directory"
            )
        if ".." in path.parts:
            raise ValueError("system_prompt_file must not contain '..'")
        return self

    @model_validator(mode="before")
    @classmethod
    def infer_capabilities(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        capabilities = dict(data.get("capabilities") or {})
        if "tools" not in capabilities and (
            data.get("search_mode", SearchMode.OFF) != SearchMode.OFF
            or data.get("enabled_tools")
        ):
            capabilities["tools"] = True
        effort = data.get("reasoning_effort", ReasoningEffort.PROVIDER_DEFAULT)
        if "reasoning" not in capabilities and effort not in {
            ReasoningEffort.PROVIDER_DEFAULT,
            ReasoningEffort.PROVIDER_DEFAULT.value,
            ReasoningEffort.OFF,
            ReasoningEffort.OFF.value,
        }:
            capabilities["reasoning"] = True
        if capabilities:
            data["capabilities"] = capabilities
        return data

    @model_validator(mode="after")
    def validate_capabilities(self) -> ProviderProfile:
        if self.search_mode == SearchMode.BUILTIN_WEB_SEARCH and self.protocol not in {
            Protocol.OPENAI_RESPONSES,
            Protocol.ANTHROPIC_MESSAGES,
        }:
            raise ValueError(
                "builtin_web_search is only supported by openai-responses "
                "or anthropic-messages"
            )
        if (
            self.search_mode == SearchMode.BUILTIN_WEB_SEARCH
            and self.max_builtin_tool_calls < 1
        ):
            raise ValueError("builtin_web_search requires max_builtin_tool_calls >= 1")
        if self.search_mode != SearchMode.OFF and not self.capabilities.tools:
            raise ValueError("search_mode requires tools capability")
        if self.enabled_tools and not self.capabilities.tools:
            raise ValueError("enabled_tools requires tools capability")
        if "web_search" in self.enabled_tools:
            raise ValueError(
                "web_search is selected by search_mode and must not be in enabled_tools"
            )
        if (
            self.reasoning_effort
            not in (ReasoningEffort.PROVIDER_DEFAULT, ReasoningEffort.OFF)
            and not self.capabilities.reasoning
        ):
            raise ValueError("reasoning_effort requires reasoning capability")
        reserved = {
            "model",
            "messages",
            "input",
            "stream",
            "stream_options",
            "tools",
            "tool_choice",
            "reasoning",
            "reasoning_effort",
            "thinking",
            "output_config",
            "store",
            "previous_response_id",
            "conversation",
            "instructions",
            "system",
            "include",
            "parallel_tool_calls",
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
            "max_tool_calls",
            "temperature",
        }
        conflict = reserved.intersection(self.extra_body)
        if conflict:
            names = ", ".join(sorted(conflict))
            raise ValueError(f"extra_body cannot override reserved fields: {names}")
        return self


@dataclass
class AgentImage:
    media_type: str
    data: bytes

    def data_url(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.media_type};base64,{encoded}"

    def base64_data(self) -> str:
        return base64.b64encode(self.data).decode("ascii")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class AgentMessage:
    role: MessageRole
    text: str = ""
    images: list[AgentImage] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_error: bool = False
    native_items: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            self.role = MessageRole(self.role)


@dataclass(frozen=True)
class Source:
    url: str
    title: str = ""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0

    def merge(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.reasoning_tokens += other.reasoning_tokens


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    native_items: list[dict[str, Any]] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    hosted_searches: int = 0
    finish_reason: str | None = None


@dataclass
class RunResult:
    text: str
    sources: list[Source]
    usage: Usage
    model_turns: int
    local_tool_calls: int
    searches: int
    actual_profile: str = ""
    run_id: str = ""
