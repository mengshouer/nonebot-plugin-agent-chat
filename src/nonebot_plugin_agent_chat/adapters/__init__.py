from __future__ import annotations

from ..errors import ConfigurationError
from ..models import Protocol, ProviderProfile
from .anthropic import AnthropicMessagesAdapter
from .base import ProtocolAdapter
from .openai_chat import OpenAIChatAdapter
from .openai_responses import OpenAIResponsesAdapter


def create_adapter(profile: ProviderProfile, api_key: str) -> ProtocolAdapter:
    if profile.protocol == Protocol.OPENAI_COMPLETIONS:
        return OpenAIChatAdapter(profile, api_key)
    if profile.protocol == Protocol.OPENAI_RESPONSES:
        return OpenAIResponsesAdapter(profile, api_key)
    if profile.protocol == Protocol.ANTHROPIC_MESSAGES:
        return AnthropicMessagesAdapter(profile, api_key)
    raise ConfigurationError(f"Unsupported protocol: {profile.protocol}")


__all__ = ["ProtocolAdapter", "create_adapter"]
