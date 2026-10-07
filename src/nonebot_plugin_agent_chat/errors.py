from __future__ import annotations


class AgentChatError(Exception):
    """Base error safe for the service layer to classify."""


class ConfigurationError(AgentChatError):
    pass


class ProfileNotFoundError(ConfigurationError):
    pass


class ProfileCredentialError(ConfigurationError):
    pass


class BusyError(AgentChatError):
    pass


class InputError(AgentChatError):
    pass


class RoomError(AgentChatError):
    pass


class ProfileRuleError(AgentChatError):
    """A profile rule points at a profile that no longer exists."""


class ToolExecutionError(AgentChatError):
    pass


class RenderError(AgentChatError):
    """Rendering an answer as an image failed; callers fall back to text."""


class ProviderError(AgentChatError):
    def __init__(
        self,
        message: str,
        *,
        retriable: bool = False,
        emitted_text: bool = False,
        status_code: int | None = None,
        error_type: str = "provider_error",
    ) -> None:
        super().__init__(message)
        self.retriable = retriable
        self.emitted_text = emitted_text
        self.status_code = status_code
        self.error_type = error_type
        self.profile_name: str | None = None


class ProviderIncompleteError(ProviderError):
    def __init__(
        self,
        message: str,
        *,
        emitted_text: bool = False,
        error_type: str = "provider_incomplete",
    ) -> None:
        super().__init__(
            message,
            retriable=False,
            emitted_text=emitted_text,
            error_type=error_type,
        )
