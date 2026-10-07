from __future__ import annotations

from .models import ReasoningEffort, SearchMode
from .profiles import LoadedProfile

MAX_FALLBACK_PROFILES = 3

_REGISTERED_EFFORTS = {
    ReasoningEffort.PROVIDER_DEFAULT,
    ReasoningEffort.OFF,
}


def _requests_search(loaded: LoadedProfile) -> bool:
    return loaded.config.search_mode != SearchMode.OFF


def _requests_reasoning(loaded: LoadedProfile) -> bool:
    return loaded.config.reasoning_effort not in _REGISTERED_EFFORTS


def is_compatible(
    root: LoadedProfile,
    candidate: LoadedProfile,
    *,
    has_images: bool,
) -> bool:
    """Whether a fallback profile can serve the root profile's request.

    The chain may add capabilities but must never silently drop one the root
    asked for: no images without vision, no losing provider search, no losing
    reasoning, and no losing an explicitly enabled tool.
    """

    if has_images and not candidate.config.capabilities.vision:
        return False
    if _requests_search(root) != _requests_search(candidate):
        return False
    if _requests_reasoning(root) and not (
        candidate.config.capabilities.reasoning and _requests_reasoning(candidate)
    ):
        return False
    return set(root.config.enabled_tools).issubset(candidate.config.enabled_tools)
