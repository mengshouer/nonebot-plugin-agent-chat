"""Rolling context compaction for Agent Room conversations.

Follows the Pi Agent / Claude Code / Codex CLI pattern: when a Room's context
crosses ``AGENT_CHAT_ROOM_COMPACT_THRESHOLD_PERCENT`` of its profile's
``context_window`` (or on ``/agent_room compact`` or a provider context-overflow
error), older exchanges are serialized as plain ``<conversation>`` text and
merged with any ``<previous-summary>`` into a structured summary while keeping
a recent tail of raw exchanges intact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .adapters import create_adapter
from .errors import ProviderIncompleteError
from .models import (
    AgentMessage,
    MessageRole,
    ProviderProfile,
    ReasoningEffort,
    SearchMode,
    Usage,
)
from .storage import AgentRoom, StoredMessage

CompactionMode = Literal["auto", "manual", "overflow"]

ESTIMATED_IMAGE_TOKENS = 1200

TRUNCATED_FINISH_REASONS = frozenset(
    {
        "length",
        "max_tokens",
        "max_output_tokens",
        "incomplete",
        "model_context_window_exceeded",
    }
)

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the "
    "following summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"
COMPACTION_SUMMARY_ACK = (
    "Understood. I have the compacted conversation context and will continue from here."
)

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI assistant, then produce a concise, "
    "structured summary in the primary language of the conversation.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in "
    "the conversation. Do NOT call any tools. ONLY output the structured summary."
)

SUMMARIZATION_FORMAT = """Use this EXACT structure:

## Goal & Topic
[What the user is working on, asking about, or trying to accomplish]

## Key Facts & Decisions
- [Important facts, conclusions, constraints, preferences, or decisions established]

## Progress & Open Items
- [What has been answered or completed, and what remains open or in progress]

## Critical Context
- [Specific names, identifiers, code snippets, parameters, or data needed to continue]

Keep each section concise and factual. Preserve exact identifiers and paths."""

SUMMARIZATION_PROMPT = (
    "Summarize the conversation above so a future turn can continue seamlessly.\n\n"
    f"{SUMMARIZATION_FORMAT}"
)

UPDATE_SUMMARIZATION_PROMPT = (
    "The messages in <conversation> are NEW turns to incorporate into the "
    "existing summary in <previous-summary>.\n"
    "- PRESERVE still-relevant goals, facts, decisions, and critical context "
    "from <previous-summary>\n"
    "- ADD new progress, decisions, and context from <conversation>\n"
    "- REMOVE obsolete details that were superseded\n\n"
    f"{SUMMARIZATION_FORMAT}"
)


@dataclass(frozen=True)
class CompactionPlan:
    """Which stored messages to summarize and which recent ones to keep."""

    previous_summary: str
    messages_to_summarize: list[StoredMessage]
    kept_messages: list[StoredMessage]
    remove_ids: list[int]
    transcript: str
    tokens_before: int


@dataclass(frozen=True)
class CompactionOutcome:
    """Result of one Room compaction pass."""

    summary: str
    summarized_messages: int
    kept_messages: int
    tokens_before: int
    tokens_after: int
    usage: Usage | None = None


def estimate_text_tokens(text: str) -> int:
    """Fast, tokenizer-free estimate suitable for CJK and Latin text alike."""

    if not text:
        return 0
    return max(1, (len(text.encode("utf-8")) + 2) // 3)


FIXED_PROMPT_OVERHEAD_TOKENS = (
    estimate_text_tokens(SUMMARIZATION_SYSTEM_PROMPT)
    + estimate_text_tokens(UPDATE_SUMMARIZATION_PROMPT)
    + 128
)


def estimate_message_tokens(message: StoredMessage) -> int:
    return (
        estimate_text_tokens(message.text)
        + len(message.images) * ESTIMATED_IMAGE_TOKENS
    )


def estimate_room_tokens(summary: str, messages: Sequence[StoredMessage]) -> int:
    total = sum(estimate_message_tokens(item) for item in messages)
    if summary.strip():
        total += estimate_text_tokens(summary) + 32
    return total


def effective_context_tokens(
    room: AgentRoom,
    messages: Sequence[StoredMessage],
) -> int:
    """Reported provider tokens when valid, else the heuristic estimate."""

    if room.context_tokens > 0:
        return room.context_tokens
    return estimate_room_tokens(room.summary, messages)


def summary_max_output_tokens(
    context_window: int,
    profile_max_output_tokens: int = 4096,
) -> int:
    """Cap summary generation output so input + output fits ``context_window``."""

    return min(profile_max_output_tokens, max(256, context_window // 5))


def transcript_token_budget(
    context_window: int,
    *,
    max_output_tokens: int = 4096,
    previous_summary: str = "",
    custom_instructions: str = "",
) -> int:
    """Token budget available for the ``<conversation>`` transcript body."""

    reserved = (
        summary_max_output_tokens(context_window, max_output_tokens)
        + FIXED_PROMPT_OVERHEAD_TOKENS
        + estimate_text_tokens(previous_summary)
        + estimate_text_tokens(custom_instructions)
    )
    return max(256, context_window - reserved)


def summary_messages(summary: str) -> list[AgentMessage]:
    """Synthetic user/assistant pair injecting the summary before kept turns."""

    cleaned = summary.strip()
    if not cleaned:
        return []
    return [
        AgentMessage(
            role=MessageRole.USER,
            text=f"{COMPACTION_SUMMARY_PREFIX}{cleaned}{COMPACTION_SUMMARY_SUFFIX}",
        ),
        AgentMessage(
            role=MessageRole.ASSISTANT,
            text=COMPACTION_SUMMARY_ACK,
        ),
    ]


def _group_exchanges(messages: Sequence[StoredMessage]) -> list[list[StoredMessage]]:
    """Group stored messages into user/assistant exchanges from oldest to newest."""

    exchanges: list[list[StoredMessage]] = []
    index = 0
    total = len(messages)
    while index < total:
        first = messages[index]
        if (
            index + 1 < total
            and first.role == MessageRole.USER.value
            and messages[index + 1].role == MessageRole.ASSISTANT.value
        ):
            exchanges.append([first, messages[index + 1]])
            index += 2
        else:
            exchanges.append([first])
            index += 1
    return exchanges


def _render_message_line(message: StoredMessage, *, max_chars: int) -> str:
    role_label = "User" if message.role == MessageRole.USER.value else "Assistant"
    if message.images:
        role_label = f"{role_label} (with {len(message.images)} image(s))"
    text = message.text.strip()
    if len(text) > max_chars:
        omitted = len(text) - max_chars
        text = f"{text[:max_chars]}\n[... {omitted} more characters truncated]"
    return f"[{role_label}]: {text}"


def serialize_conversation(
    messages: Sequence[StoredMessage],
    *,
    max_tokens: int,
) -> str:
    """Serialize a contiguous oldest prefix of messages within ``max_tokens``."""

    selected, _, text = _select_summarizable_prefix(
        _group_exchanges(messages),
        max_tokens=max_tokens,
    )
    if not selected:
        return ""
    return text


def _select_summarizable_prefix(
    candidate_exchanges: Sequence[list[StoredMessage]],
    *,
    max_tokens: int,
) -> tuple[list[list[StoredMessage]], list[list[StoredMessage]], str]:
    """Select the contiguous oldest exchanges that fit in ``max_tokens``.

    Any candidate exchanges that do not fit into this summarization pass are
    returned in ``unselected`` so the caller keeps them in the Room rather than
    deleting messages the summary model never saw.
    """

    if not candidate_exchanges:
        return [], [], ""
    per_message_char_cap = max(600, max_tokens)
    selected: list[list[StoredMessage]] = []
    blocks: list[str] = []
    used_tokens = 0

    for index, exchange in enumerate(candidate_exchanges):
        lines = [
            _render_message_line(msg, max_chars=per_message_char_cap)
            for msg in exchange
        ]
        block = "\n\n".join(lines)
        block_tokens = estimate_text_tokens(block)
        if selected and used_tokens + block_tokens > max_tokens:
            return selected, list(candidate_exchanges[index:]), "\n\n".join(blocks)
        selected.append(exchange)
        blocks.append(block)
        used_tokens += block_tokens

    return selected, [], "\n\n".join(blocks)


def plan_compaction(
    room: AgentRoom,
    messages: Sequence[StoredMessage],
    *,
    context_window: int,
    threshold_percent: int,
    max_output_tokens: int = 4096,
    custom_instructions: str = "",
    mode: CompactionMode = "auto",
) -> CompactionPlan | None:
    """Plan a compaction cut point, or return ``None`` when none is needed."""

    if not messages:
        return None
    tokens_before = effective_context_tokens(room, messages)
    threshold_tokens = max(1, context_window * threshold_percent // 100)
    if mode == "auto" and tokens_before < threshold_tokens:
        return None

    exchanges = _group_exchanges(messages)
    if mode == "auto" and len(exchanges) < 2:
        return None
    if mode in ("manual", "overflow"):
        # Manual `/agent_room compact` and pre-turn overflow recovery prioritize
        # freeing context immediately, even when only 1-2 stored turns exist.
        keep_recent_tokens = max(1, threshold_tokens // 8)
        max_keep_exchanges = 0 if len(exchanges) <= 2 else 1
    else:
        # Post-turn auto-compaction keeps ~1/4 of the trigger budget (e.g. 20%
        # of window when the trigger is 80%) and always keeps the turn just
        # completed.
        keep_recent_tokens = max(1, threshold_tokens // 4)
        max_keep_exchanges = len(exchanges) - 1

    kept_exchanges: list[list[StoredMessage]] = []
    kept_tokens = 0
    for exchange in reversed(exchanges):
        if len(kept_exchanges) >= max_keep_exchanges:
            break
        exchange_tokens = sum(estimate_message_tokens(msg) for msg in exchange)
        if kept_exchanges and kept_tokens + exchange_tokens > keep_recent_tokens:
            break
        if (
            mode in ("manual", "overflow")
            and not kept_exchanges
            and exchange_tokens > keep_recent_tokens
        ):
            break
        kept_exchanges.append(exchange)
        kept_tokens += exchange_tokens

    kept_exchanges.reverse()
    cut_exchange_index = len(exchanges) - len(kept_exchanges)
    if cut_exchange_index <= 0:
        return None

    previous_summary = room.summary.strip()
    budget = transcript_token_budget(
        context_window,
        max_output_tokens=max_output_tokens,
        previous_summary=previous_summary,
        custom_instructions=custom_instructions,
    )
    selected_exchanges, deferred_exchanges, transcript = _select_summarizable_prefix(
        exchanges[:cut_exchange_index],
        max_tokens=budget,
    )
    if not selected_exchanges:
        return None

    to_summarize = [msg for exchange in selected_exchanges for msg in exchange]
    kept = [
        msg for exchange in [*deferred_exchanges, *kept_exchanges] for msg in exchange
    ]
    remove_ids = [msg.id for msg in to_summarize if msg.id > 0]
    return CompactionPlan(
        previous_summary=previous_summary,
        messages_to_summarize=to_summarize,
        kept_messages=kept,
        remove_ids=remove_ids,
        transcript=transcript,
        tokens_before=tokens_before,
    )


def build_summarization_prompt(
    plan: CompactionPlan,
    *,
    custom_instructions: str = "",
) -> str:
    instructions = (
        UPDATE_SUMMARIZATION_PROMPT if plan.previous_summary else SUMMARIZATION_PROMPT
    )
    focus = custom_instructions.strip()
    if focus:
        instructions = f"{instructions}\n\nAdditional focus: {focus}"

    parts = [f"<conversation>\n{plan.transcript}\n</conversation>"]
    if plan.previous_summary:
        parts.append(
            f"<previous-summary>\n{plan.previous_summary}\n</previous-summary>"
        )
    parts.append(instructions)
    return "\n\n".join(parts)


def profile_for_summarization(profile: ProviderProfile) -> ProviderProfile:
    """A stripped copy of ``profile`` with tools, search, and reasoning off."""

    return profile.model_copy(
        update={
            "capabilities": profile.capabilities.model_copy(
                update={"tools": False, "vision": False}
            ),
            "search_mode": SearchMode.OFF,
            "enabled_tools": [],
            "max_output_tokens": summary_max_output_tokens(
                profile.context_window,
                profile.max_output_tokens,
            ),
            "reasoning_effort": (
                ReasoningEffort.OFF
                if profile.capabilities.reasoning
                else ReasoningEffort.PROVIDER_DEFAULT
            ),
        }
    )


async def generate_compaction_summary(
    plan: CompactionPlan,
    profile: ProviderProfile,
    api_key: str,
    *,
    custom_instructions: str = "",
) -> tuple[str, Usage]:
    """Run one tool-free summarization call against the Room's profile."""

    summary_profile = profile_for_summarization(profile)
    prompt_text = build_summarization_prompt(
        plan,
        custom_instructions=custom_instructions,
    )
    adapter = create_adapter(summary_profile, api_key)
    try:
        turn = await adapter.stream_turn(
            [AgentMessage(role=MessageRole.USER, text=prompt_text)],
            [],
            SUMMARIZATION_SYSTEM_PROMPT,
        )
    finally:
        await adapter.close()

    if turn.tool_calls:
        raise ProviderIncompleteError(
            "Summarization attempted to call a tool",
            error_type="compaction_tool_call",
        )
    if turn.finish_reason in TRUNCATED_FINISH_REASONS:
        raise ProviderIncompleteError(
            f"Summarization hit the output token cap ({turn.finish_reason})",
            error_type="compaction_length",
        )
    text = turn.text.strip()
    if not text:
        raise ProviderIncompleteError(
            "Summarization returned empty text",
            error_type="compaction_empty",
        )
    return text, turn.usage
