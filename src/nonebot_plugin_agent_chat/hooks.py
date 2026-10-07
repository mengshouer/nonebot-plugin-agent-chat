from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .models import AgentMessage, ProviderProfile, RunResult, ToolCall


@dataclass(frozen=True)
class RunContext:
    profile_name: str
    profile: ProviderProfile
    subject_key: str = ""
    context_key: str = ""


BeforeRunHook = Callable[[list[AgentMessage], RunContext], Awaitable[None]]
BeforeToolHook = Callable[[ToolCall, RunContext], Awaitable[None]]
AfterResponseHook = Callable[[RunResult, RunContext], Awaitable[None]]


class HookRegistry:
    def __init__(self) -> None:
        self.before_run: list[BeforeRunHook] = []
        self.before_tool: list[BeforeToolHook] = []
        self.after_response: list[AfterResponseHook] = []

    async def run_before_run(
        self,
        messages: list[AgentMessage],
        context: RunContext,
    ) -> None:
        for hook in self.before_run:
            await hook(messages, context)

    async def run_before_tool(self, call: ToolCall, context: RunContext) -> None:
        for hook in self.before_tool:
            await hook(call, context)

    async def run_after_response(
        self,
        result: RunResult,
        context: RunContext,
    ) -> None:
        for hook in self.after_response:
            await hook(result, context)


registry = HookRegistry()


def register_before_run(hook: BeforeRunHook) -> BeforeRunHook:
    registry.before_run.append(hook)
    return hook


def register_before_tool(hook: BeforeToolHook) -> BeforeToolHook:
    registry.before_tool.append(hook)
    return hook


def register_after_response(hook: AfterResponseHook) -> AfterResponseHook:
    registry.after_response.append(hook)
    return hook
