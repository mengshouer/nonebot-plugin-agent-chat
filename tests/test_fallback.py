from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nonebot_plugin_agent_chat.config import Config
from nonebot_plugin_agent_chat.errors import ProviderError, ProviderIncompleteError
from nonebot_plugin_agent_chat.models import AgentImage, AgentMessage, RunResult, Usage
from nonebot_plugin_agent_chat.profiles import LoadedProfile
from nonebot_plugin_agent_chat.runner import RunnerBudget
from nonebot_plugin_agent_chat.service import AgentChatService

Outcome = RunResult | BaseException


class FakeService(AgentChatService):
    def __init__(self, config: Config, outcomes: dict[str, list[Outcome]]) -> None:
        super().__init__(config)
        self.outcomes = outcomes
        self.attempts: list[str] = []
        self.budgets: list[RunnerBudget | None] = []

    async def _run_profile(
        self,
        loaded: LoadedProfile,
        messages: list[AgentMessage],
        *,
        subject_key: str = "",
        context_key: str = "",
        budget: RunnerBudget | None = None,
    ) -> RunResult:
        del messages, subject_key, context_key
        self.attempts.append(loaded.name)
        self.budgets.append(budget)
        outcome = self.outcomes[loaded.name].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        outcome.actual_profile = loaded.name
        return outcome


class FallbackTests(unittest.IsolatedAsyncioTestCase):
    def _service(
        self, directory: Path, outcomes: dict[str, list[Outcome]]
    ) -> FakeService:
        values = {
            "primary": {
                "protocol": "openai-responses",
                "model": "one",
                "capabilities": {"vision": True},
                "fallback_profiles": ["backup"],
            },
            "backup": {
                "protocol": "anthropic-messages",
                "model": "two",
            },
        }
        for name, value in values.items():
            (directory / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
        config = Config(
            agent_chat_data_dir=directory / "data",
            agent_chat_profile_dir=directory,
            agent_chat_default_profile="primary",
        )
        service = FakeService(config, outcomes)
        service.profiles.load()
        service._active_profile = "primary"
        return service

    async def test_retries_then_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def transient() -> ProviderError:
                return ProviderError("temporary", retriable=True)

            service = self._service(
                directory,
                {
                    "primary": [transient(), transient()],
                    "backup": [RunResult("ok", [], Usage(), 1, 0, 0)],
                },
            )
            result = await service._run_with_fallback(
                service.profiles.get("primary"),
                [AgentMessage(role="user", text="question")],
            )
            self.assertEqual(result.text, "ok")
            self.assertEqual(service.attempts, ["primary", "primary", "backup"])
            self.assertEqual(len({id(value) for value in service.budgets}), 1)

    async def test_non_retriable_provider_error_does_not_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self._service(
                directory,
                {
                    "primary": [
                        ProviderIncompleteError(
                            "Search-call limit reached",
                            error_type="search_limit",
                        )
                    ],
                    "backup": [RunResult("should not run", [], Usage(), 1, 0, 0)],
                },
            )
            with self.assertRaises(ProviderError):
                await service._run_with_fallback(
                    service.profiles.get("primary"),
                    [AgentMessage(role="user", text="question")],
                )
            self.assertEqual(service.attempts, ["primary"])

    async def test_incompatible_vision_fallback_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self._service(
                directory,
                {
                    "primary": [
                        ProviderError("temporary", retriable=True),
                        ProviderError("temporary", retriable=True),
                    ],
                    "backup": [RunResult("should not run", [], Usage(), 1, 0, 0)],
                },
            )
            with self.assertRaises(ProviderError):
                await service._run_with_fallback(
                    service.profiles.get("primary"),
                    [
                        AgentMessage(
                            role="user",
                            images=[AgentImage(media_type="image/png", data=b"png")],
                        )
                    ],
                )
            self.assertEqual(service.attempts, ["primary", "primary"])

    async def test_partial_stream_does_not_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            service = self._service(
                directory,
                {
                    "primary": [
                        ProviderError("partial", retriable=True, emitted_text=True)
                    ],
                    "backup": [RunResult("should not run", [], Usage(), 1, 0, 0)],
                },
            )
            with self.assertRaises(ProviderError):
                await service._run_with_fallback(
                    service.profiles.get("primary"),
                    [AgentMessage(role="user", text="question")],
                )
            self.assertEqual(service.attempts, ["primary"])


if __name__ == "__main__":
    unittest.main()
