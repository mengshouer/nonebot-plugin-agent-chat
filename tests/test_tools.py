import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from nonebot_plugin_agent_chat.errors import ConfigurationError, ToolExecutionError
from nonebot_plugin_agent_chat.models import ProviderProfile
from nonebot_plugin_agent_chat.tools import (
    ToolContext,
    ToolOutput,
    ToolRegistry,
    ToolRisk,
    ToolSpec,
    WebSearchInput,
    exa_web_search,
)


class Input(BaseModel):
    value: int


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_arguments_are_validated(self) -> None:
        seen = []

        async def handler(arguments: BaseModel, context: ToolContext) -> ToolOutput:
            parsed = Input.model_validate(arguments)
            seen.append(parsed.value)
            return ToolOutput(content="ok")

        registry = ToolRegistry()
        registry.register(ToolSpec("sample", "sample", Input, handler))
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-completions",
                "model": "test",
                "capabilities": {"tools": True},
                "enabled_tools": ["sample"],
            }
        )
        result = await registry.execute(
            "sample", '{"value":2}', ToolContext("test", profile)
        )
        self.assertEqual(result.content, "ok")
        self.assertEqual(seen, [2])

        with self.assertRaises(ToolExecutionError):
            await registry.execute(
                "sample", '{"value":"bad"}', ToolContext("test", profile)
            )

    async def test_exa_search_uses_async_sdk_and_normalizes_sources(self) -> None:
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-completions",
                "model": "test",
                "capabilities": {"tools": True},
                "search_mode": "exa",
                "exa_base_url": "https://exa.example",
            }
        )
        client = SimpleNamespace(
            search=AsyncMock(
                return_value=SimpleNamespace(
                    results=[
                        SimpleNamespace(
                            title="Result",
                            url="https://example.com/page",
                            highlights=["fresh content"],
                            text="",
                        )
                    ]
                )
            )
        )
        with (
            patch.dict(os.environ, {"EXA_API_KEY": "secret"}),
            patch("exa_py.AsyncExa", return_value=client) as constructor,
        ):
            result = await exa_web_search(
                WebSearchInput(query="current info"),
                ToolContext("test", profile),
            )

        constructor.assert_called_once_with(
            api_key="secret", api_base="https://exa.example"
        )
        client.search.assert_awaited_once()
        self.assertIn("fresh content", result.content)
        self.assertEqual(result.sources[0].url, "https://example.com/page")

    async def test_mutating_tools_are_not_exposed(self) -> None:
        async def handler(arguments: BaseModel, context: ToolContext) -> ToolOutput:
            return ToolOutput(content="ok")

        registry = ToolRegistry()
        registry.register(
            ToolSpec("danger", "danger", Input, handler, risk=ToolRisk.CONFIRMATION)
        )
        profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-completions",
                "model": "test",
                "capabilities": {"tools": True},
                "enabled_tools": ["danger"],
            }
        )
        with self.assertRaises(ConfigurationError):
            registry.for_profile(profile)
        with self.assertRaises(ToolExecutionError):
            await registry.execute(
                "danger",
                '{"value":1}',
                ToolContext("test", profile),
            )


if __name__ == "__main__":
    unittest.main()
