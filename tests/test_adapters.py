import unittest

from nonebot_plugin_agent_chat.adapters.anthropic import AnthropicMessagesAdapter
from nonebot_plugin_agent_chat.adapters.openai_chat import OpenAIChatAdapter
from nonebot_plugin_agent_chat.adapters.openai_responses import OpenAIResponsesAdapter
from nonebot_plugin_agent_chat.models import (
    AgentImage,
    AgentMessage,
    ProviderProfile,
    ToolCall,
)


class AdapterEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.messages = [
            AgentMessage(
                role="user",
                text="question",
                images=[AgentImage(media_type="image/png", data=b"png")],
            ),
            AgentMessage(
                role="assistant",
                tool_calls=[ToolCall("call-1", "lookup", '{"q":"x"}')],
            ),
            AgentMessage(role="tool", text='{"ok":true}', tool_call_id="call-1"),
        ]

    def test_chat_completions_encoding(self) -> None:
        encoded = OpenAIChatAdapter._messages(self.messages, "system")
        self.assertEqual(encoded[0], {"role": "system", "content": "system"})
        self.assertEqual(encoded[1]["content"][1]["type"], "image_url")
        self.assertEqual(encoded[2]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(encoded[3]["tool_call_id"], "call-1")

    def test_responses_encoding(self) -> None:
        encoded = OpenAIResponsesAdapter._input(self.messages)
        self.assertEqual(encoded[0]["content"][1]["type"], "input_image")
        self.assertEqual(encoded[1]["type"], "function_call")
        self.assertEqual(encoded[2]["type"], "function_call_output")

    def test_anthropic_encoding(self) -> None:
        encoded = AnthropicMessagesAdapter._messages(self.messages)
        self.assertEqual(encoded[0]["content"][1]["type"], "image")
        self.assertEqual(encoded[1]["content"][0]["type"], "tool_use")
        self.assertEqual(encoded[2]["content"][0]["type"], "tool_result")

    def test_gateway_reasoning_efforts_are_passed_through(self) -> None:
        chat_profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-completions",
                "model": "test",
                "reasoning_effort": "max",
            }
        )
        responses_profile = ProviderProfile.model_validate(
            {
                "protocol": "openai-responses",
                "model": "test",
                "reasoning_effort": "max",
            }
        )
        anthropic_profile = ProviderProfile.model_validate(
            {
                "protocol": "anthropic-messages",
                "model": "test",
                "reasoning_effort": "xhigh",
            }
        )
        chat = object.__new__(OpenAIChatAdapter)
        chat.profile = chat_profile
        responses = object.__new__(OpenAIResponsesAdapter)
        responses.profile = responses_profile
        anthropic = object.__new__(AnthropicMessagesAdapter)
        anthropic.profile = anthropic_profile

        self.assertEqual(chat._reasoning_effort(), "max")
        self.assertEqual(responses._reasoning(), {"effort": "max", "summary": "auto"})
        self.assertEqual(
            anthropic._thinking()["output_config"],
            {"effort": "xhigh"},
        )

    def test_responses_source_extraction(self) -> None:
        sources = OpenAIResponsesAdapter._sources_from_value(
            {
                "annotations": [
                    {
                        "type": "url_citation",
                        "url": "https://example.com",
                        "title": "Example",
                    }
                ]
            }
        )
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].url, "https://example.com")


if __name__ == "__main__":
    unittest.main()
