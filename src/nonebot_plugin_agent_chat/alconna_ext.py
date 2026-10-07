from __future__ import annotations

from arclet.alconna import Alconna
from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna import Extension, Text, UniMessage

from .input import strip_command_mention
from .platforms import resolve_chat_identity, uses_command_mentions


class CommandMentionExtension(Extension):
    """Accept ``/command@botname`` on platforms that declare that suffix form.

    Telegram clients append the bot username to slash commands in groups; the
    platform registry records which scopes do that. This wrapper normalizes the
    first text segment before Alconna parses the message, while leaving commands
    addressed to other bots untouched.
    """

    @property
    def priority(self) -> int:
        return 15

    @property
    def id(self) -> str:
        return "nonebot_plugin_agent_chat:telegram_command_mention"

    def validate(self, bot: Bot, event: Event) -> bool:
        identity = resolve_chat_identity(bot, event)
        return identity is not None and uses_command_mentions(identity.scope)

    async def receive_wrapper(
        self,
        bot: Bot,
        event: Event,
        command: Alconna,
        receive: UniMessage,
    ) -> UniMessage:
        username = getattr(bot, "username", None)
        if not username:
            return receive
        for index, segment in enumerate(receive):
            if not isinstance(segment, Text):
                continue
            normalized = strip_command_mention(segment.text, str(username))
            if normalized != segment.text:
                # Copy before mutating: the message may be shared by other
                # matchers through the UniSeg cache.
                receive = receive.copy()
                text_segment = receive[index]
                assert isinstance(text_segment, Text)
                text_segment.text = normalized
            break
        return receive
