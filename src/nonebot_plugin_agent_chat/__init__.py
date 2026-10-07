from __future__ import annotations

from nonebot import get_driver, require
from nonebot.plugin import PluginMetadata

# Config imports UniSeg scopes. Register the dependency first inside a Bot;
# standalone CLI/tests must remain importable without initializing NoneBot.
try:
    get_driver()
except ValueError:
    _driver_ready = False
else:
    _driver_ready = True
    require("nonebot_plugin_alconna")

from .config import Config
from .hooks import (
    RunContext,
    register_after_response,
    register_before_run,
    register_before_tool,
)
from .tools import ToolContext, ToolOutput, ToolRisk, ToolSpec, register_tool

__plugin_meta__ = PluginMetadata(
    name="Agent Chat",
    description="支持多模型与跨平台消息的 AI 问答插件，可选网页搜索",
    usage=(
        "使用配置的触发词（默认 /llm）发起单次提问。"
        "超级用户可用 /agentctl 管理插件，也可启用 /agent_room 持久对话。"
    ),
    type="application",
    config=Config,
    # Metadata only: NoneBot does not gate loading on this, and the plugin
    # accepts any adapter UniSeg can translate (see platforms.py).
    supported_adapters=None,
)

if _driver_ready:
    from . import matchers as matchers


__all__ = [
    "RunContext",
    "ToolContext",
    "ToolOutput",
    "ToolRisk",
    "ToolSpec",
    "register_after_response",
    "register_before_run",
    "register_before_tool",
    "register_tool",
]
