"""One-line documentation for every ``AGENT_CHAT_*`` setting.

This catalogue is the single source of truth for what a setting does; the CLI
listing legend, `--config-explain` / the `:config <KEY>` session command, and the
full-screen editor all read it, so the surfaces cannot drift. The
README deliberately does not repeat the table (it points here instead).

A contract test asserts that every ``Config`` field has an entry and that no
entry outlives its field.
"""

from __future__ import annotations

import typing
from enum import Enum
from pathlib import Path

from .config import Config
from .config_editor import (
    RESTART_ONLY_KEYS,
    display_value,
    unwrap_optional,
)

# Default length of a summary line in the editor's lists; the full sentence is
# always available in the detail pane.
SUMMARY_CHARS_DEFAULT = 17

# `--config-explain` reads as a sentence, so it wraps the shared labels.
SOURCE_WHERE: dict[str, str] = {
    "env": "来自环境变量",
    "file": "来自 dotenv 文件",
    "default": "内置默认",
}

# keyed by the environment-variable name, i.e. the field name upper-cased.
SETTING_HELP: dict[str, str] = {
    "AGENT_CHAT_DATA_DIR": (
        "运行时数据根目录：SQLite、图片缓存、prompts；相对路径按 bot 工作目录解析"
    ),
    "AGENT_CHAT_PROFILE_DIR": "profile JSON 目录；文件名（去掉 .json）就是 profile ID",
    "AGENT_CHAT_DEFAULT_PROFILE": (
        "启动时使用的默认 profile；留空时要求目录里只有一个 profile"
    ),
    "AGENT_CHAT_DEFAULT_SYSTEM_PROMPT_FILE": (
        "全局默认 prompt 文件（相对 <数据目录>/prompts）；"
        "profile 自己没写 prompt 时使用"
    ),
    "AGENT_CHAT_CLEANUP_INTERVAL_SECONDS": (
        "后台清理周期（秒）：过期运行记录、已关闭 Room、图片缓存；0 = 不启动后台清理"
    ),
    "AGENT_CHAT_TRIGGERS": ('触发自由提问的文本（JSON 数组，子串匹配），例如 ["/llm"]'),
    "AGENT_CHAT_ENABLE_AT": ("没写触发词时，被 @ 或明确点名也让 bot 回答"),
    "AGENT_CHAT_ENABLE_PRIVATE_AUTO_REPLY": "私聊里不写触发词也自动回答",
    "AGENT_CHAT_ALLOWED_GROUPS": (
        '群白名单（JSON 数组，如 "QQClient:123"）；平台标识区分大小写，'
        '--platform-list 可查询；"<平台标识>:*" 放行该平台全部群'
    ),
    "AGENT_CHAT_ALLOWED_USERS": (
        "私聊用户白名单（JSON 数组，同上）；NoneBot superuser 始终允许"
    ),
    "AGENT_CHAT_PRIORITY": (
        "自由提问 matcher 的优先级（数字越小越先处理）；注册期读取，改动需重启"
    ),
    "AGENT_CHAT_ROOM_ENABLED": "是否启用 Agent Room 多轮持久会话（仅 superuser 可用）",
    "AGENT_CHAT_ROOM_MAX_TURNS": "Room 每次提问最多带上的历史条数",
    "AGENT_CHAT_ROOM_MAX_CHARS": "Room 历史注入的字符上限（与上一条共同截断窗口）",
    "AGENT_CHAT_ROOM_RETENTION_DAYS": "已关闭 Room 的保留天数，超期由清理任务删除",
    "AGENT_CHAT_ROOM_IMAGE_RETENTION_DAYS": "Room 图片缓存的保留天数",
    "AGENT_CHAT_RUN_METADATA_RETENTION_DAYS": (
        "运行元数据的保留天数（/agentctl status 显示的近期运行）"
    ),
    "AGENT_CHAT_GLOBAL_CONCURRENCY": "全局同时进行的回答数上限",
    "AGENT_CHAT_USER_COOLDOWN_SECONDS": "同一用户两次提问之间的最小间隔（秒）",
    "AGENT_CHAT_DAILY_REQUEST_LIMIT": "每个会话每天可发起的请求数上限；0 = 不限",
    "AGENT_CHAT_MAX_MODEL_TURNS": "一次回答最多往返模型几次（含工具续跑）",
    "AGENT_CHAT_MAX_LOCAL_TOOL_CALLS": "一次回答最多执行多少次本地只读工具",
    "AGENT_CHAT_MAX_SEARCHES": "一次回答最多搜索几次；0 = 禁用搜索",
    "AGENT_CHAT_TOOL_TIMEOUT_SECONDS": "单个本地工具的超时（秒）",
    "AGENT_CHAT_RUN_TIMEOUT_SECONDS": "一次回答的整体超时（秒）",
    "AGENT_CHAT_MESSAGE_CHUNK_CHARS": (
        "文本分条长度（字符）；0 = 插件不主动分段（按平台上限或整段发送）；"
        "平台覆盖优先生效"
    ),
    "AGENT_CHAT_MESSAGE_CHUNK_CHARS_BY_PLATFORM": (
        "按平台覆盖分条长度；0 = 插件不主动分段，交给平台自身的上限"
    ),
    "AGENT_CHAT_MESSAGE_SEND_DELAY_SECONDS": "相邻分条之间的发送间隔（秒）",
    "AGENT_CHAT_IMAGE_REPLY_MODE": (
        "回答是否渲染成图片：off / auto（长文或表格代码块）/ always"
    ),
    "AGENT_CHAT_IMAGE_REPLY_MODE_BY_PLATFORM": "按平台覆盖图片模式（优先级最高）",
    "AGENT_CHAT_SHOW_SOURCES_TEXT": "文本回答是否附加「来源：」页脚",
    "AGENT_CHAT_SHOW_SOURCES_TEXT_BY_PLATFORM": "按平台覆盖上面的文本来源开关",
    "AGENT_CHAT_SHOW_SOURCES_IMAGE": (
        "图片回答是否渲染来源页脚；关 = 整块页脚消失（含正文里的链接 URL）"
    ),
    "AGENT_CHAT_SHOW_SOURCES_IMAGE_BY_PLATFORM": "按平台覆盖上面的图片来源开关",
    "AGENT_CHAT_IMAGE_REPLY_MIN_CHARS": "auto 模式下，回答达到多少字符才渲染成图片",
    "AGENT_CHAT_IMAGE_REPLY_MAX_HEIGHT": "渲染图片的最大高度（像素）",
    "AGENT_CHAT_IMAGE_REPLY_TIMEOUT_SECONDS": "渲染图片的超时（秒），超时自动退回文字",
    "AGENT_CHAT_MAX_REPLY_CHARS": "单条提问（含引用内容）截断到的字符上限",
    "AGENT_CHAT_MAX_IMAGES": "单条消息最多处理几张图片",
    "AGENT_CHAT_MAX_IMAGE_BYTES": (
        "单条消息内所有图片（含引用与 Room 历史）的总字节上限"
    ),
}

LEGEND: tuple[str, ...] = (
    "标记含义：",
    (
        "  [文件]     该行写在 dotenv 文件（--env-file / AGENT_CHAT_ENV_FILE，"
        "默认 .env.agent_chat）"
    ),
    "  [环境变量] 由真实环境变量提供；dotenv 文件里的同名行不会覆盖它",
    "  （无来源标记）文件与环境变量都没设，用内置默认值",
    "  [需重启]   这项在启动/matcher 注册时读取，reload 不生效，要重启 bot",
    (
        "  （无 [需重启]）支持热重载：/agentctl reload 执行时重载；"
        "CLI 的 --reload / 编辑动作在 Bot 收到下一条消息时生效"
    ),
    "  CLI 专用   只被命令行工具读取，不参与 bot 运行配置",
    "  未管理     非 AGENT_CHAT_* 的键（例如 API 密钥）：只显示键名，不读取也不修改",
    "  未知键     以 AGENT_CHAT_ 开头但没有对应设置项（拼错或已废弃），会被忽略",
    "查看单条说明：nonebot-agent-chat --config-explain <KEY>（会话内 :config <KEY>）",
)


def type_hint(annotation: object) -> str:
    """A short Chinese description of the accepted value shape."""

    resolved, optional = unwrap_optional(annotation)
    prefix = "可留空；" if optional else ""
    origin = typing.get_origin(resolved)
    args = typing.get_args(resolved)
    if origin is list:
        return prefix + "JSON 数组"
    if origin is set:
        return prefix + "JSON 数组（不重复）"
    if origin is dict:
        value = args[1] if len(args) > 1 else object
        if isinstance(value, type) and issubclass(value, Enum):
            options = "/".join(str(member.value) for member in value)
            return prefix + f"JSON 映射（平台 → {options}）"
        if value is bool:
            return prefix + "JSON 映射（平台 → true/false）"
        if value is int:
            return prefix + "JSON 映射（平台 → 整数）"
        return prefix + "JSON 映射"
    if isinstance(resolved, type) and issubclass(resolved, Enum):
        options = " / ".join(str(member.value) for member in resolved)
        return prefix + f"取值：{options}"
    if resolved is bool:
        return prefix + "true / false"
    if resolved is int:
        return prefix + "整数"
    if resolved is float:
        return prefix + "数字（可小数）"
    if isinstance(resolved, type) and issubclass(resolved, Path):
        return prefix + "路径"
    if resolved is str:
        return prefix + "字符串"
    return prefix + str(resolved)


def describe(key: str) -> str | None:
    return SETTING_HELP.get(key)


def summarize(text: str, limit: int = SUMMARY_CHARS_DEFAULT) -> str:
    """A one-glance form for list columns: first clause, bounded.

    The full sentence stays in the detail pane, so truncation never hides
    information that cannot be reached another way.
    """

    if not text:
        return ""
    for separator in ("；", "。"):
        head, found, _ = text.partition(separator)
        if found:
            text = head
            break
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def default_text(key: str) -> str:
    """The built-in default, as shown to operators (environment ignored)."""

    field = key.lower()
    if field not in Config.model_fields:
        return ""
    info = Config.model_fields[field]
    if info.is_required():
        return "（必填）"
    return display_value(info.get_default(call_default_factory=True))


def explain_lines(
    key: str,
    *,
    current: str | None = None,
    source: str | None = None,
) -> list[str]:
    """The detail block printed by ``--config-explain`` / ``:config <KEY>``."""

    field = key.lower()
    description = describe(key)
    if field not in Config.model_fields or description is None:
        return [f"{key}：未知设置项（--config-list 看全部键）"]
    lines = [key, f"  说明：{description}"]
    if current is not None:
        where = SOURCE_WHERE.get(source or "", "内置默认")
        lines.append(f"  当前：{current}（{where}）")
    lines.append(f"  类型：{type_hint(Config.model_fields[field].annotation)}")
    lines.append(f"  默认：{default_text(key)}")
    if key in RESTART_ONLY_KEYS:
        lines.append("  生效：需重启 bot（reload 不会应用它）")
    else:
        lines.append("  生效：热生效（reload 或编辑后立即生效）")
    return lines
