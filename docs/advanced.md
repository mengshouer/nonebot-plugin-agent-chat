# 高级用法

首次安装见 [README](../README.md)。所有相对路径均以 **Bot 工作目录**为基准。

## 配置与重载

平台标识区分大小写，例如 `QQClient`、`Telegram`，不是适配器包名。`nonebot-agent-chat --platform-list` 可在未配置 Bot 时查询，附加 `--json` 输出结构化列表；这不是已连接平台的清单。白名单、平台覆盖和 Profile 规则使用同一套平台标识。NoneBot `SUPERUSERS` 的适配器前缀是另一套规则，与这里的平台标识不同。

默认配置文件是 `.env.agent_chat`；Profile、Prompt 分别位于 `data/agent_chat/profiles/` 和 `data/agent_chat/prompts/`。启动前可用 `AGENT_CHAT_ENV_FILE` 指定其他配置文件。

```bash
nonebot-agent-chat --config-list
nonebot-agent-chat --config-explain AGENT_CHAT_IMAGE_REPLY_MODE
nonebot-agent-chat --config-set AGENT_CHAT_MAX_SEARCHES=5
nonebot-agent-chat --config-unset AGENT_CHAT_MAX_SEARCHES
nonebot-agent-chat --reload
```

清单显示当前值、来源和是否需重启。常用示例见 [`.env.example`](../.env.example)，安装包中对应 `nonebot_plugin_agent_chat/env.example`。配置和 Profile 命令支持 `--json` 输出。

手改文件后执行 `/agentctl reload` 或 `nonebot-agent-chat --reload`。CLI 编辑会写重载标记，Bot 在下一条消息路由前处理，并发消息等待已开始的重载完成。配置不自动轮询；解析或校验失败保留旧配置，并在 status 中记录错误。

`AGENT_CHAT_DATA_DIR`、`AGENT_CHAT_PRIORITY` 需要重启；`AGENT_CHAT_ENV_FILE` 同样在启动时固定。其余以配置查询和重载结果为准。群中显示摘要，私聊可查看差异。

## Profile 与 Prompt

| 字段 | 用途 |
|---|---|
| `protocol` / `model` | 必填，协议与服务商模型 ID |
| `api_key_env` / `base_url` | 凭据环境变量名；可选兼容网关地址 |
| `capabilities` | `vision` / `tools` / `reasoning` 开关 |
| `reasoning_effort` | `provider_default` / `off` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max` |
| `search_mode` | `off` / `builtin_web_search` / `exa` |
| `responses_store` | Responses 响应存储开关，默认 `false` |
| `system_prompt` / `system_prompt_file` | 内联提示词或 Prompt 目录中的 Markdown 文件 |
| `fallback_profiles` | 非递归的候选 Profile ID 列表 |
| `enabled_tools` | 显式启用的自定义只读工具 |

示例见 [`profiles.example`](../src/nonebot_plugin_agent_chat/profiles.example)。支持 `openai-completions`、`openai-responses`、`anthropic-messages`，搜索可用协议内建搜索或 Exa。具体模型的能力与参数支持需向服务商确认。

已有环境变量优先于 dotenv。Profile 内联的非空 `api_key` / `exa_api_key` 是例外，优先于对应 `*_env`，但会明文写在文件中。`null`、空串、纯空白回落环境变量；非空凭据的首尾空白或控制字符会被拒绝。推荐环境变量引用，不要通过命令行参数传密钥。

Prompt 优先级：非空 `system_prompt` → `system_prompt_file` → 全局默认文件（`AGENT_CHAT_DEFAULT_SYSTEM_PROMPT_FILE`，默认 `default.md`）→ 内置中立提示词。

文件使用 UTF-8 Markdown，不支持模板。全局文件不存在时使用内置提示词，存在但为空表示不使用提示词；显式引用文件缺失或编码错误会导致重载失败。路径必须在 Prompt 目录内，不能通过绝对路径或 `..` 越界。修改后重载生效，运行中的请求保留原快照。

`/agentctl use <profile>` 临时全局覆盖，重启恢复默认配置。Room 绑定独立持久化，不跟随全局切换。fallback 不发生在可见输出、本地工具执行或不可重试错误之后；Room 不自动换用其他 Profile。

## Profile 规则

管理员可以给平台或群绑定默认 Profile：

```text
/agentctl rule set qq-safe
/agentctl rule set qq-safe platform
/agentctl rule list
/agentctl rule unset
/agentctl rule unset platform
```

第一条绑定当前群，第二条绑定当前平台。指定其他群用 `group <id>`，跨平台 ID 写为 `<平台标识>:<群ID>`；私聊必须明确目标。

```bash
nonebot-agent-chat --rule-set qq-safe --scope QQClient --group 123456789
nonebot-agent-chat --rule-set qq-safe --scope QQClient
nonebot-agent-chat --rule-list --json
nonebot-agent-chat --rule-unset --scope QQClient --group 123456789
```

优先级：**Room > `/agentctl use` 临时覆盖 > 群规则 > 平台规则 > 默认 Profile**。`use` 切回配置的默认 Profile 可解除临时覆盖。规则存入 SQLite，下一次提问生效；私聊不参与，也没有 topic 级规则。被引用的 Profile 删除后会报错，不会静默换模型。群中只显示本会话规则，完整规则在私聊或 CLI 查看。

## Agent Room

设置 `AGENT_CHAT_ROOM_ENABLED=true` 并重载，仅 superuser 可使用：

```text
/agent_room new [name]
/agent_room ask <question>
/agent_room use <profile>
/agent_room status
/agent_room clear
/agent_room close
```

Room 的提问与回答存入数据库，图片使用本地缓存。`clear` 清除该 Room 历史，`close` 关闭 Room，不等同于删除全部历史数据。不需要持久化请保持默认关闭。

**同一 Room 忙时，新提问会被拒绝，不会后台排队。** 请求受并发、超时和 `/agentctl cancel` 管理。Room 按会话隔离，Telegram 论坛 topic 另行隔离；同一群/话题的 Room 不是每位提问者独享的私人会话。

## 图片与长回答

默认按 `AGENT_CHAT_MESSAGE_CHUNK_CHARS=1000` 分条，间隔由 `AGENT_CHAT_MESSAGE_SEND_DELAY_SECONDS` 控制。Telegram 默认平台覆盖为 `0`，使用平台的 4096 UTF-16 code units 上限。

`AGENT_CHAT_MESSAGE_CHUNK_CHARS_BY_PLATFORM` 可覆盖平台设置。未知平台设为 `0` 时先尝试整段发送，拒绝后自适应拆分。重试仍失败时会提示部分或全部内容发送失败。

图片回复安装见 [README](../README.md#图片回复)。`off` 仅文字，`auto` 按长度/表格/代码块转图，`always` 优先转图。常用参数：

- `AGENT_CHAT_IMAGE_REPLY_MIN_CHARS`：默认 1000。
- `AGENT_CHAT_IMAGE_REPLY_MAX_HEIGHT`：默认 8000。
- `AGENT_CHAT_IMAGE_REPLY_TIMEOUT_SECONDS`：默认 20。

覆盖优先级：**平台 > Profile > 全局**。Telegram 默认 `AGENT_CHAT_IMAGE_REPLY_MODE_BY_PLATFORM={"Telegram":"off"}`，想启用可改为 `{"Telegram":"auto"}`。Chromium 按需启动，投递结束后释放；渲染不可用时退回文字。

来源页脚由 `AGENT_CHAT_SHOW_SOURCES_TEXT` / `AGENT_CHAT_SHOW_SOURCES_IMAGE` 控制，也支持 `_BY_PLATFORM` 和 Profile 覆盖。图片来源关闭时隐藏整个图片页脚，包括正文链接的原始 URL；文字正文的链接不受这些页脚开关影响。

本地检查排版，不调用模型或发送消息：

```bash
nonebot-agent-chat --render-image out.png --render-input answer.md
```

远程 URL 图片有公共 IP 校验和流式字节限制；平台媒体由 adapter 下载后检查大小，并非所有路径都有下载中的内存硬上限。渲染禁用 JS 并阻止资源加载，但不是执行不可信代码的沙箱。

## CLI 与交互编辑器

CLI 不启动 NoneBot，默认调试数据写到 `data/agent_chat/debug/`。不带问题进入交互会话：

```bash
nonebot-agent-chat
nonebot-agent-chat --profile openai-chat --image ./image.png "描述图片"
nonebot-agent-chat --profile-new qq-safe --from openai-chat
nonebot-agent-chat --profile-set qq-safe model=replace-with-your-model
nonebot-agent-chat --profile-remove qq-safe
```

会话内可用 `:config`、`:profile`、`:reload`、`:help`、`:quit`，例如 `:config set AGENT_CHAT_MAX_SEARCHES=5`。编辑参数可重复，整批校验通过后写入；单文件原子替换。删除 Profile 前会提示规则和 Room 引用，仍需自行处理引用。

安装 `[tui]` extra 后，用 `--config-edit` / `--profile-edit [name]`，或会话内 `:config edit` / `:profile edit`。

| 按键 | 用途 |
|---|---|
| `Ctrl+←` / `Ctrl+→` | 切换设置 / Profiles 标签 |
| `Tab` / `Shift+Tab` | 切换焦点 |
| `↑` / `↓`、`PageUp` / `PageDown` | 浏览列表 |
| `/` | 过滤项目 |
| `Enter` | 编辑或确认 |
| `n` | 新建 Profile |
| `Ctrl+D` | 删除，Profile 需确认 |
| `Ctrl+S` / `s` | 保存并请求重载 |
| `Ctrl+Z` / `u` | 放弃未保存改动 |
| `Ctrl+Q` / `Ctrl+C` | 退出，未保存时确认 |

字母快捷键仅在非文本输入时生效。终端可能截获 `Ctrl+S`，可用 `s`；弹窗可点保存或使用终端支持的 `Ctrl+Enter`。改动先暂存，保存才落盘；本地会话立即重载，Bot 在下一条消息前重载。

已存储凭据不以原值预填。直接凭据字段的占位符原样保存表示保留，清空表示回落环境变量；扩展请求字段默认隐藏，不应用编辑器查看密钥。

按群规则试跑：

```bash
nonebot-agent-chat --scope QQClient --group 123456789 "你好"
```

这会真实调用模型，读取 Bot 的生产数据目录并写入运行历史，不是离线预览。可用 `--data-dir` 指定其他目录。

## 开发与扩展

开发步骤见[贡献指引](../CONTRIBUTING.md)。扩展插件先 require 再 import：

```python
from nonebot import require

require("nonebot_plugin_agent_chat")

from nonebot_plugin_agent_chat import (
    RunContext,
    ToolContext,
    ToolOutput,
    ToolRisk,
    ToolSpec,
    register_after_response,
    register_before_run,
    register_before_tool,
    register_tool,
)
```

工具受当前 Profile allowlist 和 `ToolRisk.READ_ONLY` 检查，但扩展仍是可信 Python 代码，没有进程隔离。不要加载来源不明的扩展。

`AgentChatService`、adapters、storage 属于内部实现，0.x 不承诺兼容。平台标识直接使用 UniSeg 的 `SupportScope.value`；只有需要特殊投递行为时才在 `platforms.py` 增加策略，不必为每个平台新增条目。新平台仍需验证实际消息、权限与投递。
