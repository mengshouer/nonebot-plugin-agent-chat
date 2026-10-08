# nonebot-plugin-agent-chat

给 NoneBot2 添加多模型 AI 问答：支持 OpenAI 兼容接口、Anthropic、搜索、图片输入，以及可选的图片回复和持久对话。

## 安装

需要 Python 3.10+ 和已有的 NoneBot2 Bot。在 Bot 目录用 NB-CLI 安装，它会自动把插件写入列表：

```bash
nb plugin install nonebot-plugin-agent-chat
```

也可以手动安装，再把插件追加到 Bot 的 `pyproject.toml` 列表（不要覆盖其他插件）：

```bash
python -m pip install "nonebot-plugin-agent-chat"
```

```toml
[tool.nonebot]
plugins = ["nonebot_plugin_agent_chat"]
```

可选依赖（图片回复、TUI 编辑器）需要用 pip 安装，见下文。插件沿用 Bot 已安装并注册的适配器。

如果要使用尚未发布的开发版，从 GitHub 源码安装：

```bash
git clone https://github.com/mengshouer/nonebot-plugin-agent-chat.git
python -m pip install "./nonebot-plugin-agent-chat"
```

不要同时加载本插件的目录副本，否则会重复注册。

## 最少配置

以下以一个 OpenAI Chat Completions 兼容服务为例。`.env.agent_chat` 放在 **Bot 工作目录**，Profile 与运行时数据目录按下文规则决定；已有文件请合并内容，不要直接覆盖。

### 1. 配置凭据和权限

新建 `.env.agent_chat`，将占位符替换为自己的 API key：

```dotenv
OPENAI_API_KEY=replace-with-your-key
AGENT_CHAT_DEFAULT_PROFILE=openai-chat
AGENT_CHAT_ALLOWED_GROUPS=["QQClient:123456789"]
```

白名单格式是 `<平台标识>:<群号或用户ID>`，平台标识如 `QQClient`、`Telegram`，区分大小写，不是适配器包名。这些标识由 [Alconna 的跨平台消息组件](https://github.com/nonebot/plugin-alconna/blob/master/src/nonebot_plugin_alconna/uniseg/constraint.py) 提供。

不确定填写什么时，在本地查询平台标识即可，无需启动 Bot、配置 API key 或拥有平台管理员账号：

```bash
nonebot-agent-chat --platform-list
```

复制列表中的平台标识即可用于配置，但列表不代表当前 Bot 已连接或验证过全部平台。请替换示例中的群号；Telegram 群示例为 `Telegram:-1001234567890`，指定用户可用 `AGENT_CHAT_ALLOWED_USERS=["Telegram:12345"]`。

- **默认不允许普通用户使用**。只放行你信任的群或用户，避免意外调用费用。
- NoneBot 的 `SUPERUSERS` 不受上述白名单限制，支持框架的裸 ID 和 adapter 限定 ID 写法。
- 白名单必须包含平台前缀，裸 ID 会报错；`QQClient:*` 等通配符会放行该平台全部目标，谨慎使用。
- 将凭据文件和实际使用的数据/配置目录（见下文）加入 **Bot 自己的 `.gitignore`**；插件仓库的规则不会替宿主生效。Linux/macOS 可执行 `chmod 600 .env.agent_chat` 限制文件权限。

### 2. 配置模型

创建 Profile 文件，将 `model` 替换为服务商提供的模型 ID。默认位置由 [`nonebot-plugin-localstore`](https://github.com/nonebot/plugin-localstore) 决定，Linux 上通常是 `~/.config/nonebot2/nonebot_plugin_agent_chat/profiles/`；如果 Bot 工作目录下已经存在 `data/agent_chat/` 中的既有插件数据，则继续使用 `data/agent_chat/profiles/`。不确定用哪个目录时，在 `.env.agent_chat` 里显式设置 `AGENT_CHAT_PROFILE_DIR`，这样 Bot 和 CLI 会使用同一目录：

```json
{
  "protocol": "openai-completions",
  "model": "replace-with-your-model",
  "api_key_env": "OPENAI_API_KEY"
}
```

使用兼容网关时，在 JSON 中增加服务商指定的 `base_url`。文件名 `openai-chat` 就是 Profile ID，应与 `AGENT_CHAT_DEFAULT_PROFILE` 一致。

其他协议可参考 [`profiles.example`](src/nonebot_plugin_agent_chat/profiles.example)：`openai-responses`、`anthropic-messages`。只配置自己要使用的服务即可。

### 3. 启动并提问

按原来的方式启动 Bot，在已放行的会话中使用配置的提问触发词。默认是 `/llm`：

```text
/llm 你好，请介绍一下自己
```

`/llm` 不是固定命令，可在 `.env.agent_chat` 中修改，例如：

```dotenv
AGENT_CHAT_TRIGGERS=["/ask"]
```

重载后即可发送 `/ask 你好`。可以配置多个触发词，消息中包含其中任意一个即可触发提问。默认不会自动回复所有私聊或 @ 消息；插件只发送最终结果或错误，不发送“处理中”消息。

管理员常用命令：

```text
/agentctl status
/agentctl profiles
/agentctl use openai-chat
/agentctl reload
/agentctl cancel all
```

管理命令固定为 `/agentctl`，不随提问触发词变化。
修改配置或 Profile 后执行 `/agentctl reload`；CLI 编辑会请求 Bot 在下一条消息路由前重载。部分设置需要重启，重载结果会提示。`use` 临时切换默认 Profile，重启后恢复配置值。

## 可选功能

### 图片回复

在 Bot 的 Python 环境中安装额外依赖，并安装 Chromium：

```bash
python -m pip install "nonebot-plugin-agent-chat[render-image]"
python -m playwright install chromium
# Linux 若缺系统依赖，可使用：python -m playwright install --with-deps chromium
```

在 `.env.agent_chat` 中添加 `AGENT_CHAT_IMAGE_REPLY_MODE=auto`，重载后长文、表格等会转为图片；渲染不可用时退回文字。Telegram 默认仍使用文字及原生格式化。详细阈值和平台覆盖见[高级用法](docs/advanced.md#图片与长回答)。

### 本地 CLI 与编辑器

也可以通过 `nonebot-agent-chat` 查看和修改配置，无需手动编辑文件。在 **Bot 工作目录**运行：

```bash
nonebot-agent-chat --config-list
nonebot-agent-chat --config-set AGENT_CHAT_IMAGE_REPLY_MODE=auto
nonebot-agent-chat --profile-list
nonebot-agent-chat --check --profile openai-chat
nonebot-agent-chat --profile openai-chat "你好"
```

配置修改会请求 Bot 在下一条消息前重载，需重启的设置会另行提示。最后一条会真实调用模型。可选安装 `[tui]` extra 后，用 `--config-edit` / `--profile-edit` 打开交互编辑器；安装方式与上面的 extra 相同。

### Profile 规则与 Agent Room

- 可以给平台或群绑定不同的默认模型，见 [Profile 规则](docs/advanced.md#profile-规则)。
- Agent Room 用于持久多轮对话，**默认关闭且仅 superuser 可用**。开启后会保存对话原文；同一 Room 正在处理时，新提问会被拒绝，不会后台排队。见 [Agent Room](docs/advanced.md#agent-room)。

## 隐私与限制

- 提问、引用消息和图片会发送给配置的模型服务，搜索还会使用相应搜索服务。服务商的数据留存政策需自行确认。
- 普通提问的 SQLite 只保存运行元数据，不保存问答正文；启用 Room 后，其对话原文会持久化，图片可能保存在本地缓存。
- Responses 的 `responses_store` 默认 `false`，仅关闭对应的响应存储选项，**不代表服务商不保留任何日志或数据**。
- 不要把 API key 放进命令行参数、公开日志、截图或仓库。包括 `$(...)` 命令替换，展开后的值仍会进入进程参数。优先使用受限权限的凭据文件。
- 超长回答会分段或转图；发送失败可能重试、降级或省略部分内容，并提示失败。自定义工具属于可信 Python 扩展，不是安全沙箱。

## 常见问题

- **没有回复？** 检查是否使用了配置的提问触发词（默认 `/llm`）、白名单的平台标识和 ID 是否正确，再由管理员查看 `/agentctl status`。
- **认证或模型错误？** 检查当前 Profile 的模型 ID、API key 对应的环境变量，以及兼容网关的 `base_url`。已导出的环境变量优先于 dotenv。
- **改配置没生效？** 手动修改后执行 `/agentctl reload`；提示需重启的项要重启 Bot。
- **没有转成图片？** 确认安装了 extra 和匹配的 Chromium，查看 status；Telegram 有默认关闭图片回复的平台覆盖。

全部配置说明：[`.env.agent_chat.example`](.env.agent_chat.example)、`nonebot-agent-chat --config-list`、`nonebot-agent-chat --config-explain <KEY>`。

更多功能见[高级用法](docs/advanced.md)，变更见[更新记录](CHANGELOG.md)，开发与反馈见[贡献指引](CONTRIBUTING.md)。

## 许可证

[MIT](LICENSE)
