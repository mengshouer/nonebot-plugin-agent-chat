# 更新记录

## [0.1.3] - 2026-10-14

### 新增

- Agent Room 新增滚动上下文压缩（Rolling Context Compaction）：
  - Profile 新增 `context_window`（默认 `200000` tokens），全局配置新增 `AGENT_CHAT_ROOM_COMPACT_THRESHOLD_PERCENT`（默认 `80`）。
  - 当累计上下文达到阈值或遇到服务商上下文超限报错时，自动将早期问答滚动压缩为结构化摘要并保留近期原文。
  - 新增 `/agent_room compact [focus]`（及 CLI `:compact [focus]`）手动压缩命令，并在 `/agent_room status` 中显示当前上下文占用与压缩次数。

## [0.1.2] - 2026-10-13

### 变更

- 数据与 Profile 目录统一由 `nonebot-plugin-localstore` 的插件数据目录决定（`profiles/`、`prompts/`、`images/`、`agent_chat.db` 均位于该数据目录下）：**移除 `AGENT_CHAT_DATA_DIR`、`AGENT_CHAT_PROFILE_DIR` 与 `AGENT_CHAT_DEBUG_DATA_DIR` 配置项**。要改位置请在 Bot 的全局 `.env` / `.env.prod` 中配置 localstore 自己的 `LOCALSTORE_*`（例如 `LOCALSTORE_USE_CWD=true` 或 `LOCALSTORE_DATA_DIR`）。
- CLI 会按 Bot 的 NoneBot 配置解析 localstore 目录，因此 `--reload`、`--rule-*`、`--config-*` 与 Bot 读写同一份数据。`--data-dir` / `--profiles-dir` 仍可显式覆盖（调试数据默认在 `<数据目录>/debug/`）。
- README、高级用法与示例配置改为说明 localstore 目录规则与 `LOCALSTORE_*` 用法。

### 升级注意

- 旧版 `data/agent_chat/` 不再自动优先或迁移；如果仍要使用其中的数据，请按照新的 localstore 配置手动移动到对应的插件数据目录（如 `<数据目录>/profiles/`、`<数据目录>/prompts/`）。`.env.agent_chat` 里残留的旧目录键会被当作未知键忽略。
- 未显式配置 `LOCALSTORE_*` 时文件位于系统用户数据目录（如 Linux 下的 `~/.local/share/nonebot2/nonebot_plugin_agent_chat`）；要跟随 Bot 工作目录请设置 `LOCALSTORE_USE_CWD=true`。

## [0.1.1] - 2026-10-09

### 变更

- 补齐 NoneBot 插件商店要求的元数据：新增 `homepage`，`supported_adapters` 改为继承 alconna/UniSeg 的适配器集合。
- 本地文件改由 `nonebot-plugin-localstore` 管理：Profile 默认在插件配置目录，数据库、图片缓存与 Prompt 默认在插件数据目录。旧的 `data/agent_chat/` 中若已有 `profiles/`、数据库等插件数据则继续沿用；也可用 `AGENT_CHAT_PROFILE_DIR` / `AGENT_CHAT_DATA_DIR` 固定位置。
- Telegram 适配器依赖不再精确锁定单一版本，改为 `>=0.1.0b20,<0.2`，允许后续 0.1.x beta 修复版本。
- README 补充 NB-CLI 安装方式，并说明新的默认目录规则。
- 示例配置由 `.env.example` 改名为 `.env.agent_chat.example`，与插件实际读取的 `.env.agent_chat` 对应；随包附带的副本同步为 `nonebot_plugin_agent_chat/env.agent_chat.example`。

### 升级注意

- 插件现在依赖 `nonebot-plugin-localstore`；用 NB-CLI 或 pip 安装会自动带入，手动维护虚拟环境（如 editable 安装）需要重新安装一次以补上该依赖。
- 不设置目录配置、且工作目录下没有旧布局的实际插件数据时，Profile 与数据的默认位置会变到系统用户目录；担心位置变化就显式设置两个目录项。

## [0.1.0] - 2026-10-08

首个公开版本，功能概览如下。

### 首版功能

- 支持 OpenAI Chat Completions、OpenAI Responses 和 Anthropic Messages，可使用兼容接口及多个模型配置。
- 支持跨平台消息，主要验证 OneBot V11 和 Telegram；其他聊天平台仍需兼容性测试。
- 支持文字、引用消息和图片输入，以及可选的图片回复、长文分条和投递重试。
- 支持模型服务内建网页搜索、Exa 搜索，以及可选的来源链接展示。
- 提问触发词可配置，默认 `/llm`；超级用户通过 `/agentctl` 管理插件。
- 支持 Profile、Markdown 系统提示词、备用模型配置，以及按平台或群选择默认 Profile。
- 可选的 Agent Room 提供持久多轮对话，仅超级用户可用；同一 Room 忙碌时拒绝新提问，不后台排队。
- 提供本地 CLI 和可选的 TUI 编辑器，用于查询平台标识、管理配置与 Profile、检查配置及请求重载。
- 提供白名单、并发与调用额度限制、取消和超时管理，以及可信 Python 工具扩展接口。

### 使用约定

- 白名单使用 `<平台标识>:<群号或用户ID>` 格式，例如 `QQClient:12345`、`Telegram:67890`，平台标识区分大小写。
- 配置和 Profile 在启动或明确重载时读取；CLI 编辑请求在 Bot 下一条消息到来时生效，部分设置需重启。
- 普通提问只保存运行元数据；启用 Room 后会持久保存对话原文和相关图片。模型与搜索服务可能产生费用并保留数据。
- 图片回复需要安装 `render-image` 扩展依赖及匹配的 Playwright Chromium；渲染不可用时退回文字。

安装与使用见 [README](README.md)，详细说明见[高级用法](docs/advanced.md)。
