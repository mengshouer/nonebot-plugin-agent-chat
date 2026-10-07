# 更新记录

## [Unreleased]（首版准备中）

以下为首个公开版本的功能概览，尚未发布。

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
