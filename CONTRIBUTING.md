# 贡献与反馈

这是一个 Alpha 项目，优先保证已有功能可靠。提交问题时请说明 Python、NoneBot、适配器与插件版本，提供最小复现和已脱敏的错误信息。不要上传 `.env`、Profile 凭据、数据库或真实聊天记录。

## 开发

在独立开发目录中执行，不要直接使用生产 Bot 的配置和数据：

```bash
uv sync --all-extras --group dev
uv run --all-extras playwright install chromium
./scripts/check.sh
```

Linux 缺少浏览器系统依赖时，可执行 `uv run --all-extras playwright install --with-deps chromium`。完整检查要求渲染器可用；只运行本地单元测试时，未安装可选渲染依赖的用例可以明确跳过。

`check.sh` 会同步开发环境、检查格式/类型、运行测试、重建 `dist/`，并在临时环境安装产物进行 smoke 测试。依赖和浏览器下载需要网络；测试使用假 Provider，不需要真实 API key。请勿把生产凭据导出到测试环境。

修改行为时补回归测试，并在 `CHANGELOG.md` 的 Unreleased 中记录用户可感知的变更。公开扩展 API 在 0.x 仍可能变化。CI 覆盖 Python 3.10 与 3.13。

## 安全问题

不要在公开 issue 或 PR 中贴凭据、私人数据或可直接利用的细节。若仓库提供 GitHub 私密漏洞报告入口，请使用该入口；否则先通过不含敏感细节的消息请求维护者提供私下报告渠道。不要假设仓库已经启用私密报告。

怀疑密钥泄露时，先在服务商处撤销或轮换，再分享完全脱敏的复现。删除文件或 Git 提交不能使已泄露的凭据重新安全。
