# Codex Python SDK 来源说明

本目录保留 OpenAI Codex 仓库 `rust-v0.149.0` 中 `sdk/python` 的官方源代码快照。
快照提交为 `758ef40f50c1a458425c7cfbf1eb12cbc07af0b0`，许可证为 Apache-2.0，完整文本见
同目录 `LICENSE`。

本地 `pyproject.toml` 只调整打包元数据：使用仓库标准环境已有的 setuptools，并移除
官方开发树中对 `openai-codex-cli-bin==0.147.0` 的依赖。运行时由
`third_party/codex_cli/0.149.0/` 提供固定版本的 Codex CLI；不得让 SDK 自动下载或选择其他
CLI runtime。
