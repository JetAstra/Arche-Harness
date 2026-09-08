# Codex CLI 0.149.0（Linux x86_64 musl）

本目录保存 ARCHE Harness 固定使用的 Codex CLI 运行时。它来自当前系统 npm 安装的
`@openai/codex@0.149.0-linux-x64` 平台包，对应官方仓库 `rust-v0.149.0`，许可证为
Apache-2.0，完整文本见同目录 `LICENSE`。

官方运行时不只有主 ELF：工具回合还需要与入口相邻的
`bin/codex-code-mode-host`，沙箱和命令工具还需要 `codex-resources/bwrap`、打包的 zsh
以及 `codex-path/rg`。本目录因此按官方平台包的布局追踪五个 xz 归档和一个小型布局清单；
每个归档及解包后的文件都在 `manifest.json` 中固定 SHA-256。

`linux-x64/codex` 是唯一运行入口。首次调用时它核对所有归档和布局清单，将文件原子解包到
被 Git 忽略的
`local_data/runtime/codex_cli/0.149.0/linux-x64/vendor/x86_64-unknown-linux-musl/`，然后执行
官方兼容的 `bin/codex`。这样 sidecar 和资源可以通过相对路径被发现；它不读取
`/usr/local/lib/node_modules`，也不回退到 PATH 中的其他 Codex。

可用下面的命令检查版本并触发完整运行时校验：

```bash
third_party/codex_cli/0.149.0/linux-x64/codex --version
```

该命令输出 `codex-cli 0.149.0` 后，运行副本的完整路径应为
`.../vendor/x86_64-unknown-linux-musl/bin/codex`；同一目录下应同时存在
`codex-code-mode-host`、`codex-resources/bwrap`、`codex-resources/zsh/bin/zsh` 和
`codex-path/rg`。
