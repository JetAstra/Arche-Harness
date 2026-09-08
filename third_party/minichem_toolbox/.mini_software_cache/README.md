# MiniChem 软件缓存

此目录随工具箱一起复制，但其中的二进制载荷不纳入 Git。

预期目录结构：

```text
.mini_software_cache/
├── gaussian/g16/                  # 由操作人员提供且受许可证约束的 Gaussian 16 目录
└── multiwfn/2026.7.15/            # 官方 Multiwfn noGUI 压缩包及发行内容
```

MiniChem 使用仓库根目录 `.venv/bin/python` 和
`PYTHONPATH=src:third_party/minichem_toolbox/src`；此目录仅存放软件载荷。
ASH 是源代码而不是缓存载荷，在 `vendor/ash-0.95/` 下单独进行版本管理。
Gaussian 受其许可证约束，只能在许可证允许的位置复制。
Multiwfn 载荷是官方 `Multiwfn_2026.7.15_bin_Linux_noGUI.zip` 发行包；其上游许可证和引用要求
同样适用。
