"""解析 MiniChem 使用的固定标准 Python 环境路径。"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import PROJECT_ROOT, TOOLBOX_ROOT


ARCHE_PYTHON_ENV = "ARCHE_PYTHON"
MINICHEM_RUNTIME_PYTHON_ENV = "MINICHEM_RUNTIME_PYTHON"
# Resolve the shared repository environment from the copied toolbox location;
# no machine-specific absolute path is embedded in the source tree.
STANDARD_ARCHE_PYTHON = (TOOLBOX_ROOT.parent.parent / ".venv" / "bin" / "python").resolve()
DEFAULT_ARCHE_PYTHON = STANDARD_ARCHE_PYTHON
STANDARD_RUNTIME_NAME = "arche"


def environment_path(runtime: str) -> Path:
    """返回所有 MiniChem Backend 共用的标准环境前缀。

    输入是 Backend runtime 名称；动作忽略该名称，因为当前部署只允许一个
    MiniChem Python 环境。返回值是标准解释器的环境前缀；显式配置只能指向同一
    标准解释器，其他路径会直接报错。
    """

    del runtime
    return _arche_python_path().parent.parent.resolve()


def _arche_python_path() -> Path:
    """读取 MiniChem 解释器路径，不检查文件是否存在。"""

    value = (
        os.environ.get(MINICHEM_RUNTIME_PYTHON_ENV, "").strip()
        or os.environ.get(ARCHE_PYTHON_ENV, "").strip()
    )
    if not value:
        return DEFAULT_ARCHE_PYTHON
    path = Path(value).expanduser()
    resolved = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if resolved.resolve() != STANDARD_ARCHE_PYTHON.resolve():
        raise ValueError(
            "MiniChem 必须使用标准 Python："
            f"{STANDARD_ARCHE_PYTHON}"
        )
    return STANDARD_ARCHE_PYTHON


def resolve_runtime_path(runtime: str, configured_value: str | Path) -> Path:
    """解析 Backend runtime 配置，并将标准逻辑名映射到 ARCHE 环境。

    输入是 runtime 名称和配置文件中的环境前缀。标准逻辑名 ``arche`` 使用
    统一环境；显式路径也必须解析到标准环境。其他配置直接报错，不会偷偷
    创建或使用另一套环境。
    """

    path = Path(configured_value).expanduser()
    if str(configured_value).strip() == STANDARD_RUNTIME_NAME:
        return environment_path(runtime)
    if path.is_absolute() and path.resolve() == STANDARD_ARCHE_PYTHON.parent.parent:
        return path.resolve()
    raise ValueError(
        "MiniChem runtime 必须使用标准环境："
        f"{STANDARD_ARCHE_PYTHON.parent.parent}"
    )


def resolve_configured_path(value: str | Path) -> Path:
    """解析配置路径，并支持迁移后的 MiniChem 运行根目录。"""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()
