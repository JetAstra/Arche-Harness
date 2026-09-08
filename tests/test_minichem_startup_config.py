"""验证仓库记录的 MiniChem 随附服务启动配置。"""

import json
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "src"
    / "chemistry_multiagent"
    / "tools"
    / "minichem_mcp_startup_config.json"
)


def load_minichem_startup_config() -> dict:
    """读取启动配置；配置损坏时让 JSON 解析异常直接暴露。"""

    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_minichem_startup_config_points_to_bundled_executable() -> None:
    """确认配置从候选仓库根解析随附 Toolbox 的启动脚本。"""

    config = load_minichem_startup_config()
    deployment = config["deployment"]
    toolbox_root = REPOSITORY_ROOT / deployment["toolbox_root"]
    startup_script = REPOSITORY_ROOT / config["server"]["startup_script"]

    assert config["transport"] == "streamable_http"
    assert deployment["type"] == "bundled_toolbox"
    assert not Path(deployment["toolbox_root"]).is_absolute()
    assert not Path(config["server"]["startup_script"]).is_absolute()
    assert startup_script.is_file()
    assert os.access(startup_script, os.X_OK)
    assert startup_script.parent.parent == toolbox_root


def test_minichem_startup_config_matches_local_service_contract() -> None:
    """确认端点、端口和服务资源上限与当前 MiniChem 部署约定一致。"""

    config = load_minichem_startup_config()
    server = config["server"]
    environment = server["environment"]

    assert config["endpoint"] == "http://127.0.0.1:9010/mcp"
    assert config["adapter"]["default_endpoint"] == config["endpoint"]
    assert config["adapter"]["endpoint_environment"] == "MINICHEM_MCP_ENDPOINT"
    assert server["arguments"] == [
        "--transport",
        "streamable_http",
        "--host",
        "127.0.0.1",
        "--port",
        "9010",
    ]
    assert environment == {
        "RESEARCHCHEMBENCH_AVAILABLE_CPU_CORES": "60",
        "RESEARCHCHEMBENCH_AVAILABLE_MEMORY_MB": "320000",
        "RESEARCHCHEMBENCH_AVAILABLE_GPU_COUNT": "1",
    }
