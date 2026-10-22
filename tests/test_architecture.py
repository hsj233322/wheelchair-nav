"""架构守卫：core 必须保持纯函数、不得依赖配置或 runner。"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_core_does_not_import_runner_or_config() -> None:
    for path in sorted(CORE.glob("*.py")):
        for module in imports_of(path):
            assert module.split(".")[0] not in {"runner", "config"}, (
                f"{path.name} 不得 import {module}"
            )
            assert module.split(".")[0] != "yaml", f"{path.name} 不得读取 YAML 配置"


def test_core_does_not_open_files() -> None:
    for path in sorted(CORE.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "open(" not in source, f"{path.name} 不应做文件 IO"
        assert "Path(" not in source, f"{path.name} 不应做文件 IO"
