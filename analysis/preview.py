"""一条命令出图：户型栅格 + 膨胀层 + 起终点 + A* 路径 + 地形分区。

```bash
uv run python -m analysis.preview config/scenarios/houseexpo_7e80c5f4.yaml
uv run python -m analysis.preview config/scenarios/houseexpo_*.yaml --out-dir docs/preview
```

左图：占据栅格（深色为障碍/轮廓外）+ 膨胀层（红色带，宽度来自
``platform.body_radius + platform.inflation``，同时记录在场景的 ``map.inflation_m``）
+ A* 路径 + 起终点；图上标注 resolution、场景内容哈希与规划量。
右图：地形分区（房间类型映射的假设材质），用于核对标签是否落在合理房间上。

只画静态 PNG，不做动画 / GUI。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "preview"
DEFAULT_ALGO = "configs/frozen_A.yaml"

# 用户级 matplotlib 缓存目录不一定可写；统一放到项目内（.gitignore 已忽略）。
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from runner import config as config_module  # noqa: E402
from runner import experiment  # noqa: E402

TERRAIN_COLORS = {
    "tile": "#9ecae1",
    "wood": "#e0b878",
    "concrete": "#cfcfcf",
    "default": "#eeeeee",
}
_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
)


def _setup_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.monospace"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def terrain_class_grid(
    scene: experiment.Scene, scenario: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    """把场景重建出的 ``c_rr`` 反映射成材质名网格（同一 c_rr 视为同一材质）。"""
    name_by_value = {float(scenario["terrain"]["default"]["c_rr"]): "default"}
    for patch in scenario["terrain"].get("patches", []):
        name_by_value.setdefault(float(patch["c_rr"]), str(patch["id"]))
    values = np.unique(scene.c_rr)
    names = [name_by_value.get(float(value), f"c_rr={value:g}") for value in values]
    index_grid = np.zeros(scene.c_rr.shape, dtype=int)
    for index, value in enumerate(values):
        index_grid[scene.c_rr == value] = index
    return index_grid, names


def render(
    scenario_path: str | Path,
    *,
    algo: str | Path = DEFAULT_ALGO,
    platform: str | Path | None = None,
    config_root: str | Path | None = None,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    dpi: int = 150,
) -> Path:
    """渲染一个场景，返回 PNG 路径。"""
    resolved = config_module.load_resolved(
        algo, scenario_path, platform_path=platform, config_root=config_root
    )
    scenario = resolved.scenario
    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    quality = scenario.get("quality", {}) or {}
    declared_clearance = quality.get("verified_clearance_m")
    if declared_clearance is not None and float(declared_clearance) < setup.astar_params.d_min:
        # 窄门场景按自己的验证口径出图（frozen_A 的 d_min 过不去）
        setup = replace(
            setup,
            astar_params=replace(
                setup.astar_params, d_min=float(declared_clearance)
            ),
        )
    result = experiment.plan_path(scene, setup)

    rows, cols = scene.occupancy.shape
    res = scene.resolution
    ox, oy = scene.origin
    extent = (ox, ox + cols * res, oy, oy + rows * res)
    inflation_m = float(
        scenario["map"].get("inflation_m", setup.astar_params.d_min)
    )
    content_hash = str(scenario.get("content_hash", "-"))

    _setup_fonts()
    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)

    # 左图：栅格 + 膨胀层 + 路径
    left.imshow(
        scene.occupancy.astype(float),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#ffffff", "#3b3b3b"]),
        interpolation="nearest",
    )
    inflated = (~scene.occupancy) & (scene.clearance + 1e-9 < inflation_m)
    overlay = np.zeros((rows, cols, 4), dtype=float)
    overlay[inflated] = (0.85, 0.10, 0.10, 0.30)
    left.imshow(overlay, origin="lower", extent=extent, interpolation="nearest")
    if result.success:
        left.plot(
            result.xy[:, 0],
            result.xy[:, 1],
            "-",
            color="#1f77b4",
            linewidth=2.0,
            label=f"A* 路径 {result.length_m:.2f} m",
        )
    left.plot(
        scene.start_xy[0], scene.start_xy[1], "o", color="#2ca02c", markersize=9,
        label="起点",
    )
    left.plot(
        scene.goal_xy[0], scene.goal_xy[1], "*", color="#d62728", markersize=16,
        label="终点",
    )
    grade = str(quality.get("grade", "-"))
    doors = quality.get("door_crossings", "-")
    rooms = quality.get("rooms_visited", "-")
    info = (
        f"resolution = {res} m    栅格 {rows} x {cols}\n"
        f"content_hash = {content_hash[:12]}…\n"
        f"inflation = {inflation_m:.2f} m (body_radius + inflation)\n"
        f"d_min = {setup.astar_params.d_min:.2f} m   e_budget = "
        f"{setup.astar_params.e_budget:.0f} J\n"
        f"难度 = {grade}   跨门 = {doors}   房间 = {rooms}"
    )
    if quality.get("group") == "narrow_door":
        info += (
            f"\n窄门宽度 = {quality.get('narrow_door_width_m')} m"
            f"（实测 {quality.get('measured_door_width_m')} m）"
            f"\nfrozen_A 可通行 = {quality.get('feasible_under_frozen_A')}"
        )
    if result.success:
        info += (
            f"\n路径长度 = {result.length_m:.2f} m   能耗 = "
            f"{result.total_energy_j:.1f} J\n最小离障 = {result.clearance.min():.2f} m"
        )
    else:
        info += f"\n规划失败: {result.reason}"
    left.text(
        0.02, 0.02, info, transform=left.transAxes, va="bottom", ha="left",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6"),
    )
    left.set_title(
        f"{scenario['id']}：占据栅格 + 膨胀层 + A* 路径\n"
        f"难度 {grade} | 跨门 {doors} | 房间 {rooms} | res {res} m | hash "
        f"{content_hash[:8]}",
        fontsize=11,
    )
    left.set_xlabel("x [m]")
    left.set_ylabel("y [m]")
    left.set_aspect("equal")
    left.legend(loc="upper right", fontsize=8)

    # 右图：地形分区（房间类型 -> 材质的假设映射）
    index_grid, names = terrain_class_grid(scene, scenario)
    colors = [TERRAIN_COLORS.get(name, "#c7c7c7") for name in names]
    right.imshow(
        index_grid,
        origin="lower",
        extent=extent,
        cmap=ListedColormap(colors),
        interpolation="nearest",
        vmin=0,
        vmax=max(1, len(names) - 1),
    )
    walls = np.ma.masked_where(~scene.occupancy, scene.occupancy)
    right.imshow(
        walls, origin="lower", extent=extent, cmap=ListedColormap(["#3b3b3b"]),
        interpolation="nearest",
    )
    right.plot(scene.start_xy[0], scene.start_xy[1], "o", color="#2ca02c", markersize=9)
    right.plot(scene.goal_xy[0], scene.goal_xy[1], "*", color="#d62728", markersize=16)
    right.set_title(
        "地形分区（基于房间类型的假设，非实测数据）\n"
        "rules: config/terrain_rules.yaml",
        fontsize=11,
    )
    right.set_xlabel("x [m]")
    right.set_ylabel("y [m]")
    right.set_aspect("equal")
    right.legend(
        handles=[Patch(facecolor=color, label=name) for color, name in zip(colors, names)],
        loc="upper right",
        fontsize=8,
    )

    out_path = Path(out_dir) / f"{scenario['id']}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.preview",
        description="场景预览图：栅格 + 膨胀层 + 起终点 + A* 路径 + 地形分区",
    )
    parser.add_argument("scenarios", nargs="+", help="场景 YAML 路径（可给多个）")
    parser.add_argument("--algo", default=DEFAULT_ALGO)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failed = 0
    for scenario_path in args.scenarios:
        try:
            out_path = render(
                scenario_path,
                algo=args.algo,
                platform=args.platform,
                config_root=args.config_root,
                out_dir=args.out_dir,
                dpi=args.dpi,
            )
        except config_module.ConfigError as exc:
            failed += 1
            print(f"FAIL {scenario_path}: {exc}", file=sys.stderr)
            continue
        print(f"OK   {out_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
