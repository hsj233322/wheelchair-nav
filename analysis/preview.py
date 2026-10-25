"""一条命令出图：户型栅格 + 膨胀层 + 起终点 + A* 路径 + 地形分区。

```bash
uv run python -m analysis.preview config/scenarios/houseexpo_7e80c5f4.yaml
uv run python -m analysis.preview config/scenarios/houseexpo_*.yaml --out-dir docs/preview
```

左图：占据栅格（深色为障碍/轮廓外）+ **物理半径层**（深红，来自场景
``map.inflation_m = platform.body_radius``）+ **策略裕度层**（橙色，物理半径到
``algo.astar.d_min`` 之间）+ A* 路径 + 起终点；图上标注 resolution、场景内容哈希、
两个半径与规划量。指定非默认 ``--algo``（如 ``configs/dmin_sweep/dmin_110.yaml``）时
输出文件名会加 ``__dmin_110`` 后缀，方便同一场景跨档位对比。
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

from core import astar  # noqa: E402
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
    start_xy: tuple[float, float] | None = None,
    goal_xy: tuple[float, float] | None = None,
    tag: str | None = None,
) -> Path:
    """渲染一个场景，返回 PNG 路径。

    ``start_xy`` / ``goal_xy`` 可以覆盖场景里冻结的起终点，用来渲染
    "按档位重新采样"后的任务（坐标取自 docs/dmin_sweep.csv 的 resampled_* 列）。
    """
    resolved = config_module.load_resolved(
        algo, scenario_path, platform_path=platform, config_root=config_root
    )
    scenario = resolved.scenario
    scene = experiment.build_scene(scenario)
    if start_xy is not None and goal_xy is not None:
        scene = replace(
            scene,
            start_xy=(float(start_xy[0]), float(start_xy[1])),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            start_cell=astar.world_to_cell(start_xy, scene.resolution, scene.origin),
            goal_cell=astar.world_to_cell(goal_xy, scene.resolution, scene.origin),
        )
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
    # 场景层只声明物理半径；d_min 来自 algo，两者分开画
    physical_m = float(scenario["map"].get("inflation_m", 0.0))
    policy_m = float(setup.astar_params.d_min)
    content_hash = str(scenario.get("content_hash", "-"))

    _setup_fonts()
    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 7.5), constrained_layout=True)

    # 左图：栅格 + 物理半径层 + 策略裕度层 + 路径
    left.imshow(
        scene.occupancy.astype(float),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#ffffff", "#3b3b3b"]),
        interpolation="nearest",
    )
    overlay = np.zeros((rows, cols, 4), dtype=float)
    free = ~scene.occupancy
    collision = free & (scene.clearance + 1e-9 < physical_m)
    policy = free & (scene.clearance + 1e-9 >= physical_m) & (
        scene.clearance + 1e-9 < policy_m
    )
    overlay[collision] = (0.75, 0.05, 0.05, 0.45)   # 车身进不去
    overlay[policy] = (0.95, 0.65, 0.15, 0.35)      # 物理可行、但被 d_min 排除
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
    left.plot([], [], "s", color=(0.75, 0.05, 0.05, 0.45),
              label=f"物理半径 {physical_m:.2f} m（车身）")
    left.plot([], [], "s", color=(0.95, 0.65, 0.15, 0.35),
              label=f"策略裕度至 d_min {policy_m:.2f} m")
    grade = str(quality.get("grade", "-"))
    label_changes = quality.get("room_label_changes", "-")
    rooms = quality.get("rooms_visited", "-")
    info = (
        f"resolution = {res} m    栅格 {rows} x {cols}\n"
        f"content_hash = {content_hash[:12]}…\n"
        f"physical (scenario) = {physical_m:.2f} m   d_min (algo) = {policy_m:.2f} m\n"
        f"e_budget = {setup.astar_params.e_budget:.0f} J\n"
        f"难度 = {grade}  房间标签变化 = {label_changes}  房间数 = {rooms}"
    )
    if quality.get("group") == "narrow_door":
        info += (
            f"\n窄门宽度 = {quality.get('narrow_door_width_m')} m"
            f"（实测 {quality.get('measured_door_width_m')} m）"
            f"\n物理半径可通行 = {quality.get('feasible_at_physical_radius')}"
            f"  基准 d_min 可通行 = {quality.get('feasible_at_base_d_min')}"
        )
    if result.success:
        below = quality.get("clearance_below_090_ratio")
        info += (
            f"\n路径长度 = {result.length_m:.2f} m   能耗 = "
            f"{result.total_energy_j:.1f} J\n最小离障 = {result.clearance.min():.2f} m"
        )
        if below is not None:
            info += f"   clearance<0.9 占比 = {float(below):.2f}"
    else:
        info += f"\n规划失败: {result.reason}"
    left.text(
        0.02, 0.02, info, transform=left.transAxes, va="bottom", ha="left",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6"),
    )
    left.set_title(
        f"{scenario['id']}：栅格 + 物理半径 + 策略裕度 + A* 路径\n"
        f"难度 {grade} | d_min {policy_m:.2f} m | 房间标签变化 {label_changes} | "
        f"房间 {rooms} | res {res} m | hash {content_hash[:8]}",
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

    # 非默认 algo（例如 configs/dmin_sweep/dmin_058.yaml）时给文件名加后缀，
    # 便于同一场景在不同 d_min 档位下并存对比。
    if tag:
        suffix = f"__{tag}"
    else:
        suffix = "" if str(algo) == DEFAULT_ALGO else f"__{Path(algo).stem}"
    out_path = Path(out_dir) / f"{scenario['id']}{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return out_path


def _parse_xy(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    parts = [item.strip() for item in text.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"--start/--goal 需要 \"x,y\"，实际为 {text!r}")
    return float(parts[0]), float(parts[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.preview",
        description="场景预览图：栅格 + 物理半径 + 策略裕度 + 起终点 + A* 路径 + 地形",
    )
    parser.add_argument("scenarios", nargs="+", help="场景 YAML 路径（可给多个）")
    parser.add_argument("--algo", default=DEFAULT_ALGO)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--start", default=None, help="覆盖起点 x,y（世界坐标）")
    parser.add_argument("--goal", default=None, help="覆盖终点 x,y（世界坐标）")
    parser.add_argument("--tag", default=None, help="输出文件名后缀")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failed = 0
    for scenario_path in args.scenarios:
        try:
            start_xy = _parse_xy(args.start)
            goal_xy = _parse_xy(args.goal)
            out_path = render(
                scenario_path,
                algo=args.algo,
                platform=args.platform,
                config_root=args.config_root,
                out_dir=args.out_dir,
                dpi=args.dpi,
                start_xy=start_xy,
                goal_xy=goal_xy,
                tag=args.tag,
            )
        except config_module.ConfigError as exc:
            failed += 1
            print(f"FAIL {scenario_path}: {exc}", file=sys.stderr)
            continue
        print(f"OK   {out_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
