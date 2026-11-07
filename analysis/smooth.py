"""B 样条平滑的可视化与统计（交接文档 7.7 第 6、7 条）。

```bash
uv run python -m analysis.smooth --tier easy                 # easy 档全部场景
uv run python -m analysis.smooth --tier all --out-dir docs/smooth
```

产出：

1. 每个场景一张**三合一叠图**：原始栅格路径 / RDP 折线 / B 样条；
2. 每个场景一张**曲率剖面**：``kappa(s)`` 随弧长，画 ±``kappa_max`` 限值线；
3. 每个场景一张**平滑前后对比**：长度 / 能耗 / 最小 clearance；
4. ``docs/bspline_report.csv``：逐场景的平滑前后数值与平滑报告；
5. ``docs/bspline_feasibility.csv``：按难度档汇总的**曲率不可满足率**。

注意：回退到折线**不是"降级成功"，是曲率约束违反**——折线拐点曲率无界，
不是可跟踪的参考轨迹，不得交给局部规划器。因此报告口径以
``curvature_infeasible`` 为主指标，``produced_spline`` 与 ``curvature_satisfied``
分两列，不允许用前者掩盖后者。

统计口径与 ``core/bspline.py`` 的 ``SmoothingReport`` 一致，不做二次加工。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "smooth"
DEFAULT_ALGO = "configs/frozen_A.yaml"
REPORT_CSV = PROJECT_ROOT / "docs" / "bspline_report.csv"
FEASIBILITY_CSV = PROJECT_ROOT / "docs" / "bspline_feasibility.csv"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from core import astar, bspline  # noqa: E402
from runner import config as config_module  # noqa: E402
from runner import experiment  # noqa: E402

_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
)
COLOR_ASTAR = "#1f77b4"
COLOR_RDP = "#ff7f0e"
COLOR_SPLINE = "#2ca02c"
COLOR_FALLBACK = "#d62728"


def _setup_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.monospace"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class SceneSmoothing:
    """一个场景的完整平滑结果（绘图与统计共用）。"""

    scenario_id: str
    grade: str
    bottleneck_width_m: float
    astar: astar.PathResult
    smooth: bspline.SmoothPath
    occupancy: np.ndarray
    resolution: float
    origin: tuple[float, float]
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    d_min: float
    kappa_max: float
    e_budget: float
    physical_radius_m: float
    """物理外接圆半径，**从场景的 ``map.inflation_m`` 读**（= platform.body_radius）。

    旧口径用 ``d_min - 0.12`` 反推物理半径，只在基准档恰好等于 0.58；在 d_min 扫描的
    0.58 档会算出 0.46、1.10 档会算出 0.98，两档都错。物理半径是**平台层**物理量，
    与策略裕度无关，必须从场景读。
    """
    clearance_margin_m: float
    """曲率裕度前馈量 [m]；``0.0`` 表示未启用前馈（不额外预留）。"""
    control_spacing_m: float
    """控制点**目标**间距 [m]（配置值 ``bspline.control_spacing_m``）。

    注意：这是**目标值**，不是实际均距。实际均距由
    ``length / (n_control - 1)`` 决定，可能显著大于目标值——原因见
    ``core.bspline.control_points_for``：控制点优先取 **RDP 顶点子集**，
    而 RDP 顶点数往往少于 ``length / spacing + 1``，于是间距被 RDP 顶点
    锁住而无法达到目标。图例里两个数都要显示，避免把"目标"读成"实际"。
    """
    content_hash: str
    config_hash: str


def run_scene(
    scenario_path: str | Path,
    *,
    algo: str | Path = DEFAULT_ALGO,
    platform: str | Path | None = None,
    config_root: str | Path | None = None,
) -> SceneSmoothing:
    """跑一条完整链路：A* -> B 样条平滑。"""
    resolved = config_module.load_resolved(
        algo, scenario_path, platform_path=platform, config_root=config_root
    )
    scenario = resolved.scenario
    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    result = experiment.plan_path(scene, setup)
    if not result.success:
        raise RuntimeError(f"{scenario['id']}: A* 无解（{result.reason}），无法平滑")
    smooth = experiment.smooth_result(result, scene, setup)
    quality = scenario.get("quality", {}) or {}
    # 物理外接圆半径从**场景层**读（= platform.body_radius），不从 d_min 反推。
    physical_m = float(scenario.get("map", {}).get("inflation_m", 0.0))
    return SceneSmoothing(
        scenario_id=str(scenario["id"]),
        grade=str(quality.get("grade", "-")),
        bottleneck_width_m=float(quality.get("bottleneck_width_m", float("nan"))),
        astar=result,
        smooth=smooth,
        occupancy=scene.occupancy,
        resolution=scene.resolution,
        origin=scene.origin,
        start_xy=scene.start_xy,
        goal_xy=scene.goal_xy,
        d_min=float(setup.bspline_params.d_min),
        kappa_max=float(setup.astar_params.kappa_max),
        e_budget=float(setup.astar_params.e_budget),
        physical_radius_m=physical_m,
        clearance_margin_m=float(setup.clearance_margin_m),
        control_spacing_m=float(setup.bspline_params.control_spacing_m),
        content_hash=str(scenario.get("content_hash", "-")),
        config_hash=resolved.config_hash,
    )


def list_scenarios(
    tier: str | None, *, config_root: Path = CONFIG_ROOT
) -> list[Path]:
    """按难度档挑场景（档位来自场景文件自带的 quality.grade）。"""
    paths: list[Path] = []
    for path in sorted((config_root / "scenarios").glob("houseexpo_*.yaml")):
        if tier in (None, "all"):
            paths.append(path)
            continue
        doc = config_module.read_config(path)
        grade = str((doc.get("quality") or {}).get("grade", ""))
        if grade == tier:
            paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# 三类图
# ---------------------------------------------------------------------------


def _extent(item: SceneSmoothing) -> tuple[float, float, float, float]:
    rows, cols = item.occupancy.shape
    ox, oy = item.origin
    return (ox, ox + cols * item.resolution, oy, oy + rows * item.resolution)


def plot_overlay(item: SceneSmoothing, out_path: Path) -> Path:
    """图 a：原始栅格路径 / RDP 折线 / B 样条，三合一叠图。"""
    _setup_fonts()
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    extent = _extent(item)

    axis.imshow(
        item.occupancy.astype(float),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#ffffff", "#3b3b3b"]),
        interpolation="nearest",
    )
    rows, cols = item.occupancy.shape
    clearance = astar.clearance_field(item.occupancy, item.resolution)
    overlay = np.zeros((rows, cols, 4), dtype=float)
    free = ~item.occupancy
    # 深红 = 物理半径内（车身进不去）；橙 = 物理半径到 d_min 的策略裕度。
    # 物理半径取自场景 map.inflation_m，与 d_min 无关（d_min 是 algo 层变量）。
    physical = item.physical_radius_m
    collision = free & (clearance + 1e-9 < physical)
    policy = free & (clearance + 1e-9 >= physical) & (clearance + 1e-9 < item.d_min)
    overlay[collision] = (0.75, 0.05, 0.05, 0.30)
    overlay[policy] = (0.95, 0.65, 0.15, 0.30)
    axis.imshow(overlay, origin="lower", extent=extent, interpolation="nearest")

    axis.plot(
        item.astar.xy[:, 0], item.astar.xy[:, 1], "-",
        color=COLOR_ASTAR, linewidth=1.0, alpha=0.75,
        label=f"A* 栅格路径 {item.astar.length_m:.2f} m",
    )
    axis.plot(
        item.smooth.polyline[:, 0], item.smooth.polyline[:, 1], "--",
        color=COLOR_RDP, linewidth=1.4,
        label=f"RDP 折线（{item.smooth.report.rdp_vertices} 顶点）",
    )
    if item.smooth.is_spline and item.smooth.curve is not None:
        axis.plot(
            item.smooth.xy[:, 0], item.smooth.xy[:, 1], "-",
            color=COLOR_SPLINE, linewidth=2.4,
            label=f"B 样条 {item.smooth.length_m:.2f} m",
        )
        axis.plot(
            item.smooth.curve.control_points[:, 0],
            item.smooth.curve.control_points[:, 1],
            "o", color="#17becf", markersize=4.5,
            label=(
                f"控制点 {item.smooth.report.control_points} 个"
                f"（均距 {item.smooth.length_m / max(1, item.smooth.report.control_points - 1):.2f} m，"
                f"目标 {item.control_spacing_m:.2f} m）"
            ),
        )
    else:
        axis.plot(
            item.smooth.xy[:, 0], item.smooth.xy[:, 1], "-",
            color=COLOR_FALLBACK, linewidth=2.4,
            label=f"回退：折线（{item.smooth.report.reason}）",
        )

    axis.plot(*item.start_xy, "o", color="#2ca02c", markersize=10, label="起点")
    axis.plot(*item.goal_xy, "*", color="#d62728", markersize=17, label="终点")
    axis.plot([], [], "s", color=(0.75, 0.05, 0.05, 0.55),
              label=f"物理半径内 {item.physical_radius_m:.2f} m（车身）")
    margin_note = (
        f"（前馈 +{item.clearance_margin_m * 1000:.0f} mm）"
        if item.clearance_margin_m > 0
        else ""
    )
    axis.plot([], [], "s", color=(0.95, 0.65, 0.15, 0.55),
              label=f"策略裕度至 d_min {item.d_min:.2f} m{margin_note}")

    status = "B 样条" if item.smooth.is_spline else "回退折线（曲率违反）"
    info = (
        f"档位 = {item.grade}   瓶颈 = {item.bottleneck_width_m:.2f} m\n"
        f"物理半径 = {item.physical_radius_m:.2f} m   d_min = {item.d_min:.2f} m\n"
        f"kappa_max = {item.kappa_max:.2f} 1/m   前馈裕度 = "
        f"{item.clearance_margin_m * 1000:.0f} mm\n"
        f"状态 = {status}（{item.smooth.report.reason}）  轮数 = {item.smooth.report.rounds}\n"
        f"曲率约束 = "
        f"{'满足' if item.smooth.curvature_ok else '违反（不可作参考轨迹）'}\n"
        f"插入 = {item.smooth.report.inserted_points}  放稀 = "
        f"{item.smooth.report.removed_points}"
    )
    axis.text(
        0.02, 0.02, info, transform=axis.transAxes, va="bottom", ha="left",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.6"),
    )
    axis.set_title(
        f"{item.scenario_id}：栅格路径 / RDP 折线 / B 样条\n"
        f"hash {item.content_hash[:8]} | res {item.resolution} m", fontsize=11
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal")
    axis.legend(loc="upper right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


def plot_curvature(item: SceneSmoothing, out_path: Path) -> Path:
    """图 b：曲率随弧长剖面，画出 ±kappa_max 限值线。"""
    _setup_fonts()
    figure, axis = plt.subplots(figsize=(10, 4.6), constrained_layout=True)

    if item.smooth.is_spline:
        s = item.smooth.s
        curvature = item.smooth.curvature
        axis.plot(s, curvature, "-", color=COLOR_SPLINE, linewidth=1.6,
                  label="B 样条曲率 kappa(s)")
        axis.fill_between(s, curvature, 0, color=COLOR_SPLINE, alpha=0.15)
    else:
        s = item.smooth.s
        corners = item.smooth.polyline
        axis.plot(item.smooth.s, np.zeros_like(item.smooth.s), "-",
                  color=COLOR_FALLBACK, linewidth=1.2,
                  label="回退折线：段内曲率为 0")
        arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(corners, axis=0), axis=1))]
        )
        for value in arc[1:-1]:
            axis.axvline(value, color=COLOR_FALLBACK, linestyle=":", linewidth=1.2,
                         alpha=0.8)
        axis.plot([], [], ":", color=COLOR_FALLBACK,
                  label="顶点处曲率无界（不可导）")

    axis.axhline(item.kappa_max, color="#d62728", linestyle="--", linewidth=1.6,
                 label=f"+kappa_max = {item.kappa_max:.2f} 1/m")
    axis.axhline(-item.kappa_max, color="#d62728", linestyle="--", linewidth=1.6,
                 label=f"-kappa_max = {item.kappa_max:.2f} 1/m")
    axis.axhline(0.0, color="0.6", linewidth=0.8)

    max_kappa = item.smooth.max_abs_curvature
    shown = "无界（回退）" if np.isinf(max_kappa) else f"{max_kappa:.3f}"
    axis.set_title(
        f"{item.scenario_id}：曲率随弧长剖面\n"
        f"max|kappa| = {shown}   限值 {item.kappa_max:.2f} 1/m   "
        f"长度 {item.smooth.length_m:.2f} m",
        fontsize=11,
    )
    axis.set_xlabel("弧长 s [m]")
    axis.set_ylabel("曲率 kappa [1/m]")
    if item.smooth.is_spline:
        limit = max(float(np.max(np.abs(item.smooth.curvature))) * 1.25, item.kappa_max * 1.2)
        axis.set_ylim(-limit, limit)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


def plot_comparison(item: SceneSmoothing, out_path: Path) -> Path:
    """图 c：平滑前后的长度 / 能耗 / 最小 clearance 对比。"""
    _setup_fonts()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)

    before_length = item.astar.length_m
    after_length = item.smooth.length_m
    before_energy = item.astar.total_energy_j
    after_energy = item.smooth.energy_j
    before_clearance = float(item.astar.clearance.min())
    after_clearance = item.smooth.min_clearance
    before_kappa = float("nan")  # 折线曲率尖峰，不给出有限值
    after_kappa = item.smooth.max_abs_curvature

    panels = (
        ("路径长度 [m]", before_length, after_length, "{:.2f}"),
        ("能耗 [J]", before_energy, after_energy, "{:.1f}"),
        ("最小离障 [m]", before_clearance, after_clearance, "{:.3f}"),
    )
    colors_before = COLOR_ASTAR
    colors_after = COLOR_SPLINE if item.smooth.is_spline else COLOR_FALLBACK

    for axis, (title, before, after, fmt) in zip(axes, panels):
        bars = axis.bar(
            ["A* 折线", "平滑后"], [before, after],
            color=[colors_before, colors_after], width=0.55,
        )
        for bar, value in zip(bars, (before, after)):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                fmt.format(value),
                ha="center", va="bottom", fontsize=10,
            )
        axis.set_title(title, fontsize=11)
        axis.grid(axis="y", alpha=0.25)
        if title.startswith("最小离障"):
            axis.axhline(item.d_min, color="#d62728", linestyle="--", linewidth=1.4,
                         label=f"d_min = {item.d_min:.2f} m")
            axis.legend(fontsize=8)

    delta_length = after_length - before_length
    delta_energy = after_energy - before_energy
    delta_clearance = after_clearance - before_clearance
    footnote = (
        f"长度 {delta_length:+.2f} m（{delta_length / before_length * 100:+.1f}%）   "
        f"能耗 {delta_energy:+.1f} J   离障 {delta_clearance:+.3f} m"
    )
    figure.suptitle(
        f"{item.scenario_id}：平滑前后对比（{item.grade}，"
        f"{'B 样条' if item.smooth.is_spline else '回退折线'}）\n{footnote}",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# 统计表
# ---------------------------------------------------------------------------


REPORT_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "grade",
    "bottleneck_width_m",
    "physical_radius_m",
    "d_min_m",
    "clearance_margin_m",
    "kappa_max",
    "smooth_status",
    "smooth_reason",
    # 下面两列是**独立概念**，不允许用前者掩盖后者：
    # produced_spline = 几何链路是否产出了 B 样条曲线；
    # curvature_satisfied = 曲率硬约束是否真的成立（回退恒为 0）。
    "produced_spline",
    "curvature_satisfied",
    "curvature_infeasible",
    "trackable",
    "inserted_points",
    "removed_points",
    "depth",
    "rounds",
    "rdp_vertices",
    "control_points",
    "control_spacing_target_m",
    "control_spacing_actual_m",
    "astar_length_m",
    "smooth_length_m",
    "delta_length_m",
    "astar_energy_j",
    "smooth_energy_j",
    "delta_energy_j",
    "astar_min_clearance_m",
    "smooth_min_clearance_m",
    "smooth_max_abs_curvature",
    "energy_within_budget",
    "clearance_within_limit",
    "config_hash",
)


def _row_of(item: SceneSmoothing) -> dict[str, Any]:
    smooth = item.smooth
    return {
        "scenario_id": item.scenario_id,
        "grade": item.grade,
        "bottleneck_width_m": f"{item.bottleneck_width_m:.4f}",
        "physical_radius_m": f"{item.physical_radius_m:.3f}",
        "d_min_m": f"{item.d_min:.3f}",
        "clearance_margin_m": f"{item.clearance_margin_m:.4f}",
        "kappa_max": f"{item.kappa_max:.3f}",
        "smooth_status": smooth.status,
        "smooth_reason": smooth.report.reason,
        "produced_spline": int(smooth.is_spline),
        "curvature_satisfied": int(smooth.curvature_ok),
        "curvature_infeasible": int(smooth.curvature_infeasible),
        "trackable": int(smooth.trackable),
        "inserted_points": smooth.report.inserted_points,
        "removed_points": smooth.report.removed_points,
        "depth": smooth.report.depth,
        "rounds": smooth.report.rounds,
        "rdp_vertices": smooth.report.rdp_vertices,
        "control_points": smooth.report.control_points,
        "control_spacing_target_m": f"{item.control_spacing_m:.3f}",
        "control_spacing_actual_m": (
            f"{smooth.length_m / max(1, smooth.report.control_points - 1):.3f}"
        ),
        "astar_length_m": f"{item.astar.length_m:.4f}",
        "smooth_length_m": f"{smooth.length_m:.4f}",
        "delta_length_m": f"{smooth.length_m - item.astar.length_m:+.4f}",
        "astar_energy_j": f"{item.astar.total_energy_j:.3f}",
        "smooth_energy_j": f"{smooth.energy_j:.3f}",
        "delta_energy_j": f"{smooth.energy_j - item.astar.total_energy_j:+.3f}",
        "astar_min_clearance_m": f"{float(item.astar.clearance.min()):.4f}",
        "smooth_min_clearance_m": f"{smooth.min_clearance:.4f}",
        "smooth_max_abs_curvature": (
            "inf" if np.isinf(smooth.max_abs_curvature)
            else f"{smooth.max_abs_curvature:.4f}"
        ),
        "energy_within_budget": int(smooth.energy_j <= item.e_budget + 1e-9),
        "clearance_within_limit": int(smooth.min_clearance >= item.d_min - 1e-9),
        "config_hash": item.config_hash,
    }


def write_report(items: Sequence[SceneSmoothing], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for item in items:
            writer.writerow(_row_of(item))
    return path


def summarize_feasibility(
    items: Sequence[SceneSmoothing],
) -> list[dict[str, Any]]:
    """按难度档汇总**曲率可满足性**（7.7 第 6 条，2026-10-29 重新定性）。

    主指标是 ``curvature_infeasible_rate``，不是"几何链路成功率"：
    回退折线的曲率无界，必须计入不可满足，不能用"产出了结果"掩盖约束违反。
    """
    rows: list[dict[str, Any]] = []
    grades = sorted({item.grade for item in items})
    for grade in grades:
        subset = [item for item in items if item.grade == grade]
        stats = bspline.feasibility_stats([item.smooth.report for item in subset])
        lengths_before = [item.astar.length_m for item in subset]
        lengths_after = [item.smooth.length_m for item in subset]
        energy_before = [item.astar.total_energy_j for item in subset]
        energy_after = [item.smooth.energy_j for item in subset]
        trackable = [item for item in subset if item.smooth.trackable]
        rows.append(
            {
                "grade": grade,
                "tasks": stats["tasks"],
                "curvature_satisfied": stats["curvature_satisfied"],
                "curvature_infeasible": stats["curvature_infeasible"],
                "curvature_infeasible_rate": f"{stats['curvature_infeasible_rate']:.3f}",
                "inserted": stats["inserted"],
                "insertion_rate": f"{stats['insertion_rate']:.3f}",
                "pure_spline_rate": f"{stats['pure_spline_rate']:.3f}",
                "median_length_before_m": f"{np.median(lengths_before):.3f}",
                "median_length_after_m": f"{np.median(lengths_after):.3f}",
                "median_energy_before_j": f"{np.median(energy_before):.1f}",
                "median_energy_after_j": f"{np.median(energy_after):.1f}",
                "spline_median_max_kappa": (
                    f"{np.median([i.smooth.max_abs_curvature for i in trackable]):.3f}"
                    if trackable
                    else "-"
                ),
                "min_clearance_after_m": (
                    f"{min(i.smooth.min_clearance for i in subset):.3f}"
                ),
            }
        )
    return rows


FEASIBILITY_COLUMNS: tuple[str, ...] = (
    "grade",
    "tasks",
    "curvature_satisfied",
    "curvature_infeasible",
    "curvature_infeasible_rate",
    "inserted",
    "insertion_rate",
    "pure_spline_rate",
    "median_length_before_m",
    "median_length_after_m",
    "median_energy_before_j",
    "median_energy_after_j",
    "spline_median_max_kappa",
    "min_clearance_after_m",
)


def write_feasibility(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FEASIBILITY_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def render_scene(item: SceneSmoothing, out_dir: Path) -> list[Path]:
    return [
        plot_overlay(item, out_dir / f"{item.scenario_id}__overlay.png"),
        plot_curvature(item, out_dir / f"{item.scenario_id}__curvature.png"),
        plot_comparison(item, out_dir / f"{item.scenario_id}__compare.png"),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.smooth",
        description="B 样条平滑：三类图 + 曲率可满足性统计 + 平滑前后对比表",
    )
    parser.add_argument("--tier", default="all",
                        help="难度档：easy / medium / hard / all（默认 all）")
    parser.add_argument("--scenario", nargs="*", default=None,
                        help="直接指定场景 YAML（给了就忽略 --tier）")
    parser.add_argument("--algo", default=DEFAULT_ALGO)
    parser.add_argument("--platform", default=None)
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--report", default=str(REPORT_CSV))
    parser.add_argument("--feasibility-report", default=str(FEASIBILITY_CSV))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-plots", action="store_true", help="只出表，不出图")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_root = Path(args.config_root) if args.config_root else CONFIG_ROOT
    if args.scenario:
        paths = [Path(item) for item in args.scenario]
    else:
        paths = list_scenarios(args.tier, config_root=config_root)
    if not paths:
        print(f"没有找到场景（tier={args.tier}）", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    items: list[SceneSmoothing] = []
    failures = 0
    for path in paths:
        try:
            item = run_scene(
                path, algo=args.algo, platform=args.platform, config_root=args.config_root
            )
        except (RuntimeError, config_module.ConfigError) as exc:
            failures += 1
            print(f"FAIL {path}: {exc}", file=sys.stderr)
            continue
        items.append(item)
        if not args.no_plots:
            for output in render_scene(item, out_dir):
                print(f"OK   {output}")
        kappa_text = (
            "inf" if np.isinf(item.smooth.max_abs_curvature)
            else f"{item.smooth.max_abs_curvature:.3f}"
        )
        print(
            f"{item.scenario_id} [{item.grade}] "
            f"d_min={item.d_min:.2f}(+{item.clearance_margin_m * 1000:.0f}mm) "
            f"{item.smooth.status}/{item.smooth.report.reason} "
            f"curvature_{'OK' if item.smooth.curvature_ok else 'INFEASIBLE'} "
            f"insert={item.smooth.report.inserted_points} "
            f"thin={item.smooth.report.removed_points} "
            f"kappa={kappa_text} "
            f"clr={item.smooth.min_clearance:.3f} "
            f"len {item.astar.length_m:.2f}->{item.smooth.length_m:.2f}"
        )

    if items:
        print(f"report: {write_report(items, Path(args.report))}")
        summary = summarize_feasibility(items)
        print(f"feasibility: {write_feasibility(summary, Path(args.feasibility_report))}")
        header = (
            f"{'档位':<8}{'任务':>4}{'满足':>5}{'违反':>5}"
            f"{'违反率':>8}{'插入率':>8}{'一次通过':>9}"
        )
        print(header)
        for row in summary:
            print(
                f"{row['grade']:<8}{row['tasks']:>4}{row['curvature_satisfied']:>5}"
                f"{row['curvature_infeasible']:>5}"
                f"{row['curvature_infeasible_rate']:>8}"
                f"{row['insertion_rate']:>8}{row['pure_spline_rate']:>9}"
            )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
