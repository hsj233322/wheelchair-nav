"""第 3 章结果生成：三张表 + 两张图（不新增任何算法能力）。

```bash
uv run python -m analysis.chapter3                # 三件事全跑
uv run python -m analysis.chapter3 --only table   # 规划器对比表
uv run python -m analysis.chapter3 --only pareto  # (长度, 能耗) 前沿
uv run python -m analysis.chapter3 --only soft    # 软惩罚 vs 硬约束
```

三件事分别对应：

1. **规划器对比表**：传统 A*（仅长度）/ 能耗加权 A*（ESA*） 与本文约束化 A*，
   在同一场景集、同一 seed、参数冻结下对比。基线按文献做法只保证**物理半径**
   （``d_min = body_radius = 0.58 m``），本文把**策略半径** 0.70 m 写成硬约束；
   因此表里的"约束不可行任务数"就是"不满足 0.70 m 策略裕度"的任务数。
2. **(长度, 能耗) Pareto 前沿**：用 ``collect_pareto=True`` 取算法自己的非支配集，
   叠加"多同伦类候选路线"（经房间航路点枚举）的点云，并标出给定偏好下的选点。
3. **软惩罚 vs 硬约束**：``enforce_dmin=False`` + 扫 ``soft_penalty_weight``。

口径提醒（论文必须写）：

* 到达时间按**参考轨迹口径** ``t = L / v_cruise``（全局层是恒速参考轨迹，不是控制器输出）；
* 违反率按**单元中心 clearance** 判定（与全局层既有口径一致），不做连续扫掠；
* 软惩罚模式下"违反率"与"物理穿行（clearance < body_radius）"分开报。
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_ALGO = "configs/frozen_A.yaml"
REVIEW_DIR = PROJECT_ROOT / "docs" / "review"
OUT_DIR = PROJECT_ROOT / "docs" / "chapter3"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from core import astar, energy as energy_module  # noqa: E402
from env import scene_quality  # noqa: E402
from runner import config as config_module  # noqa: E402
from runner import experiment  # noqa: E402

_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
)
COLORS = {
    "length": "#1f77b4",
    "esa": "#ff7f0e",
    "constrained": "#2ca02c",
    "soft": "#d62728",
    "route": "#7f7f7f",
}
BOUNDARY_TOL = 1e-6


def _setup_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.monospace"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _git_commit() -> str:
    """当前 git 短哈希（只用于报告，取不到就返回 "-"）。"""
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "-"


def scenario_paths(tier: str = "all", *, config_root: Path = CONFIG_ROOT) -> list[Path]:
    paths: list[Path] = []
    for path in sorted((config_root / "scenarios").glob("houseexpo_*.yaml")):
        if tier == "all":
            paths.append(path)
            continue
        doc = config_module.read_config(path)
        if str((doc.get("quality") or {}).get("grade", "")) == tier:
            paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# 一次规划 = 一条记录
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanRecord:
    """一个 (规划器, 场景) 的输出与约束评估。"""

    planner: str
    planner_label: str
    scenario_id: str
    grade: str
    algo_config: str
    config_hash: str
    seed: int
    e_budget_used_j: float
    d_min_search_m: float
    """搜索时使用的离障门槛（基线 0.58 / 本文 0.70）。"""
    d_min_ref_m: float
    """评估"约束可行"用的**本文策略半径**（恒为 0.70）。"""
    body_radius_m: float
    success: bool
    reason: str
    length_m: float
    energy_j: float
    time_s: float
    min_clearance_m: float
    mean_clearance_m: float
    violation_length_m: float
    violation_rate: float
    """路径长度加权、``clearance < d_min_ref`` 的比例。"""
    physical_violation_length_m: float
    physical_violation_rate: float
    """``clearance < body_radius`` 的比例（穿墙/穿车体）。"""
    feasible: bool
    feasible_at_physical: bool
    pareto_size: int
    soft_penalty: float
    expanded: int
    cells: np.ndarray = field(repr=False)
    xy: np.ndarray = field(repr=False)
    clearance: np.ndarray = field(repr=False)
    step_length: np.ndarray = field(repr=False)

    @property
    def is_physical(self) -> bool:
        return self.physical_violation_length_m <= 1e-9


def _evaluate(
    *,
    planner: str,
    planner_label: str,
    scenario: dict[str, Any],
    resolved: Any,
    result: astar.PathResult,
    body_radius: float,
    d_min_ref: float,
    d_min_search: float,
    e_budget_used: float,
    cruise_speed: float,
    seed: int,
    pareto_size: int = 0,
) -> PlanRecord:
    """把一条 ``PathResult`` 折算成表里的指标（违反率按长度加权）。"""
    grade = str((scenario.get("quality") or {}).get("grade", "-"))
    if not result.success or result.step_length.size == 0:
        return PlanRecord(
            planner=planner,
            planner_label=planner_label,
            scenario_id=str(scenario["id"]),
            grade=grade,
            algo_config=str(resolved.algo.get("id", "-")),
            config_hash=str(resolved.config_hash),
            seed=int(seed),
            e_budget_used_j=float(e_budget_used),
            d_min_search_m=float(d_min_search),
            d_min_ref_m=float(d_min_ref),
            body_radius_m=float(body_radius),
            success=False,
            reason=str(result.reason),
            length_m=0.0,
            energy_j=0.0,
            time_s=0.0,
            min_clearance_m=0.0,
            mean_clearance_m=0.0,
            violation_length_m=0.0,
            violation_rate=0.0,
            physical_violation_length_m=0.0,
            physical_violation_rate=0.0,
            feasible=False,
            feasible_at_physical=False,
            pareto_size=int(pareto_size),
            soft_penalty=0.0,
            expanded=int(result.expanded),
            cells=np.zeros((0, 2), dtype=np.int64),
            xy=np.zeros((0, 2), dtype=float),
            clearance=np.zeros(0),
            step_length=np.zeros(0),
        )

    steps = np.asarray(result.step_length, dtype=float)
    clear = np.asarray(result.clearance, dtype=float)
    total = float(steps.sum())
    length_weights = steps if total > 0 else np.ones_like(steps)
    viol_mask = clear < float(d_min_ref) - BOUNDARY_TOL
    phys_mask = clear < float(body_radius) - BOUNDARY_TOL
    viol_length = float(length_weights[viol_mask].sum())
    phys_length = float(length_weights[phys_mask].sum())
    min_clearance = float(clear.min()) if clear.size else 0.0
    feasible = bool(
        min_clearance + BOUNDARY_TOL >= d_min_ref
        and float(result.total_energy_j) <= float(e_budget_used) + 1e-9
    )
    return PlanRecord(
        planner=planner,
        planner_label=planner_label,
        scenario_id=str(scenario["id"]),
        grade=grade,
        algo_config=str(resolved.algo.get("id", "-")),
        config_hash=str(resolved.config_hash),
        seed=int(seed),
        e_budget_used_j=float(e_budget_used),
        d_min_search_m=float(d_min_search),
        d_min_ref_m=float(d_min_ref),
        body_radius_m=float(body_radius),
        success=True,
        reason=str(result.reason),
        length_m=float(result.length_m),
        energy_j=float(result.total_energy_j),
        time_s=float(result.length_m) / float(cruise_speed),
        min_clearance_m=min_clearance,
        mean_clearance_m=float(clear.mean()) if clear.size else 0.0,
        violation_length_m=viol_length,
        violation_rate=viol_length / total if total > 0 else 0.0,
        physical_violation_length_m=phys_length,
        physical_violation_rate=phys_length / total if total > 0 else 0.0,
        feasible=feasible,
        feasible_at_physical=bool(min_clearance + BOUNDARY_TOL >= body_radius),
        pareto_size=int(pareto_size),
        soft_penalty=float(result.soft_penalty),
        expanded=int(result.expanded),
        cells=np.asarray(result.cells, dtype=np.int64),
        xy=np.asarray(result.xy, dtype=float),
        clearance=clear,
        step_length=steps,
    )


def _plan(
    scene: experiment.Scene,
    setup: experiment.PlannerSetup,
    params: astar.AStarParams,
) -> astar.PathResult:
    return astar.find_path(
        scene.occupancy,
        scene.resolution,
        scene.start_xy,
        scene.goal_xy,
        params,
        energy_params=setup.energy_params,
        speed=setup.cruise_speed,
        c_rr=scene.c_rr,
        slope_rad=scene.slope_rad,
        origin=scene.origin,
        clearance=scene.clearance,
    )


# ---------------------------------------------------------------------------
# 1. 三种规划器对比
# ---------------------------------------------------------------------------

PLANNERS: tuple[tuple[str, str, str], ...] = (
    ("length", "传统 A*（仅长度）", "configs/chapter3/baseline_length_a.yaml"),
    ("esa", "能耗加权 A*（ESA*）", "configs/chapter3/baseline_esa_a.yaml"),
    ("constrained", "本文约束化 A*", DEFAULT_ALGO),
)


def run_planner_scenario(
    scenario_path: str,
    planner: str,
    label: str,
    algo: str,
    *,
    config_root: str,
    seed: int,
    d_min_ref: float,
) -> list[PlanRecord]:
    """跑一个 (规划器, 场景)，返回 1~2 条记录（ESA* 会额外跑一次紧能耗预算）。"""
    root = Path(config_root)
    resolved = config_module.load_resolved(algo, Path(scenario_path), config_root=root)
    scenario = resolved.scenario
    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    body_radius = float(resolved.platform["body_radius"])
    cruise = float(setup.cruise_speed)
    base = setup.astar_params

    records: list[PlanRecord] = []
    if planner == "esa":
        # 1) 收前沿（非支配集），取能耗最小点
        front_result = _plan(
            scene,
            setup,
            astar.AStarParams(
                e_budget=base.e_budget,
                d_min=base.d_min,
                kappa_max=base.kappa_max,
                kappa_window_m=base.kappa_window_m,
                max_labels_per_state=base.max_labels_per_state,
                max_expansions=base.max_expansions,
                collect_pareto=True,
            ),
        )
        if not front_result.success or front_result.pareto_energy_j.size == 0:
            records.append(
                _evaluate(
                    planner=planner, planner_label=label, scenario=scenario,
                    resolved=resolved, result=front_result, body_radius=body_radius,
                    d_min_ref=d_min_ref, d_min_search=float(base.d_min),
                    e_budget_used=float(base.e_budget), cruise_speed=cruise,
                    seed=seed,
                )
            )
            return records
        index = int(np.argmin(front_result.pareto_energy_j))
        e_min = float(front_result.pareto_energy_j[index])
        # 2) 把能耗预算压到前沿最小能耗 -> 反解出那条路径本身
        tight = _plan(
            scene,
            setup,
            astar.AStarParams(
                e_budget=e_min * (1.0 + 1e-9) + 1e-9,
                d_min=base.d_min,
                kappa_max=base.kappa_max,
                kappa_window_m=base.kappa_window_m,
                max_labels_per_state=base.max_labels_per_state,
                max_expansions=base.max_expansions,
            ),
        )
        if not tight.success:
            tight = front_result
        records.append(
            _evaluate(
                planner=planner, planner_label=label, scenario=scenario,
                resolved=resolved, result=tight, body_radius=body_radius,
                d_min_ref=d_min_ref, d_min_search=float(base.d_min),
                e_budget_used=e_min, cruise_speed=cruise, seed=seed,
                pareto_size=int(front_result.pareto_length_m.size),
            )
        )
        return records

    result = _plan(scene, setup, base)
    records.append(
        _evaluate(
            planner=planner, planner_label=label, scenario=scenario, resolved=resolved,
            result=result, body_radius=body_radius, d_min_ref=d_min_ref,
            d_min_search=float(base.d_min), e_budget_used=float(base.e_budget),
            cruise_speed=cruise, seed=seed,
            pareto_size=int(result.pareto_length_m.size),
        )
    )
    return records


TASK_COLUMNS: tuple[str, ...] = (
    "planner",
    "planner_label",
    "scenario_id",
    "grade",
    "algo_config",
    "config_hash",
    "seed",
    "success",
    "reason",
    "d_min_search_m",
    "d_min_ref_m",
    "body_radius_m",
    "e_budget_used_j",
    "length_m",
    "energy_j",
    "time_s",
    "min_clearance_m",
    "mean_clearance_m",
    "violation_length_m",
    "violation_rate",
    "physical_violation_length_m",
    "physical_violation_rate",
    "feasible",
    "feasible_at_physical",
    "pareto_size",
    "soft_penalty",
    "expanded",
)


def _task_row(record: PlanRecord) -> dict[str, Any]:
    return {
        "planner": record.planner,
        "planner_label": record.planner_label,
        "scenario_id": record.scenario_id,
        "grade": record.grade,
        "algo_config": record.algo_config,
        "config_hash": record.config_hash,
        "seed": record.seed,
        "success": int(record.success),
        "reason": record.reason,
        "d_min_search_m": f"{record.d_min_search_m:.3f}",
        "d_min_ref_m": f"{record.d_min_ref_m:.3f}",
        "body_radius_m": f"{record.body_radius_m:.3f}",
        "e_budget_used_j": f"{record.e_budget_used_j:.3f}",
        "length_m": f"{record.length_m:.4f}",
        "energy_j": f"{record.energy_j:.3f}",
        "time_s": f"{record.time_s:.3f}",
        "min_clearance_m": f"{record.min_clearance_m:.4f}",
        "mean_clearance_m": f"{record.mean_clearance_m:.4f}",
        "violation_length_m": f"{record.violation_length_m:.4f}",
        "violation_rate": f"{record.violation_rate:.4f}",
        "physical_violation_length_m": f"{record.physical_violation_length_m:.4f}",
        "physical_violation_rate": f"{record.physical_violation_rate:.4f}",
        "feasible": int(record.feasible),
        "feasible_at_physical": int(record.feasible_at_physical),
        "pareto_size": record.pareto_size,
        "soft_penalty": f"{record.soft_penalty:.4f}",
        "expanded": record.expanded,
    }


def write_csv(rows: Sequence[dict[str, Any]], path: Path, columns: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _median(values: Sequence[float]) -> float:
    data = [v for v in values if np.isfinite(v)]
    return float(np.median(data)) if data else float("nan")


SUMMARY_COLUMNS: tuple[str, ...] = (
    "planner",
    "planner_label",
    "algo_config",
    "config_hash_ref_scenario",
    "tasks",
    "success",
    "length_median_m",
    "length_total_m",
    "energy_median_j",
    "energy_total_j",
    "time_median_s",
    "min_clearance_min_m",
    "min_clearance_median_m",
    "violation_rate_length_weighted",
    "violation_tasks",
    "physical_violation_tasks",
    "feasible_tasks",
    "feasible_rate",
    "pareto_size_median",
)


def summarise_planners(
    records: Sequence[PlanRecord], *, ref_scenario: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for planner, label, algo in PLANNERS:
        subset = [r for r in records if r.planner == planner]
        if not subset:
            continue
        total_length = sum(r.length_m for r in subset)
        viol_length = sum(r.violation_length_m for r in subset)
        ref = next(
            (r for r in subset if r.scenario_id == ref_scenario), subset[0]
        )
        feasible = sum(1 for r in subset if r.feasible)
        rows.append(
            {
                "planner": planner,
                "planner_label": label,
                "algo_config": algo,
                "config_hash_ref_scenario": f"{ref.config_hash}",
                "tasks": len(subset),
                "success": sum(1 for r in subset if r.success),
                "length_median_m": f"{_median([r.length_m for r in subset]):.4f}",
                "length_total_m": f"{total_length:.4f}",
                "energy_median_j": f"{_median([r.energy_j for r in subset]):.3f}",
                "energy_total_j": f"{sum(r.energy_j for r in subset):.3f}",
                "time_median_s": f"{_median([r.time_s for r in subset]):.3f}",
                "min_clearance_min_m": (
                    f"{min(r.min_clearance_m for r in subset):.4f}"
                    if subset
                    else "nan"
                ),
                "min_clearance_median_m": (
                    f"{_median([r.min_clearance_m for r in subset]):.4f}"
                ),
                "violation_rate_length_weighted": (
                    f"{(viol_length / total_length if total_length else 0.0):.4f}"
                ),
                "violation_tasks": sum(1 for r in subset if r.violation_length_m > 1e-9),
                "physical_violation_tasks": sum(
                    1 for r in subset if r.physical_violation_length_m > 1e-9
                ),
                "feasible_tasks": feasible,
                "feasible_rate": f"{feasible / len(subset):.4f}",
                "pareto_size_median": f"{_median([r.pareto_size for r in subset]):.0f}",
            }
        )
    return rows


def format_planner_table(rows: Sequence[dict[str, Any]], *, d_min_ref: float) -> str:
    header = (
        f"{'规划器':<22}{'任务':>4}{'成功':>5}{'长度中位[m]':>12}{'能耗中位[J]':>12}"
        f"{'到达时间中位[s]':>15}{'最小离障最小[m]':>15}"
        f"{'违反率(按长度)':>15}{'不可行任务':>11}{'物理穿行':>9}"
    )
    lines = [
        f"第 3 章规划器对比（约束口径：clearance >= d_min = {d_min_ref:.2f} m，"
        f"body_radius = 0.58 m）",
        header,
        "-" * len(header),
    ]
    for row in rows:
        lines.append(
            f"{row['planner_label']:<22}{row['tasks']:>4}{row['success']:>5}"
            f"{row['length_median_m']:>12}{row['energy_median_j']:>12}"
            f"{row['time_median_s']:>15}{row['min_clearance_min_m']:>15}"
            f"{row['violation_rate_length_weighted']:>15}"
            f"{row['violation_tasks']:>11}{row['physical_violation_tasks']:>9}"
        )
    return "\n".join(lines)


def plot_planner_table(
    records: Sequence[PlanRecord], out_path: Path, *, d_min_ref: float
) -> Path:
    """图 1：三种规划器的（长度, 能耗）逐任务散点 + 约束可行性着色。"""
    _setup_fonts()
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    axis = axes[0]
    for planner, label, _algo in PLANNERS:
        subset = [r for r in records if r.planner == planner and r.success]
        if not subset:
            continue
        feasible = [r for r in subset if r.feasible]
        infeasible = [r for r in subset if not r.feasible]
        axis.scatter(
            [r.length_m for r in feasible], [r.energy_j for r in feasible],
            s=46, color=COLORS[planner], marker="o", label=f"{label}（满足约束）",
            zorder=3,
        )
        axis.scatter(
            [r.length_m for r in infeasible], [r.energy_j for r in infeasible],
            s=54, color=COLORS[planner], marker="x",
            linewidths=1.6, label=f"{label}（违反 {d_min_ref:.2f} m）", zorder=3,
        )
    axis.set_xlabel("路径长度 [m]")
    axis.set_ylabel("电能 [J]")
    axis.set_title("逐任务：路径长度 vs 能耗（× = 不满足策略安全半径）", fontsize=11)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1]
    labels, viol_rates, physical_rates = [], [], []
    for planner, label, _algo in PLANNERS:
        subset = [r for r in records if r.planner == planner]
        if not subset:
            continue
        labels.append(label)
        viol_rates.append(
            sum(r.violation_length_m for r in subset)
            / max(1e-12, sum(r.length_m for r in subset))
        )
        physical_rates.append(
            sum(r.physical_violation_length_m for r in subset)
            / max(1e-12, sum(r.length_m for r in subset))
        )
    positions = np.arange(len(labels))
    axis.bar(positions - 0.18, viol_rates, width=0.36, color="#ff7f0e",
             label=f"违反策略半径 {d_min_ref:.2f} m")
    axis.bar(positions + 0.18, physical_rates, width=0.36, color="#d62728",
             label="物理穿行（< body_radius 0.58 m）")
    for index, value in enumerate(viol_rates):
        axis.text(positions[index] - 0.18, value, f"{value:.3f}", ha="center",
                  va="bottom", fontsize=9)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=9)
    axis.set_ylabel("按长度加权的违反率")
    axis.set_title("约束违反率（跨 20 个任务加总）", fontsize=11)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# 2. (长度, 能耗) Pareto 前沿
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteCandidate:
    """一条**原图上的合法路径**（多同伦候选）。

    枚举方式：先跑一次 A*，把这条路线**最窄处**的一小段在栅格**副本**上临时填成障碍，
    再跑一次；重复到无解或达到条数上限。因为只在副本上**增加**障碍：

    * 每条候选在原图上都满足全部硬约束（``clearance >= d_min``、能耗预算、曲率窗口）；
    * 离障按**原图**的距离场计算，与主路线同一口径；
    * 不存在"两段拼接、连接处曲率不可行"的伪候选。

    因此候选点云里的每个 (长度, 能耗) 都对应一条真实可行路径，
    "前沿退化"就不是枚举artifact，而是这些户型任务的真实性质。
    """

    route_index: int
    length_m: float
    energy_j: float
    energy_per_m: float
    min_clearance_m: float
    feasible: bool
    cells: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    step_length: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    xy: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2)))

    def explicit_segments(self) -> tuple[np.ndarray, np.ndarray]:
        """这条候选路线的 ``(cells, step_length)``，用于按路段长度统计地形占比。"""
        return np.asarray(self.cells, dtype=np.int64), np.asarray(self.step_length, dtype=float)


def enumerate_routes(
    scene: experiment.Scene,
    setup: experiment.PlannerSetup,
    params: astar.AStarParams,
    *,
    d_min_ref: float,
    max_routes: int = 6,
    block_radius_cells: int = 1,
) -> list[RouteCandidate]:
    """多同伦候选路线枚举（只用 ``core.astar.find_path``，不新增算法能力）。"""
    occupancy = np.asarray(scene.occupancy, dtype=bool).copy()
    clearance = scene.clearance
    start_cell = astar.world_to_cell(scene.start_xy, scene.resolution, scene.origin)
    goal_cell = astar.world_to_cell(scene.goal_xy, scene.resolution, scene.origin)
    radius = max(0, int(block_radius_cells))
    candidates: list[RouteCandidate] = []
    for index in range(max_routes):
        result = astar.find_path(
            occupancy, scene.resolution, scene.start_xy, scene.goal_xy, params,
            energy_params=setup.energy_params, speed=setup.cruise_speed,
            c_rr=scene.c_rr, slope_rad=scene.slope_rad, origin=scene.origin,
            clearance=clearance,
        )
        if not result.success:
            break
        min_clearance = (
            float(result.clearance.min()) if result.clearance.size else 0.0
        )
        candidates.append(
            RouteCandidate(
                route_index=index,
                length_m=float(result.length_m),
                energy_j=float(result.total_energy_j),
                energy_per_m=(
                    float(result.total_energy_j) / float(result.length_m)
                    if result.length_m > 0
                    else float("nan")
                ),
                min_clearance_m=min_clearance,
                feasible=bool(min_clearance + BOUNDARY_TOL >= d_min_ref),
                cells=np.asarray(result.cells, dtype=np.int64),
                step_length=np.asarray(result.step_length, dtype=float),
                xy=np.asarray(result.xy, dtype=float),
            )
        )
        cells = np.asarray(result.cells, dtype=np.int64)
        if cells.size == 0:
            break
        values = clearance[cells[:, 0], cells[:, 1]]
        narrowest = int(np.argmin(values))
        blocked = 0
        for row, col in cells[max(0, narrowest - 2): narrowest + 3]:
            if (int(row), int(col)) in (start_cell, goal_cell):
                continue
            occupancy[
                max(0, row - radius): row + radius + 1,
                max(0, col - radius): col + radius + 1,
            ] = True
            blocked += 1
        if blocked == 0:
            break
    return candidates


def pareto_front(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """(长度, 能耗) 的非支配前沿（按长度升序、能耗严格下降）。"""
    ordered = sorted(points)
    front: list[tuple[float, float]] = []
    for length, energy in ordered:
        if not front or energy < front[-1][1] - 1e-9:
            front.append((length, energy))
    return front


def _preference_point(
    front: Sequence[tuple[float, float]], beta: float
) -> tuple[float, float]:
    """给定偏好 β 的选点：min (1-β)·L/L0 + β·E/E0（归一化到前沿端点）。"""
    lengths = [p[0] for p in front]
    energies = [p[1] for p in front]
    l0, l1 = min(lengths), max(lengths)
    e0, e1 = min(energies), max(energies)
    l_span = (l1 - l0) or 1.0
    e_span = (e1 - e0) or 1.0
    return min(
        front,
        key=lambda p: (1.0 - beta) * (p[0] - l0) / l_span + beta * (p[1] - e0) / e_span,
    )


PARETO_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "grade",
    "source",
    "route_index",
    "length_m",
    "energy_j",
    "energy_per_m",
    "min_clearance_m",
    "on_front",
    "feasible",
)


def collect_pareto_data(
    scenario_path: str,
    *,
    config_root: str,
    algo: str = DEFAULT_ALGO,
    d_min_ref: float = 0.7,
) -> dict[str, Any]:
    """一个场景的 Pareto 数据：算法前沿 + 多同伦候选点云 + 偏好选点。"""
    root = Path(config_root)
    resolved = config_module.load_resolved(algo, Path(scenario_path), config_root=root)
    scenario = resolved.scenario
    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    base = setup.astar_params
    front_result = _plan(
        scene,
        setup,
        astar.AStarParams(
            e_budget=base.e_budget, d_min=base.d_min, kappa_max=base.kappa_max,
            kappa_window_m=base.kappa_window_m,
            max_labels_per_state=base.max_labels_per_state,
            max_expansions=base.max_expansions, collect_pareto=True,
        ),
    )
    algo_front = [
        (float(l), float(e))
        for l, e in zip(front_result.pareto_length_m, front_result.pareto_energy_j)
    ]
    routes = enumerate_routes(scene, setup, base, d_min_ref=d_min_ref)
    return {
        "scenario_id": str(scenario["id"]),
        "grade": str((scenario.get("quality") or {}).get("grade", "-")),
        "config_hash": str(resolved.config_hash),
        "algo_front": algo_front,
        "routes": routes,
        "length_m": float(front_result.length_m) if front_result.success else 0.0,
        "energy_j": (
            float(front_result.total_energy_j) if front_result.success else 0.0
        ),
    }


def synthetic_two_route_dataset() -> dict[str, Any]:
    """**合成**双通道地图：上门洞短但高阻、下门洞长但低阻。

    用途只有一个：展示"非退化前沿"长什么样（真实 20 个户型场景的前沿全部退化为
    单点）。它**不参与任何指标统计**，图上也标注为合成。
    """
    res = 0.1
    rows, cols = 40, 80
    occ = np.zeros((rows, cols), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    occ[:, 40] = True
    occ[15:18, 40] = False      # 上门洞（短，高阻）
    occ[30:36, 40] = False      # 下门洞（长，低阻）
    clearance = astar.clearance_field(occ, res)
    c_rr = np.full((rows, cols), 0.015)
    c_rr[:20, :] = 0.06
    c_rr[20:30, :] = 0.03
    params = energy_module.EnergyParams(mass=100.0, g=9.81, eta=0.85, p_aux=15.0)
    result = astar.find_path(
        occ, res, (0.4, 2.0), (7.6, 2.0),
        astar.AStarParams(e_budget=1e9, d_min=0.0, kappa_max=1.5,
                          kappa_window_m=0.6, collect_pareto=True),
        energy_params=params, speed=1.0, c_rr=c_rr, slope_rad=0.0,
        origin=(0.0, 0.0), clearance=clearance,
    )
    return {
        "scenario_id": "synthetic_two_doors",
        "grade": "合成",
        "config_hash": "-",
        "algo_front": [
            (float(l), float(e))
            for l, e in zip(result.pareto_length_m, result.pareto_energy_j)
        ],
        "routes": [],
        "length_m": float(result.length_m) if result.success else 0.0,
        "energy_j": float(result.total_energy_j) if result.success else 0.0,
    }


def plot_pareto_panels(
    datasets: Sequence[dict[str, Any]], out_path: Path, *,
    title: str, d_min_ref: float = 0.7,
) -> Path:
    """每个场景一个面板：候选路线点云 + 前沿线 + 给定偏好选点。"""
    _setup_fonts()
    count = len(datasets)
    if count == 0:
        raise ValueError("没有可画的 Pareto 数据")
    figure, axes = plt.subplots(
        1, count, figsize=(5.6 * count, 5.1), constrained_layout=True, squeeze=False
    )
    for axis, data in zip(axes[0], datasets):
        routes = data["routes"]
        raw_points = [(r.length_m, r.energy_j) for r in routes]
        raw_points.extend(data["algo_front"])
        if not raw_points:
            axis.text(0.5, 0.5, "无数据", ha="center", va="center",
                      transform=axis.transAxes)
            continue
        # 归一化到各自的坐标最小值（同一张图上比较"谁被谁支配"最清楚）
        l_ref = min(p[0] for p in raw_points)
        e_ref = min(p[1] for p in raw_points)
        to_xy = lambda p: (p[0] / l_ref, p[1] / e_ref)  # noqa: E731
        front = pareto_front(raw_points)
        axis.axvspan(1.0, 2.0, ymin=0.0, color="0.85", alpha=0.35, zorder=0)
        axis.axhline(1.0, color="0.6", linestyle=":", linewidth=1.0, zorder=1)
        axis.axvline(1.0, color="0.6", linestyle=":", linewidth=1.0, zorder=1)
        if routes:
            rate = [r.energy_per_m for r in routes]
            axis.scatter(
                [to_xy((r.length_m, r.energy_j))[0] for r in routes],
                [to_xy((r.length_m, r.energy_j))[1] for r in routes],
                s=46, color=COLORS["route"], alpha=0.85, zorder=2,
                label=(
                    f"多同伦候选路线（{len(routes)} 条，"
                    f"{min(rate):.1f}~{max(rate):.1f} J/m）"
                ),
            )
        if data["algo_front"]:
            axis.scatter(
                [to_xy(p)[0] for p in data["algo_front"]],
                [to_xy(p)[1] for p in data["algo_front"]],
                s=200, marker="*", color=COLORS["constrained"], zorder=5,
                label=f"collect_pareto 非支配集（{len(data['algo_front'])} 点）",
            )
        if len(front) > 1:
            axis.plot(
                [to_xy(p)[0] for p in front], [to_xy(p)[1] for p in front], "-o",
                color=COLORS["soft"], markersize=7, linewidth=1.9, zorder=4,
                label=f"Pareto 前沿（{len(front)} 点）",
            )
            for beta, marker, color, name in (
                (0.0, "v", "#1f77b4", "偏好 β=0（最短）"),
                (0.5, "^", "#9467bd", "偏好 β=0.5"),
                (1.0, "s", "#8c564b", "偏好 β=1（最省电）"),
            ):
                point = to_xy(_preference_point(front, beta))
                axis.scatter(
                    [point[0]], [point[1]], s=130, marker=marker, color=color,
                    edgecolors="black", linewidths=0.7, zorder=6, label=name,
                )
        else:
            point = to_xy(front[0])
            axis.scatter(
                [point[0]], [point[1]], s=210, marker="D",
                facecolors="none", edgecolors=COLORS["soft"], linewidths=2.0,
                zorder=6, label="所有偏好选点重合（前沿退化）",
            )
            axis.annotate(
                "前沿退化：最短路径同时是能耗最低路径，\n"
                "替代路线被 (1,1) 完全支配 —— 无权衡空间",
                xy=(0.98, 0.03), xycoords="axes fraction", ha="right",
                fontsize=8.5, color=COLORS["soft"],
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.92,
                          edgecolor=COLORS["soft"]),
            )
        axis.set_title(f"{data['scenario_id']}（{data['grade']}）", fontsize=10.5)
        axis.set_xlabel("路径长度 / 最优长度 [—]")
        axis.grid(alpha=0.25)
        handles, labels_ = axis.get_legend_handles_labels()
        unique: dict[str, Any] = {}
        for handle, name in zip(handles, labels_):
            unique.setdefault(name, handle)
        axis.legend(unique.values(), unique.keys(), fontsize=7.5, loc="upper right")
    axes[0][0].set_ylabel("电能 / 最低电能 [—]")
    figure.suptitle(title, fontsize=11.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# 3. 软惩罚 vs 硬约束
# ---------------------------------------------------------------------------

SOFT_WEIGHTS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
"""扫描点。**必须在 1.0 以下加密**：本数据集上 w >= 1 时软惩罚解与硬约束解
逐字段重合（违反率 0），只有 0~0.5 之间才看得到"违反率随 w 下降"的过程。"""


def soft_config_for(weight: float) -> str:
    mapping = {
        0.0: "configs/chapter3/soft_w000.yaml",
        0.1: "configs/chapter3/soft_w010.yaml",
        0.25: "configs/chapter3/soft_w025.yaml",
        0.5: "configs/chapter3/soft_w050.yaml",
        1.0: "configs/chapter3/soft_w100.yaml",
        2.0: "configs/chapter3/soft_w200.yaml",
        5.0: "configs/chapter3/soft_w500.yaml",
    }
    return mapping[float(weight)]


SOFT_COLUMNS: tuple[str, ...] = (
    "scheme",
    "soft_penalty_weight",
    "enforce_dmin",
    "algo_config",
    "config_hash_ref_scenario",
    "tasks",
    "success",
    "length_median_m",
    "length_total_m",
    "energy_median_j",
    "energy_total_j",
    "violation_rate_length_weighted",
    "violation_tasks",
    "physical_violation_rate_length_weighted",
    "physical_violation_tasks",
    "min_clearance_min_m",
    "min_clearance_median_m",
    "mean_soft_penalty",
)


def summarise_soft(
    records: Sequence[PlanRecord], *, ref_scenario: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = [("soft", w) for w in SOFT_WEIGHTS] + [("hard", None)]
    for scheme, weight in keys:
        if scheme == "soft":
            subset = [
                r for r in records
                if r.planner == f"soft_w{weight:.2f}"
            ]
            label = f"软惩罚 w={weight:g}"
            enforce = 0
            algo_config = soft_config_for(float(weight))
            penalty = _median([r.soft_penalty for r in subset])
        else:
            subset = [r for r in records if r.planner == "hard"]
            label = "硬约束（本文）"
            enforce = 1
            algo_config = DEFAULT_ALGO
            penalty = 0.0
        if not subset:
            continue
        total_length = sum(r.length_m for r in subset)
        ref = next((r for r in subset if r.scenario_id == ref_scenario), subset[0])
        rows.append(
            {
                "scheme": label,
                "soft_penalty_weight": "" if weight is None else f"{weight:g}",
                "enforce_dmin": enforce,
                "algo_config": algo_config,
                "config_hash_ref_scenario": ref.config_hash,
                "tasks": len(subset),
                "success": sum(1 for r in subset if r.success),
                "length_median_m": f"{_median([r.length_m for r in subset]):.4f}",
                "length_total_m": f"{total_length:.4f}",
                "energy_median_j": f"{_median([r.energy_j for r in subset]):.3f}",
                "energy_total_j": f"{sum(r.energy_j for r in subset):.3f}",
                "violation_rate_length_weighted": (
                    f"{(sum(r.violation_length_m for r in subset) / total_length if total_length else 0.0):.4f}"
                ),
                "violation_tasks": sum(
                    1 for r in subset if r.violation_length_m > 1e-9
                ),
                "physical_violation_rate_length_weighted": (
                    f"{(sum(r.physical_violation_length_m for r in subset) / total_length if total_length else 0.0):.4f}"
                ),
                "physical_violation_tasks": sum(
                    1 for r in subset if r.physical_violation_length_m > 1e-9
                ),
                "min_clearance_min_m": f"{min(r.min_clearance_m for r in subset):.4f}",
                "min_clearance_median_m": (
                    f"{_median([r.min_clearance_m for r in subset]):.4f}"
                ),
                "mean_soft_penalty": f"{penalty:.4f}",
            }
        )
    return rows


def format_soft_table(rows: Sequence[dict[str, Any]], *, d_min_ref: float) -> str:
    header = (
        f"{'方案':<16}{'任务':>4}{'长度中位[m]':>12}{'能耗中位[J]':>12}"
        f"{'违反率(按长度)':>15}{'违反任务':>9}{'物理穿行率':>11}"
        f"{'最小离障最小[m]':>15}"
    )
    lines = [
        f"软惩罚 vs 硬约束（违反 = clearance < d_min = {d_min_ref:.2f} m，按长度加权）",
        header,
        "-" * len(header),
    ]
    for row in rows:
        lines.append(
            f"{row['scheme']:<16}{row['tasks']:>4}{row['length_median_m']:>12}"
            f"{row['energy_median_j']:>12}{row['violation_rate_length_weighted']:>15}"
            f"{row['violation_tasks']:>9}"
            f"{row['physical_violation_rate_length_weighted']:>11}"
            f"{row['min_clearance_min_m']:>15}"
        )
    return "\n".join(lines)


def plot_soft_vs_hard(
    rows: Sequence[dict[str, Any]], out_path: Path, *, d_min_ref: float
) -> Path:
    """图 3：违反率 / 长度 / 能耗 随软惩罚权重 w 的变化（硬约束为水平线）。"""
    _setup_fonts()
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    soft = [r for r in rows if r["enforce_dmin"] == 0]
    hard = next((r for r in rows if r["enforce_dmin"] == 1), None)
    weights = [float(r["soft_penalty_weight"]) for r in soft]
    series = [
        ("violation_rate_length_weighted", "违反率（按长度）", axes[0]),
        ("length_median_m", "路径长度中位 [m]", axes[1]),
        ("energy_median_j", "电能中位 [J]", axes[2]),
    ]
    for key, ylabel, axis in series:
        values = [float(r[key]) for r in soft]
        axis.plot(weights, values, "-o", color=COLORS["soft"], label="软惩罚（关硬约束）")
        # 只在首末两点标注：中间点密集时会互相压字（尤其是长度/能耗两个面板）
        for index in (0, len(weights) - 1):
            axis.annotate(
                f"{values[index]:.3f}", (weights[index], values[index]),
                textcoords="offset points",
                xytext=(6 if index else -28, 7), ha="center", fontsize=8.5,
            )
        if hard is not None:
            axis.axhline(
                float(hard[key]), color=COLORS["constrained"], linestyle="--",
                linewidth=1.8, label="硬约束（本文，与 w 无关）",
            )
        axis.set_xlabel("软惩罚权重 w")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8.5)
    axes[0].set_title(
        f"违反率随 w 单调下降；w 足够大时与硬约束解重合\n"
        f"（阈值 {d_min_ref:.2f} m；硬约束恒为 0，且**不依赖 w 的取值**）",
        fontsize=10.5,
    )
    axes[1].set_title("长度随 w 上升并收敛到硬约束解", fontsize=10.5)
    axes[2].set_title("能耗随 w 上升并收敛到硬约束解", fontsize=10.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# 并行入口
# ---------------------------------------------------------------------------


def _planner_task(
    scenario_path: str, planner: str, label: str, algo: str,
    config_root: str, seed: int, d_min_ref: float,
) -> list[PlanRecord]:
    return run_planner_scenario(
        scenario_path, planner, label, algo, config_root=config_root, seed=seed,
        d_min_ref=d_min_ref,
    )


def _soft_task(
    scenario_path: str, algo: str, config_root: str, seed: int, d_min_ref: float,
    weight: float,
) -> list[PlanRecord]:
    return run_planner_scenario(
        scenario_path, f"soft_w{float(weight):.2f}", f"软惩罚 w={float(weight):g}",
        algo, config_root=config_root, seed=seed, d_min_ref=d_min_ref,
    )


def _run_parallel(tasks, worker, jobs: int) -> list[Any]:
    results: list[Any] = []
    workers = jobs if jobs > 0 else min(len(tasks), os.cpu_count() or 1)
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, *task) for task in tasks]
            for future in as_completed(futures):
                results.extend(future.result())
    else:
        for task in tasks:
            results.extend(worker(*task))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.chapter3",
        description="第 3 章结果：规划器对比表 / Pareto 前沿 / 软硬对照",
    )
    parser.add_argument("--only", choices=("all", "table", "pareto", "soft"), default="all")
    parser.add_argument("--tier", default="all")
    parser.add_argument("--algo", default=DEFAULT_ALGO, help="本文规划器用的 algo 配置")
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-min-ref", type=float, default=0.7)
    parser.add_argument("--d-min-baseline", type=float, default=0.58)
    parser.add_argument("--pareto-scenarios", nargs="*", default=None,
                        help="默认按档位各取一个代表场景")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--review", action="store_true", help="把图复制到 docs/review/")
    parser.add_argument("--review-start", type=int, default=23)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def _representative_scenarios(paths: Sequence[Path]) -> list[Path]:
    """按档位各取一个代表场景（路径长度中位附近那个）。"""
    chosen: list[Path] = []
    for tier in ("easy", "medium", "hard"):
        subset = [
            (p, float((config_module.read_config(p).get("quality") or {}).get(
                "path_len_m", 0.0)))
            for p in paths
            if str((config_module.read_config(p).get("quality") or {}).get("grade", "")) == tier
        ]
        if not subset:
            continue
        subset.sort(key=lambda item: item[1])
        chosen.append(subset[len(subset) // 2][0])
    return chosen


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_root = Path(args.config_root) if args.config_root else CONFIG_ROOT
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = scenario_paths(args.tier, config_root=config_root)
    if not paths:
        print("没有找到场景", flush=True)
        return 2
    ref_scenario = paths[0].stem
    jobs = int(args.jobs) if args.jobs else 0
    review_index = int(args.review_start)

    if args.only in ("all", "table"):
        tasks = [
            (str(path), planner, label, algo, str(config_root), int(args.seed),
             float(args.d_min_ref))
            for path in paths
            for planner, label, algo in PLANNERS
        ]
        records = _run_parallel(tasks, _planner_task, jobs)
        task_csv = write_csv(
            [_task_row(r) for r in sorted(records, key=lambda r: (r.planner, r.scenario_id))],
            PROJECT_ROOT / "docs" / "chapter3_planner_tasks.csv",
            TASK_COLUMNS,
        )
        rows = summarise_planners(records, ref_scenario=ref_scenario)
        summary_csv = write_csv(
            rows, PROJECT_ROOT / "docs" / "chapter3_planner_summary.csv", SUMMARY_COLUMNS
        )
        print(f"\n{format_planner_table(rows, d_min_ref=float(args.d_min_ref))}")
        print(f"\n逐任务: {task_csv}\n汇总:   {summary_csv}")
        if not args.no_plots:
            figure = plot_planner_table(
                records, out_dir / "planner_compare.png", d_min_ref=float(args.d_min_ref)
            )
            print(f"OK   {figure}")
            if args.review:
                target = REVIEW_DIR / f"{review_index:02d}_alltier_ch3_planner_compare.png"
                REVIEW_DIR.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copyfile(figure, target)
                review_index += 1
                print(f"REVIEW {target}")

    if args.only in ("all", "pareto"):
        targets = (
            [Path(p) for p in args.pareto_scenarios]
            if args.pareto_scenarios
            else _representative_scenarios(paths)
        )
        datasets = [
            collect_pareto_data(
                str(path), config_root=str(config_root), algo=args.algo,
                d_min_ref=float(args.d_min_ref),
            )
            for path in targets
        ]
        rows: list[dict[str, Any]] = []
        for data in datasets:
            route_points = [(r.length_m, r.energy_j) for r in data["routes"]]
            route_front = pareto_front(route_points)
            merged_front = pareto_front(route_points + list(data["algo_front"]))

            def _on(points: Sequence[tuple[float, float]],
                    point: tuple[float, float], *, tol: float = 1e-6) -> bool:
                return any(
                    abs(point[0] - l) <= tol and abs(point[1] - e) <= tol
                    for l, e in points
                )

            for r in data["routes"]:
                rows.append({
                    "scenario_id": data["scenario_id"], "grade": data["grade"],
                    "source": "route", "route_index": r.route_index,
                    "length_m": f"{r.length_m:.4f}",
                    "energy_j": f"{r.energy_j:.3f}",
                    "energy_per_m": f"{r.energy_per_m:.3f}",
                    "min_clearance_m": f"{r.min_clearance_m:.4f}",
                    "on_front": int(
                        _on(route_front, (r.length_m, r.energy_j))
                    ),
                    "feasible": int(r.feasible),
                })
            for l, e in data["algo_front"]:
                rows.append({
                    "scenario_id": data["scenario_id"], "grade": data["grade"],
                    "source": "collect_pareto", "route_index": "",
                    "length_m": f"{l:.4f}", "energy_j": f"{e:.3f}",
                    "energy_per_m": f"{(e / l if l > 0 else float('nan')):.3f}",
                    "min_clearance_m": "",
                    "on_front": int(_on(merged_front, (l, e))),
                    "feasible": 1,
                })
        pareto_csv = write_csv(
            rows, PROJECT_ROOT / "docs" / "chapter3_pareto_points.csv", PARETO_COLUMNS
        )
        print("\nPareto（collect_pareto 前沿 + 多同伦候选路线）")
        for data in datasets:
            front = pareto_front(
                [(r.length_m, r.energy_j) for r in data["routes"]] + data["algo_front"]
            )
            print(
                f"  {data['scenario_id']:<22}{data['grade']:<7}"
                f"算法前沿={len(data['algo_front'])} 点  候选路线={len(data['routes'])} 条  "
                f"合并前沿={len(front)} 点"
            )
        if not args.no_plots:
            figure = plot_pareto_panels(
                datasets, out_dir / "pareto_scenarios.png",
                title=(
                    "第 3 章 (路径长度, 电能) 权衡：算法前沿 vs 多同伦候选路线\n"
                    "★ = collect_pareto 非支配集；▲/■/● = 给定偏好下的选点"
                ),
                d_min_ref=float(args.d_min_ref),
            )
            print(f"OK   {figure}")
            synthetic = synthetic_two_route_dataset()
            synthetic_figure = plot_pareto_panels(
                [synthetic], out_dir / "pareto_synthetic.png",
                title=(
                    "合成双通道地图（非论文数据）：当地形阻力跨度足够大时，"
                    "(长度, 电能) 前沿才不退化为单点"
                ),
                d_min_ref=float(args.d_min_ref),
            )
            print(
                f"OK   {synthetic_figure}"
                f"（合成前沿 {len(synthetic['algo_front'])} 点 —— 真实场景全部为 1 点）"
            )
            print(f"点集: {pareto_csv}")
            if args.review:
                import shutil

                REVIEW_DIR.mkdir(parents=True, exist_ok=True)
                for source, name in (
                    (figure, "pareto_scenarios"),
                    (synthetic_figure, "pareto_synthetic"),
                ):
                    target = REVIEW_DIR / f"{review_index:02d}_ch3_{name}.png"
                    shutil.copyfile(source, target)
                    review_index += 1
                    print(f"REVIEW {target}")

    if args.only in ("all", "soft"):
        tasks = [
            (str(path), soft_config_for(weight), str(config_root), int(args.seed),
             float(args.d_min_ref), float(weight))
            for path in paths
            for weight in SOFT_WEIGHTS
        ]
        records = _run_parallel(tasks, _soft_task, jobs)
        # 硬约束行：直接复用本文配置（frozen_A）
        hard_tasks = [
            (str(path), "hard", "硬约束（本文）", args.algo, str(config_root),
             int(args.seed), float(args.d_min_ref))
            for path in paths
        ]
        hard_records = _run_parallel(hard_tasks, _planner_task, jobs)
        all_records = records + hard_records
        task_csv = write_csv(
            [_task_row(r) for r in sorted(all_records, key=lambda r: (r.planner, r.scenario_id))],
            PROJECT_ROOT / "docs" / "chapter3_soft_tasks.csv", TASK_COLUMNS,
        )
        rows = summarise_soft(all_records, ref_scenario=ref_scenario)
        summary_csv = write_csv(
            rows, PROJECT_ROOT / "docs" / "chapter3_soft_summary.csv", SOFT_COLUMNS
        )
        print(f"\n{format_soft_table(rows, d_min_ref=float(args.d_min_ref))}")
        print(f"\n逐任务: {task_csv}\n汇总:   {summary_csv}")
        if not args.no_plots:
            figure = plot_soft_vs_hard(
                rows, out_dir / "soft_vs_hard.png", d_min_ref=float(args.d_min_ref)
            )
            print(f"OK   {figure}")
            if args.review:
                import shutil

                REVIEW_DIR.mkdir(parents=True, exist_ok=True)
                target = REVIEW_DIR / f"{review_index:02d}_alltier_ch3_soft_vs_hard.png"
                shutil.copyfile(figure, target)
                review_index += 1
                print(f"REVIEW {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
