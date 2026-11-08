"""多端点对任务：同一户型多组起终点，专门制造「短-高阻 vs 长-低阻」同伦类候选。

```bash
uv run python -m analysis.multi_endpoint                        # 每场景 2 个附加任务
uv run python -m analysis.multi_endpoint --ports-per-scenario 3
uv run python -m analysis.multi_endpoint --scenarios houseexpo_eef4fbdd houseexpo_277f75f1
```

为什么需要这一层
----------------
受控材质因子（``env/terrain_partition.py``）已经把 ``c_rr`` 跨度从 1.67 倍拉到
4 倍，硬档任务的 ``(长度, 能耗)`` 前沿出现多点（最多 7 点、能耗差 11%）。
但 easy / medium 任务的冻结起终点**只有一条必经路线**（瓶颈 1.7~2.0 m 对
``d_min = 0.70`` 只留 0.3~0.6 m 余量），同伦类候选本身就不存在 ——
这是几何事实，不是参数问题。

本模块的做法：**不改冻结场景、不改 core 算法**，只在同一户型的**同一连通分量**内
重新采样若干组起终点（同一难度档结构），逐组计算：

* ``collect_pareto`` 的 ``(长度, 能耗)`` 前沿点数与跨度；
* 能耗加权 A*（ESA*）与仅长度 A* 的配对差。

它回答的是："把任务换成同户型里的另一对端点，权衡空间会不会出现？"
—— 即把**任务端点**也当成一个可能的影响因素来检验，而不是靠调参凑图。

口径：所有附加端点对都经过 ``core.astar`` 在冻结 ``algo.astar.d_min`` 下验证有解；
只有可行任务进入统计；失败类型（``start_violates_clearance`` / ``no_path``）单列。
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_ALGO = "configs/frozen_A.yaml"
OUT_CSV = PROJECT_ROOT / "docs" / "chapter3_multi_endpoint.csv"
OUT_DIR = PROJECT_ROOT / "docs" / "chapter3"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from core import astar  # noqa: E402
from env import scene_quality, terrain_partition  # noqa: E402
from runner import config as config_module  # noqa: E402
from runner import experiment  # noqa: E402
from analysis import chapter3  # noqa: E402

_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
)
_STRUCTURE = np.ones((3, 3), dtype=int)
COLORS = {"task": "#1f77b4", "multi": "#2ca02c", "single": "#bbbbbb"}


def _setup_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.monospace"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class TaskOutcome:
    scenario_id: str
    grade: str
    terrain_tier: str
    port: str
    """``frozen`` = 场景自带起终点；``p1/p2/...`` = 附加端点对。"""
    start: tuple[float, float]
    goal: tuple[float, float]
    feasible: bool
    reason: str
    length_m: float
    energy_j: float
    energy_per_m: float
    min_clearance_m: float
    front_points: int
    front_length_span_m: float
    front_energy_span_j: float
    esa_length_m: float
    esa_energy_j: float
    len_length_m: float
    len_energy_j: float
    esa_minus_len_length_m: float
    esa_minus_len_energy_j: float


def _largest_component_cells(clearance: np.ndarray, d_min: float) -> np.ndarray:
    feasible = clearance + 1e-9 >= float(d_min)
    labels, count = ndimage.label(feasible, structure=_STRUCTURE)
    if count == 0:
        return np.zeros((0, 2), dtype=int)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return np.argwhere(labels == int(sizes.argmax()))


def sample_ports(
    clearance: np.ndarray,
    labels: np.ndarray,
    *,
    resolution: float,
    d_min: float,
    rng: np.random.Generator,
    ports: int,
    min_distance_m: float = 8.0,
    max_distance_m: float = 24.0,
    max_attempts: int = 400,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """在同一连通分量内采样 ``ports`` 组附加起终点。

    与冻结场景的采样口径一致（同样的可行域、同样的"必须同分量"约束），
    并把**直线距离**限制在 ``[min_distance_m, max_distance_m]``：
    冻结任务的路径长度是 8~22 m，附加任务要与之可比，不能造出 40 m 的巨型任务
    （那会把"同户型另一对端点"变成"另一个量级的任务"）。
    """
    cells = _largest_component_cells(clearance, d_min)
    if cells.shape[0] < 2:
        return []
    res = float(resolution)
    samples: list[tuple[tuple[int, int], tuple[int, int]]] = []
    guard = 0
    while len(samples) < ports and guard < max_attempts:
        guard += 1
        first = int(rng.integers(cells.shape[0]))
        second = int(rng.integers(cells.shape[0]))
        if first == second:
            continue
        cell_a = cells[first]
        cell_b = cells[second]
        distance = float(np.hypot(*(cell_b - cell_a))) * res
        if distance < min_distance_m or distance > max_distance_m:
            continue
        samples.append(
            ((int(cell_a[0]), int(cell_a[1])), (int(cell_b[0]), int(cell_b[1])))
        )
    return samples


def _trial_scene(
    scene: experiment.Scene,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> experiment.Scene:
    """把场景的起终点换成给定端点（单元随之重算）。"""
    return replace(
        scene,
        start_xy=start_xy,
        goal_xy=goal_xy,
        start_cell=astar.world_to_cell(start_xy, scene.resolution, scene.origin),
        goal_cell=astar.world_to_cell(goal_xy, scene.resolution, scene.origin),
    )


def _evaluate_length_only(
    scene: experiment.Scene,
    baseline_setup: experiment.PlannerSetup,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> tuple[float, float] | None:
    """仅长度 A*（``d_min = body_radius``）的 ``(length_m, energy_j)``。"""
    result = chapter3._plan(
        _trial_scene(scene, start_xy, goal_xy), baseline_setup, baseline_setup.astar_params
    )
    if not result.success:
        return None
    return float(result.length_m), float(result.total_energy_j)


def _evaluate_esa(
    scene: experiment.Scene,
    baseline_setup: experiment.PlannerSetup,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> tuple[float, float] | None:
    """能耗加权 A*（ESA*）的 ``(length_m, energy_j)``。

    复现口径与 ``analysis/chapter3.run_planner_scenario`` 一致：先收
    ``collect_pareto`` 非支配集取最小能耗点，再把能耗预算压到该点重跑一次，
    反解出那条路径本身。
    """
    trial = _trial_scene(scene, start_xy, goal_xy)
    base = baseline_setup.astar_params
    front = chapter3._plan(trial, baseline_setup, replace(base, collect_pareto=True))
    if not front.success or front.pareto_energy_j.size == 0:
        return None
    e_min = float(front.pareto_energy_j[int(np.argmin(front.pareto_energy_j))])
    tight = chapter3._plan(
        trial,
        baseline_setup,
        replace(base, e_budget=e_min * (1.0 + 1e-9) + 1e-9),
    )
    if not tight.success:
        tight = front
    return float(tight.length_m), float(tight.total_energy_j)


def _evaluate_port(
    *,
    scene: experiment.Scene,
    setup: experiment.PlannerSetup,
    scenario: dict[str, Any],
    resolved: Any,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    port: str,
    body_radius: float,
    d_min_ref: float,
) -> TaskOutcome:
    trial = replace(
        scene,
        start_xy=start_xy,
        goal_xy=goal_xy,
        start_cell=astar.world_to_cell(start_xy, scene.resolution, scene.origin),
        goal_cell=astar.world_to_cell(goal_xy, scene.resolution, scene.origin),
    )
    base = setup.astar_params
    tier = str(
        ((scenario.get("terrain") or {}).get("partition") or {}).get("tier", "-")
    )
    grade = str((scenario.get("quality") or {}).get("grade", "-"))
    blank = dict(
        scenario_id=str(scenario["id"]),
        grade=grade,
        terrain_tier=tier,
        port=port,
        start=(round(start_xy[0], 4), round(start_xy[1], 4)),
        goal=(round(goal_xy[0], 4), round(goal_xy[1], 4)),
    )

    constrained = chapter3._plan(_trial_scene(scene, start_xy, goal_xy), setup, base)
    if not constrained.success:
        return TaskOutcome(
            **blank, feasible=False, reason=str(constrained.reason),
            length_m=0.0, energy_j=0.0, energy_per_m=0.0, min_clearance_m=0.0,
            front_points=0, front_length_span_m=0.0, front_energy_span_j=0.0,
            esa_length_m=0.0, esa_energy_j=0.0, len_length_m=0.0, len_energy_j=0.0,
            esa_minus_len_length_m=0.0, esa_minus_len_energy_j=0.0,
        )

    baseline_setup = replace(
        setup, astar_params=replace(base, d_min=body_radius, collect_pareto=False)
    )
    length_only = _evaluate_length_only(scene, baseline_setup, start_xy, goal_xy)
    front = chapter3._plan(
        _trial_scene(scene, start_xy, goal_xy),
        baseline_setup,
        replace(baseline_setup.astar_params, collect_pareto=True),
    )
    points = list(zip(front.pareto_length_m, front.pareto_energy_j))
    esa = _evaluate_esa(scene, baseline_setup, start_xy, goal_xy)
    esa_length, esa_energy = esa if esa is not None else (0.0, 0.0)
    len_length, len_energy = length_only if length_only is not None else (0.0, 0.0)

    return TaskOutcome(
        **blank,
        feasible=True,
        reason="ok",
        length_m=float(constrained.length_m),
        energy_j=float(constrained.total_energy_j),
        energy_per_m=(
            float(constrained.total_energy_j) / float(constrained.length_m)
            if constrained.length_m > 0
            else float("nan")
        ),
        min_clearance_m=float(constrained.clearance.min()) if constrained.clearance.size else 0.0,
        front_points=int(len(points)),
        front_length_span_m=(
            float(max(p[0] for p in points) - min(p[0] for p in points)) if points else 0.0
        ),
        front_energy_span_j=(
            float(max(p[1] for p in points) - min(p[1] for p in points)) if points else 0.0
        ),
        esa_length_m=esa_length,
        esa_energy_j=esa_energy,
        len_length_m=len_length,
        len_energy_j=len_energy,
        esa_minus_len_length_m=esa_length - len_length,
        esa_minus_len_energy_j=esa_energy - len_energy,
    )


def run_scenario(
    scenario_path: str | Path,
    *,
    config_root: Path = CONFIG_ROOT,
    algo: str = DEFAULT_ALGO,
    ports_per_scenario: int = 2,
    seed: int = 42,
    d_min_ref: float = 0.7,
) -> list[TaskOutcome]:
    resolved = config_module.load_resolved(algo, scenario_path, config_root=config_root)
    scenario = resolved.scenario
    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    body_radius = float(resolved.platform["body_radius"])
    d_min = float(setup.astar_params.d_min)
    segmentation = scene_quality.segment_rooms(scene.clearance, core_clearance_m=0.9)
    rng = np.random.default_rng(
        terrain_partition.stable_seed(str(scenario["id"]), f"ports{seed}")
    )

    outcomes = [
        _evaluate_port(
            scene=scene, setup=setup, scenario=scenario, resolved=resolved,
            start_xy=scene.start_xy, goal_xy=scene.goal_xy, port="frozen",
            body_radius=body_radius, d_min_ref=d_min_ref,
        )
    ]
    candidates = sample_ports(
        scene.clearance,
        segmentation.labels,
        resolution=scene.resolution,
        d_min=d_min,
        rng=rng,
        ports=ports_per_scenario,
    )
    for index, (cell_a, cell_b) in enumerate(candidates, start=1):
        start_xy = _world(cell_a, scene)
        goal_xy = _world(cell_b, scene)
        outcomes.append(
            _evaluate_port(
                scene=scene, setup=setup, scenario=scenario, resolved=resolved,
                start_xy=start_xy, goal_xy=goal_xy, port=f"p{index}",
                body_radius=body_radius, d_min_ref=d_min_ref,
            )
        )
    return outcomes


def _world(cell: Sequence[int], scene: experiment.Scene) -> tuple[float, float]:
    return (
        float(scene.origin[0]) + (int(cell[1]) + 0.5) * scene.resolution,
        float(scene.origin[1]) + (int(cell[0]) + 0.5) * scene.resolution,
    )


CSV_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "grade",
    "terrain_tier",
    "port",
    "start",
    "goal",
    "feasible",
    "reason",
    "length_m",
    "energy_j",
    "energy_per_m",
    "min_clearance_m",
    "front_points",
    "front_length_span_m",
    "front_energy_span_j",
    "esa_length_m",
    "esa_energy_j",
    "len_length_m",
    "len_energy_j",
    "esa_minus_len_length_m",
    "esa_minus_len_energy_j",
)


def row_for(outcome: TaskOutcome) -> dict[str, Any]:
    return {
        "scenario_id": outcome.scenario_id,
        "grade": outcome.grade,
        "terrain_tier": outcome.terrain_tier,
        "port": outcome.port,
        "start": f"{outcome.start[0]},{outcome.start[1]}",
        "goal": f"{outcome.goal[0]},{outcome.goal[1]}",
        "feasible": int(outcome.feasible),
        "reason": outcome.reason,
        "length_m": f"{outcome.length_m:.4f}",
        "energy_j": f"{outcome.energy_j:.3f}",
        "energy_per_m": f"{outcome.energy_per_m:.3f}",
        "min_clearance_m": f"{outcome.min_clearance_m:.4f}",
        "front_points": outcome.front_points,
        "front_length_span_m": f"{outcome.front_length_span_m:.4f}",
        "front_energy_span_j": f"{outcome.front_energy_span_j:.3f}",
        "esa_length_m": f"{outcome.esa_length_m:.4f}",
        "esa_energy_j": f"{outcome.esa_energy_j:.3f}",
        "len_length_m": f"{outcome.len_length_m:.4f}",
        "len_energy_j": f"{outcome.len_energy_j:.3f}",
        "esa_minus_len_length_m": f"{outcome.esa_minus_len_length_m:.4f}",
        "esa_minus_len_energy_j": f"{outcome.esa_minus_len_energy_j:.3f}",
    }


def summarise(outcomes: Sequence[TaskOutcome]) -> list[str]:
    feasible = [o for o in outcomes if o.feasible]
    lines: list[str] = []
    total = len(outcomes)
    lines.append(
        f"任务总数 {total}，可行 {len(feasible)}（"
        f"失败 {Counter(o.reason for o in outcomes if not o.feasible)}）"
    )
    dist = Counter(o.front_points for o in feasible)
    multi = sum(count for points, count in dist.items() if points > 1)
    lines.append(
        f"前沿点数分布 {dict(sorted(dist.items()))}；多点任务 {multi}/{len(feasible)}"
        f"（{100.0 * multi / len(feasible) if feasible else 0:.1f}%）"
    )
    for label, key in (("frozen", "frozen"), ("附加端点对", "port")):
        subset = (
            [o for o in feasible if o.port == key]
            if key == "frozen"
            else [o for o in feasible if o.port != "frozen"]
        )
        if not subset:
            continue
        spans = [o.front_energy_span_j for o in subset]
        lines.append(
            f"  {label:<10} 任务 {len(subset):>3}  多点 "
            f"{sum(1 for o in subset if o.front_points > 1):>3}  能耗跨度中位 "
            f"{float(np.median(spans)):.2f} J  最大 {max(spans):.2f} J"
        )
    by_grade: dict[str, list[TaskOutcome]] = {}
    for o in feasible:
        by_grade.setdefault(o.grade, []).append(o)
    for grade_name in sorted(by_grade):
        subset = by_grade[grade_name]
        lines.append(
            f"  {grade_name:<10} 任务 {len(subset):>3}  多点 "
            f"{sum(1 for o in subset if o.front_points > 1):>3}"
            f"  ESA* 与仅长度不同 "
            f"{sum(1 for o in subset if abs(o.esa_minus_len_energy_j) > 1e-6 or abs(o.esa_minus_len_length_m) > 1e-6):>3}"
        )
    return lines


def plot_multi_endpoint(
    outcomes: Sequence[TaskOutcome], out_path: Path
) -> Path:
    """图：逐任务前沿点数（按档位分面板）+ 能耗跨度；绿色 = 非退化。"""
    _setup_fonts()
    feasible = [o for o in outcomes if o.feasible]
    grades = [g for g in ("easy", "medium", "hard") if any(o.grade == g for o in feasible)]
    figure = plt.figure(figsize=(16.5, 8.8), constrained_layout=True)
    grid = figure.add_gridspec(2, len(grades), height_ratios=[1.15, 1.0])
    for column, grade_name in enumerate(grades):
        subset = sorted(
            (o for o in feasible if o.grade == grade_name),
            key=lambda o: (o.scenario_id, o.port),
        )
        axis = figure.add_subplot(grid[0, column])
        x = np.arange(len(subset))
        colors = [
            COLORS["multi"] if o.front_points > 1 else COLORS["single"] for o in subset
        ]
        axis.bar(x, [o.front_points for o in subset], color=colors)
        for index, outcome in enumerate(subset):
            if outcome.front_points > 1:
                axis.annotate(
                    str(outcome.front_points), (index, outcome.front_points),
                    ha="center", va="bottom", fontsize=8,
                )
        axis.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.3)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [f"{o.scenario_id.replace('houseexpo_', '')}\n{o.port}" for o in subset],
            fontsize=7.0,
        )
        axis.set_title(
            f"{grade_name}（{len(subset)} 个任务，多点 "
            f"{sum(1 for o in subset if o.front_points > 1)}）",
            fontsize=10.5,
        )
        axis.set_ylabel("(长度, 能耗) 前沿点数")
        axis.grid(axis="y", alpha=0.25)
    figure.axes[0].legend(
        handles=[
            Patch(facecolor=COLORS["multi"], label="非退化（> 1 点）"),
            Patch(facecolor=COLORS["single"], label="退化（单点）"),
        ],
        fontsize=8.5, loc="upper left",
    )

    axis = figure.add_subplot(grid[1, :])
    ordered = sorted(feasible, key=lambda o: (o.grade, o.scenario_id, o.port))
    x = np.arange(len(ordered))
    axis.bar(
        x,
        [o.front_energy_span_j for o in ordered],
        color=[
            COLORS["multi"] if o.front_points > 1 else COLORS["single"]
            for o in ordered
        ],
    )
    axis.set_ylabel("前沿两端能耗差 [J]")
    axis.set_xticks(x)
    axis.set_xticklabels(
        [
            f"{o.scenario_id.replace('houseexpo_', '')}\n{o.port}"
            for o in ordered
        ],
        fontsize=6.2,
    )
    axis.set_title("前沿两端点的能耗差（0 = 退化为单点）；横轴顺序 easy -> medium -> hard",
                   fontsize=10.5)
    axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "同一户型多组起终点（多端点对任务）：\n"
        "冻结端点只有一条必经路线时前沿退化，附加端点把多点任务扩到 easy/medium/hard 三档",
        fontsize=11.5,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


def _scenario_task(
    path: str,
    *,
    config_root: str,
    algo: str,
    ports_per_scenario: int,
    seed: int,
    d_min_ref: float,
) -> list[TaskOutcome]:
    """并行入口（必须是模块级函数：``ProcessPoolExecutor`` 需要可 pickle）。"""
    return run_scenario(
        path,
        config_root=Path(config_root),
        algo=algo,
        ports_per_scenario=int(ports_per_scenario),
        seed=int(seed),
        d_min_ref=float(d_min_ref),
    )


def _scenario_task_star(task: tuple[Any, ...]) -> list[TaskOutcome]:
    """``ProcessPoolExecutor.map`` 的单参数包装（逐项按关键字转发）。"""
    path, config_root, algo, ports_per_scenario, seed, d_min_ref = task
    return _scenario_task(
        str(path),
        config_root=str(config_root),
        algo=str(algo),
        ports_per_scenario=int(ports_per_scenario),
        seed=int(seed),
        d_min_ref=float(d_min_ref),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.multi_endpoint",
        description="同一户型多组起终点：制造「短-高阻 vs 长-低阻」同伦类候选",
    )
    parser.add_argument("--algo", default=DEFAULT_ALGO)
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--tier", default="all")
    parser.add_argument("--ports-per-scenario", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-min-ref", type=float, default=0.7)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_root = Path(args.config_root) if args.config_root else CONFIG_ROOT
    paths = (
        [Path(item) for item in args.scenarios]
        if args.scenarios
        else chapter3.scenario_paths(args.tier, config_root=config_root)
    )
    if not paths:
        print("没有找到场景", flush=True)
        return 2

    from concurrent.futures import ProcessPoolExecutor

    workers = int(args.jobs) if args.jobs else min(len(paths), os.cpu_count() or 1)
    tasks = [
        (
            str(path), str(config_root), args.algo,
            int(args.ports_per_scenario), int(args.seed), float(args.d_min_ref),
        )
        for path in paths
    ]

    outcomes: list[TaskOutcome] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_scenario_task_star, tasks):
                outcomes.extend(result)
    else:
        for task in tasks:
            outcomes.extend(_scenario_task(*task))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for outcome in outcomes:
            writer.writerow(row_for(outcome))
    print("\n多端点对任务（同一户型多组起终点）")
    for line in summarise(outcomes):
        print("  " + line)
    print(f"\n明细: {OUT_CSV}")
    if not args.no_plots:
        figure = plot_multi_endpoint(outcomes, Path(args.out_dir) / "multi_endpoint.png")
        print(f"OK   {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
