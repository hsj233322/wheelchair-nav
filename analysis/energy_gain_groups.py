"""能耗收益的分组分析：收益是否随「主路线高阻材质占比」单调上升。

```bash
uv run python -m analysis.energy_gain_groups                     # 60 个任务（20 冻结 + 40 附加）
uv run python -m analysis.energy_gain_groups --ports-per-scenario 0   # 只看 20 个冻结任务
uv run python -m analysis.energy_gain_groups --p-aux 0           # 简化能耗模型对照（去掉辅助功率）
```

**为什么需要这张表**：能耗加权 A*（ESA*）相对「仅长度 A*」的收益不是常数，
它取决于**任务是否经过高阻地形**——只有当"绕远走低阻"这条路线真实存在时，
优化能耗才有空间。分组依据是**主路线（本文约束化 A* 的路径）上高阻材质
（``c_rr >= 0.04``）的**路段长度加权**占比**：

| 组 | 高阻占比 |
|---|---|
| 低 | < 10% |
| 中 | 10% ~ 40% |
| 高 | > 40% |

每组报：任务数、ESA* 相对仅长度 A* 的**配对**能耗收益 Δ（中位数与总和）、
该组高阻占比范围。**配对**是硬要求（交接文档 7.8 口径规则 1）：同一任务、
同一可行域（两者都用 ``d_min = body_radius``），只换目标函数，因此 Δ 不会
混进策略半径或子集成分变化的影响。

**与已发表论文的 26% 不可并列对比**（README 有完整口径段）：那个数字来自
随机栅格上高阻近似满铺 + 简化能耗模型（无辅助功率项）；本框架是真实户型 +
完整能耗模型（辅助功率约占参考能耗 1/3）。本模块的 ``--p-aux 0`` 就是用于
**量化**这条差异：去掉辅助功率项后，同一批任务的收益会明显放大。
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_ALGO = "configs/frozen_A.yaml"
OUT_CSV = PROJECT_ROOT / "docs" / "energy_gain_groups.csv"
OUT_SUMMARY = PROJECT_ROOT / "docs" / "energy_gain_groups_summary.csv"
OUT_DIR = PROJECT_ROOT / "docs" / "chapter3"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from env import build_scenarios, terrain_partition  # noqa: E402
from runner import config as config_module  # noqa: E402
from runner import experiment  # noqa: E402
from analysis import chapter3, multi_endpoint  # noqa: E402

_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
)
HIGH_C_RR = 0.04
GROUP_EDGES: tuple[tuple[str, float, float], ...] = (
    ("low", 0.0, 0.10),
    ("mid", 0.10, 0.40),
    ("high", 0.40, 1.01),
)
GROUP_LABELS = {
    "low": "低（高阻 < 10%）",
    "mid": "中（10% ~ 40%）",
    "high": "高（> 40%）",
}
COLORS = {"low": "#9ecae1", "mid": "#fdae6b", "high": "#e6550d"}


def _setup_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.monospace"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def group_of(high_ratio: float) -> str:
    """高阻占比 -> 组名（区间左闭右开，最后一组含 1.0）。"""
    for name, low, high in GROUP_EDGES:
        if low <= high_ratio < high or (name == "high" and high_ratio >= 0.40):
            return name
    return "high"


@dataclass(frozen=True)
class TaskGain:
    scenario_id: str
    grade: str
    terrain_tier: str
    port: str
    high_ratio: float
    """主路线（本文约束化 A*）上 c_rr >= 0.04 的路段长度占比。"""
    length_m: float
    energy_j: float
    esa_length_m: float
    esa_energy_j: float
    len_length_m: float
    len_energy_j: float
    delta_length_m: float
    """ESA* - 仅长度 A*（长度差，正值 = 多走）。"""
    delta_energy_j: float
    """ESA* - 仅长度 A*（能耗差，负值 = 省电，即收益）。"""
    delta_energy_pct: float
    c_rr_hist: dict[float, float]

    @property
    def group(self) -> str:
        return group_of(self.high_ratio)

    @property
    def gain_j(self) -> float:
        """收益 [J]（正值 = 省电）。"""
        return -self.delta_energy_j


def _route_c_rr(
    scene: experiment.Scene,
    scenario: dict[str, Any],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    setup: experiment.PlannerSetup,
) -> tuple[Any, dict[float, float]]:
    """跑本文约束化 A*，返回 ``(PathResult, 路径上的 c_rr 长度加权分布)``。"""
    result = chapter3._plan(
        multi_endpoint._trial_scene(scene, start_xy, goal_xy), setup, setup.astar_params
    )
    class_grid = build_scenarios.terrain_partition_grid_from_scenario(scenario, scene)
    mapping = {
        name: float(props["c_rr"])
        for name, props in scenario["terrain"]["partition"]["classes"].items()
    }
    if not result.success:
        return result, {}
    histogram = terrain_partition.path_c_rr_histogram(
        result.cells, result.step_length, class_grid, mapping
    )
    return result, histogram


def run_scenario(
    scenario_path: str | Path,
    *,
    config_root: Path = CONFIG_ROOT,
    algo: str = DEFAULT_ALGO,
    ports_per_scenario: int = 2,
    seed: int = 42,
    d_min_ref: float = 0.7,
    p_aux: float | None = None,
) -> list[TaskGain]:
    """一个场景的 ``1 + ports_per_scenario`` 个任务：高阻占比 + ESA*/仅长度配对差。"""
    resolved = config_module.load_resolved(algo, scenario_path, config_root=config_root)
    scenario = resolved.scenario
    platform = dict(resolved.platform)
    if p_aux is not None:
        platform["p_aux"] = float(p_aux)
    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(platform, resolved.algo)
    body_radius = float(platform["body_radius"])
    setup_baseline = replace(
        setup,
        astar_params=replace(
            setup.astar_params, d_min=body_radius, collect_pareto=False
        ),
    )
    from env import scene_quality

    segmentation = scene_quality.segment_rooms(scene.clearance, core_clearance_m=0.9)
    rng = np.random.default_rng(
        terrain_partition.stable_seed(str(scenario["id"]), f"ports{seed}")
    )

    ports: list[tuple[str, tuple[float, float], tuple[float, float]]] = [
        ("frozen", scene.start_xy, scene.goal_xy)
    ]
    if ports_per_scenario > 0:
        for index, (cell_a, cell_b) in enumerate(
            multi_endpoint.sample_ports(
                scene.clearance,
                segmentation.labels,
                resolution=scene.resolution,
                d_min=float(setup.astar_params.d_min),
                rng=rng,
                ports=ports_per_scenario,
            ),
            start=1,
        ):
            ports.append(
                (
                    f"p{index}",
                    multi_endpoint._world(cell_a, scene),
                    multi_endpoint._world(cell_b, scene),
                )
            )

    tier = str(((scenario.get("terrain") or {}).get("partition") or {}).get("tier", "-"))
    grade = str((scenario.get("quality") or {}).get("grade", "-"))
    gains: list[TaskGain] = []
    for port, start_xy, goal_xy in ports:
        constrained, histogram = _route_c_rr(scene, scenario, start_xy, goal_xy, setup)
        if not constrained.success:
            continue
        esa = multi_endpoint._evaluate_esa(scene, setup_baseline, start_xy, goal_xy)
        baseline = multi_endpoint._evaluate_length_only(
            scene, setup_baseline, start_xy, goal_xy
        )
        if esa is None or baseline is None:
            continue
        high_ratio = float(
            sum(value for key, value in histogram.items() if key >= HIGH_C_RR - 1e-12)
        )
        energy_delta = esa[1] - baseline[1]
        gains.append(
            TaskGain(
                scenario_id=str(scenario["id"]),
                grade=grade,
                terrain_tier=tier,
                port=port,
                high_ratio=high_ratio,
                length_m=float(constrained.length_m),
                energy_j=float(constrained.total_energy_j),
                esa_length_m=float(esa[0]),
                esa_energy_j=float(esa[1]),
                len_length_m=float(baseline[0]),
                len_energy_j=float(baseline[1]),
                delta_length_m=float(esa[0] - baseline[0]),
                delta_energy_j=float(energy_delta),
                delta_energy_pct=(
                    100.0 * energy_delta / baseline[1] if baseline[1] > 0 else 0.0
                ),
                c_rr_hist=histogram,
            )
        )
    return gains


CSV_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "grade",
    "terrain_tier",
    "port",
    "group",
    "high_ratio",
    "length_m",
    "energy_j",
    "len_length_m",
    "len_energy_j",
    "esa_length_m",
    "esa_energy_j",
    "delta_length_m",
    "delta_energy_j",
    "gain_j",
    "delta_energy_pct",
    "hist_c_rr_0.015",
    "hist_c_rr_0.030",
    "hist_c_rr_0.040",
    "hist_c_rr_0.060",
)


def row_for(gain: TaskGain) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario_id": gain.scenario_id,
        "grade": gain.grade,
        "terrain_tier": gain.terrain_tier,
        "port": gain.port,
        "group": gain.group,
        "high_ratio": f"{gain.high_ratio:.4f}",
        "length_m": f"{gain.length_m:.4f}",
        "energy_j": f"{gain.energy_j:.3f}",
        "len_length_m": f"{gain.len_length_m:.4f}",
        "len_energy_j": f"{gain.len_energy_j:.3f}",
        "esa_length_m": f"{gain.esa_length_m:.4f}",
        "esa_energy_j": f"{gain.esa_energy_j:.3f}",
        "delta_length_m": f"{gain.delta_length_m:.4f}",
        "delta_energy_j": f"{gain.delta_energy_j:.3f}",
        "gain_j": f"{gain.gain_j:.3f}",
        "delta_energy_pct": f"{gain.delta_energy_pct:.4f}",
    }
    for level in (0.015, 0.030, 0.040, 0.060):
        row[f"hist_c_rr_{level:.3f}"] = f"{gain.c_rr_hist.get(level, 0.0):.4f}"
    return row


@dataclass(frozen=True)
class GroupSummary:
    group: str
    label: str
    tasks: int
    high_ratio_min: float
    high_ratio_max: float
    high_ratio_median: float
    gain_median_j: float
    gain_total_j: float
    gain_pct_median: float
    gain_pct_total: float
    length_delta_median_m: float
    improved_tasks: int
    """ESA* 能耗严格更低的任务数。"""
    longer_tasks: int
    """ESA* 走得更长的任务数。"""


def summarise(gains: Sequence[TaskGain]) -> list[GroupSummary]:
    rows: list[GroupSummary] = []
    for name, _low, _high in GROUP_EDGES:
        subset = [gain for gain in gains if gain.group == name]
        if not subset:
            rows.append(
                GroupSummary(name, GROUP_LABELS[name], 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
            )
            continue
        ratios = [gain.high_ratio for gain in subset]
        gains_j = [gain.gain_j for gain in subset]
        pcts = [gain.delta_energy_pct for gain in subset]
        base_total = sum(gain.len_energy_j for gain in subset)
        rows.append(
            GroupSummary(
                group=name,
                label=GROUP_LABELS[name],
                tasks=len(subset),
                high_ratio_min=float(min(ratios)),
                high_ratio_max=float(max(ratios)),
                high_ratio_median=float(np.median(ratios)),
                gain_median_j=float(np.median(gains_j)),
                gain_total_j=float(sum(gains_j)),
                gain_pct_median=float(np.median(pcts)),
                gain_pct_total=(
                    100.0 * sum(gains_j) / base_total if base_total > 0 else 0.0
                ),
                length_delta_median_m=float(
                    np.median([gain.delta_length_m for gain in subset])
                ),
                improved_tasks=sum(1 for gain in subset if gain.gain_j > 1e-9),
                longer_tasks=sum(1 for gain in subset if gain.delta_length_m > 1e-9),
            )
        )
    return rows


SUMMARY_COLUMNS: tuple[str, ...] = (
    "group",
    "label",
    "tasks",
    "high_ratio_min",
    "high_ratio_max",
    "high_ratio_median",
    "gain_median_j",
    "gain_total_j",
    "gain_pct_median",
    "gain_pct_total",
    "length_delta_median_m",
    "improved_tasks",
    "longer_tasks",
)


def summary_row(summary: GroupSummary) -> dict[str, Any]:
    return {
        "group": summary.group,
        "label": summary.label,
        "tasks": summary.tasks,
        "high_ratio_min": f"{summary.high_ratio_min:.4f}",
        "high_ratio_max": f"{summary.high_ratio_max:.4f}",
        "high_ratio_median": f"{summary.high_ratio_median:.4f}",
        "gain_median_j": f"{summary.gain_median_j:.3f}",
        "gain_total_j": f"{summary.gain_total_j:.3f}",
        "gain_pct_median": f"{summary.gain_pct_median:.4f}",
        "gain_pct_total": f"{summary.gain_pct_total:.4f}",
        "length_delta_median_m": f"{summary.length_delta_median_m:.4f}",
        "improved_tasks": summary.improved_tasks,
        "longer_tasks": summary.longer_tasks,
    }


def format_table(
    summaries: Sequence[GroupSummary], *, tasks: int, label: str
) -> str:
    header = (
        f"{'组':<16}{'任务':>5}{'高阻占比范围':>18}{'Δ能耗中位[J]':>13}"
        f"{'Δ能耗总和[J]':>14}{'Δ%中位':>9}{'Δ%总和':>9}{'多走中位[m]':>12}"
        f"{'省电任务':>9}{'走更长':>8}"
    )
    lines = [
        f"能耗收益分组（{label}，共 {tasks} 个任务；收益 = 仅长度 A* 能耗 − ESA* 能耗，正值 = 省电）",
        header,
        "-" * len(header),
    ]
    for summary in summaries:
        lines.append(
            f"{summary.label:<16}{summary.tasks:>5}"
            f"{f'{summary.high_ratio_min:.2f}~{summary.high_ratio_max:.2f}':>18}"
            f"{summary.gain_median_j:>13.2f}{summary.gain_total_j:>14.2f}"
            f"{summary.gain_pct_median:>9.2f}{summary.gain_pct_total:>9.2f}"
            f"{summary.length_delta_median_m:>12.3f}"
            f"{summary.improved_tasks:>9}{summary.longer_tasks:>8}"
        )
    return "\n".join(lines)


def plot_groups(gains: Sequence[TaskGain], summaries: Sequence[GroupSummary], out_path: Path) -> Path:
    """图：高阻占比 vs 配对收益散点（左）+ 分组收益柱状（右）。"""
    _setup_fonts()
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    axis = axes[0]
    for name, _low, _high in GROUP_EDGES:
        subset = [gain for gain in gains if gain.group == name]
        if not subset:
            continue
        axis.scatter(
            [gain.high_ratio for gain in subset],
            [gain.gain_j for gain in subset],
            s=42, color=COLORS[name], label=GROUP_LABELS[name], zorder=3,
        )
    axis.axhline(0.0, color="0.4", linestyle="--", linewidth=1.1)
    axis.set_xlabel("主路线高阻材质（c_rr >= 0.04）长度占比")
    axis.set_ylabel("ESA* 相对仅长度 A* 的能耗收益 [J]（正 = 省电）")
    axis.set_title("收益随高阻占比上升（同一可行域、只换目标函数）", fontsize=10.5)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8.5)

    axis = axes[1]
    positions = np.arange(len(summaries))
    axis.bar(
        positions - 0.2,
        [summary.gain_median_j for summary in summaries],
        width=0.4, color="#1f77b4", label="Δ能耗中位 [J]",
    )
    axis.bar(
        positions + 0.2,
        [summary.gain_total_j for summary in summaries],
        width=0.4, color="#2ca02c", label="Δ能耗总和 [J]",
    )
    for index, summary in enumerate(summaries):
        axis.text(positions[index] - 0.2, summary.gain_median_j, f"{summary.gain_median_j:.1f}",
                  ha="center", va="bottom", fontsize=8.5)
        axis.text(positions[index] + 0.2, summary.gain_total_j, f"{summary.gain_total_j:.0f}",
                  ha="center", va="bottom", fontsize=8.5)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{summary.label}\n({summary.tasks} 个任务)" for summary in summaries], fontsize=9
    )
    axis.set_ylabel("能耗收益 [J]")
    axis.set_title("分组收益：中位数与总和", fontsize=10.5)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8.5)
    figure.suptitle(
        "能耗优化空间取决于任务是否经过高阻地形（真实户型 + 完整能耗模型）",
        fontsize=11.5,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.energy_gain_groups",
        description="按主路线高阻材质占比分组统计 ESA* 的配对能耗收益",
    )
    parser.add_argument("--algo", default=DEFAULT_ALGO)
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--tier", default="all")
    parser.add_argument("--ports-per-scenario", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-min-ref", type=float, default=0.7)
    parser.add_argument(
        "--p-aux", type=float, default=None,
        help="覆盖 platform.p_aux [W]；0 = 简化能耗模型（无辅助功率项）对照",
    )
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--suffix", default="", help="输出文件名后缀（用于对照）")
    return parser


def _task(task: tuple[Any, ...]) -> list[TaskGain]:
    path, config_root, algo, ports, seed, d_min_ref, p_aux = task
    return run_scenario(
        str(path),
        config_root=Path(str(config_root)),
        algo=str(algo),
        ports_per_scenario=int(ports),
        seed=int(seed),
        d_min_ref=float(d_min_ref),
        p_aux=None if p_aux is None else float(p_aux),
    )


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

    tasks = [
        (
            str(path), str(config_root), args.algo, int(args.ports_per_scenario),
            int(args.seed), float(args.d_min_ref),
            None if args.p_aux is None else float(args.p_aux),
        )
        for path in paths
    ]

    from concurrent.futures import ProcessPoolExecutor

    workers = int(args.jobs) if args.jobs else min(len(tasks), os.cpu_count() or 1)
    gains: list[TaskGain] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_task, tasks):
                gains.extend(result)
    else:
        for task in tasks:
            gains.extend(_task(task))

    if not gains:
        print("没有任何可用任务", flush=True)
        return 2

    model_label = (
        f"P_aux = {args.p_aux:g} W" if args.p_aux is not None else "P_aux = 15 W（platform 冻结值）"
    )
    suffix = args.suffix
    details = OUT_CSV.with_name(f"energy_gain_groups{suffix}.csv")
    summary_path = OUT_SUMMARY.with_name(f"energy_gain_groups_summary{suffix}.csv")
    with details.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for gain in gains:
            writer.writerow(row_for(gain))
    summaries = summarise(gains)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary_row(summary))

    print(f"\n{format_table(summaries, tasks=len(gains), label=model_label)}")
    print(f"\n逐任务: {details}\n分组汇总: {summary_path}")
    if not args.no_plots:
        figure = plot_groups(
            gains, summaries, Path(args.out_dir) / f"energy_gain_groups{suffix}.png"
        )
        print(f"OK   {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
