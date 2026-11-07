"""曲率裕度前馈对照：frozen_A（无前馈）vs frozen_A_feedforward（+55 mm）。

背景（交接文档 7.7 的收尾项）：平滑后的 B 样条在 45° 转弯处必然向内切掉一段深度

    Δ = r (1/cos(Δθ/2) − 1),   r = 1/κ_max

κ_max = 1.5 1/m 时 r = 0.667 m、Δθ = 45° ⇒ Δ ≈ 55 mm。

所以**若 A* 折线的离障只有 d_min + 11.6 mm**，任何满足 κ ≤ κ_max 的平滑曲线都必然
跌破 d_min —— 这不是算法没调好，是几何上无解。前馈就是让全局层按
``d_min + clearance_margin_m`` 规划，把这段已知会被切掉的深度提前留出来。

```bash
uv run python -m analysis.feedforward                 # 20 个户型场景，两组对照
uv run python -m analysis.feedforward --tier medium
```

输出 ``docs/feedforward_compare.csv``（逐场景）与
``docs/feedforward_summary.csv``（按档位汇总）。
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core import astar, bspline, energy
from runner import config as config_module
from runner import experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
BASELINE_ALGO = "configs/frozen_A.yaml"
FEEDFORWARD_ALGO = "configs/frozen_A_feedforward.yaml"
COMPARE_CSV = PROJECT_ROOT / "docs" / "feedforward_compare.csv"
SUMMARY_CSV = PROJECT_ROOT / "docs" / "feedforward_summary.csv"


@dataclass(frozen=True)
class RunOutcome:
    """一组配置在一个场景上的结果。"""

    algo: str
    scenario_id: str
    grade: str
    plan_ok: bool
    plan_reason: str
    d_min_policy: float
    d_min_astar: float
    clearance_margin_m: float
    astar_length_m: float
    astar_clearance_min_m: float
    smooth_status: str
    smooth_reason: str
    curvature_ok: bool
    smooth_clearance_min_m: float
    smooth_energy_j: float
    smooth_length_m: float
    max_abs_curvature: float
    inserted_points: int


def run_one(
    scenario_path: Path, algo: str, *, config_root: Path = CONFIG_ROOT
) -> RunOutcome:
    """在一组 algo 配置下跑 A* + 平滑，记录两组离障口径与曲率结果。"""
    resolved = config_module.load_resolved(
        algo, scenario_path, config_root=config_root
    )
    scenario = resolved.scenario
    quality = scenario.get("quality", {}) or {}
    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    result = experiment.plan_path(scene, setup)

    base = dict(
        algo=str(resolved.algo["id"]),
        scenario_id=str(scenario["id"]),
        grade=str(quality.get("grade", "-")),
        d_min_policy=float(setup.d_min_policy),
        d_min_astar=float(setup.astar_params.d_min),
        clearance_margin_m=float(setup.clearance_margin_m),
    )

    if not result.success:
        # A* 在前馈档下可能直接判不可行——这正是"从违反曲率变成明确不可行"。
        return RunOutcome(
            **base,
            plan_ok=False,
            plan_reason=result.reason,
            astar_length_m=0.0,
            astar_clearance_min_m=0.0,
            smooth_status="skipped",
            smooth_reason="plan_failed",
            curvature_ok=False,
            smooth_clearance_min_m=0.0,
            smooth_energy_j=0.0,
            smooth_length_m=0.0,
            max_abs_curvature=float("nan"),
            inserted_points=0,
        )

    smooth = experiment.smooth_result(result, scene, setup)
    return RunOutcome(
        **base,
        plan_ok=True,
        plan_reason=result.reason,
        astar_length_m=float(result.length_m),
        astar_clearance_min_m=float(result.clearance.min()),
        smooth_status=smooth.status,
        smooth_reason=smooth.report.reason,
        curvature_ok=bool(smooth.curvature_ok),
        smooth_clearance_min_m=float(smooth.min_clearance),
        smooth_energy_j=float(smooth.energy_j),
        smooth_length_m=float(smooth.length_m),
        max_abs_curvature=float(smooth.max_abs_curvature),
        inserted_points=smooth.report.inserted_points,
    )


def list_scenarios(tier: str, *, config_root: Path = CONFIG_ROOT) -> list[Path]:
    paths: list[Path] = []
    for path in sorted((config_root / "scenarios").glob("houseexpo_*.yaml")):
        if tier == "all":
            paths.append(path)
            continue
        doc = config_module.read_config(path)
        if str((doc.get("quality") or {}).get("grade", "")) == tier:
            paths.append(path)
    return paths


COMPARE_COLUMNS: tuple[str, ...] = (
    "group",
    "algo",
    "scenario_id",
    "grade",
    "d_min_policy_m",
    "d_min_astar_m",
    "clearance_margin_m",
    "plan_ok",
    "plan_reason",
    "astar_length_m",
    "astar_min_clearance_m",
    "smooth_status",
    "smooth_reason",
    "curvature_satisfied",
    "curvature_infeasible",
    "smooth_min_clearance_m",
    "smooth_energy_j",
    "smooth_length_m",
    "smooth_max_abs_curvature",
    "inserted_points",
)


def _compare_row(group: str, outcome: RunOutcome) -> dict[str, Any]:
    kappa = outcome.max_abs_curvature
    return {
        "group": group,
        "algo": outcome.algo,
        "scenario_id": outcome.scenario_id,
        "grade": outcome.grade,
        "d_min_policy_m": f"{outcome.d_min_policy:.4f}",
        "d_min_astar_m": f"{outcome.d_min_astar:.4f}",
        "clearance_margin_m": f"{outcome.clearance_margin_m:.4f}",
        "plan_ok": int(outcome.plan_ok),
        "plan_reason": outcome.plan_reason,
        "astar_length_m": f"{outcome.astar_length_m:.4f}",
        "astar_min_clearance_m": f"{outcome.astar_clearance_min_m:.4f}",
        "smooth_status": outcome.smooth_status,
        "smooth_reason": outcome.smooth_reason,
        "curvature_satisfied": int(outcome.curvature_ok),
        "curvature_infeasible": int(not outcome.curvature_ok),
        "smooth_min_clearance_m": f"{outcome.smooth_clearance_min_m:.4f}",
        "smooth_energy_j": f"{outcome.smooth_energy_j:.3f}",
        "smooth_length_m": f"{outcome.smooth_length_m:.4f}",
        "smooth_max_abs_curvature": (
            "nan" if np.isnan(kappa)
            else ("inf" if np.isinf(kappa) else f"{kappa:.4f}")
        ),
        "inserted_points": outcome.inserted_points,
    }


SUMMARY_COLUMNS: tuple[str, ...] = (
    "grade",
    "group",
    "algo",
    "tasks",
    "plan_ok_tasks",
    "curvature_satisfied",
    "curvature_infeasible",
    "curvature_infeasible_rate",
    "median_smooth_clearance_m",
    "median_smooth_energy_j",
    "median_astar_length_m",
    "median_smooth_length_m",
    "median_max_abs_curvature",
)


def summarize(
    outcomes: Sequence[RunOutcome], grades: Sequence[str], groups: Sequence[str]
) -> list[dict[str, Any]]:
    """按档位 + 组别汇总：违反率 / 可行任务数 / 平滑后离障与能耗。

    **口径警告**：``median_smooth_energy_j`` 与 ``median_smooth_clearance_m`` 只在
    **曲率满足**的子集上取中位数。这个子集的成分在两组之间会变（前馈档救回/丢失
    不同任务），所以**跨组的该两列不可直接横向比较**——它反映的是"可平滑任务集合"
    的差异，不是单个任务的增减。配对比较见 ``paired_deltas``。
    """
    rows: list[dict[str, Any]] = []
    for grade in grades:
        for group in groups:
            subset = [
                o for o in outcomes if o.grade == grade and o.algo == group
            ]
            if not subset:
                continue
            plan_ok = [o for o in subset if o.plan_ok]
            satisfied = [o for o in subset if o.curvature_ok]
            feasible = plan_ok
            trackable = satisfied
            clearances = [o.smooth_clearance_min_m for o in trackable]
            energies = [o.smooth_energy_j for o in trackable]
            lengths_before = [o.astar_length_m for o in feasible]
            lengths_after = [o.smooth_length_m for o in feasible]
            kappas = [o.max_abs_curvature for o in trackable]
            rows.append(
                {
                    "grade": grade,
                    "group": group,
                    "algo": subset[0].algo,
                    "tasks": len(subset),
                    # 可行任务数 = A* 能给出路径的任务数（前馈档会因门槛抬高而变小）
                    "plan_ok_tasks": len(feasible),
                    "curvature_satisfied": len(satisfied),
                    "curvature_infeasible": len(subset) - len(satisfied),
                    "curvature_infeasible_rate": (
                        f"{(len(subset) - len(satisfied)) / len(subset):.3f}"
                    ),
                    "median_smooth_clearance_m": (
                        f"{np.median(clearances):.4f}" if clearances else "-"
                    ),
                    "median_smooth_energy_j": (
                        f"{np.median(energies):.1f}" if energies else "-"
                    ),
                    "median_astar_length_m": (
                        f"{np.median(lengths_before):.3f}" if lengths_before else "-"
                    ),
                    "median_smooth_length_m": (
                        f"{np.median(lengths_after):.3f}" if lengths_after else "-"
                    ),
                    "median_max_abs_curvature": (
                        f"{np.median(kappas):.3f}" if kappas else "-"
                    ),
                }
            )
    return rows


@dataclass(frozen=True)
class PairedDelta:
    """同一场景在两组配置下的配对差异（避免"子集成分变化"造成的假象）。"""

    scenario_id: str
    grade: str
    baseline_ok: bool
    feedforward_ok: bool
    baseline_plan_ok: bool
    feedforward_plan_ok: bool
    delta_clearance_m: float
    delta_energy_j: float
    delta_length_m: float

    @property
    def comparable(self) -> bool:
        """两组是否**都**产出了可比较的平滑结果。

        A* 在前馈档直接无解时，``smooth_*`` 字段是 0 占位，此时算 Δ 会得到
        "-0.7 m 离障""−963 J 能耗"这类**伪差**，必须排除在配对差值统计之外。
        """
        return self.baseline_plan_ok and self.feedforward_plan_ok

    @property
    def transition(self) -> str:
        if self.baseline_plan_ok != self.feedforward_plan_ok:
            return "plan_ok->plan_fail" if self.baseline_plan_ok else "plan_fail->plan_ok"
        if not self.feedforward_plan_ok:
            return "both_plan_fail"
        if self.baseline_ok and self.feedforward_ok:
            return "kept_feasible"
        if not self.baseline_ok and self.feedforward_ok:
            return "rescued"
        if self.baseline_ok and not self.feedforward_ok:
            return "broken"
        return "both_infeasible"


def paired_deltas(
    outcomes: Sequence[RunOutcome], baseline: str, feedforward: str
) -> list[PairedDelta]:
    """把同一场景的两组结果配对，逐任务算差异。"""
    by_key: dict[tuple[str, str], RunOutcome] = {
        (o.scenario_id, o.algo): o for o in outcomes
    }
    deltas: list[PairedDelta] = []
    for scenario_id in sorted({o.scenario_id for o in outcomes}):
        base = by_key.get((scenario_id, baseline))
        feed = by_key.get((scenario_id, feedforward))
        if base is None or feed is None:
            continue
        deltas.append(
            PairedDelta(
                scenario_id=scenario_id,
                grade=base.grade,
                baseline_ok=base.curvature_ok,
                feedforward_ok=feed.curvature_ok,
                baseline_plan_ok=base.plan_ok,
                feedforward_plan_ok=feed.plan_ok,
                delta_clearance_m=(
                    feed.smooth_clearance_min_m - base.smooth_clearance_min_m
                ),
                delta_energy_j=feed.smooth_energy_j - base.smooth_energy_j,
                delta_length_m=feed.smooth_length_m - base.smooth_length_m,
            )
        )
    return deltas


def _write(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.feedforward",
        description="曲率裕度前馈对照：无前馈 vs d_min+55 mm",
    )
    parser.add_argument("--tier", default="all",
                        help="难度档：easy / medium / hard / all（默认 all）")
    parser.add_argument("--baseline", default=BASELINE_ALGO)
    parser.add_argument("--feedforward", default=FEEDFORWARD_ALGO)
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--compare-report", default=str(COMPARE_CSV))
    parser.add_argument("--summary-report", default=str(SUMMARY_CSV))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_root = Path(args.config_root) if args.config_root else CONFIG_ROOT
    paths = list_scenarios(args.tier, config_root=config_root)
    if not paths:
        print(f"没有找到场景（tier={args.tier}）", file=sys.stderr)
        return 2

    outcomes: list[RunOutcome] = []
    failures = 0
    for path in paths:
        for algo in (args.baseline, args.feedforward):
            try:
                outcomes.append(run_one(path, algo, config_root=config_root))
            except config_module.ConfigError as exc:
                failures += 1
                print(f"FAIL {path} / {algo}: {exc}", file=sys.stderr)

    if not outcomes:
        return 1

    grades = sorted({o.grade for o in outcomes})
    groups = sorted({o.algo for o in outcomes})
    # 配对比较以 algo 的 id（不是文件路径）为准
    baseline_id = config_module.load_resolved(
        args.baseline, paths[0], config_root=config_root
    ).algo["id"]
    feedforward_id = config_module.load_resolved(
        args.feedforward, paths[0], config_root=config_root
    ).algo["id"]
    compare_rows = [
        _compare_row(algo, o) for algo in groups for o in outcomes if o.algo == algo
    ]
    print(f"compare: {_write(Path(args.compare_report), COMPARE_COLUMNS, compare_rows)}")

    summary = summarize(outcomes, grades, groups)
    print(f"summary: {_write(Path(args.summary_report), SUMMARY_COLUMNS, summary)}")

    print(
        f"\n{'档位':<8}{'配置':<24}{'任务':>4}{'A*可行':>7}"
        f"{'曲率满足':>9}{'违反率':>8}{'离障中位':>10}{'能耗中位':>10}"
    )
    for row in summary:
        print(
            f"{row['grade']:<8}{row['algo']:<24}{row['tasks']:>4}"
            f"{row['plan_ok_tasks']:>7}{row['curvature_satisfied']:>9}"
            f"{row['curvature_infeasible_rate']:>8}"
            f"{row['median_smooth_clearance_m']:>10}"
            f"{row['median_smooth_energy_j']:>10}"
        )

    deltas = paired_deltas(outcomes, baseline_id, feedforward_id)
    if deltas:
        print("\n配对转移（同一场景在前馈前后的状态变化）：")
        transitions: dict[str, int] = {}
        for delta in deltas:
            transitions[delta.transition] = transitions.get(delta.transition, 0) + 1
        for name, count in sorted(transitions.items()):
            print(f"  {name:<20}{count:>4}")
        changed = [
            d for d in deltas if d.comparable and abs(d.delta_clearance_m) > 1e-9
        ]
        if changed:
            print("\n逐任务离障变化（仅列两组都产出结果的场景）：")
            print(f"  {'场景':<24}{'档位':<8}{'Δ离障[m]':>10}{'Δ能耗[J]':>10}")
            for delta in sorted(changed, key=lambda d: -d.delta_clearance_m):
                print(
                    f"  {delta.scenario_id:<24}{delta.grade:<8}"
                    f"{delta.delta_clearance_m:>+10.4f}{delta.delta_energy_j:>+10.1f}"
                )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
