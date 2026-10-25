"""d_min 参数扫描：把 algo.astar.d_min 当作实验变量，逐场景逐档跑约束化 A*。

平台/约束解耦后：

* 场景层只有物理半径 ``platform.body_radius``（写进场景 ``map.inflation_m``）；
* 策略安全半径 ``algo.astar.d_min`` 是**算法层**的可变参数，
  档位配置在 ``config/configs/dmin_sweep/dmin_{058,070,090,110}.yaml``。

每档给出**两个口径**（同一行两组列，都写进 ``docs/dmin_sweep.csv``）：

1. **固定起终点**（``frozen_*``）：沿用场景里冻结的起终点。在高档位下它可能直接
   ``start_violates_clearance`` —— 这只能说明"冻结任务在新档不合法"，**不能**说明
   "该档没有可行任务"。
2. **按档位重新采样**（``resample_*``/``resampled_*``）：在该档可行域
   （``clearance >= d_min``）内、同一连通分量内重新采样起终点，并保持场景原有的
   难度档结构（``env.scene_quality.sample_task_cells``）。``resample_tier_ok`` 表示
   该档存在该档结构的任务（采样成功率），``resampled_feasible`` 表示采样后 A* 有解。

```bash
uv run python -m analysis.dmin_sweep     # -> docs/dmin_sweep.csv + 终端汇总表
```

难度档只由 ``bottleneck_width_m``（起终点之间的最宽瓶颈宽度）决定，与 d_min、
与 A* 走哪条路都无关，所以同一场景在四个档位下的档位标签一致。
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from env import scene_quality
from runner import config as config_module
from runner import experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "dmin_sweep.csv"

CSV_COLUMNS = (
    "algo_id",
    "d_min_m",
    "scenario",
    "tier",
    # 固定起终点
    "frozen_feasible",
    "frozen_reason",
    "frozen_path_len_m",
    "frozen_energy_j",
    "frozen_clearance_min_m",
    "frozen_below_090_ratio",
    "frozen_bottleneck_width_m",
    # 按档位重新采样
    "resample_tier_ok",
    "resample_attempts",
    "resampled_feasible",
    "resampled_reason",
    "resampled_start",
    "resampled_goal",
    "resampled_path_len_m",
    "resampled_energy_j",
    "resampled_clearance_min_m",
    "resampled_below_090_ratio",
    "resampled_bottleneck_width_m",
)


def _sweep_configs(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"{directory} 里没有 d_min 档位配置")
    return files


def _point_mask(shape: tuple[int, int], cell) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[int(cell[0]), int(cell[1])] = True
    return mask


def _world(
    cell: Sequence[int], resolution: float, origin: Sequence[float]
) -> tuple[float, float]:
    return (
        float(origin[0]) + (int(cell[1]) + 0.5) * resolution,
        float(origin[1]) + (int(cell[0]) + 0.5) * resolution,
    )


def _profile(
    result, scene: experiment.Scene, threshold_m: float
) -> dict[str, float]:
    return scene_quality.clearance_profile(
        result.cells, scene.clearance, result.step_length, threshold_m=threshold_m
    )


def _frozen_row(resolved, scene, setup, threshold_m: float) -> dict[str, Any]:
    result = experiment.plan_path(scene, setup)
    row: dict[str, Any] = {
        "frozen_feasible": int(result.success),
        "frozen_reason": result.reason,
        "frozen_path_len_m": round(result.length_m, 3),
        "frozen_energy_j": round(result.total_energy_j, 1),
        "frozen_clearance_min_m": 0.0,
        "frozen_below_090_ratio": 0.0,
        "frozen_bottleneck_width_m": 0.0,
    }
    if result.success:
        profile = _profile(result, scene, threshold_m)
        row.update(
            {
                "frozen_clearance_min_m": round(profile["clearance_min_m"], 3),
                "frozen_below_090_ratio": round(
                    profile["clearance_below_090_ratio"], 4
                ),
                "frozen_bottleneck_width_m": round(
                    scene_quality.widest_connection_m(
                        scene.clearance,
                        _point_mask(scene.clearance.shape, result.cells[0]),
                        _point_mask(scene.clearance.shape, result.cells[-1]),
                    ),
                    3,
                ),
            }
        )
    return row


def _resampled_row(
    resolved,
    scene,
    setup,
    *,
    tier: str,
    threshold_m: float,
    min_start_goal_m: float,
    max_samples: int,
) -> dict[str, Any]:
    d_min = float(setup.astar_params.d_min)
    labels = scene_quality.segment_rooms(scene.clearance, core_clearance_m=0.9).labels
    feasible_mask = scene.clearance + 1e-9 >= d_min
    seed = zlib.crc32(f"{resolved.algo['id']}:{resolved.scenario['id']}".encode("utf-8"))
    rng = np.random.default_rng(seed)
    samples = scene_quality.sample_task_cells(
        scene.clearance,
        labels,
        feasible_mask,
        scene.resolution,
        tier=tier,
        rng=rng,
        min_start_goal_m=min_start_goal_m,
        max_samples=max_samples,
    )
    row: dict[str, Any] = {
        "resample_tier_ok": 0,
        "resample_attempts": 0,
        "resampled_feasible": 0,
        "resampled_reason": "no_tier_task" if not samples else "no_feasible_sample",
        "resampled_start": "",
        "resampled_goal": "",
        "resampled_path_len_m": 0.0,
        "resampled_energy_j": 0.0,
        "resampled_clearance_min_m": 0.0,
        "resampled_below_090_ratio": 0.0,
        "resampled_bottleneck_width_m": 0.0,
    }
    for cell_a, cell_b in samples:
        row["resample_attempts"] += 1
        start = _world(cell_a, scene.resolution, scene.origin)
        goal = _world(cell_b, scene.resolution, scene.origin)
        trial = replace(
            scene, start_xy=start, goal_xy=goal, start_cell=cell_a, goal_cell=cell_b
        )
        result = experiment.plan_path(trial, setup)
        if not result.success:
            continue
        width = scene_quality.widest_connection_m(
            scene.clearance,
            _point_mask(scene.clearance.shape, cell_a),
            _point_mask(scene.clearance.shape, cell_b),
        )
        if scene_quality.grade_scene(bottleneck_width_m=width) != tier:
            continue  # 结构不符（墙壁更窄/更宽），继续采
        profile = _profile(result, scene, threshold_m)
        row.update(
            {
                "resample_tier_ok": 1,
                "resampled_feasible": 1,
                "resampled_reason": result.reason,
                "resampled_start": f"{start[0]:.3f},{start[1]:.3f}",
                "resampled_goal": f"{goal[0]:.3f},{goal[1]:.3f}",
                "resampled_path_len_m": round(result.length_m, 3),
                "resampled_energy_j": round(result.total_energy_j, 1),
                "resampled_clearance_min_m": round(profile["clearance_min_m"], 3),
                "resampled_below_090_ratio": round(
                    profile["clearance_below_090_ratio"], 4
                ),
                "resampled_bottleneck_width_m": round(width, 3),
            }
        )
        break
    return row


def run_sweep(
    *,
    config_dir: Path,
    scenario_paths: Sequence[Path],
    platform_path: Path | None = None,
    config_root: Path = CONFIG_ROOT,
    threshold_m: float = 0.9,
    min_start_goal_m: float = 4.0,
    max_samples: int = 12,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algo_path in _sweep_configs(config_dir):
        for scenario_path in scenario_paths:
            resolved = config_module.load_resolved(
                str(algo_path),
                str(scenario_path),
                platform_path=platform_path,
                config_root=config_root,
            )
            scene = experiment.build_scene(resolved.scenario)
            setup = experiment.planner_setup(resolved.platform, resolved.algo)
            quality = resolved.scenario.get("quality", {}) or {}
            tier = str(quality.get("grade", scene_quality.EASY))
            row: dict[str, Any] = {
                "algo_id": resolved.algo["id"],
                "d_min_m": f"{float(setup.astar_params.d_min):.2f}",
                "scenario": resolved.scenario["id"],
                "tier": tier,
            }
            row.update(_frozen_row(resolved, scene, setup, threshold_m))
            row.update(
                _resampled_row(
                    resolved,
                    scene,
                    setup,
                    tier=tier,
                    threshold_m=threshold_m,
                    min_start_goal_m=min_start_goal_m,
                    max_samples=max_samples,
                )
            )
            rows.append(row)
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_algo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_algo.setdefault(str(row["algo_id"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for algo_id, group in sorted(
        by_algo.items(), key=lambda item: float(item[1][0]["d_min_m"])
    ):
        def medians(prefix: str, field: str) -> float:
            values = [
                float(row[f"{prefix}{field}"])
                for row in group
                if int(row[f"{prefix}feasible"]) == 1
            ]
            return statistics.median(values) if values else 0.0

        summary.append(
            {
                "algo_id": algo_id,
                "d_min_m": float(group[0]["d_min_m"]),
                "n_total": len(group),
                "frozen_feasible": sum(int(row["frozen_feasible"]) for row in group),
                "frozen_path_len_median": medians("frozen_", "path_len_m"),
                "frozen_energy_median": medians("frozen_", "energy_j"),
                "frozen_clearance_median": medians("frozen_", "clearance_min_m"),
                "resample_tier_ok": sum(int(row["resample_tier_ok"]) for row in group),
                "resampled_feasible": sum(
                    int(row["resampled_feasible"]) for row in group
                ),
                "resampled_path_len_median": medians("resampled_", "path_len_m"),
                "resampled_energy_median": medians("resampled_", "energy_j"),
                "resampled_clearance_median": medians("resampled_", "clearance_min_m"),
            }
        )
    return summary


def format_table(summary: Sequence[dict[str, Any]]) -> str:
    header = (
        "| d_min [m] | 固定:可行 | 固定:路径中位 [m] | 固定:能耗中位 [J] | "
        "固定:离障中位 [m] | 重采样:成功率 | 重采样:可行任务 | 重采样:路径中位 [m] | "
        "重采样:能耗中位 [J] | 重采样:离障中位 [m] |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|---|---|"]
    for item in summary:
        lines.append(
            f"| {item['d_min_m']:.2f} | "
            f"{item['frozen_feasible']}/{item['n_total']} | "
            f"{item['frozen_path_len_median']:.2f} | "
            f"{item['frozen_energy_median']:.1f} | "
            f"{item['frozen_clearance_median']:.3f} | "
            f"{item['resample_tier_ok']}/{item['n_total']} | "
            f"{item['resampled_feasible']}/{item['n_total']} | "
            f"{item['resampled_path_len_median']:.2f} | "
            f"{item['resampled_energy_median']:.1f} | "
            f"{item['resampled_clearance_median']:.3f} |"
        )
    return "\n".join(lines)


def _resolve_scenarios(pattern_text: str, config_root: Path) -> list[Path]:
    pattern = Path(pattern_text)
    if pattern.is_absolute():
        return sorted(pattern.parent.glob(pattern.name))
    found = sorted(Path().glob(pattern_text))
    if found:
        return found
    return sorted(config_root.glob(pattern.name))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.dmin_sweep",
        description="按 d_min 档位扫描全部场景（固定起终点 + 按档位重新采样），输出对比表",
    )
    parser.add_argument(
        "--configs", default=str(CONFIG_ROOT / "configs" / "dmin_sweep")
    )
    parser.add_argument(
        "--scenarios", default=str(CONFIG_ROOT / "scenarios" / "houseexpo_*.yaml")
    )
    parser.add_argument("--platform", default=None)
    parser.add_argument("--config-root", default=str(CONFIG_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--threshold-m", type=float, default=0.9)
    parser.add_argument("--min-start-goal-m", type=float, default=4.0)
    parser.add_argument("--max-samples", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_root = Path(args.config_root)
    scenario_paths = _resolve_scenarios(args.scenarios, config_root)
    if not scenario_paths:
        print(f"没有找到场景: {args.scenarios}", file=sys.stderr)
        return 2
    config_dir = Path(args.configs)
    if not config_dir.is_dir():
        print(f"没有找到 d_min 档位目录: {config_dir}", file=sys.stderr)
        return 2

    rows = run_sweep(
        config_dir=config_dir,
        scenario_paths=scenario_paths,
        platform_path=Path(args.platform) if args.platform else None,
        config_root=config_root,
        threshold_m=args.threshold_m,
        min_start_goal_m=args.min_start_goal_m,
        max_samples=args.max_samples,
    )
    summary = summarize(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(format_table(summary))
    print(f"\n{len(rows)} runs -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
