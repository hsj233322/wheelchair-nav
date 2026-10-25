"""d_min 参数扫描：把 algo.astar.d_min 当作实验变量，逐场景逐档跑一遍约束化 A*。

平台/约束解耦后：

* 场景层只有物理半径 ``platform.body_radius``（写进场景 ``map.inflation_m``）；
* 策略安全半径 ``algo.astar.d_min`` 是**算法层**的可变参数，
  档位配置在 ``config/configs/dmin_sweep/dmin_{058,070,090,110}.yaml``。

```bash
uv run python -m analysis.dmin_sweep            # -> docs/dmin_sweep.csv + 终端汇总表
```

每行（场景 x 档位）记录：是否可行、路径长度、能耗、路径 clearance 的最小值与分布、
最宽瓶颈宽度（分割无关）、房间标签变化次数（依赖分割）。汇总表给出每档的
可行场景数与各指标中位数，直接作为第 3 章 Pareto 分析的一个轴。
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
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
    "grade",
    "feasible",
    "reason",
    "path_len_m",
    "energy_j",
    "clearance_min_m",
    "clearance_median_m",
    "clearance_below_090_ratio",
    "bottleneck_width_m",
    "bottleneck_crossings",
    "room_label_changes",
    "expanded",
)


def _sweep_configs(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"{directory} 里没有 d_min 档位配置")
    return files


def _point_mask(shape: tuple[int, int], cell: Sequence[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[int(cell[0]), int(cell[1])] = True
    return mask


def run_sweep(
    *,
    config_dir: Path,
    scenario_paths: Sequence[Path],
    platform_path: Path | None = None,
    config_root: Path = CONFIG_ROOT,
    threshold_m: float = 0.9,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algo_path in _sweep_configs(config_dir):
        for scenario_path in scenario_paths:
            resolved = config_module.load_resolved(
                str(algo_path), str(scenario_path),
                platform_path=platform_path, config_root=config_root,
            )
            scene = experiment.build_scene(resolved.scenario)
            setup = experiment.planner_setup(resolved.platform, resolved.algo)
            result = experiment.plan_path(scene, setup)
            quality = resolved.scenario.get("quality", {}) or {}
            row: dict[str, Any] = {
                "algo_id": resolved.algo["id"],
                "d_min_m": float(setup.astar_params.d_min),
                "scenario": resolved.scenario["id"],
                "grade": quality.get("grade", "-"),
                "feasible": int(result.success),
                "reason": result.reason,
                "path_len_m": round(result.length_m, 3),
                "energy_j": round(result.total_energy_j, 1),
                "clearance_min_m": 0.0,
                "clearance_median_m": 0.0,
                "clearance_below_090_ratio": 0.0,
                "bottleneck_width_m": 0.0,
                "bottleneck_crossings": 0,
                "room_label_changes": 0,
                "expanded": int(result.expanded),
            }
            if result.success:
                profile = scene_quality.clearance_profile(
                    result.cells,
                    scene.clearance,
                    result.step_length,
                    threshold_m=threshold_m,
                )
                row.update(
                    {
                        "clearance_min_m": round(profile["clearance_min_m"], 3),
                        "clearance_median_m": round(profile["clearance_median_m"], 3),
                        "clearance_below_090_ratio": round(
                            profile["clearance_below_090_ratio"], 4
                        ),
                        "bottleneck_width_m": round(
                            scene_quality.widest_connection_m(
                                scene.clearance,
                                _point_mask(scene.clearance.shape, result.cells[0]),
                                _point_mask(scene.clearance.shape, result.cells[-1]),
                            ),
                            3,
                        ),
                        "bottleneck_crossings": int(profile["bottleneck_crossings"]),
                        "room_label_changes": int(
                            scene_quality.path_metadata(
                                result.cells,
                                labels=scene_quality.segment_rooms(
                                    scene.clearance, core_clearance_m=0.9
                                ).labels,
                                clearance=scene.clearance,
                                free_mask=scene.clearance + 1e-9
                                >= setup.astar_params.d_min,
                            )["room_label_changes"]
                        ),
                    }
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
        feasible = [row for row in group if row["feasible"]]
        per_grade: dict[str, str] = {}
        for grade in (scene_quality.EASY, scene_quality.MEDIUM, scene_quality.HARD):
            total = [row for row in group if row["grade"] == grade]
            ok = [row for row in total if row["feasible"]]
            if total:
                per_grade[grade] = f"{len(ok)}/{len(total)}"
        summary.append(
            {
                "algo_id": algo_id,
                "d_min_m": float(group[0]["d_min_m"]),
                "n_total": len(group),
                "n_feasible": len(feasible),
                "per_grade": per_grade,
                "path_len_median_m": statistics.median(
                    [row["path_len_m"] for row in feasible]
                ) if feasible else 0.0,
                "energy_median_j": statistics.median(
                    [row["energy_j"] for row in feasible]
                ) if feasible else 0.0,
                "clearance_min_median_m": statistics.median(
                    [row["clearance_min_m"] for row in feasible]
                ) if feasible else 0.0,
                "below_090_median": statistics.median(
                    [row["clearance_below_090_ratio"] for row in feasible]
                ) if feasible else 0.0,
                "bottleneck_median_m": statistics.median(
                    [row["bottleneck_width_m"] for row in feasible]
                ) if feasible else 0.0,
            }
        )
    return summary


def format_table(summary: Sequence[dict[str, Any]]) -> str:
    header = (
        "| d_min [m] | 可行场景（总） | 可行 易/中/难 | 路径长中位数 [m] | "
        "能耗中位数 [J] | 最小离障中位数 [m] | clearance<0.9 长度占比中位数 | "
        "瓶颈宽度中位数 [m] |"
    )
    lines = [
        header,
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in summary:
        per_grade = item["per_grade"]
        grades = ", ".join(
            per_grade.get(name, "0/0")
            for name in (scene_quality.EASY, scene_quality.MEDIUM, scene_quality.HARD)
        )
        lines.append(
            f"| {item['d_min_m']:.2f} | {item['n_feasible']}/{item['n_total']} | "
            f"{grades} | "
            f"{item['path_len_median_m']:.2f} | {item['energy_median_j']:.1f} | "
            f"{item['clearance_min_median_m']:.3f} | "
            f"{item['below_090_median']:.3f} | {item['bottleneck_median_m']:.2f} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.dmin_sweep",
        description="按 d_min 档位扫描全部场景，输出对比表",
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
    return parser


def _resolve_scenarios(pattern_text: str, config_root: Path) -> list[Path]:
    pattern = Path(pattern_text)
    if pattern.is_absolute():
        return sorted(pattern.parent.glob(pattern.name))
    found = sorted(Path().glob(pattern_text))
    if found:
        return found
    return sorted(config_root.glob(pattern.name))


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
