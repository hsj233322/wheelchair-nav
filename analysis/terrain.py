"""受控材质因子的事实核验与敏感性分析（第 2 章建模 -> 第 3 章前沿的支撑材料）。

```bash
uv run python -m analysis.terrain                  # 分布表 + 直方图 + 图
uv run python -m analysis.terrain --sensitivity    # 附 ±30% 与 p_aux 敏感性
```

四件事：

1. **``c_rr`` 分布直方图 + 实际面积占比**（`docs/terrain_分布.csv`）：
   逐场景给自由区与"侵蚀后可行区"的材质面积占比，以及每个材质类别的实际占比与
   目标占比的偏差（受控因子的兑现度）；
2. **候选路线上的 ``c_rr`` 覆盖**（`docs/terrain_route_coverage.csv`）：
   把 A* 主路线与多同伦候选路线上的材质按**路段长度加权**统计，确认高阻地形
   真的落在候选路线上，而不是躺在没人去的地方；
3. **±30% 敏感性 + ``p_aux`` 敏感性**（`docs/terrain_sensitivity.csv`）：
   `c_rr` 整体缩放 ×{0.7, 1.0, 1.3}，`P_aux` ∈ {7.5, 15, 22.5, 30} W，
   报告每档的路径长度/能耗/前沿点数/ESA* 是否与仅长度 A* 有差异；
4. 图：`docs/terrain/c_rr_histogram.png`（逐场景堆叠占比）与
   `docs/terrain/route_overlay_*.png`（材质分区 + 候选路线叠图）。

口径（论文必须写）：**材质空间分布是受控实验因子，不是实测数据**；
`c_rr` 取自公开文献范围，`P_aux` / 电池参数取自平台层的冻结值。
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_ALGO = "configs/frozen_A.yaml"
OUT_DIR = PROJECT_ROOT / "docs" / "terrain"
REVIEW_DIR = PROJECT_ROOT / "docs" / "review"
RATIO_CSV = PROJECT_ROOT / "docs" / "terrain_ratios.csv"
ROUTE_CSV = PROJECT_ROOT / "docs" / "terrain_route_coverage.csv"
SENSITIVITY_CSV = PROJECT_ROOT / "docs" / "terrain_sensitivity.csv"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from core import astar, energy as energy_module  # noqa: E402
from env import build_scenarios, terrain_partition  # noqa: E402
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

# 材质配色（按 c_rr 升高：越暖越高阻）
C_RR_COLORS: dict[float, str] = {
    0.015: "#a6cee3",
    0.030: "#b2df8a",
    0.040: "#fdbf6f",
    0.060: "#e31a1c",
}
C_RR_LABELS: dict[float, str] = {
    0.015: "0.015 瓷砖/抛光水泥",
    0.030: "0.030 水泥地面",
    0.040: "0.040 短绒地毯/防滑垫",
    0.060: "0.060 地面格栅/粗糙防滑区",
}
ROUTE_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf")
P_AUX_VALUES: tuple[float, ...] = (7.5, 15.0, 22.5, 30.0)
C_RR_SCALES: tuple[float, ...] = (0.7, 1.0, 1.3)


def _setup_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.monospace"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


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
# 1. c_rr 分布直方图与面积占比
# ---------------------------------------------------------------------------


def c_rr_by_class(scenario: dict[str, Any]) -> dict[str, float]:
    """场景里出现的 ``材质名 -> c_rr``（含 default 与 partition.classes）。"""
    mapping: dict[str, float] = {}
    partition = (scenario.get("terrain") or {}).get("partition") or {}
    for name, props in (partition.get("classes") or {}).items():
        mapping[str(name)] = float(props["c_rr"])
    for patch in (scenario.get("terrain") or {}).get("patches", []):
        mapping.setdefault(str(patch["id"]), float(patch["c_rr"]))
    default = (scenario.get("terrain") or {}).get("default") or {}
    if default:
        mapping.setdefault("default", float(default["c_rr"]))
    return mapping


@dataclass(frozen=True)
class TerrainScenarioReport:
    scenario_id: str
    grade: str
    terrain_tier: str
    terrain_tier_label: str
    expected_c_rr: float
    zones: int
    free_area_hist: dict[float, float]
    feasible_area_hist: dict[float, float]
    target_shares: dict[str, float]
    realized_shares: dict[str, float]
    c_rr_min: float
    c_rr_max: float
    c_rr_free_mean: float
    c_rr_feasible_mean: float
    high_ratio_free: float
    """``c_rr >= 0.04`` 的自由区面积占比（报告里叫"高阻地形占比"）。"""
    high_ratio_feasible: float


def terrain_histogram_for(
    scenario: dict[str, Any],
    *,
    scene: experiment.Scene,
    d_min: float,
) -> TerrainScenarioReport:
    """逐场景材质占比：自由区 + 侵蚀后可行区 + 高阻占比。"""
    class_grid = build_scenarios.terrain_partition_grid_from_scenario(scenario, scene)
    mapping = c_rr_by_class(scenario)
    free = ~scene.occupancy
    feasible = scene.clearance + 1e-9 >= float(d_min)

    free_hist = terrain_partition.c_rr_histogram(class_grid, mapping, mask=free)
    feasible_hist = (
        terrain_partition.c_rr_histogram(class_grid, mapping, mask=feasible)
        if feasible.any()
        else {}
    )
    class_values = np.zeros(class_grid.shape, dtype=float)
    for name, value in mapping.items():
        class_values[class_grid == name] = float(value)

    free_values = class_values[free]
    feasible_values = class_values[feasible] if feasible.any() else np.zeros(0)
    partition = (scenario.get("terrain") or {}).get("partition") or {}
    tier_name = str(partition.get("tier", "-"))
    names, counts = np.unique(class_grid[free], return_counts=True)
    total = int(counts.sum()) or 1
    realized = {str(name): float(count) / total for name, count in zip(names, counts)}
    return TerrainScenarioReport(
        scenario_id=str(scenario["id"]),
        grade=str((scenario.get("quality") or {}).get("grade", "-")),
        terrain_tier=tier_name,
        terrain_tier_label=str(partition.get("tier_label_zh", tier_name)),
        expected_c_rr=float(partition.get("expected_c_rr", float("nan"))),
        zones=int(partition.get("zones", 0)),
        free_area_hist=free_hist,
        feasible_area_hist=feasible_hist,
        target_shares={
            str(name): float(share)
            for name, share in (partition.get("target_shares") or {}).items()
        },
        realized_shares=realized,
        c_rr_min=float(free_values.min()) if free_values.size else float("nan"),
        c_rr_max=float(free_values.max()) if free_values.size else float("nan"),
        c_rr_free_mean=float(free_values.mean()) if free_values.size else float("nan"),
        c_rr_feasible_mean=(
            float(feasible_values.mean()) if feasible_values.size else float("nan")
        ),
        high_ratio_free=float((free_values >= 0.04 - 1e-12).mean())
        if free_values.size
        else float("nan"),
        high_ratio_feasible=float((feasible_values >= 0.04 - 1e-12).mean())
        if feasible_values.size
        else float("nan"),
    )


RATIO_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "grade",
    "terrain_tier",
    "terrain_tier_label",
    "expected_c_rr",
    "zones",
    "free_c_rr_0.015",
    "free_c_rr_0.030",
    "free_c_rr_0.040",
    "free_c_rr_0.060",
    "feasible_c_rr_0.015",
    "feasible_c_rr_0.030",
    "feasible_c_rr_0.040",
    "feasible_c_rr_0.060",
    "free_high_ratio",
    "feasible_high_ratio",
    "c_rr_min",
    "c_rr_max",
    "c_rr_free_mean",
    "c_rr_feasible_mean",
    "target_tile",
    "target_concrete",
    "target_wood",
    "target_grating",
    "realized_tile",
    "realized_concrete",
    "realized_wood",
    "realized_grating",
    "max_share_deviation",
)

_C_RR_LEVELS: tuple[float, ...] = (0.015, 0.030, 0.040, 0.060)
_CLASS_ORDER: tuple[str, ...] = ("tile", "concrete", "wood", "grating")


def ratio_row(report: TerrainScenarioReport) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario_id": report.scenario_id,
        "grade": report.grade,
        "terrain_tier": report.terrain_tier,
        "terrain_tier_label": report.terrain_tier_label,
        "expected_c_rr": f"{report.expected_c_rr:.5f}",
        "zones": report.zones,
        "free_high_ratio": f"{report.high_ratio_free:.4f}",
        "feasible_high_ratio": f"{report.high_ratio_feasible:.4f}",
        "c_rr_min": f"{report.c_rr_min:.4f}",
        "c_rr_max": f"{report.c_rr_max:.4f}",
        "c_rr_free_mean": f"{report.c_rr_free_mean:.5f}",
        "c_rr_feasible_mean": f"{report.c_rr_feasible_mean:.5f}",
    }
    for level in _C_RR_LEVELS:
        row[f"free_c_rr_{level:.3f}"] = f"{report.free_area_hist.get(level, 0.0):.4f}"
        row[f"feasible_c_rr_{level:.3f}"] = (
            f"{report.feasible_area_hist.get(level, 0.0):.4f}"
        )
    deviations: list[float] = []
    for name in _CLASS_ORDER:
        target = float(report.target_shares.get(name, float("nan")))
        realized = float(report.realized_shares.get(name, float("nan")))
        row[f"target_{name}"] = "" if not np.isfinite(target) else f"{target:.4f}"
        row[f"realized_{name}"] = "" if not np.isfinite(realized) else f"{realized:.4f}"
        if np.isfinite(target) and np.isfinite(realized):
            deviations.append(abs(target - realized))
    row["max_share_deviation"] = f"{max(deviations):.4f}" if deviations else ""
    return row


# ---------------------------------------------------------------------------
# 2. 候选路线上的 c_rr 覆盖
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteCoverage:
    scenario_id: str
    grade: str
    terrain_tier: str
    route_index: int
    source: str
    length_m: float
    energy_j: float
    energy_per_m: float
    min_clearance_m: float
    on_front: bool
    hist: dict[float, float]
    high_ratio: float

    def row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "grade": self.grade,
            "terrain_tier": self.terrain_tier,
            "route_index": self.route_index,
            "source": self.source,
            "length_m": f"{self.length_m:.4f}",
            "energy_j": f"{self.energy_j:.3f}",
            "energy_per_m": f"{self.energy_per_m:.3f}",
            "min_clearance_m": f"{self.min_clearance_m:.4f}",
            "on_front": int(self.on_front),
            "route_high_ratio": f"{self.high_ratio:.4f}",
        }
        for level in _C_RR_LEVELS:
            row[f"route_c_rr_{level:.3f}"] = f"{self.hist.get(level, 0.0):.4f}"
        return row


def route_coverages(
    scenario_path: str, *, config_root: Path = CONFIG_ROOT, algo: str = DEFAULT_ALGO
) -> tuple[list[RouteCoverage], Any]:
    """主路线 + 多同伦候选路线的材质覆盖（按路段长度加权）。"""
    resolved = config_module.load_resolved(algo, Path(scenario_path), config_root=config_root)
    scenario = resolved.scenario
    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    class_grid = build_scenarios.terrain_partition_grid_from_scenario(scenario, scene)
    mapping = c_rr_by_class(scenario)
    tier = str(((scenario.get("terrain") or {}).get("partition") or {}).get("tier", "-"))

    base = setup.astar_params
    main = chapter3._plan(scene, setup, base)
    routes = chapter3.enumerate_routes(
        scene, setup, base, d_min_ref=float(base.d_min)
    )
    route_points = [(r.length_m, r.energy_j) for r in routes]
    front = chapter3.pareto_front(route_points)

    def _on(point: tuple[float, float]) -> bool:
        return any(
            abs(point[0] - l) <= 1e-6 and abs(point[1] - e) <= 1e-6 for l, e in front
        )

    def _hist(cells: np.ndarray, steps: np.ndarray) -> dict[float, float]:
        return terrain_partition.path_c_rr_histogram(
            cells, steps, class_grid, mapping
        )

    coverages: list[RouteCoverage] = []
    if main.success:
        hist = _hist(main.cells, main.step_length)
        coverages.append(
            RouteCoverage(
                scenario_id=str(scenario["id"]),
                grade=str((scenario.get("quality") or {}).get("grade", "-")),
                terrain_tier=tier,
                route_index=-1,
                source="main",
                length_m=float(main.length_m),
                energy_j=float(main.total_energy_j),
                energy_per_m=(
                    float(main.total_energy_j) / float(main.length_m)
                    if main.length_m > 0
                    else float("nan")
                ),
                min_clearance_m=float(main.clearance.min()) if main.clearance.size else 0.0,
                on_front=True,
                hist=hist,
                high_ratio=float(sum(v for k, v in hist.items() if k >= 0.04 - 1e-12)),
            )
        )
    for index, route in enumerate(routes):
        cells, steps = route.explicit_segments()
        hist = _hist(cells, steps)
        coverages.append(
            RouteCoverage(
                scenario_id=str(scenario["id"]),
                grade=str((scenario.get("quality") or {}).get("grade", "-")),
                terrain_tier=tier,
                route_index=index,
                source="route",
                length_m=route.length_m,
                energy_j=route.energy_j,
                energy_per_m=route.energy_per_m,
                min_clearance_m=route.min_clearance_m,
                on_front=_on((route.length_m, route.energy_j)),
                hist=hist,
                high_ratio=float(sum(v for k, v in hist.items() if k >= 0.04 - 1e-12)),
            )
        )
    return coverages, resolved


ROUTE_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "grade",
    "terrain_tier",
    "route_index",
    "source",
    "length_m",
    "energy_j",
    "energy_per_m",
    "min_clearance_m",
    "on_front",
    "route_high_ratio",
    "route_c_rr_0.015",
    "route_c_rr_0.030",
    "route_c_rr_0.040",
    "route_c_rr_0.060",
)


# ---------------------------------------------------------------------------
# 3. 敏感性分析
# ---------------------------------------------------------------------------


def scaled_energy(
    c_rr: np.ndarray,
    slope_rad: np.ndarray,
    *,
    scale: float,
    p_aux: float,
    platform: dict[str, Any],
    speed: float,
    step_length: np.ndarray,
) -> float:
    """一条**已规划路径**在给定 ``c_rr`` 缩放与 ``p_aux`` 下的参考能耗 [J]。

    与 A* 的 ``energy.step_energy`` 同一口径（稳态巡航、``accel = 0``、不回收），
    只是把每个单元的 ``c_rr`` 乘以 ``scale``、把辅助功率换成 ``p_aux``。
    用途：路径不变、只改材质取值时的能耗复核（回答"材质阻力整体偏 30% 会怎样"）。
    """
    params = energy_module.EnergyParams(
        mass=float(platform["mass"]),
        g=float(platform["g"]),
        eta=float(platform["eta"]),
        p_aux=float(p_aux),
        battery_voltage=float(platform["battery_voltage"]),
        battery_resistance=float(platform["battery_resistance"]),
    )
    total = 0.0
    steps = np.asarray(step_length, dtype=float).reshape(-1)
    values = np.asarray(c_rr, dtype=float).reshape(-1)
    slopes = np.asarray(slope_rad, dtype=float).reshape(-1)
    for distance, value, slope in zip(steps, values, slopes):
        if distance <= 0:
            continue
        total += energy_module.step_energy(
            float(distance),
            float(speed),
            accel=0.0,
            c_rr=float(value) * float(scale),
            slope_rad=float(slope),
            params=params,
        )
    return float(total)


SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "axis",
    "value",
    "terrain_tier",
    "tasks",
    "length_median_m",
    "energy_median_j",
    "energy_total_j",
    "energy_per_m_median",
    "pareto_tasks_multipoint",
    "pareto_points_total",
    "esa_differs_from_length",
    "esa_energy_lower_tasks",
    "esa_longer_tasks",
)


def sensitivity_table(
    paths: Sequence[Path],
    *,
    config_root: Path,
    algo: str,
    d_min_ref: float,
    jobs: int,
) -> list[dict[str, Any]]:
    """±30% ``c_rr`` 与 ``p_aux`` 扫描：逐场景重算能耗、重跑前沿。

    两种轴的实现方式不同，必须写清：

    * **``c_rr`` ±30%**：直接缩放场景里每个 patch 的 ``c_rr``（等价于换一套材质
      取值），重新规划 + 重算前沿。它改变的是**路径选择**；
    * **``p_aux`` 扫描**：只改平台层的辅助功率，场景不动，重新规划 + 重算前沿。
      它改变的是"时间项与地形项的配比"，是能耗结构的敏感性。
    """
    rows: list[dict[str, Any]] = []
    for scale in C_RR_SCALES:
        records, fronts = _run_with_override(
            paths,
            config_root=config_root,
            algo=algo,
            d_min_ref=d_min_ref,
            c_rr_scale=scale,
            p_aux=None,
            jobs=jobs,
        )
        rows.extend(
            _summarise_axis(
                f"c_rr x{scale:g}", f"{scale:g}", records, fronts
            )
        )
    for p_aux in P_AUX_VALUES:
        records, fronts = _run_with_override(
            paths,
            config_root=config_root,
            algo=algo,
            d_min_ref=d_min_ref,
            c_rr_scale=1.0,
            p_aux=float(p_aux),
            jobs=jobs,
        )
        rows.extend(
            _summarise_axis(f"P_aux={p_aux:g} W", f"{p_aux:g}", records, fronts)
        )
    return rows


def _scaled_scenario(scenario: dict[str, Any], scale: float) -> dict[str, Any]:
    import copy

    doc = copy.deepcopy(scenario)
    terrain = doc["terrain"]
    terrain["default"]["c_rr"] = round(float(terrain["default"]["c_rr"]) * scale, 8)
    for patch in terrain.get("patches", []):
        patch["c_rr"] = round(float(patch["c_rr"]) * scale, 8)
    partition = terrain.get("partition")
    if partition:
        partition["c_rr_scale"] = float(scale)
        for props in partition.get("classes", {}).values():
            props["c_rr"] = round(float(props["c_rr"]) * scale, 8)
        partition["expected_c_rr"] = round(float(partition.get("expected_c_rr", 0.0)) * scale, 8)
    doc["content_hash"] = config_module.compute_scenario_hash(doc)
    return doc


def _run_with_override(
    paths: Sequence[Path],
    *,
    config_root: Path,
    algo: str,
    d_min_ref: float,
    c_rr_scale: float,
    p_aux: float | None,
    jobs: int,
) -> tuple[list[chapter3.PlanRecord], dict[str, int]]:
    """在一组覆盖参数下逐场景跑三个规划器：本文硬约束 / 仅长度 / ESA*。

    ESA* 与仅长度都用 ``d_min = body_radius``（同一可行域），这样两者的差异
    **只来自"长度 vs 能耗"这个目标函数**，不会混进策略半径的影响。
    """
    records: list[chapter3.PlanRecord] = []
    fronts: dict[str, int] = {}
    for path in paths:
        resolved = config_module.load_resolved(algo, path, config_root=config_root)
        scenario = resolved.scenario
        if c_rr_scale != 1.0:
            scenario = _scaled_scenario(scenario, c_rr_scale)
        platform = dict(resolved.platform)
        if p_aux is not None:
            platform["p_aux"] = float(p_aux)
        scene = experiment.build_scene(scenario)
        setup = experiment.planner_setup(platform, resolved.algo)
        base = setup.astar_params
        body_radius = float(platform["body_radius"])

        def _record(planner: str, label: str, result: astar.PathResult, *, used_budget: float,
                    d_min_search: float, pareto_size: int = 0) -> chapter3.PlanRecord:
            return chapter3._evaluate(
                planner=planner, planner_label=label, scenario=scenario,
                resolved=resolved, result=result, body_radius=body_radius,
                d_min_ref=d_min_ref, d_min_search=d_min_search,
                e_budget_used=used_budget, cruise_speed=float(setup.cruise_speed),
                seed=0, pareto_size=pareto_size,
            )

        constrained = chapter3._plan(scene, setup, base)
        records.append(
            _record(
                "constrained", "本文约束化 A*", constrained,
                used_budget=float(base.e_budget), d_min_search=float(base.d_min),
            )
        )

        baseline_setup = replace(
            setup,
            astar_params=replace(base, d_min=body_radius, collect_pareto=False),
        )
        baseline = chapter3._plan(scene, baseline_setup, baseline_setup.astar_params)
        records.append(
            _record(
                "length", "传统 A*（仅长度）", baseline,
                used_budget=float(base.e_budget),
                d_min_search=float(baseline_setup.astar_params.d_min),
            )
        )

        front_result = chapter3._plan(
            scene,
            baseline_setup,
            replace(baseline_setup.astar_params, collect_pareto=True),
        )
        fronts[str(scenario["id"])] = int(front_result.pareto_length_m.size)
        if front_result.success and front_result.pareto_energy_j.size:
            index = int(np.argmin(front_result.pareto_energy_j))
            e_min = float(front_result.pareto_energy_j[index])
            tight = chapter3._plan(
                scene,
                baseline_setup,
                replace(baseline_setup.astar_params, e_budget=e_min * (1.0 + 1e-9) + 1e-9),
            )
            if not tight.success:
                tight = front_result
            records.append(
                _record(
                    "esa", "能耗加权 A*（ESA*）", tight, used_budget=e_min,
                    d_min_search=float(baseline_setup.astar_params.d_min),
                    pareto_size=int(front_result.pareto_length_m.size),
                )
            )
    return records, fronts


def _summarise_axis(
    axis: str,
    value: str,
    records: Sequence[chapter3.PlanRecord],
    fronts: dict[str, int],
) -> list[dict[str, Any]]:
    constrained = [r for r in records if r.planner == "constrained"]
    by_grade: dict[str, list[chapter3.PlanRecord]] = {}
    for record in constrained:
        by_grade.setdefault(record.grade, []).append(record)
    by_grade["all"] = list(constrained)

    def _difference(planner_a: str, planner_b: str) -> int:
        count = 0
        for record in constrained:
            first = next(
                (r for r in records
                 if r.planner == planner_a and r.scenario_id == record.scenario_id),
                None,
            )
            second = next(
                (r for r in records
                 if r.planner == planner_b and r.scenario_id == record.scenario_id),
                None,
            )
            if first is None or second is None or not (first.success and second.success):
                continue
            if (
                abs(first.length_m - second.length_m) > 1e-6
                or abs(first.energy_j - second.energy_j) > 1e-6
            ):
                count += 1
        return count

    esa_differs = _difference("esa", "length")
    esa_energy_lower = 0
    esa_length_longer = 0
    for record in constrained:
        esa = next(
            (r for r in records if r.planner == "esa" and r.scenario_id == record.scenario_id),
            None,
        )
        base = next(
            (r for r in records if r.planner == "length" and r.scenario_id == record.scenario_id),
            None,
        )
        if esa is None or base is None or not (esa.success and base.success):
            continue
        esa_energy_lower += int(esa.energy_j < base.energy_j - 1e-6)
        esa_length_longer += int(esa.length_m > base.length_m + 1e-6)

    rows: list[dict[str, Any]] = []
    for tier_name in sorted(by_grade, key=lambda item: (item != "all", item)):
        subset = by_grade[tier_name]
        if not subset:
            continue
        lengths = [r.length_m for r in subset]
        energies = [r.energy_j for r in subset]
        per_m = [r.energy_j / r.length_m for r in subset if r.length_m > 0]
        rows.append(
            {
                "axis": axis,
                "value": value,
                "terrain_tier": tier_name,
                "tasks": len(subset),
                "length_median_m": f"{float(np.median(lengths)):.4f}",
                "energy_median_j": f"{float(np.median(energies)):.3f}",
                "energy_total_j": f"{float(np.sum(energies)):.3f}",
                "energy_per_m_median": f"{float(np.median(per_m)):.3f}",
                "pareto_tasks_multipoint": sum(
                    1 for points in fronts.values() if points > 1
                ),
                "pareto_points_total": int(sum(fronts.values())),
                "esa_differs_from_length": esa_differs,
                "esa_energy_lower_tasks": esa_energy_lower,
                "esa_longer_tasks": esa_length_longer,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 4. 图
# ---------------------------------------------------------------------------


def plot_c_rr_histogram(
    reports: Sequence[TerrainScenarioReport], out_path: Path
) -> Path:
    """图 1：逐场景的 c_rr 面积占比堆叠图（自由区），四档材质分色。"""
    _setup_fonts()
    order = sorted(reports, key=lambda item: (item.grade, item.scenario_id))
    figure, axes = plt.subplots(2, 1, figsize=(15.0, 8.6), constrained_layout=True)
    x = np.arange(len(order))
    levels = sorted(
        {
            level
            for report in order
            for level in report.free_area_hist
            if level > 0
            and (report.free_area_hist.get(level, 0.0) > 0 or
                 report.feasible_area_hist.get(level, 0.0) > 0)
        }
    )
    for axis, key, title in (
        (axes[0], "free_area_hist", "自由区（clearance > 0）"),
        (axes[1], "feasible_area_hist", "侵蚀后可行区（clearance >= d_min = 0.70 m）"),
    ):
        bottom = np.zeros(len(order))
        for level in levels:
            values = np.array([getattr(r, key).get(level, 0.0) for r in order])
            axis.bar(
                x, values, bottom=bottom, width=0.78,
                color=C_RR_COLORS.get(round(level, 3), "#cccccc"),
                label=C_RR_LABELS.get(round(level, 3), f"c_rr = {level:g}"),
            )
            bottom += values
        axis.set_xticks(x)
        axis.set_xticklabels(
            [f"{r.scenario_id.replace('houseexpo_', '')}\n{r.grade} / {r.terrain_tier}"
             for r in order],
            fontsize=7.0,
        )
        axis.set_ylabel("面积占比 [—]")
        axis.set_ylim(0, 1.0)
        axis.set_title(f"c_rr 面积占比（{title}）", fontsize=11)
        axis.grid(axis="y", alpha=0.25)
        handles, labels_ = axis.get_legend_handles_labels()
        unique: dict[str, Any] = {}
        for handle, name in zip(handles, labels_):
            unique.setdefault(name, handle)
        axis.legend(unique.values(), unique.keys(), fontsize=8.2, ncol=4, loc="upper center",
                    bbox_to_anchor=(0.5, -0.42))
    figure.suptitle(
        "受控材质因子：c_rr 四档（0.015/0.03/0.04/0.06）在真实户型自由区上的实际面积占比\n"
        "材质空间分布是受控实验因子（非实测）；tier = 低阻为主 / 混合 / 高阻为主",
        fontsize=11.5,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


def plot_route_overlay(
    scenario_path: Path,
    out_path: Path,
    *,
    config_root: Path = CONFIG_ROOT,
    algo: str = DEFAULT_ALGO,
) -> Path:
    """图 2：材质分区 + 多同伦候选路线叠图（看高阻地形是否在候选路线上）。"""
    _setup_fonts()
    resolved = config_module.load_resolved(algo, scenario_path, config_root=config_root)
    scenario = resolved.scenario
    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    from env import build_scenarios

    class_grid = build_scenarios.terrain_partition_grid_from_scenario(scenario, scene)
    mapping = c_rr_by_class(scenario)
    values = np.zeros(class_grid.shape, dtype=float)
    for name, value in mapping.items():
        values[class_grid == name] = float(value)
    levels = sorted(C_RR_COLORS)
    index_grid = np.zeros(class_grid.shape, dtype=int)
    for index, level in enumerate(levels):
        index_grid[np.isclose(values, level, atol=1e-9)] = index

    rows, cols = class_grid.shape
    ox, oy = scene.origin
    extent = (ox, ox + cols * scene.resolution, oy, oy + rows * scene.resolution)
    figure, axis = plt.subplots(figsize=(7.6, 7.6), constrained_layout=True)
    axis.imshow(
        index_grid, origin="lower", extent=extent,
        cmap=ListedColormap([C_RR_COLORS[level] for level in levels]),
        interpolation="nearest", vmin=0, vmax=len(levels) - 1, zorder=1,
    )
    walls = np.ma.masked_where(~scene.occupancy, scene.occupancy)
    axis.imshow(walls, origin="lower", extent=extent,
                cmap=ListedColormap(["#2b2b2b"]), interpolation="nearest", zorder=2)

    params = setup.astar_params
    candidates = chapter3.enumerate_routes(
        scene, setup, params, d_min_ref=float(params.d_min), max_routes=6
    )
    for index, route in enumerate(candidates):
        color = ROUTE_COLORS[index % len(ROUTE_COLORS)]
        axis.plot(
            route.xy[:, 0], route.xy[:, 1], "-", color=color, linewidth=2.1,
            zorder=4 + index,
            label=(
                f"路线 {index}：{route.length_m:.2f} m / {route.energy_j:.1f} J / "
                f"{route.energy_per_m:.1f} J/m"
            ),
        )
    axis.plot(scene.start_xy[0], scene.start_xy[1], "o", color="#2ca02c",
              markersize=11, zorder=20)
    axis.plot(scene.goal_xy[0], scene.goal_xy[1], "*", color="#111111",
              markersize=17, zorder=20)
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    partition = (scenario.get("terrain") or {}).get("partition") or {}
    axis.set_title(
        f"{scenario['id']}（{(scenario.get('quality') or {}).get('grade', '-')} / "
        f"{partition.get('tier_label_zh', '-')}）\n"
        f"受控材质分区 + 多同伦候选路线（★终点，●起点）",
        fontsize=11,
    )
    handles = [Patch(facecolor=C_RR_COLORS[level], label=C_RR_LABELS[level]) for level in levels]
    handles += axis.get_legend_handles_labels()[0]
    labels_ = [C_RR_LABELS[level] for level in levels] + axis.get_legend_handles_labels()[1]
    unique: dict[str, Any] = {}
    for handle, name in zip(handles, labels_):
        unique.setdefault(name, handle)
    axis.legend(unique.values(), unique.keys(), fontsize=7.6, loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


def plot_sensitivity(rows: Sequence[dict[str, Any]], out_path: Path) -> Path:
    """图 3：±30% c_rr 与 p_aux 扫描下的能耗/前沿点数。"""
    _setup_fonts()
    figure, axes = plt.subplots(1, 4, figsize=(19.0, 4.6), constrained_layout=True)
    c_rr_rows = [r for r in rows if r["axis"].startswith("c_rr") and r["terrain_tier"] == "all"]
    p_aux_rows = [r for r in rows if r["axis"].startswith("P_aux") and r["terrain_tier"] == "all"]
    panels = (
        (c_rr_rows, "c_rr 缩放系数", axes[0], axes[1]),
        (p_aux_rows, "P_aux [W]", axes[2], axes[3]),
    )
    for group, xlabel, energy_axis, points_axis in panels:
        x = [float(r["value"]) for r in group]
        energy_axis.plot(
            x, [float(r["energy_median_j"]) for r in group], "-o", color="#1f77b4"
        )
        energy_axis.set_xlabel(xlabel)
        energy_axis.set_ylabel("能耗中位 [J]")
        energy_axis.grid(alpha=0.25)
        energy_axis.set_title("能耗中位", fontsize=10.5)
        points_axis.bar(
            [str(value) for value in x],
            [int(r["pareto_points_total"]) for r in group],
            color="#2ca02c",
        )
        points_axis.set_xlabel(xlabel)
        points_axis.set_ylabel("前沿点数合计")
        points_axis.set_title("(长度, 能耗) 前沿点数合计（20 任务）", fontsize=10.5)
        points_axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "敏感性：c_rr 整体 ±30% 与 P_aux ∈ {7.5, 15, 22.5, 30} W\n"
        "（P_aux = 辅助负载功率，占参考能耗约 1/3，是稀释地形影响的直接原因）",
        fontsize=11.5,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# 并行与外层
# ---------------------------------------------------------------------------


def _ratio_task(path: str, config_root: str, algo: str) -> TerrainScenarioReport:
    resolved = config_module.load_resolved(algo, Path(path), config_root=Path(config_root))
    scenario = resolved.scenario
    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    return terrain_histogram_for(
        scenario, scene=scene, d_min=float(setup.astar_params.d_min)
    )


def _route_task(path: str, config_root: str, algo: str) -> list[RouteCoverage]:
    coverages, _ = route_coverages(path, config_root=Path(config_root), algo=algo)
    return coverages


def write_csv(rows: Sequence[dict[str, Any]], path: Path, columns: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def format_ratio_table(reports: Sequence[TerrainScenarioReport]) -> str:
    header = (
        f"{'场景':<22}{'档':<7}{'材质档':<16}{'期望c_rr':>9}"
        f"{'0.015':>8}{'0.030':>8}{'0.040':>8}{'0.060':>8}{'高阻占比':>9}{'最大偏差':>9}"
    )
    lines = [
        "受控材质因子：逐场景 c_rr 实际面积占比（自由区；高阻占比 = c_rr >= 0.04）",
        header,
        "-" * len(header),
    ]
    for report in sorted(reports, key=lambda item: (item.grade, item.scenario_id)):
        deviations = [
            abs(float(report.target_shares.get(name, 0.0)) - float(report.realized_shares.get(name, 0.0)))
            for name in _CLASS_ORDER
            if name in report.target_shares
        ]
        lines.append(
            f"{report.scenario_id:<22}{report.grade:<7}{report.terrain_tier:<16}"
            f"{report.expected_c_rr:>9.4f}"
            f"{report.free_area_hist.get(0.015, 0.0):>8.3f}"
            f"{report.free_area_hist.get(0.030, 0.0):>8.3f}"
            f"{report.free_area_hist.get(0.040, 0.0):>8.3f}"
            f"{report.free_area_hist.get(0.060, 0.0):>8.3f}"
            f"{report.high_ratio_free:>9.3f}"
            f"{(max(deviations) if deviations else float('nan')):>9.3f}"
        )
    counts = Counter(report.terrain_tier for report in reports)
    lines.append(
        "材质档分布：" + "，".join(f"{name} {counts[name]}/20" for name in sorted(counts))
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.terrain",
        description="受控材质因子：c_rr 分布/路线覆盖 + ±30% 与 P_aux 敏感性 + 图",
    )
    parser.add_argument("--tier", default="all")
    parser.add_argument("--algo", default=DEFAULT_ALGO)
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--sensitivity", action="store_true", help="附跑 ±30% 与 P_aux 扫描")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--review", action="store_true", help="把图复制到 docs/review/")
    parser.add_argument("--review-start", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--overlay-scenarios", nargs="*", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_root = Path(args.config_root) if args.config_root else CONFIG_ROOT
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = scenario_paths(args.tier, config_root=config_root)
    if not paths:
        print("没有找到场景", flush=True)
        return 2

    from concurrent.futures import ProcessPoolExecutor

    workers = int(args.jobs) if args.jobs else min(len(paths), os.cpu_count() or 1)
    reports: list[TerrainScenarioReport] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_ratio_task, str(path), str(config_root), args.algo)
                for path in paths
            ]
            reports = [future.result() for future in futures]
    else:
        reports = [
            _ratio_task(str(path), str(config_root), args.algo) for path in paths
        ]
    ratio_csv = write_csv(
        [ratio_row(report) for report in reports], RATIO_CSV, RATIO_COLUMNS
    )
    print(f"\n{format_ratio_table(reports)}")
    print(f"\n分布明细: {ratio_csv}")

    coverages: list[RouteCoverage] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_route_task, str(path), str(config_root), args.algo)
                for path in paths
            ]
            for future in futures:
                coverages.extend(future.result())
    else:
        for path in paths:
            coverages.extend(_route_task(str(path), str(config_root), args.algo))
    route_csv = write_csv(
        [coverage.row() for coverage in coverages], ROUTE_CSV, ROUTE_COLUMNS
    )
    frontier_routes = sum(1 for coverage in coverages if coverage.on_front)
    print(
        f"候选路线覆盖: {len(coverages)} 条（{frontier_routes} 条在候选前沿上）-> {route_csv}"
    )
    high_on_route = [
        coverage.high_ratio for coverage in coverages if coverage.source == "main"
    ]
    if high_on_route:
        array = np.asarray(high_on_route, dtype=float)
        print(
            "主路线上的高阻（c_rr >= 0.04）长度占比："
            f"中位 {float(np.median(array)):.3f}，均值 {float(array.mean()):.3f}，"
            f"最小 {float(array.min()):.3f}，最大 {float(array.max()):.3f}"
        )
        print(
            "  高阻占比 > 10% 的任务 "
            f"{int((array > 0.10).sum())}/{array.size}；> 30% 的 "
            f"{int((array > 0.30).sum())}/{array.size}；"
            f"> 60% 的 {int((array > 0.60).sum())}/{array.size}"
        )

    figures: list[Path] = []
    if not args.no_plots:
        figures.append(plot_c_rr_histogram(reports, out_dir / "c_rr_histogram.png"))
        overlay_targets = (
            [Path(item) for item in args.overlay_scenarios]
            if args.overlay_scenarios
            else chapter3._representative_scenarios(paths)
        )
        for path in overlay_targets:
            figures.append(
                plot_route_overlay(
                    path,
                    out_dir / f"route_overlay_{path.stem}.png",
                    config_root=config_root,
                    algo=args.algo,
                )
            )

    if args.sensitivity:
        rows = sensitivity_table(
            paths,
            config_root=config_root,
            algo=args.algo,
            d_min_ref=0.7,
            jobs=workers,
        )
        sensitivity_csv = write_csv(rows, SENSITIVITY_CSV, SENSITIVITY_COLUMNS)
        print(f"\n敏感性: {sensitivity_csv}")
        for row in rows:
            if row["terrain_tier"] != "all":
                continue
            print(
                f"  {row['axis']:<16} 长度中位 {row['length_median_m']:>8} m  "
                f"能耗中位 {row['energy_median_j']:>9} J  "
                f"前沿多点任务 {row['pareto_tasks_multipoint']}/20  "
                f"ESA* 与仅长度差异任务 {row['esa_differs_from_length']}/20"
            )
        if not args.no_plots:
            figures.append(plot_sensitivity(rows, out_dir / "sensitivity.png"))

    for figure in figures:
        print(f"OK   {figure}")
    if args.review:
        import shutil

        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        index = int(args.review_start)
        for figure in figures:
            target = REVIEW_DIR / f"{index:02d}_terrain_{figure.stem}.png"
            shutil.copyfile(figure, target)
            index += 1
            print(f"REVIEW {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
