"""一个 spec -> 一批 run：加载三层配置、规划、采样轨迹、落盘。

每次运行产出 ``results/<run_id>/``：

* ``config.json``：三层配置快照 + 配置哈希 + git commit + seed；
* ``metrics.csv``：首列 config hash，每次重复一行；
* ``trajectory.npz``（第 2 次起的重复为 ``trajectory_001.npz`` ...）：
  t / xy / yaw / speed / yaw_rate / energy_j / clearance / collision。

``metrics.csv`` 与 ``config.json`` 都不含时间戳：同一 seed 的两次运行，
指标逐位一致，run 目录名里的时间戳只用于区分目录。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core import astar, bspline, energy, metrics
from env import floorplan
from runner import config as config_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DT = 0.05

CSV_COLUMNS: tuple[str, ...] = (
    "config_hash",
    "run_id",
    "repeat",
    "seed",
    "success",
    "reason",
    *metrics.METRIC_NAMES,
    "smooth_status",
    "smooth_reason",
    "smooth_fell_back",
    "smooth_curvature_satisfied",
    "smooth_curvature_infeasible",
    "smooth_trackable",
    "smooth_inserted_points",
    "smooth_removed_points",
    "smooth_depth",
    "smooth_rounds",
    "smooth_control_points",
    "smooth_max_abs_curvature",
    "smooth_min_clearance_m",
    "smooth_energy_j",
    "smooth_length_m",
    "expanded",
    "labels_created",
)


@dataclass(frozen=True)
class Scene:
    occupancy: np.ndarray
    resolution: float
    origin: tuple[float, float]
    clearance: np.ndarray
    c_rr: np.ndarray
    slope_rad: np.ndarray
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    yaw_rate: np.ndarray
    energy_j: np.ndarray
    clearance: np.ndarray
    collision: np.ndarray
    violation: np.ndarray


@dataclass(frozen=True)
class ExperimentResult:
    run_dir: Path
    run_id: str
    config_hash: str
    seed: int
    repeats: int
    rows: tuple[dict[str, Any], ...]
    success: bool


# ---------------------------------------------------------------------------
# 场景栅格化：矩形墙（手写场景）或户型轮廓多边形（HouseExpo）
# ---------------------------------------------------------------------------


def build_scene(scenario: dict[str, Any]) -> Scene:
    map_doc = scenario["map"]
    resolution = float(map_doc["resolution"])
    rows, cols = int(map_doc["shape"][0]), int(map_doc["shape"][1])
    origin = map_doc.get("origin", [0.0, 0.0])
    ox, oy = float(origin[0]), float(origin[1])

    if "polygons" in map_doc:
        # 户型外轮廓：单元中心在多边形内即自由（零厚度轮廓，非有厚度的墙）。
        occupancy = floorplan.rasterize_polygons(
            map_doc["polygons"], resolution, (rows, cols), (ox, oy)
        )
    else:
        occupancy = np.zeros((rows, cols), dtype=bool)
        for x0, y0, x1, y1 in map_doc["walls"]:
            c0 = max(0, int(math.floor((float(x0) - ox) / resolution)))
            c1 = min(cols, int(math.ceil((float(x1) - ox) / resolution)))
            r0 = max(0, int(math.floor((float(y0) - oy) / resolution)))
            r1 = min(rows, int(math.ceil((float(y1) - oy) / resolution)))
            if c1 > c0 and r1 > r0:
                # 保守栅格化：与矩形有任何重叠的单元都视为障碍，薄墙不会漏。
                occupancy[r0:r1, c0:c1] = True

    default_terrain = scenario["terrain"]["default"]
    c_rr = np.full((rows, cols), float(default_terrain["c_rr"]), dtype=float)
    slope_rad = np.full(
        (rows, cols), math.radians(float(default_terrain["slope_deg"])), dtype=float
    )
    xs = ox + (np.arange(cols) + 0.5) * resolution
    ys = oy + (np.arange(rows) + 0.5) * resolution
    for patch in scenario["terrain"].get("patches", []):
        x0, y0, x1, y1 = (float(value) for value in patch["rect"])
        mask = ((xs >= x0) & (xs <= x1))[None, :] & ((ys >= y0) & (ys <= y1))[:, None]
        c_rr[mask] = float(patch["c_rr"])
        slope_rad[mask] = math.radians(float(patch["slope_deg"]))

    start_xy = (float(scenario["start"][0]), float(scenario["start"][1]))
    goal_xy = (float(scenario["goal"][0]), float(scenario["goal"][1]))
    start_cell = astar.world_to_cell(start_xy, resolution, (ox, oy))
    goal_cell = astar.world_to_cell(goal_xy, resolution, (ox, oy))
    for name, (row, col), xy in (
        ("start", start_cell, start_xy),
        ("goal", goal_cell, goal_xy),
    ):
        if not (0 <= row < rows and 0 <= col < cols):
            raise config_module.ConfigError(
                f"scenario.{name} = {xy} 超出地图范围（单元 {row, col}）"
            )
        if occupancy[row, col]:
            raise config_module.ConfigError(
                f"scenario.{name} = {xy} 落在障碍物内（单元 {row, col}）；"
                "起点/终点必须在自由空间，否则规划无从谈起"
            )

    return Scene(
        occupancy=occupancy,
        resolution=resolution,
        origin=(ox, oy),
        clearance=astar.clearance_field(occupancy, resolution),
        c_rr=c_rr,
        slope_rad=slope_rad,
        start_xy=start_xy,
        goal_xy=goal_xy,
        start_cell=start_cell,
        goal_cell=goal_cell,
    )


# ---------------------------------------------------------------------------
# 规划器装配（runner 与 analysis/preview 共用同一条路径）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerSetup:
    astar_params: astar.AStarParams
    bspline_params: bspline.BSplineParams
    energy_params: energy.EnergyParams
    cruise_speed: float
    d_min_policy: float = 0.0
    """策略安全半径（平滑后必须守住）[m]。"""

    clearance_margin_m: float = 0.0
    """曲率裕度前馈量 [m]；``astar_params.d_min = d_min_policy + 该值``。"""


def planner_setup(
    platform: dict[str, Any], algo: dict[str, Any]
) -> PlannerSetup:
    """把已校验的 platform / algo 层转成 core 的显式参数对象。

    **曲率裕度前馈**在这里落地：``AStarParams.d_min`` 取
    ``d_min + clearance_margin_m``（全局层走得保守一点），而
    ``BSplineParams.d_min`` 仍是 ``d_min``（平滑后必须守住的策略半径）。
    两者由 ``d_min_policy`` / ``clearance_margin_m`` 显式记录，便于报告区分。
    """
    effective = config_module.astar_effective_params(algo, platform)
    smooth = config_module.bspline_effective_params(algo, platform)
    return PlannerSetup(
        astar_params=astar.AStarParams(
            e_budget=float(effective["e_budget"]),
            d_min=float(effective["d_min_astar"]),
            kappa_max=float(effective["kappa_max"]),
            kappa_window_m=float(effective["kappa_window_m"]),
            max_labels_per_state=int(effective["max_labels_per_state"]),
            max_expansions=int(effective["max_expansions"]),
            collect_pareto=bool(effective.get("collect_pareto", False)),
            enforce_dmin=bool(effective.get("enforce_dmin", True)),
            soft_penalty_weight=float(effective.get("soft_penalty_weight", 0.0)),
        ),
        bspline_params=bspline.BSplineParams(
            kappa_max=float(smooth["kappa_max"]),
            d_min=float(smooth["d_min"]),
            e_budget=float(smooth["e_budget"]),
            control_spacing_m=float(smooth["control_spacing_m"]),
            rdp_epsilon_m=float(smooth["rdp_epsilon_m"]),
            sample_step_m=float(smooth["sample_step_m"]),
            max_insert_depth=int(smooth["max_insert_depth"]),
        ),
        energy_params=energy.EnergyParams(
            mass=float(platform["mass"]),
            g=float(platform["g"]),
            eta=float(platform["eta"]),
            p_aux=float(platform["p_aux"]),
            battery_voltage=float(platform["battery_voltage"]),
            battery_resistance=float(platform["battery_resistance"]),
        ),
        cruise_speed=float(effective["v_cruise"]),
        d_min_policy=float(effective["d_min"]),
        clearance_margin_m=float(effective["clearance_margin_m"]),
    )


def plan_path(scene: Scene, setup: PlannerSetup) -> astar.PathResult:
    """用约束化 A* 规划一条路径；场景与算法参数都来自已校验的配置。"""
    return astar.find_path(
        scene.occupancy,
        scene.resolution,
        scene.start_xy,
        scene.goal_xy,
        setup.astar_params,
        energy_params=setup.energy_params,
        speed=setup.cruise_speed,
        c_rr=scene.c_rr,
        slope_rad=scene.slope_rad,
        origin=scene.origin,
        clearance=scene.clearance,
    )


def smooth_result(
    result: astar.PathResult, scene: Scene, setup: PlannerSetup
) -> bspline.SmoothPath:
    """把 A* 折线交给 B 样条平滑；约束口径与 A* 完全一致（共用参数）。"""
    return bspline.smooth_path(
        result.xy,
        params=setup.bspline_params,
        clearance_field=scene.clearance,
        resolution=scene.resolution,
        speed=setup.cruise_speed,
        energy_params=setup.energy_params,
        c_rr=scene.c_rr,
        slope_rad=scene.slope_rad,
        origin=scene.origin,
    )


# ---------------------------------------------------------------------------
# 参考轨迹
# ---------------------------------------------------------------------------


def _empty_trajectory() -> Trajectory:
    return Trajectory(
        t=np.zeros(0),
        xy=np.zeros((0, 2)),
        yaw=np.zeros(0),
        speed=np.zeros(0),
        yaw_rate=np.zeros(0),
        energy_j=np.zeros(0),
        clearance=np.zeros(0),
        collision=np.zeros(0, dtype=bool),
        violation=np.zeros(0, dtype=bool),
    )


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def build_trajectory(
    result: astar.PathResult,
    *,
    dt: float,
    cruise_speed: float,
    w_max: float,
    body_radius: float,
    d_min: float,
    e_budget: float,
    smooth: bspline.SmoothPath | None = None,
) -> Trajectory:
    """把路径按恒定巡航速度采样成参考轨迹，朝向受 ``w_max`` 限速。

    说明：这一层是**参考轨迹**，不是控制器输出。速度恒定意味着
    ``jerk`` 为 0、角速度为纯几何量；真正的速度/朝向平滑由
    core/bspline.py 与 core/dwa.py 在后续里程碑给出。全局层的硬约束
    （离障、能耗）在这里逐点复核，违反率应当是 0。

    ``smooth`` 不为 ``None`` 时用平滑后的几何替换 A* 折线：穿越的每个采样点
    都来自 B 样条（或回退折线），离障与能耗在平滑阶段已逐点复核，这里再按
    采样步长复核一次，作为"约束违反率为零"的最后一道关口。
    """
    if not result.success:
        return _empty_trajectory()

    if smooth is not None:
        points_array = np.asarray(smooth.xy, dtype=float)
        yaw_array = np.asarray(smooth.yaw, dtype=float)
        clearance_array = np.asarray(smooth.clearance, dtype=float)
        steps = np.linalg.norm(np.diff(points_array, axis=0), axis=1)
        step_length = np.concatenate([[0.0], steps])
        total_length = float(steps.sum())
        # 平滑后的电能按段均摊：口径与 curve_energy 一致，总量守恒。
        segment_energy = (
            steps / total_length * float(smooth.energy_j)
            if total_length > 0
            else np.zeros_like(steps)
        )
        segment_energy = np.concatenate([[0.0], segment_energy])
    else:
        points_array = result.xy
        yaw_array = result.yaw
        clearance_array = result.clearance
        step_length = result.step_length
        segment_energy = result.step_energy

    times = [0.0]
    points = [points_array[0].copy()]
    raw_yaw = [float(yaw_array[0])]
    energies = [0.0]
    clearances = [float(clearance_array[0])]

    for index in range(1, len(points_array)):
        distance = float(step_length[index])
        if distance <= 0.0:
            continue
        substeps = max(1, int(math.ceil(distance / (dt * cruise_speed))))
        for step in range(1, substeps + 1):
            fraction = step / substeps
            times.append(times[-1] + distance / (substeps * cruise_speed))
            points.append(
                points_array[index - 1]
                + fraction * (points_array[index] - points_array[index - 1])
            )
            raw_yaw.append(float(yaw_array[index]))
            energies.append(
                energies[-1] + float(segment_energy[index]) / substeps
            )
            clearances.append(
                clearance_array[index - 1]
                + fraction * (clearance_array[index] - clearance_array[index - 1])
            )

    t = np.asarray(times, dtype=float)
    xy = np.asarray(points, dtype=float)
    raw = np.asarray(raw_yaw, dtype=float)

    yaw = np.empty_like(raw)
    yaw[0] = raw[0]
    for index in range(1, raw.size):
        max_step = w_max * (t[index] - t[index - 1])
        delta = _wrap(raw[index] - yaw[index - 1])
        yaw[index] = yaw[index - 1] + max(-max_step, min(max_step, delta))

    yaw_rate = np.zeros_like(t)
    if t.size > 1:
        yaw_rate[1:] = np.array(
            [_wrap(yaw[i] - yaw[i - 1]) for i in range(1, yaw.size)]
        ) / np.diff(t)

    speed = np.full_like(t, float(cruise_speed))
    energy_array = np.asarray(energies, dtype=float)
    clearance_values = np.asarray(clearances, dtype=float)
    collision = clearance_values < float(body_radius)
    violation = (
        (clearance_values + 1e-9 < float(d_min))
        | (energy_array > float(e_budget) + 1e-9)
        | collision
    )
    return Trajectory(
        t=t,
        xy=xy,
        yaw=yaw,
        speed=speed,
        yaw_rate=yaw_rate,
        energy_j=energy_array,
        clearance=clearance_values,
        collision=collision,
        violation=violation,
    )


# ---------------------------------------------------------------------------
# 运行与落盘
# ---------------------------------------------------------------------------


def _git_metadata(cwd: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if commit.returncode != 0 or not commit.stdout.strip():
        return None, None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return commit.stdout.strip(), None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return commit.stdout.strip(), dirty


def _check_dt(value: float) -> float:
    dt = float(value)
    if not math.isfinite(dt) or not 0.0 < dt <= 0.5:
        raise config_module.ConfigError(f"dt 必须在 (0, 0.5] 内，实际为 {value}")
    return dt


def _unique_run_dir(out_root: Path, base_name: str) -> tuple[Path, str]:
    candidate = out_root / base_name
    suffix = 1
    while candidate.exists():
        candidate = out_root / f"{base_name}-{suffix}"
        suffix += 1
    return candidate, candidate.name


def run_experiment(
    algo_path: str | Path,
    scenario_path: str | Path,
    *,
    platform_path: str | Path | None = None,
    run_path: str | Path | None = None,
    config_root: str | Path | None = None,
    seed: int | None = None,
    repeats: int | None = None,
    out_root: str | Path | None = None,
    dt: float | None = None,
) -> ExperimentResult:
    resolved = config_module.load_resolved(
        algo_path,
        scenario_path,
        platform_path=platform_path,
        run_path=run_path,
        config_root=config_root,
        seed=seed,
        repeats=repeats,
    )
    scene = build_scene(resolved.scenario)
    setup = planner_setup(resolved.platform, resolved.algo)

    base_seed = int(resolved.run["seed"])
    repeat_count = int(resolved.run["repeats"])
    dt_effective = _check_dt(dt if dt is not None else resolved.run.get("dt", DEFAULT_DT))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{resolved.algo['id']}-{resolved.config_hash[:8]}-s{base_seed}-{stamp}"
    out_root_path = Path(out_root) if out_root is not None else Path(
        resolved.run.get("out_dir", "results")
    )
    run_dir, run_id = _unique_run_dir(out_root_path, base_name)

    rows: list[dict[str, Any]] = []
    trajectories: list[Trajectory] = []
    smooth_paths: list[bspline.SmoothPath] = []
    for repeat in range(repeat_count):
        repeat_seed = base_seed + repeat
        result = plan_path(scene, setup)
        smooth = smooth_result(result, scene, setup) if result.success else None
        trajectory = build_trajectory(
            result,
            dt=dt_effective,
            cruise_speed=setup.cruise_speed,
            w_max=float(resolved.platform["w_max"]),
            body_radius=float(resolved.platform["body_radius"]),
            # 违反判据用**策略安全半径**（平滑后必须守住的 d_min），
            # 不是 A* 的规划门槛：启用曲率裕度前馈后，A* 的门槛是
            # d_min + clearance_margin_m，用它判违反会把"合法但比前馈门槛贴墙"
            # 的平滑路径误判成约束违反（实测 12f7c0b0 会报 25.3% 违反率）。
            d_min=float(setup.bspline_params.d_min),
            e_budget=float(setup.astar_params.e_budget),
            smooth=smooth,
        )
        summary = metrics.summarize(
            t=trajectory.t,
            xy=trajectory.xy,
            yaw=trajectory.yaw,
            speed=trajectory.speed,
            energy_j=trajectory.energy_j,
            clearance=trajectory.clearance,
            collision=trajectory.collision,
            violations=trajectory.violation,
        )
        smooth_fields: dict[str, Any] = {
            "smooth_status": "skipped",
            "smooth_reason": "plan_failed",
            "smooth_fell_back": 0,
            "smooth_curvature_satisfied": 0,
            "smooth_curvature_infeasible": 0,
            "smooth_trackable": 0,
            "smooth_inserted_points": 0,
            "smooth_removed_points": 0,
            "smooth_depth": 0,
            "smooth_rounds": 0,
            "smooth_control_points": 0,
            "smooth_max_abs_curvature": 0.0,
            "smooth_min_clearance_m": 0.0,
            "smooth_energy_j": 0.0,
            "smooth_length_m": 0.0,
        }
        if smooth is not None:
            smooth_fields = {
                "smooth_status": smooth.status,
                "smooth_reason": smooth.report.reason,
                "smooth_fell_back": int(smooth.fell_back),
                # 曲率可满足性与"是否产出样条"分开记录：回退产物违反曲率约束，
                # 不是可跟踪的参考轨迹（见 core/bspline.py 的模块说明）。
                "smooth_curvature_satisfied": int(smooth.curvature_ok),
                "smooth_curvature_infeasible": int(smooth.curvature_infeasible),
                "smooth_trackable": int(smooth.trackable),
                "smooth_inserted_points": smooth.report.inserted_points,
                "smooth_removed_points": smooth.report.removed_points,
                "smooth_depth": smooth.report.depth,
                "smooth_rounds": smooth.report.rounds,
                "smooth_control_points": smooth.report.control_points,
                "smooth_max_abs_curvature": smooth.max_abs_curvature,
                "smooth_min_clearance_m": smooth.min_clearance,
                "smooth_energy_j": smooth.energy_j,
                "smooth_length_m": smooth.length_m,
            }
        row: dict[str, Any] = {
            "config_hash": resolved.config_hash,
            "run_id": run_id,
            "repeat": repeat,
            "seed": repeat_seed,
            "success": int(result.success),
            "reason": result.reason,
            **summary,
            **smooth_fields,
            "expanded": result.expanded,
            "labels_created": result.labels_created,
        }
        rows.append(row)
        trajectories.append(trajectory)
        smooth_paths.append(smooth)

    run_dir.mkdir(parents=True, exist_ok=False)
    git_commit, git_dirty = _git_metadata(PROJECT_ROOT)
    snapshot = resolved.snapshot(git_commit=git_commit, git_dirty=git_dirty)
    (run_dir / "config.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for repeat, trajectory in enumerate(trajectories):
        filename = "trajectory.npz" if repeat == 0 else f"trajectory_{repeat:03d}.npz"
        smooth_entry = smooth_paths[repeat]
        extras: dict[str, np.ndarray] = {
            "start_xy": np.asarray(scene.start_xy, dtype=float),
            "goal_xy": np.asarray(scene.goal_xy, dtype=float),
            "config_hash": np.asarray(resolved.config_hash),
            "seed": np.asarray(rows[repeat]["seed"]),
        }
        if smooth_entry is not None:
            extras.update(
                smooth_xy=smooth_entry.xy,
                smooth_s=smooth_entry.s,
                smooth_curvature=smooth_entry.curvature,
                smooth_yaw=smooth_entry.yaw,
                smooth_clearance=smooth_entry.clearance,
                smooth_polyline=smooth_entry.polyline,
            )
        np.savez(
            run_dir / filename,
            t=trajectory.t,
            xy=trajectory.xy,
            yaw=trajectory.yaw,
            speed=trajectory.speed,
            yaw_rate=trajectory.yaw_rate,
            energy_j=trajectory.energy_j,
            clearance=trajectory.clearance,
            collision=trajectory.collision,
            violation=trajectory.violation,
            **extras,
        )

    return ExperimentResult(
        run_dir=run_dir,
        run_id=run_id,
        config_hash=resolved.config_hash,
        seed=base_seed,
        repeats=repeat_count,
        rows=tuple(rows),
        success=all(bool(row["success"]) for row in rows),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m runner.experiment",
        description="按 algo 配置 + 场景跑一轮（或一批）实验并落盘",
    )
    parser.add_argument("algo", help="算法配置，如 configs/frozen_A.yaml")
    parser.add_argument("scenario", help="场景配置，如 scenarios/house_01.yaml")
    parser.add_argument("--platform", default=None, help="平台配置，默认 config/platform.yaml")
    parser.add_argument("--run-config", default=None, help="执行配置，默认 config/run.yaml")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 run.seed")
    parser.add_argument("--repeats", type=int, default=None, help="覆盖 run.repeats")
    parser.add_argument("--out", default=None, help="结果根目录，默认取 run.out_dir")
    parser.add_argument("--config-root", default=None, help="config 根目录")
    parser.add_argument("--dt", type=float, default=None, help="参考轨迹采样步长 [s]")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_experiment(
            args.algo,
            args.scenario,
            platform_path=args.platform,
            run_path=args.run_config,
            config_root=args.config_root,
            seed=args.seed,
            repeats=args.repeats,
            out_root=args.out,
            dt=args.dt,
        )
    except config_module.ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    first = result.rows[0]
    print(f"run_dir: {result.run_dir}")
    print(f"config_hash: {result.config_hash}")
    print(f"repeats: {result.repeats}  seed: {result.seed}")
    print(
        f"success: {int(first['success'])} ({first['reason']})  "
        f"length={first['path_length_m']:.3f} m  "
        f"energy={first['energy_j']:.1f} J  "
        f"time={first['completion_time_s']:.2f} s"
    )
    print(
        f"min_clearance={first['min_clearance_m']:.3f} m  "
        f"violations={first['constraint_violation_rate']:.3f}"
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
