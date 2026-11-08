"""ETH/UCY 真实行人轨迹接入：锚点标定、可用性统计、动态层三类指标、两类图。

```bash
uv run python -m analysis.pedestrians --jobs 8                 # 全流程（含 DWA episode）
uv run python -m analysis.pedestrians --anchor-only --jobs 8   # 只做锚点标定 + 可用性统计
uv run python -m analysis.pedestrians --sequences eth students003 --jobs 4
```

产出：

| 文件 | 内容 |
|---|---|
| `docs/pedestrian_anchor.csv` | 逐 (场景, 序列) 的**锚点映射**（原点 / 旋转 / 缩放）与拟合冲突率 |
| `docs/pedestrian_usability.csv` | **可用性统计**：可用比例 / 出界比例 / 穿墙比例 / 被丢弃行人 |
| `docs/dwa_metrics_eth_ucy.csv` | 三类指标（主动碰撞 / 被动接触 / 未完成），口径与手工轨迹表**完全一致** |
| `docs/pedestrians/contact_decomposition.png` | 接触帧二维分解（成因 × 几何可行性） |
| `docs/pedestrians/episode_<场景>_<序列>.png` | 一个 episode 的机器人 + 行人**联合轨迹** |
| `docs/pedestrians/trajectories_<序列>.png` | 行人轨迹叠在户型栅格上（按时间着色） |

口径（论文必须写，交接文档 7.12 与 7.15 沿用）：

1. **成功 = 到达终点 ∧ 无物理违反**；``timeout`` 不算成功；
2. 接触帧**二维分解**：成因（本车是否在动）× 几何可行性（``W < 1.76 m``）；
   三类结论并列：① 主动碰撞（必须 0）② 被动接触 ③ 未完成（超时）；
3. **不使用"零碰撞"这一说法**；
4. **真人不避让机器人**：被动接触会大幅上升，这是外部效度的必然代价。
   不得为了指标好看去修正行人轨迹或放宽判据。
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_ALGO = "configs/frozen_A.yaml"
SEQUENCE_CONFIG = "pedestrian_sequences.yaml"
ANCHOR_CSV = PROJECT_ROOT / "docs" / "pedestrian_anchor.csv"
USABILITY_CSV = PROJECT_ROOT / "docs" / "pedestrian_usability.csv"
METRICS_CSV = PROJECT_ROOT / "docs" / "dwa_metrics_eth_ucy.csv"
OUT_DIR = PROJECT_ROOT / "docs" / "pedestrians"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from core import astar, bspline, dwa  # noqa: E402
from env import pedestrians as ped  # noqa: E402
from runner import config as config_module  # noqa: E402
from runner import experiment  # noqa: E402
from analysis import dwa as dwa_analysis  # noqa: E402

_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
)
COLOR_ROBOT = "#2ca02c"
COLOR_REF = "#1f77b4"
COLOR_PED = "#d62728"


def _setup_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.monospace"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceSpec:
    """一个序列的声明（来自 ``config/pedestrian_sequences.yaml``）。"""

    name: str
    file: str
    venue: str
    fps: float
    frame_step: int
    extent_m: tuple[float, float]
    note: str

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / "data" / "eth_ucy" / self.file


@dataclass(frozen=True)
class PedestrianConfig:
    radius_m: float
    scale: float
    rotation_step_deg: float
    offset_half_range_m: float
    offset_step_m: float
    free_clearance_m: float
    max_conflict_ratio: float
    dt_s: float
    max_consecutive_wall_crossings: int
    max_observation_gap_s: float
    min_samples: int
    velocity_window: int
    window_s: float
    target_pedestrians_in_window: int
    max_pedestrians_in_window: int
    sequences: Mapping[str, SequenceSpec]

    def spec(self, name: str) -> SequenceSpec:
        if name not in self.sequences:
            raise config_module.ConfigError(
                f"pedestrian_sequences.yaml 里没有序列 {name!r}；"
                f"可选：{', '.join(sorted(self.sequences))}"
            )
        return self.sequences[name]


def load_pedestrian_config(path: str | Path) -> PedestrianConfig:
    """读取并校验 ``config/pedestrian_sequences.yaml``。"""
    doc = config_module.read_config(path)
    unknown = sorted(
        set(doc)
        - {"version", "pedestrian_radius_m", "anchor", "time_alignment", "sequences", "task_design"}
    )
    if unknown:
        raise config_module.ConfigError(f"pedestrian_sequences 含未知字段: {', '.join(unknown)}")
    anchor = doc.get("anchor") or {}
    timing = doc.get("time_alignment") or {}
    design = doc.get("task_design") or {}
    raw_sequences = doc.get("sequences")
    if not isinstance(raw_sequences, Mapping) or not raw_sequences:
        raise config_module.ConfigError("pedestrian_sequences.sequences 必须是非空映射")
    sequences: dict[str, SequenceSpec] = {}
    for name, entry in raw_sequences.items():
        if not isinstance(entry, Mapping):
            raise config_module.ConfigError(f"sequences.{name} 必须是映射")
        missing = {"file", "venue"} - set(entry)
        if missing:
            raise config_module.ConfigError(
                f"sequences.{name} 缺少必填字段: {', '.join(sorted(missing))}"
            )
        extent = entry.get("extent_m", [0.0, 0.0])
        sequences[str(name)] = SequenceSpec(
            name=str(name),
            file=str(entry["file"]),
            venue=str(entry["venue"]),
            fps=float(entry.get("fps", ped.DATASET_FPS)),
            frame_step=int(entry.get("frame_step", ped.DATASET_FRAME_STEP)),
            extent_m=(float(extent[0]), float(extent[1])),
            note=str(entry.get("note", "")),
        )
    return PedestrianConfig(
        radius_m=float(doc.get("pedestrian_radius_m", ped.DEFAULT_PEDESTRIAN_RADIUS_M)),
        scale=float(anchor.get("scale", 1.0)),
        rotation_step_deg=float(anchor.get("rotation_step_deg", 5.0)),
        offset_half_range_m=float(anchor.get("offset_half_range_m", 4.0)),
        offset_step_m=float(anchor.get("offset_step_m", 1.0)),
        free_clearance_m=float(anchor.get("free_clearance_m", 0.9)),
        max_conflict_ratio=float(anchor.get("max_conflict_ratio", 0.25)),
        dt_s=float(timing.get("dt_s", 0.1)),
        max_consecutive_wall_crossings=int(timing.get("max_consecutive_wall_crossings", 3)),
        max_observation_gap_s=float(
            timing.get("max_observation_gap_s", 1.5 * ped.DATASET_DT_S)
        ),
        min_samples=int(timing.get("min_samples", 10)),
        velocity_window=int(timing.get("velocity_window", 5)),
        window_s=float(design.get("window_s", 30.0)),
        target_pedestrians_in_window=int(design.get("target_pedestrians_in_window", 15)),
        max_pedestrians_in_window=int(design.get("max_pedestrians_in_window", 25)),
        sequences=sequences,
    )


# ---------------------------------------------------------------------------
# 锚点标定缓存（逐 (场景, 序列)）
# ---------------------------------------------------------------------------

_ANCHOR_CACHE: dict[tuple[str, str, float, float], tuple[ped.AnchorMapping, float]] = {}


def anchor_for(
    scenario_id: str,
    sequence: str,
    *,
    raw: ped.RawSequence,
    clearance: np.ndarray,
    occupancy: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    config: PedestrianConfig,
    body_radius: float,
) -> tuple[ped.AnchorMapping, float]:
    """``fit_anchor`` 的带缓存包装（同一进程内对同一 (场景, 序列) 只算一次）。

    锚点只依赖 (场景几何, 序列, 搜索参数)，与 seed / 时间窗无关；缓存主要收益在
    **调试与重复调用**时（实测 ``students001`` 单次拟合 ≈ 1 s，重复调用直接命中）。
    """
    key = (scenario_id, sequence, float(config.rotation_step_deg), float(config.scale))
    if key not in _ANCHOR_CACHE:
        _ANCHOR_CACHE[key] = ped.fit_anchor(
            raw,
            clearance_field=clearance,
            occupancy=occupancy,
            resolution=resolution,
            origin=origin,
            free_clearance_m=config.free_clearance_m,
            body_radius=body_radius,
            rotation_step_deg=config.rotation_step_deg,
            offset_half_range_m=config.offset_half_range_m,
            offset_step_m=config.offset_step_m,
            scale=config.scale,
        )
    return _ANCHOR_CACHE[key]


def clear_anchor_cache() -> None:
    _ANCHOR_CACHE.clear()


# ---------------------------------------------------------------------------
# 一个 episode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PedestrianEpisode:
    """一个 (场景, 序列) 的完整结果 = DWA Episode + 锚点 + 可用性 + 行人轨迹。"""

    episode: dwa_analysis.Episode
    sequence: str
    venue: str
    anchor: ped.AnchorMapping
    conflict_ratio: float
    usability: ped.UsabilityReport
    pedestrian_count: int
    window_start_s: float
    window_length_s: float
    scenario_path: str
    pedestrian_crowd_dropped: int = 0
    """因**拥挤度受控**被排除在 episode 之外的行人条数（离参考路径较远的那批）。"""
    window_tracks: tuple[ped.PedestrianTrack, ...] = ()
    window_usability: ped.UsabilityReport | None = None


def _sequence_pedestrian_count(spec: SequenceSpec) -> int:
    """序列的一行规模（用于成本代理；读不到就退化为 1）。"""
    cached = _PEDESTRIAN_COUNT_CACHE.get(spec.name)
    if cached is not None:
        return cached
    try:
        raw = ped.load_sequence(spec.path, name=spec.name)
        count = len(raw.pedestrian_ids)
    except ped.PedestrianDataError:
        count = 1
    _PEDESTRIAN_COUNT_CACHE[spec.name] = count
    return count


_PEDESTRIAN_COUNT_CACHE: dict[str, int] = {}


def _limit_crowd(
    tracks: Sequence[ped.PedestrianTrack],
    *,
    reference_xy: np.ndarray,
    limit: int,
) -> tuple[tuple[ped.PedestrianTrack, ...], int]:
    """把窗口内行人限制到**最可能与本车发生交互**的 ``limit`` 条。

    为什么需要：拥挤度是**受控实验因子**，但它在时间窗层面不可控 —— 实测
    ``students003`` 的*每一个* 40 s 窗口里都有 93~134 个不同行人（场地本来就那么忙），
    所以只能按轨迹筛选。判据取"该行人轨迹到参考路径的**最小距离**"：
    距离越近越可能逼停/迫使绕行，也就是真正进入指标的那批人。

    同时这也是本轮的性能开关：局部层对每个候选**逐个行人**做标量运算，
    成本 ∝ 候选数 × 行人数 × 周期数（实测 93 人 = 76 s/episode，12 人 ≈ 10 s）。

    被剔除的条数单独返回并写进指标表，**不隐藏**。
    """
    if limit <= 0 or len(tracks) <= limit:
        return tuple(tracks), 0
    reference = np.asarray(reference_xy, dtype=float)
    if reference.shape[0] > 400:  # 参考路径抽稀，距离只用于排序
        keep = np.linspace(0, reference.shape[0] - 1, 400).astype(int)
        reference = reference[keep]
    scored: list[tuple[float, int]] = []
    for index, track in enumerate(tracks):
        xy = np.asarray(track.xy, dtype=float)
        if xy.shape[0] > 200:
            xy = xy[np.linspace(0, xy.shape[0] - 1, 200).astype(int)]
        # 到参考路径的最近距离（点集到点集，规模已抽稀，直接广播）
        distance = np.sqrt(
            ((xy[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        ).min()
        scored.append((float(distance), index))
    scored.sort(key=lambda item: (item[0], item[1]))
    chosen = sorted(index for _distance, index in scored[:limit])
    return tuple(tracks[index] for index in chosen), len(tracks) - len(chosen)


def run_episode(
    scenario_path: str | Path,
    sequence: str,
    *,
    config: PedestrianConfig,
    algo: str = DEFAULT_ALGO,
    config_root: Path = CONFIG_ROOT,
    seed: int = 42,
) -> PedestrianEpisode:
    """跑一个 (场景, 序列)：锚点 -> 行人轨迹 -> DWA -> 三类指标。"""
    scenario_path = Path(scenario_path)
    resolved = config_module.load_resolved(
        algo, scenario_path, config_root=config_root
    )
    scenario = resolved.scenario
    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    body_radius = float(resolved.platform["body_radius"])
    spec = config.spec(sequence)

    result = experiment.plan_path(scene, setup)
    if not result.success:
        raise RuntimeError(f"{scenario['id']}: A* 无解（{result.reason}）")
    smooth = experiment.smooth_result(result, scene, setup)
    if not smooth.trackable:
        raise dwa.NotTrackableError(
            f"{scenario['id']}: 参考轨迹不可跟踪（{smooth.status}/{smooth.report.reason}）"
        )

    raw = ped.load_sequence(spec.path, name=spec.name)
    anchor, conflict = anchor_for(
        str(scenario["id"]),
        spec.name,
        raw=raw,
        clearance=scene.clearance,
        occupancy=scene.occupancy,
        resolution=scene.resolution,
        origin=scene.origin,
        config=config,
        body_radius=body_radius,
    )
    tracks = ped.build_sequence_tracks(
        raw,
        scene_occupancy=scene.occupancy,
        clearance_field=scene.clearance,
        resolution=scene.resolution,
        origin=scene.origin,
        anchor=anchor,
        dt_s=config.dt_s,
        body_radius=body_radius,
        pedestrian_radius_m=config.radius_m,
        max_consecutive_wall_crossings=config.max_consecutive_wall_crossings,
        max_observation_gap_s=config.max_observation_gap_s,
        min_samples=config.min_samples,
        velocity_window=config.velocity_window,
    )
    tracks = ped.with_conflict_ratio(tracks, conflict)

    # 行人时间窗：起点由 seed 决定（计入哈希），长度取配置值
    clipped, window_start, window_length = _window_for(
        tracks.tracks, config=config, seed=seed, scenario_id=str(scenario["id"]),
        sequence=spec.name,
    )
    # 拥挤度受控：只保留离参考路径最近的 N 条行人轨迹（见 _limit_crowd 的理由）
    clipped, crowd_dropped = _limit_crowd(
        clipped,
        reference_xy=np.asarray(smooth.xy, dtype=float),
        limit=int(config.max_pedestrians_in_window),
    )
    window_report, _ = ped.window_usability(
        tracks.tracks,
        window_start_s=window_start,
        window_length_s=window_length,
        dt_s=config.dt_s,
        body_radius=body_radius,
    )
    tracks = ped.with_conflict_ratio(tracks, conflict)

    params_dict = config_module.dwa_effective_params(resolved.algo, resolved.platform)
    params = dwa.DwaParams(
        v_max=params_dict["v_max"],
        w_max=params_dict["w_max"],
        a_max=params_dict["a_max"],
        alpha_max=params_dict["alpha_max"],
        horizon_s=params_dict["horizon_s"],
        dt_s=params_dict["dt_s"],
        v_samples=params_dict["v_samples"],
        w_samples=params_dict["w_samples"],
        d_min=params_dict["d_min"],
        body_radius=params_dict["body_radius"],
        v_nominal=params_dict["v_nominal"],
        a_y_max=params_dict["a_y_max"],
        jerk_weight=params_dict["jerk_weight"],
        w_heading=params_dict["w_heading"],
        w_dist=params_dict["w_dist"],
        w_vel=params_dict["w_vel"],
        w_clearance=params_dict["w_clearance"],
        gap_head_on_m=params_dict["gap_head_on_m"],
        gap_overtake_m=params_dict["gap_overtake_m"],
        lateral_deviation_max_m=params_dict["lateral_deviation_max_m"],
    )
    reference = dwa.reference_from_smooth(smooth)
    obstacles = ped.to_obstacle_tracks(clipped)
    planner = dwa.DwaPlanner(
        reference=reference,
        params=params,
        clearance_field=scene.clearance,
        resolution=scene.resolution,
        origin=scene.origin,
        start_xy=np.asarray(scene.start_xy, dtype=float),
    )
    outcome = planner.simulate(obstacles)

    commands = outcome.commands
    times = np.array([c.t for c in commands], dtype=float) if commands else np.zeros(0)
    actual = (
        np.array([c.xy for c in commands], dtype=float) if commands else np.zeros((0, 2))
    )
    speeds = np.array([c.v for c in commands], dtype=float) if commands else np.zeros(0)
    yaw_rates = np.array([c.w for c in commands], dtype=float) if commands else np.zeros(0)
    a_y_series = np.abs(speeds * yaw_rates) if commands else np.zeros(0)

    # ---- 执行口径圆心距：逐周期对全部行人取最小 ----
    radii = [float(track.radius_m) for track in clipped]
    obstacle_radius_m = min(radii) if radii else config.radius_m
    contact_gap_m = float(body_radius) + float(obstacle_radius_m)
    meeting_width_required_m = dwa_analysis.MEETING_WIDTH_FACTOR * contact_gap_m
    quality = scenario.get("quality") or {}
    bottleneck_width_m = float(quality.get("bottleneck_width_m", float("nan")))
    traces: list[np.ndarray] = []
    realized_gaps = np.zeros(0, dtype=float)
    nearest_index = np.zeros(0, dtype=int)
    if times.size and clipped:
        for track in clipped:
            traces.append(
                track_xyz := np.asarray(track.xy, dtype=float)[
                    np.clip(
                        np.searchsorted(
                            np.asarray(track.times_s, dtype=float), times, side="right"
                        )
                        - 1,
                        0,
                        max(0, np.asarray(track.times_s).shape[0] - 1),
                    )
                ]
            )
            del track_xyz
        gap_matrix = np.stack(
            [np.linalg.norm(actual - trace, axis=1) for trace in traces], axis=1
        )
        realized_gaps = gap_matrix.min(axis=1)
        nearest_index = gap_matrix.argmin(axis=1)
        obstacle_trace = traces[int(nearest_index[np.argmin(realized_gaps)])]
        min_gap_realized_m = float(realized_gaps.min())
        dynamic_contact_periods = int(np.sum(realized_gaps < contact_gap_m - 1e-9))
        contact_breakdown = dwa_analysis.decompose_contact(
            realized_gaps,
            speeds,
            contact_gap_m=contact_gap_m,
            bottleneck_width_m=bottleneck_width_m,
            meeting_width_required_m=meeting_width_required_m,
        )
    else:
        obstacle_trace = np.zeros((0, 2))
        min_gap_realized_m = float("inf")
        dynamic_contact_periods = 0
        contact_breakdown = dwa_analysis.ContactBreakdown(0, 0, 0, 0, 0, (0, 0, 0, 0))

    path_length = (
        float(np.sum(np.linalg.norm(np.diff(actual, axis=0), axis=1)))
        if actual.shape[0] > 1
        else 0.0
    )
    energy_j = _path_energy(scene, actual, speeds, params, setup)

    episode = dwa_analysis.Episode(
        scenario_id=str(scenario["id"]),
        grade=str(quality.get("grade", "-")),
        obstacle_kind=f"eth_ucy:{spec.name}",
        obstacle_label=f"{spec.venue}（{spec.name}）",
        trackable=True,
        smooth_status=str(smooth.status),
        reached_goal=bool(outcome.reached_goal),
        reason=str(outcome.reason),
        task_ok=bool(outcome.success),
        periods=int(outcome.periods),
        violation_periods=int(outcome.violation_periods),
        violation_rate_per_period=float(outcome.constraint_violation_rate_per_period),
        violation_rate_per_task=float(outcome.constraint_violation_rate_per_task),
        collision_frames=int(outcome.collision_frames),
        physical_violation_periods=int(outcome.physical_violation_periods),
        policy_violation_periods=int(outcome.policy_violation_periods),
        dynamic_violation_periods=int(outcome.dynamic_violation_periods),
        fell_back_periods=int(outcome.fallback_periods),
        body_radius=float(params.body_radius),
        d_min=float(params.d_min),
        min_clearance_m=float(outcome.min_clearance_m),
        min_gap_dynamic_m=float(outcome.min_gap_dynamic_m),
        obstacle_radius_m=float(obstacle_radius_m),
        contact_gap_m=float(contact_gap_m),
        min_gap_realized_m=float(min_gap_realized_m),
        dynamic_contact_periods=int(dynamic_contact_periods),
        bottleneck_width_m=bottleneck_width_m,
        meeting_width_required_m=float(meeting_width_required_m),
        contact=contact_breakdown,
        max_clearance_deficit_m=float(outcome.max_clearance_deficit_m),
        max_physical_deficit_m=float(outcome.max_physical_deficit_m),
        max_gap_deficit_m=float(outcome.max_gap_deficit_m),
        max_a_y_deficit=float(outcome.max_a_y_deficit),
        completion_time_s=float(times[-1]) if times.size else 0.0,
        path_length_m=float(path_length),
        reference_length_m=float(reference.length_m),
        energy_j=float(energy_j),
        a_y_max=float(a_y_series.max()) if a_y_series.size else 0.0,
        a_y_limit=float(params.a_y_max),
        jerk_rms=_jerk_rms(times, speeds),
        v_mean=float(np.mean(np.abs(speeds))) if speeds.size else 0.0,
        w_abs_mean=float(np.mean(np.abs(yaw_rates))) if yaw_rates.size else 0.0,
        exit_distance_m=float(np.linalg.norm(np.asarray(actual[-1]) - np.asarray(reference.xy[-1])))
        if actual.shape[0]
        else 0.0,
        reference=np.asarray(reference.xy, dtype=float),
        actual=np.asarray(actual, dtype=float),
        obstacle_trace=np.asarray(obstacle_trace, dtype=float),
        times=times,
        speeds=speeds,
        yaw_rates=yaw_rates,
        a_y_series=a_y_series,
        config_hash=str(resolved.config_hash),
    )
    return PedestrianEpisode(
        episode=episode,
        sequence=spec.name,
        venue=spec.venue,
        anchor=anchor,
        conflict_ratio=float(conflict),
        usability=tracks.report,
        pedestrian_count=len(clipped),
        window_start_s=float(window_start),
        window_length_s=float(window_length),
        scenario_path=str(scenario_path),
        pedestrian_crowd_dropped=int(crowd_dropped),
        window_tracks=tuple(clipped),
        window_usability=window_report,
    )


def _path_energy(
    scene: experiment.Scene,
    actual: np.ndarray,
    speeds: np.ndarray,
    params: dwa.DwaParams,
    setup: experiment.PlannerSetup,
) -> float:
    """按实际几何长度复算能耗（口径与手工轨迹表一致：稳态巡航、逐段 c_rr）。"""
    from core import energy as energy_module

    if actual.shape[0] < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(actual, axis=0), axis=1)
    midpoints = 0.5 * (actual[:-1] + actual[1:])
    c_rr_values = bspline.sample_field(
        scene.c_rr, midpoints, resolution=scene.resolution, origin=scene.origin
    )
    slope_values = bspline.sample_field(
        scene.slope_rad, midpoints, resolution=scene.resolution, origin=scene.origin
    )
    deltas = np.diff(actual, axis=0)
    direction_x = deltas[:, 0] / np.maximum(steps, 1e-12)
    total = 0.0
    for index in range(steps.shape[0]):
        if steps[index] <= 1e-12:
            continue
        speed = float(speeds[index + 1]) if speeds.shape[0] > index + 1 else params.v_nominal
        if abs(speed) <= 1e-12:
            continue
        total += energy_module.step_energy(
            float(steps[index]),
            abs(speed),
            c_rr=float(c_rr_values[index]),
            slope_rad=float(slope_values[index]),
            direction_x=float(direction_x[index]),
            params=setup.energy_params,
        )
    return float(total)


def _jerk_rms(times: np.ndarray, speeds: np.ndarray) -> float:
    if times.shape[0] < 3:
        return 0.0
    accel = np.gradient(speeds, times, edge_order=1)
    jerk = np.gradient(accel, times, edge_order=1)
    return float(np.sqrt(np.mean(jerk * jerk)))


def _window_for(
    tracks: Sequence[ped.PedestrianTrack],
    *,
    config: PedestrianConfig,
    seed: int,
    scenario_id: str,
    sequence: str,
) -> tuple[tuple[ped.PedestrianTrack, ...], float, float]:
    """截取行人公共时间窗：起点由 ``(scenario_id, sequence, seed)`` 确定性派生。

    返回 ``(窗口内轨迹, 窗口起点, 窗口长度)``；时间基**不重设**（整条序列已对齐）。

    起点不是"在整段时长上均匀随机"：序列里常有长时间空档（实测 Hotel 的
    观测集中在前后两段），均匀取样会让窗口落进空档、episode 里一个行人都没有
    （实测 ``window_sample_usable_ratio = 0``）。做法是：
    **先求"最多行人同时在观测"的最早时刻**（这保证窗口非空且被充分覆盖），
    再把窗口起点在这个可行区间内按 seed 取一个确定性的位置。
    """
    if not tracks:
        return ((), 0.0, 0.0)
    start_limit, end_limit = ped.episode_time_range(tracks)
    grid = np.arange(start_limit, end_limit + config.dt_s, config.dt_s)
    if grid.size == 0:
        return ((), start_limit, 0.0)
    active = np.zeros(grid.shape[0], dtype=int)
    for track in tracks:
        times = np.asarray(track.times_s, dtype=float)
        lo = int(np.searchsorted(grid, times[0], side="left"))
        hi = int(np.searchsorted(grid, times[-1], side="right"))
        active[max(0, lo): max(0, hi)] += 1
    span = max(1, int(round(config.window_s / config.dt_s)))
    if active.size <= span:
        offset = float(grid[0])
    else:
        # 拥挤度 = **窗口内出现的不同行人数**（不是"同时在观测的平均人数"）。
        # 为什么用它：① 它同时决定指标的可比性与局部层的计算量
        #   （`core/dwa.evaluate_candidate` 对每个候选**逐个行人**做标量运算，
        #    成本 ∝ 候选数 × 窗口内行人数 × 周期数）；② 实测两者差别很大 ——
        #    `students003` 的 40 s 窗口里"不同行人"93 个，但多数只穿过几秒。
        # 用两个前缀和精确统计"与窗口相交的轨迹条数"：
        #   相交数(i) = #(首样本 <= i+span) − #(末样本 < i)
        starts = np.zeros(active.size + 1, dtype=np.int64)
        ends = np.zeros(active.size + 1, dtype=np.int64)
        for track in tracks:
            times = np.asarray(track.times_s, dtype=float)
            lo = int(np.searchsorted(grid, times[0], side="left"))
            hi = int(np.searchsorted(grid, times[-1], side="right")) - 1
            if hi < lo:
                continue
            starts[min(lo, active.size)] += 1
            ends[min(max(hi, 0), active.size - 1)] += 1
        cumulative_start = np.cumsum(starts)
        cumulative_end = np.cumsum(ends)
        index = np.arange(active.size - span)
        coverage = (
            cumulative_start[index + span] - cumulative_end[np.maximum(index - 1, 0)]
        ).astype(float)
        # **拥挤度是受控量**：窗口内"同时在观测"的行人数应当与目标值接近，而不是
        # "越挤越好"。两个理由：
        #   ① 科学上：不同序列/场景的 episode 必须拥挤度可比，否则指标差异里混进了
        #      "哪个窗口更挤"这个混杂因素；
        #   ② 工程上：`core/dwa.evaluate_candidate` 对每个候选**逐个行人**调用 numpy
        #      标量运算，成本 ∝ 候选数 × 行人数 × 周期数。实测一个 122 人的窗口要
        #      337 s、`relative_direction` 被调用 490 万次；降到 ~15 人后单个
        #      episode 回到几十秒量级。
        # 选法：|coverage − target| 最小；并列区间用 seed 派生的位置打破，
        # 保持"同 seed 可复现、不同 seed 有差异"。
        target = int(config.target_pedestrians_in_window)
        cap = int(config.max_pedestrians_in_window)
        candidates = np.arange(coverage.size)
        if cap > 0:
            allowed = candidates[coverage[candidates] <= cap]
            if allowed.size:
                candidates = allowed
        if candidates.size == 0:
            candidates = np.arange(coverage.size)
        deviation = np.abs(coverage[candidates] - target)
        # 注意不要 int() 截断：最小偏差常是 0.4 这类小数，截断成 0 会让
        # `deviation <= 0` 恒假、短名单为空（本轮踩过）。
        best = float(deviation.min())
        shortlist = candidates[deviation <= best + 1e-9]
        derived = ped.stable_seed(f"{scenario_id}|{sequence}", f"ped{seed}")
        offset = float(grid[int(shortlist[derived % shortlist.size])])
    return ped.common_window(tracks, window_s=config.window_s, t_offset_s=offset)


# ---------------------------------------------------------------------------
# 表与图
# ---------------------------------------------------------------------------

ANCHOR_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "grade",
    "sequence",
    "venue",
    "scale",
    "rotation_deg",
    "origin_x",
    "origin_y",
    "conflict_ratio",
    "fits",
    "seq_extent_x_m",
    "seq_extent_y_m",
    "free_rect_x_m",
    "free_rect_y_m",
)

USABILITY_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "grade",
    "sequence",
    "venue",
    "pedestrians_total",
    "pedestrians_used",
    "pedestrians_usable_ratio",
    "samples_raw",
    "samples_resampled",
    "samples_out_of_bounds",
    "samples_wall_crossing",
    "samples_used",
    "out_of_bounds_ratio",
    "wall_crossing_ratio",
    "sample_usable_ratio",
    "dropped_consecutive_wall_crossing",
    "dropped_too_few_samples",
    "wiped_windows",
    "anchor_conflict_ratio",
    "duration_s",
    "fits",
    "window_pedestrians_used",
    "window_sample_usable_ratio",
    "pedestrians_in_episode",
)

METRICS_COLUMNS: tuple[str, ...] = tuple(dwa_analysis.METRIC_COLUMNS) + (
    "sequence",
    "venue",
    "anchor_rotation_deg",
    "anchor_origin_x",
    "anchor_origin_y",
    "anchor_conflict_ratio",
    "pedestrians_in_window",
    "pedestrians_crowd_dropped",
    "pedestrian_samples_usable_ratio",
    "window_start_s",
    "window_length_s",
)


def anchor_row(episode: PedestrianEpisode, fits: bool) -> dict[str, Any]:
    return {
        "scenario_id": episode.episode.scenario_id,
        "grade": episode.episode.grade,
        "sequence": episode.sequence,
        "venue": episode.venue,
        "scale": f"{episode.anchor.scale:.4f}",
        "rotation_deg": f"{episode.anchor.rotation_deg:.2f}",
        "origin_x": f"{episode.anchor.origin_xy[0]:.4f}",
        "origin_y": f"{episode.anchor.origin_xy[1]:.4f}",
        "conflict_ratio": f"{episode.conflict_ratio:.4f}",
        "fits": int(fits),
        "seq_extent_x_m": "",
        "seq_extent_y_m": "",
        "free_rect_x_m": "",
        "free_rect_y_m": "",
    }


def usability_row(episode: PedestrianEpisode, fits: bool) -> dict[str, Any]:
    """可用性统计行：**全序列**指标 + **观察窗内**指标（两者分母不同，必须分列）。"""
    row = episode.usability.to_dict()
    window = episode.window_usability
    row.update(
        {
            "scenario_id": episode.episode.scenario_id,
            "grade": episode.episode.grade,
            "venue": episode.venue,
            "fits": int(fits),
            "window_pedestrians_used": (window.pedestrians_used if window else 0),
            "window_sample_usable_ratio": (
                round(float(window.sample_usable_ratio), 4) if window else 0.0
            ),
            "pedestrians_in_episode": episode.pedestrian_count,
        }
    )
    return row


def metrics_row(episode: PedestrianEpisode) -> dict[str, Any]:
    row = dwa_analysis._row(episode.episode)
    window = episode.window_usability
    row.update(
        {
            "sequence": episode.sequence,
            "venue": episode.venue,
            "anchor_rotation_deg": f"{episode.anchor.rotation_deg:.2f}",
            "anchor_origin_x": f"{episode.anchor.origin_xy[0]:.4f}",
            "anchor_origin_y": f"{episode.anchor.origin_xy[1]:.4f}",
            "anchor_conflict_ratio": f"{episode.conflict_ratio:.4f}",
            "pedestrians_in_window": episode.pedestrian_count,
            "pedestrians_crowd_dropped": episode.pedestrian_crowd_dropped,
            "pedestrian_samples_usable_ratio": (
                f"{window.sample_usable_ratio:.4f}" if window else ""
            ),
            "window_start_s": f"{episode.window_start_s:.2f}",
            "window_length_s": f"{episode.window_length_s:.2f}",
        }
    )
    return row


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    return path


def format_usability_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        f"{'序列':<14}{'组合':>5}{'可用行人':>10}{'可用比例':>9}"
        f"{'出界比例':>9}{'穿墙比例':>9}{'样本可用':>9}{'丢弃段':>8}"
    )
    lines = [
        "ETH/UCY 可用性统计（三条比例同一分母：连续观测段内按 dt 重采样后的样本数）",
        header,
        "-" * len(header),
    ]
    by_sequence: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_sequence.setdefault(str(row["sequence"]), []).append(row)
    for sequence in sorted(by_sequence):
        subset = by_sequence[sequence]
        usable = np.mean([float(r["pedestrians_usable_ratio"]) for r in subset])
        oob = np.mean([float(r["out_of_bounds_ratio"]) for r in subset])
        wall = np.mean([float(r["wall_crossing_ratio"]) for r in subset])
        samples = np.mean([float(r["sample_usable_ratio"]) for r in subset])
        dropped = sum(int(r["dropped_consecutive_wall_crossing"]) for r in subset)
        lines.append(
            f"{sequence:<14}{len(subset):>5}{usable:>10.3f}{usable:>9.3f}"
            f"{oob:>9.3f}{wall:>9.3f}{samples:>9.3f}{dropped:>8}"
        )
    return "\n".join(lines)


def plot_trajectory_map(
    episode: PedestrianEpisode,
    scene: experiment.Scene,
    out_path: Path,
) -> Path:
    """图 1：行人轨迹叠在户型栅格上（按时间着色）+ 本车参考轨迹与起终点。"""
    _setup_fonts()
    figure, axis = plt.subplots(figsize=(8.6, 7.4), constrained_layout=True)
    rows, cols = scene.occupancy.shape
    ox, oy = scene.origin
    extent = (ox, ox + cols * scene.resolution, oy, oy + rows * scene.resolution)
    axis.imshow(
        np.where(scene.occupancy, 1, 0), origin="lower", extent=extent,
        cmap=ListedColormap(["#ffffff", "#3b3b3b"]), interpolation="nearest", zorder=1,
    )
    cmap = plt.get_cmap("viridis")
    for track in episode.window_tracks:
        times = np.asarray(track.times_s, dtype=float)
        span = max(1e-9, float(times[-1] - times[0]))
        axis.scatter(
            track.xy[:, 0], track.xy[:, 1],
            c=(times - times[0]) / span, cmap=cmap, s=7, vmin=0.0, vmax=1.0,
            zorder=3, alpha=0.9,
        )
        axis.add_patch(
            Circle(
                (float(track.xy[-1, 0]), float(track.xy[-1, 1])), float(track.radius_m),
                facecolor="none", edgecolor=COLOR_PED, linewidth=0.7, zorder=4,
            )
        )
    axis.plot(
        episode.episode.reference[:, 0], episode.episode.reference[:, 1],
        "-", color=COLOR_REF, linewidth=1.8, zorder=5, label="A* 参考轨迹",
    )
    axis.plot(
        episode.episode.reference[0, 0], episode.episode.reference[0, 1],
        "o", color="#2ca02c", markersize=10, zorder=6, label="起点",
    )
    axis.plot(
        episode.episode.reference[-1, 0], episode.episode.reference[-1, 1],
        "*", color="#111111", markersize=16, zorder=6, label="终点",
    )
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(
        f"{episode.episode.scenario_id}（{episode.episode.grade}）× {episode.sequence}"
        f"（{episode.venue}）\n"
        f"时间窗 {episode.window_start_s:.1f}~"
        f"{episode.window_start_s + episode.window_length_s:.1f} s，"
        f"{episode.pedestrian_count} 条行人轨迹（颜色 = 窗口内归一化时间）",
        fontsize=10.5,
    )
    scalar = plt.cm.ScalarMappable(cmap=cmap)
    scalar.set_array(np.linspace(0.0, 1.0, 8))
    bar = figure.colorbar(scalar, ax=axis, fraction=0.045, pad=0.03)
    bar.set_label("窗口内归一化时间")
    axis.legend(fontsize=8.5, loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


def plot_joint_episode(
    episodes: Sequence[PedestrianEpisode], out_path: Path
) -> Path:
    """图 2：一个 episode 的机器人 + 行人联合轨迹（带圆环足迹）。"""
    _setup_fonts()
    if not episodes:
        raise ValueError("没有可画的 episode")
    count = len(episodes)
    figure, axes = plt.subplots(
        1, count, figsize=(6.6 * count, 6.2), constrained_layout=True, squeeze=False
    )
    for axis, episode in zip(axes[0], episodes):
        result = episode.episode
        for track in episode.window_tracks:
            axis.plot(track.xy[:, 0], track.xy[:, 1], "-", color=COLOR_PED,
                      linewidth=0.9, alpha=0.6, zorder=2,
                      label="行人轨迹" if track is episode.window_tracks[0] else None)
            axis.add_patch(
                Circle(
                    (float(track.xy[0, 0]), float(track.xy[0, 1])),
                    float(track.radius_m), facecolor=COLOR_PED, alpha=0.25,
                    edgecolor="none", zorder=2,
                )
            )
        axis.plot(
            result.reference[:, 0], result.reference[:, 1], "--", color=COLOR_REF,
            linewidth=1.5, label="参考轨迹", zorder=3,
        )
        if result.actual.shape[0]:
            axis.plot(
                result.actual[:, 0], result.actual[:, 1], "-", color=COLOR_ROBOT,
                linewidth=2.4, label="本车执行轨迹", zorder=4,
            )
            axis.plot(result.actual[0, 0], result.actual[0, 1], "o", color="#2ca02c",
                      markersize=9, zorder=6)
            axis.plot(result.actual[-1, 0], result.actual[-1, 1], "*", color="#111111",
                      markersize=15, zorder=6)
        axis.set_aspect("equal")
        axis.set_xlabel("x [m]")
        axis.grid(alpha=0.2)
        contact = result.contact
        axis.set_title(
            f"{episode.episode.scenario_id} × {episode.sequence}\n"
            f"{episode.episode.reason}：周期 {episode.episode.periods}，"
            f"接触帧 {contact.frames}"
            f"（主动 {contact.moving_frames} / 被动 {contact.stopped_frames}）",
            fontsize=9.8,
        )
        axis.legend(fontsize=8, loc="upper right")
    axes[0][0].set_ylabel("y [m]")
    figure.suptitle(
        "一个 episode 的机器人 + 行人联合轨迹\n"
        "真人不避让机器人：被动接触是外部效度的必然代价，主动碰撞必须仍为 0",
        fontsize=11.5,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.pedestrians",
        description="ETH/UCY 接入：锚点标定 + 可用性统计 + 动态层三类指标 + 两类图",
    )
    parser.add_argument("--algo", default=DEFAULT_ALGO)
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--tier", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--anchor-only", action="store_true", help="只做锚点与可用性统计")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个 (场景, 序列) 组合")
    return parser


def _task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        scenario_path, sequence, config_root, algo, seed, anchor_only,
    ) = task
    config = load_pedestrian_config(Path(str(config_root)) / SEQUENCE_CONFIG)
    try:
        episode = run_episode(
            str(scenario_path),
            str(sequence),
            config=config,
            algo=str(algo),
            config_root=Path(str(config_root)),
            seed=int(seed),
        )
    except (dwa.NotTrackableError, ped.PedestrianDataError, RuntimeError) as exc:
        return {
            "skipped": True,
            "reason": str(exc),
            "scenario": Path(str(scenario_path)).stem,
            "sequence": str(sequence),
        }
    fits = episode.conflict_ratio <= config.max_conflict_ratio
    # 进度输出：全量跑要几分钟，没有进度就只能盲等（本轮踩过）。
    print(
        f"[done] {Path(str(scenario_path)).stem} x {sequence}: "
        f"{episode.episode.reason} 周期={episode.episode.periods} "
        f"行人={episode.pedestrian_count}(剔{episode.pedestrian_crowd_dropped}) "
        f"主动碰撞={episode.episode.contact.moving_frames} "
        f"被动接触={episode.episode.contact.stopped_frames}",
        flush=True,
    )
    return {
        "skipped": False,
        "fits": fits,
        "anchor": anchor_row(episode, fits),
        "usability": usability_row(episode, fits),
        "metrics": None if anchor_only else metrics_row(episode),
        "figure": {
            "episode": episode.episode,
            "sequence": episode.sequence,
            "venue": episode.venue,
            "scenario_path": episode.scenario_path,
            "window_tracks": episode.window_tracks,
            "pedestrian_count": episode.pedestrian_count,
            "window_start_s": episode.window_start_s,
            "window_length_s": episode.window_length_s,
        },
    }


def _load_scene_for(path: str) -> experiment.Scene:
    resolved = config_module.load_resolved(
        DEFAULT_ALGO, Path(path), config_root=CONFIG_ROOT
    )
    return experiment.build_scene(resolved.scenario)


def _as_episode(payload: Mapping[str, Any]) -> PedestrianEpisode:
    """把 ``_task`` 的 payload 还原成画图需要的 ``PedestrianEpisode``（锚点/可用性略）。"""
    return PedestrianEpisode(
        episode=payload["episode"],
        sequence=str(payload["sequence"]),
        venue=str(payload["venue"]),
        anchor=ped.AnchorMapping(origin_xy=(0.0, 0.0)),
        conflict_ratio=float("nan"),
        usability=None,  # type: ignore[arg-type]
        pedestrian_count=int(payload["pedestrian_count"]),
        window_start_s=float(payload["window_start_s"]),
        window_length_s=float(payload["window_length_s"]),
        scenario_path=str(payload["scenario_path"]),
        window_tracks=tuple(payload["window_tracks"]),
    )


def _representative(payloads: Sequence[Mapping[str, Any]], sequence: str) -> Sequence[Mapping[str, Any]]:
    """按序列挑代表性 episode：优先"有被动接触且到达"的，其次接触最多的。"""
    subset = [item for item in payloads if str(item["sequence"]) == sequence]
    if not subset:
        return []

    def rank(item: Mapping[str, Any]) -> tuple[int, int, float]:
        result = item["episode"]
        return (
            1 if (result.reached_goal and result.contact.frames > 0) else 0,
            int(result.contact.frames),
            -abs(float(result.exit_distance_m)),
        )

    return sorted(subset, key=rank, reverse=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_root = Path(args.config_root) if args.config_root else CONFIG_ROOT
    config = load_pedestrian_config(config_root / SEQUENCE_CONFIG)
    sequences = list(args.sequences) if args.sequences else sorted(config.sequences)
    paths = (
        [Path(item) for item in args.scenarios]
        if args.scenarios
        else list_scenarios(args.tier, config_root=config_root)
    )
    if not paths:
        print("没有找到场景", flush=True)
        return 2
    tasks = [
        (str(path), sequence, str(config_root), args.algo, int(args.seed),
         bool(args.anchor_only))
        for path in paths
        for sequence in sequences
    ]
    if args.limit:
        tasks = tasks[: int(args.limit)]

    # **按预估成本降序下发**（LPT 调度）。为什么必须这么做：单个 episode 的成本
    # ≈ 周期数 × 候选数 × 行人数 × 步数，而周期数随参考路径长度线性增长，
    # 于是 hard 场景（20 m 路径）比 easy（8 m）贵 2~3 倍。FIFO 下发会让
    # "先做完 easy 的 worker 空转等 hard"，实测把总时长从 ~25 min 拖到 90 min 以上。
    # 成本代理：路径长度中位数/实测值（场景 quality.path_len_m）× 序列规模平方根。
    sequence_scale = {
        spec.name: float(np.sqrt(max(1, _sequence_pedestrian_count(spec))))
        for spec in config.sequences.values()
    }
    def cost_proxy(task: tuple[Any, ...]) -> float:
        scenario_path, sequence = str(task[0]), str(task[1])
        quality = (
            config_module.read_config(Path(scenario_path)).get("quality") or {}
        )
        length = float(quality.get("path_len_m", 10.0) or 10.0)
        return length * sequence_scale.get(sequence, 1.0)

    tasks.sort(key=cost_proxy, reverse=True)

    from concurrent.futures import ProcessPoolExecutor

    workers = int(args.jobs) if args.jobs else min(len(tasks), os.cpu_count() or 1)
    results: list[dict[str, Any]] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_task, tasks))
    else:
        results = [_task(task) for task in tasks]

    skipped = [item for item in results if item.get("skipped")]
    done = [item for item in results if not item.get("skipped")]
    anchor_path = write_csv([item["anchor"] for item in done], ANCHOR_CSV, ANCHOR_COLUMNS)
    usability_path = write_csv(
        [item["usability"] for item in done], USABILITY_CSV, USABILITY_COLUMNS
    )
    print(f"\n锚点映射: {anchor_path}\n可用性统计: {usability_path}")
    print(f"\n{format_usability_table([item['usability'] for item in done])}")
    if skipped:
        print(f"\n跳过 {len(skipped)} 个组合（不可跟踪 / 数据缺失）：")
        for item in skipped[:20]:
            print(f"  {item['scenario']} x {item['sequence']}: {item['reason'][:90]}")

    figures: list[Path] = []
    if not args.anchor_only:
        metrics_path = write_csv(
            [item["metrics"] for item in done], METRICS_CSV, METRICS_COLUMNS
        )
        print(f"\n动态层三类指标: {metrics_path}")
        print(_format_contact_table([item["metrics"] for item in done]))

        if not args.no_plots:
            out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
            payloads = [item["figure"] for item in done]
            # 图 1：每个序列各挑一个代表性 episode 出"行人在户型上"的叠图
            for sequence in sequences:
                ranked = _representative(payloads, sequence)
                if not ranked:
                    continue
                payload = ranked[0]
                scene = _load_scene_for(str(payload["scenario_path"]))
                figures.append(
                    plot_trajectory_map(
                        _as_episode(payload), scene,
                        out_dir / f"trajectories_{sequence}.png",
                    )
                )
            # 图 2：联合轨迹（最多 3 个：优先有接触且到达的）
            joint = sorted(
                payloads,
                key=lambda item: (
                    1 if item["episode"].reached_goal and item["episode"].contact.frames > 0 else 0,
                    int(item["episode"].contact.frames),
                ),
                reverse=True,
            )[:3]
            if joint:
                figures.append(
                    plot_joint_episode(
                        [_as_episode(item) for item in joint],
                        out_dir / "joint_episode.png",
                    )
                )
            # 接触帧二维分解（复用第 6.6 项的画法）
            contact_figure = dwa_analysis.plot_contact_decomposition(
                [item["figure"]["episode"] for item in done],
                out_dir / "contact_decomposition.png",
            )
            figures.append(contact_figure)
            for figure in figures:
                print(f"OK   {figure}")
    del figures
    return 0


def _format_contact_table(rows: Sequence[Mapping[str, Any]]) -> str:
    header = (
        f"{'序列':<14}{'任务':>5}{'到达':>5}{'超时':>5}{'主动碰撞帧':>11}"
        f"{'被动接触帧':>11}{'已停∧W<1.76':>13}{'接触任务':>9}{'最小圆心距':>11}"
    )
    lines = [
        "ETH/UCY 动态层三类指标（成功 = 到达 ∧ 无物理违反；timeout 不算成功）",
        header,
        "-" * len(header),
    ]
    by_sequence: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_sequence.setdefault(str(row["sequence"]), []).append(row)
    for sequence in sorted(by_sequence):
        subset = by_sequence[sequence]
        reached = sum(1 for r in subset if int(r["reached_goal"]) == 1)
        timeout = sum(1 for r in subset if str(r["reason"]) == "timeout")
        moving = sum(int(r["contact_moving_frames"]) for r in subset)
        stopped = sum(int(r["contact_stopped_frames"]) for r in subset)
        stopped_narrow = sum(int(r["contact_stopped_narrow"]) for r in subset)
        contact_tasks = sum(1 for r in subset if int(r["dynamic_contact_periods"]) > 0)
        gaps = [
            float(r["min_gap_realized_m"])
            for r in subset
            if np.isfinite(float(r["min_gap_realized_m"]))
        ]
        lines.append(
            f"{sequence:<14}{len(subset):>5}{reached:>5}{timeout:>5}{moving:>11}"
            f"{stopped:>11}{stopped_narrow:>13}{contact_tasks:>9}"
            f"{(min(gaps) if gaps else float('nan')):>11.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
