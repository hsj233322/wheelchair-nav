"""DWA 局部规划的验收图与指标表（交接文档 7.9 第 6、8 条）。

```bash
uv run python -m analysis.dwa                       # 全部可跟踪任务，三类动态障碍
uv run python -m analysis.dwa --tier easy --review   # 只跑 easy + 复制到 docs/review/
```

产出：

1. **轨迹叠图**：参考轨迹 vs 实际执行轨迹 + 动态障碍足迹；
2. **时间曲线**：速度 / 角速度 / 横向加速度 ``a_y`` 三联图，画出硬约束限值线；
3. **指标表**：碰撞帧数、最小离障、到达时间、路径长度、能耗、``a_y`` 最大值、
   jerk RMS、**约束违反率（按周期 + 按任务双口径，声明主口径）**。

前置条件：只跑 ``trackable = True`` 的任务（7.9 红线），不可跟踪的任务在表里
单独记为 ``skipped_untrackable``，**不进入指标分母**，且必须在报告里写明。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "dwa"
REVIEW_DIR = PROJECT_ROOT / "docs" / "review"
DEFAULT_ALGO = "configs/frozen_A.yaml"
METRICS_CSV = PROJECT_ROOT / "docs" / "dwa_metrics.csv"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from core import astar, dwa  # noqa: E402
from runner import config as config_module  # noqa: E402
from runner import experiment  # noqa: E402

_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
)
COLOR_REF = "#1f77b4"
COLOR_ACTUAL = "#2ca02c"
COLOR_OBSTACLE = "#d62728"
COLOR_LIMIT = "#d62728"


def _setup_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.monospace"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 场景与障碍构造
# ---------------------------------------------------------------------------

OBSTACLE_KINDS: tuple[tuple[str, str], ...] = (
    ("none", "无障碍"),
    ("head_on", "迎面"),
    ("crossing", "横穿"),
    ("same_direction", "同向慢速"),
)


# ---------------------------------------------------------------------------
# 接触帧的二维分解（2026-10-31：成因 × 几何可行性）
# ---------------------------------------------------------------------------

V_STOPPED_MPS = 0.01
"""「本车已停」的速度阈值 [m/s] = 标称巡航 1.0 m/s 的 1%。

把接触帧分成"主动/被动"必须有阈值。取 1% 而不是严格 0：``v = 0`` 只在停车兜底
完全刹停后才出现，而 DWA 候选速度可以低到 ``a_max * dt = 0.08 m/s``，
用严格 0 会把"几乎停住但在蠕动"的帧算成主动碰撞。本模块同时输出 |v| 分箱，
便于检验结论不随阈值翻转（见 ``ContactBreakdown.speed_bin_counts``）。
"""

CONTACT_SPEED_BINS: tuple[tuple[float, float], ...] = (
    (0.0, V_STOPPED_MPS),
    (V_STOPPED_MPS, 0.10),
    (0.10, 0.30),
    (0.30, math.inf),
)
CONTACT_SPEED_BIN_LABELS: tuple[str, ...] = ("<=0.01", "0.01~0.10", "0.10~0.30", ">0.30")

MEETING_WIDTH_FACTOR = 2.0
"""会车净宽系数：``W_required = 2 * (body_radius + r_o)``。

闭式推导：两圆不接触要求中心距 ≥ ``R + r_o``；本车中心离墙至少 ``R``、
障碍中心离墙至少 ``r_o``，两中心可达到的最大横向间距是 ``W - R - r_o``，
令 ``W - R - r_o ≥ R + r_o`` 得 ``W ≥ 2(R + r_o)``。
本平台 = 2 × (0.58 + 0.30) = **1.76 m**。
"""


@dataclass(frozen=True)
class ContactBreakdown:
    """接触帧的**二维分解**：成因（本车在动 / 已停）× 几何可行性（窄 / 宽）。

    判定用的两个量都是**逐周期实测**的（执行口径）：

    * 成因：该周期指令速度 ``|v| > V_STOPPED_MPS`` → 本车仍在推进（主动）；
      否则本车已停（被动）。
    * 几何：任务瓶颈 ``bottleneck_width_m < 2(R + r_o)`` → 该通道**几何上无法会车**，
      接触属结构性不可避；否则规划器本可避免。

    论文口径（三类结论）：

    * ``moving_frames`` = **主动碰撞**，必须为 **0**，否则是规划器 bug；
    * ``stopped_frames`` = **被动接触**（车已停、障碍扫过它）；
    * ``stopped_narrow`` = 被动接触中"本车已停 **且** 瓶颈 < 1.76 m"的部分。
    """

    frames: int
    moving_narrow: int
    moving_wide: int
    stopped_narrow: int
    stopped_wide: int
    speed_bin_counts: tuple[int, int, int, int]

    @property
    def moving_frames(self) -> int:
        """**主动碰撞**帧数（本车仍在推进时发生重叠）。"""
        return self.moving_narrow + self.moving_wide

    @property
    def stopped_frames(self) -> int:
        """**被动接触**帧数（本车已停，障碍仍扫过它的位置）。"""
        return self.stopped_narrow + self.stopped_wide

    @property
    def narrow_frames(self) -> int:
        """落在"几何上无法会车"（W < 1.76 m）任务上的接触帧。"""
        return self.moving_narrow + self.stopped_narrow

    @property
    def wide_frames(self) -> int:
        """落在"规划器本可避免"（W >= 1.76 m）任务上的接触帧。"""
        return self.moving_wide + self.stopped_wide

    @property
    def stopped_narrow_share(self) -> float:
        """「本车已停 ∧ 瓶颈 < 1.76 m」占**全部接触帧**的比例。"""
        return 0.0 if self.frames <= 0 else self.stopped_narrow / self.frames

    @property
    def has_active_collision(self) -> bool:
        return self.moving_frames > 0


def decompose_contact(
    realized_gaps: np.ndarray,
    speeds: np.ndarray,
    *,
    contact_gap_m: float,
    bottleneck_width_m: float,
    meeting_width_required_m: float,
) -> ContactBreakdown:
    """把**接触帧**按"本车是否在动"×"瓶颈是否允许会车"分解（纯函数，可单测）。

    ``realized_gaps`` 与 ``speeds`` 必须**逐周期对齐**（同一时间基）。
    接触判据：``实际圆心距 < contact_gap_m``（两圆重叠）。
    """
    gaps = np.asarray(realized_gaps, dtype=float).reshape(-1)
    velocity = np.abs(np.asarray(speeds, dtype=float).reshape(-1))
    count = min(gaps.shape[0], velocity.shape[0])
    if count == 0:
        return ContactBreakdown(0, 0, 0, 0, 0, (0, 0, 0, 0))
    contact = gaps[:count] < contact_gap_m - 1e-9
    moving = velocity[:count] > V_STOPPED_MPS
    narrow = bool(
        np.isfinite(bottleneck_width_m)
        and bottleneck_width_m < meeting_width_required_m - 1e-9
    )
    moving_contact = int(np.sum(contact & moving))
    stopped_contact = int(np.sum(contact & ~moving))
    # |v| 分箱：第一箱的左端是 0（v 恰为 0 的帧必须落进来，用 v > 0 会漏掉它们）
    speed_bins = tuple(
        int(
            np.sum(
                contact
                & (velocity[:count] > low if low > 0 else np.ones(count, dtype=bool))
                & (velocity[:count] <= high)
            )
        )
        for low, high in CONTACT_SPEED_BINS
    )
    return ContactBreakdown(
        frames=int(np.sum(contact)),
        moving_narrow=moving_contact if narrow else 0,
        moving_wide=0 if narrow else moving_contact,
        stopped_narrow=stopped_contact if narrow else 0,
        stopped_wide=0 if narrow else stopped_contact,
        speed_bin_counts=speed_bins,  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class Episode:
    """一个 (任务 × 障碍类型) 的运行结果。

    字段顺序：**无默认值在前**（dataclass 要求），派生/分级统计在后。
    """

    # ---- 标识与状态 ----
    scenario_id: str
    grade: str
    obstacle_kind: str
    obstacle_label: str
    trackable: bool
    smooth_status: str
    reached_goal: bool
    reason: str
    task_ok: bool
    """任务成功 = **到达终点** 且无物理违反（2026-10-31 口径修正）。

    ``timeout``（周期预算用尽）**不再算成功** —— 成功率 = 到达率。
    因此本字段与 ``reached_goal`` 只在"到达但穿墙"这一种情形下不同。
    """

    # ---- 周期与违反 ----
    periods: int
    violation_periods: int
    violation_rate_per_period: float
    violation_rate_per_task: float
    collision_frames: int
    """**物理碰撞帧数**（``clearance < body_radius``）：车身穿墙的周期数。

    与 ``policy_violation_periods``（策略裕度不足）**不是同一个量**。
    """
    physical_violation_periods: int
    policy_violation_periods: int
    dynamic_violation_periods: int
    fell_back_periods: int
    """停车兜底（v=0）的周期数。"""

    # ---- 离障与间距 ----
    body_radius: float
    """物理外接圆半径 [m]（碰撞判据的分界）。"""
    d_min: float
    min_clearance_m: float
    min_gap_dynamic_m: float
    """最小**中心距** [m]（车体中心到障碍中心）——**预测口径**。

    取自被选中候选的预测轨迹（时域 0.6 s），因此它是"规划器认为自己保持的间距"，
    对停车兜底候选记为 ``inf``（兜底不做预测）。**执行口径**见
    ``min_gap_realized_m``。
    """

    obstacle_radius_m: float
    """动态障碍半径 [m]（无动态障碍时为 0）。"""
    contact_gap_m: float
    """**两圆不接触**的圆心距下限 [m] = ``body_radius + obstacle_radius_m``。

    障碍半径 0.30 m 时为 0.88 m。圆心距小于该值即两圆重叠（物理穿透）。
    """
    min_gap_realized_m: float
    """最小**实际圆心距** [m]——**执行口径**（车体中心 ↔ 障碍中心，逐周期实测）。

    与 ``min_gap_dynamic_m`` 的区别：后者是规划器的预测值，停车兜底周期记 ``inf``；
    本字段直接由执行轨迹与障碍轨迹**逐周期对齐**算出，因此能捕捉"车已停住、
    障碍撞上来"这种规划器预测不到的接触。
    """
    dynamic_contact_periods: int
    """**动态接触周期数**：``实际圆心距 < contact_gap_m`` 的周期数（两圆重叠）。

    与 ``collision_frames``（车身穿墙，静态）**不是同一个量**，两者必须分列。
    """

    bottleneck_width_m: float
    """任务（冻结起终点）的**最宽瓶颈**宽度 [m]（来自场景 ``quality``，与算法无关）。

    见 ``env.scene_quality.grade_scene``：对 clearance 做二分求最大连通阈值 x2。
    """
    meeting_width_required_m: float
    """**会车净宽判据** [m] = ``2 * (body_radius + obstacle_radius_m)``（默认 1.76 m）。

    闭式来源：两车中心距至少要 ``R + r_o``；本车中心离墙至少 ``R``、障碍中心离墙
    至少 ``r_o``，因此通道净宽必须满足 ``W >= (R + r_o) + R + r_o = 2(R + r_o)``。
    ``bottleneck_width_m < 该值`` 的任务在几何上**无法完成会车**，
    此时发生的接触属**结构性不可避**，不是规划器放过的。
    """
    contact: "ContactBreakdown"
    """接触帧的**二维分解**（成因 × 几何可行性），见 :class:`ContactBreakdown`。"""

    # ---- 超出幅度（布尔率会掩盖"差 1 mm"与"差 30 cm"的差别）----
    max_clearance_deficit_m: float
    max_physical_deficit_m: float
    max_gap_deficit_m: float
    max_a_y_deficit: float

    # ---- 任务指标 ----
    completion_time_s: float
    path_length_m: float
    reference_length_m: float
    energy_j: float
    a_y_max: float
    a_y_limit: float
    jerk_rms: float
    v_mean: float
    w_abs_mean: float
    exit_distance_m: float

    # ---- 轨迹数据 ----
    reference: np.ndarray
    actual: np.ndarray
    obstacle_trace: np.ndarray
    times: np.ndarray
    speeds: np.ndarray
    yaw_rates: np.ndarray
    a_y_series: np.ndarray
    config_hash: str


def _build_obstacle(
    kind: str,
    *,
    reference_xy: np.ndarray,
    reference_length: float,
    params: dwa.DwaParams,
) -> dwa.ObstacleTrack | None:
    """按参考轨迹的几何构造确定性手工障碍（迎面 / 横穿 / 同向慢速）。

    位置全部相对参考轨迹算出来，因此对任何场景都成立。
    """
    if kind == "none":
        return None
    start = reference_xy[0]
    goal = reference_xy[-1]
    travel = goal - start
    norm = float(np.linalg.norm(travel))
    if norm < 1e-9:
        return None
    direction = travel / norm

    if kind == "head_on":
        # 从终点一侧迎面驶来，**越过本车后继续驶离**：
        # 障碍必须在任务时间内离开走廊，否则"等障碍通过"会变成永久等待
        # （实测：障碍 30 s 内只前进 15 m，规划器在 t=12 起 v 恒为 0 直到超时）。
        spawn = goal + direction * 1.0
        travel_distance = reference_length + 8.0
        return dwa.make_scenario_obstacle(
            "head_on",
            start_xy=(float(spawn[0]), float(spawn[1])),
            direction=(float(-direction[0]), float(-direction[1])),
            speed=0.6,
            radius_m=0.30,
            duration_s=travel_distance / 0.6 + 1.0,
        )
    if kind == "crossing":
        # 在参考轨迹中点附近横向切入，并**横穿到另一侧之外**。
        # 障碍必须在任务时间内完全离开参考路径周边：它的轨迹终点若落在参考
        # 路径上（实测 duration 太短时停在 (7.99, 4.85) 正好压在参考线上），
        # 就变成一个永久的静止障碍，车永远绕不过去。
        mid = reference_xy[len(reference_xy) // 2]
        lateral = np.array([-direction[1], direction[0]])
        spawn = mid + lateral * 3.0
        return dwa.make_scenario_obstacle(
            "crossing",
            start_xy=(float(spawn[0]), float(spawn[1])),
            direction=(float(-lateral[0]), float(-lateral[1])),
            speed=0.5,
            radius_m=0.30,
            duration_s=14.0,   # 0.5 m/s * 14 s = 7 m，足以横穿 3 m 并离开 4 m
        )
    # same_direction：同向慢速，跑在本车前。
    # 必须让它驶**出参考路径末端之外**，否则会在终点附近停住并永久堵住终点
    # （实测 duration 按 3 倍路径长给值时，障碍正好停在终点前）。
    spawn = start + direction * 2.0
    travel_distance = reference_length + 6.0
    return dwa.make_scenario_obstacle(
        "same_direction",
        start_xy=(float(spawn[0]), float(spawn[1])),
        direction=(float(direction[0]), float(direction[1])),
        speed=0.3,
        radius_m=0.30,
        duration_s=(travel_distance + 2.0) / 0.3 + 1.0,
    )


def run_episode(
    scenario_path: Path,
    obstacle_kind: str,
    *,
    algo: str = DEFAULT_ALGO,
    config_root: Path = CONFIG_ROOT,
) -> Episode:
    resolved = config_module.load_resolved(
        algo, scenario_path, config_root=config_root
    )
    scenario = resolved.scenario
    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    result = experiment.plan_path(scene, setup)
    if not result.success:
        raise RuntimeError(f"{scenario['id']}: A* 无解（{result.reason}）")
    smooth = experiment.smooth_result(result, scene, setup)

    params_dict = config_module.dwa_effective_params(
        resolved.algo, resolved.platform
    )
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

    # 7.9 前置红线：不可跟踪不得进入局部规划
    if not smooth.trackable:
        return Episode(
            scenario_id=str(scenario["id"]),
            grade=str((scenario.get("quality") or {}).get("grade", "-")),
            obstacle_kind=obstacle_kind,
            obstacle_label=dict(OBSTACLE_KINDS).get(obstacle_kind, obstacle_kind),
            trackable=False,
            smooth_status=str(smooth.status),
            reached_goal=False,
            reason="skipped_untrackable",
            # 不可跟踪任务：未执行，任务未完成，且**不计入**任何约束分母
            task_ok=False,
            periods=0,
            violation_periods=0,
            violation_rate_per_period=0.0,
            violation_rate_per_task=0.0,
            collision_frames=0,
            physical_violation_periods=0,
            policy_violation_periods=0,
            dynamic_violation_periods=0,
            fell_back_periods=0,
            body_radius=float(params.body_radius),
            d_min=float(params.d_min),
            min_clearance_m=0.0,
            min_gap_dynamic_m=0.0,
            obstacle_radius_m=0.0,
            contact_gap_m=0.0,
            min_gap_realized_m=float("inf"),
            dynamic_contact_periods=0,
            bottleneck_width_m=float(
                ((scenario.get("quality") or {}).get("bottleneck_width_m", float("nan")))
            ),
            meeting_width_required_m=0.0,
            contact=ContactBreakdown(0, 0, 0, 0, 0, (0, 0, 0, 0)),
            max_clearance_deficit_m=0.0,
            max_physical_deficit_m=0.0,
            max_gap_deficit_m=0.0,
            max_a_y_deficit=0.0,
            completion_time_s=0.0,
            path_length_m=0.0,
            reference_length_m=float(smooth.length_m),
            energy_j=0.0,
            a_y_max=0.0,
            a_y_limit=float(params.a_y_max),
            jerk_rms=0.0,
            v_mean=0.0,
            w_abs_mean=0.0,
            exit_distance_m=0.0,
            reference=np.asarray(smooth.xy, dtype=float),
            actual=np.zeros((0, 2)),
            obstacle_trace=np.zeros((0, 2)),
            times=np.zeros(0),
            speeds=np.zeros(0),
            yaw_rates=np.zeros(0),
            a_y_series=np.zeros(0),
            config_hash=resolved.config_hash,
        )

    reference = dwa.reference_from_smooth(smooth)
    obstacle = _build_obstacle(
        obstacle_kind,
        reference_xy=reference.xy,
        reference_length=reference.length_m,
        params=params,
    )
    obstacles = () if obstacle is None else (obstacle,)
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
        np.array([c.xy for c in commands], dtype=float)
        if commands
        else np.zeros((0, 2))
    )
    speeds = np.array([c.v for c in commands], dtype=float) if commands else np.zeros(0)
    yaw_rates = np.array([c.w for c in commands], dtype=float) if commands else np.zeros(0)
    a_y_series = (
        np.abs(speeds * yaw_rates) if commands else np.zeros(0)
    )

    # 路径长度：按逐周期位移累加
    path_length = (
        float(np.sum(np.linalg.norm(np.diff(actual, axis=0), axis=1)))
        if actual.shape[0] > 1
        else 0.0
    )
    # 能耗：用与全局层一致的稳态巡航口径，按实际几何长度复算
    energy_j = 0.0
    if actual.shape[0] > 1:
        from core import energy as energy_module

        steps = np.linalg.norm(np.diff(actual, axis=0), axis=1)
        midpoints = 0.5 * (actual[:-1] + actual[1:])
        from core.bspline import sample_field

        c_rr_values = sample_field(
            scene.c_rr, midpoints, resolution=scene.resolution, origin=scene.origin
        )
        slope_values = sample_field(
            scene.slope_rad, midpoints, resolution=scene.resolution, origin=scene.origin
        )
        deltas = np.diff(actual, axis=0)
        direction_x = deltas[:, 0] / np.maximum(steps, 1e-12)
        total = 0.0
        for index in range(steps.shape[0]):
            if steps[index] <= 1e-12:
                continue
            # 速度取绝对值（兜底可能倒车，v<0），并**跳过零速度段**：
            # 停车兜底（v=0）时 `step_energy` 会因 `distance>0 且 speed=0` 抛 ValueError。
            # 语义上"速度为 0 却发生位移"不该出现在物理轨迹里；即便出现，
            # 也不应把该段计入能耗（保守处理为 0）。
            speed = abs(
                float(speeds[index + 1])
                if speeds.shape[0] > index + 1
                else params.v_nominal
            )
            if speed <= 1e-12:
                continue
            total += energy_module.step_energy(
                float(steps[index]),
                speed,
                c_rr=float(c_rr_values[index]),
                slope_rad=float(slope_values[index]),
                direction_x=float(direction_x[index]),
                params=setup.energy_params,
            )
        energy_j = float(total)

    jerk_rms = 0.0
    if times.shape[0] >= 3:
        accel = np.gradient(speeds, times, edge_order=1)
        jerk = np.gradient(accel, times, edge_order=1)
        jerk_rms = float(np.sqrt(np.mean(jerk * jerk)))

    obstacle_trace = (
        obstacle.predict(0.0, np.arange(times.shape[0]) * params.dt_s)
        if obstacle is not None
        else np.zeros((0, 2))
    )

    # ---- 圆心距（**执行口径**）与动态接触周期数 ----
    #
    # `actual[i]` 与 `obstacle_trace[i]` 的时间基一致（都是第 i 个周期的 t），
    # 所以可以直接逐周期相减。这一口径补充了规划器预测口径的盲区：
    # 停车兜底周期的 `min_gap_dynamic` 记为 inf，无法反映"车停住后被障碍撞上"。
    obstacle_radius_m = float(obstacle.radius_m) if obstacle is not None else 0.0
    contact_gap_m = float(params.body_radius) + obstacle_radius_m
    # 会车净宽判据（任务级属性，与算法无关）：W >= 2(R + r_o) 才可能完成会车
    meeting_width_required_m = MEETING_WIDTH_FACTOR * contact_gap_m
    quality = scenario.get("quality") or {}
    bottleneck_width_m = float(quality.get("bottleneck_width_m", float("nan")))
    if obstacle is not None and actual.shape[0] > 0 and actual.shape[0] == obstacle_trace.shape[0]:
        realized_gaps = np.linalg.norm(actual - obstacle_trace, axis=1)
        min_gap_realized_m = float(realized_gaps.min())
        dynamic_contact_periods = int(
            np.sum(realized_gaps < contact_gap_m - 1e-9)
        )
        contact_breakdown = decompose_contact(
            realized_gaps,
            speeds,
            contact_gap_m=contact_gap_m,
            bottleneck_width_m=bottleneck_width_m,
            meeting_width_required_m=meeting_width_required_m,
        )
    else:
        min_gap_realized_m = float("inf")
        dynamic_contact_periods = 0
        contact_breakdown = ContactBreakdown(0, 0, 0, 0, 0, (0, 0, 0, 0))

    return Episode(
        scenario_id=str(scenario["id"]),
        grade=str((scenario.get("quality") or {}).get("grade", "-")),
        obstacle_kind=obstacle_kind,
        obstacle_label=dict(OBSTACLE_KINDS).get(obstacle_kind, obstacle_kind),
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
        obstacle_radius_m=obstacle_radius_m,
        contact_gap_m=contact_gap_m,
        min_gap_realized_m=min_gap_realized_m,
        dynamic_contact_periods=dynamic_contact_periods,
        bottleneck_width_m=bottleneck_width_m,
        meeting_width_required_m=meeting_width_required_m,
        contact=contact_breakdown,
        max_clearance_deficit_m=float(outcome.max_clearance_deficit_m),
        max_physical_deficit_m=float(outcome.max_physical_deficit_m),
        max_gap_deficit_m=float(outcome.max_gap_deficit_m),
        max_a_y_deficit=float(outcome.max_a_y_deficit),
        completion_time_s=float(times[-1]) if times.size else 0.0,
        path_length_m=path_length,
        reference_length_m=float(reference.length_m),
        energy_j=energy_j,
        a_y_max=float(a_y_series.max()) if a_y_series.size else 0.0,
        a_y_limit=float(params.a_y_max),
        jerk_rms=jerk_rms,
        v_mean=float(speeds.mean()) if speeds.size else 0.0,
        w_abs_mean=float(np.abs(yaw_rates).mean()) if yaw_rates.size else 0.0,
        exit_distance_m=float(
            np.linalg.norm(actual[-1] - reference.xy[-1])
        ) if actual.shape[0] else 0.0,
        reference=np.asarray(reference.xy, dtype=float),
        actual=actual,
        obstacle_trace=obstacle_trace,
        times=times,
        speeds=speeds,
        yaw_rates=yaw_rates,
        a_y_series=a_y_series,
        config_hash=resolved.config_hash,
    )


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------


def _resolve_world_bounds(
    episode: Episode, occupancy: np.ndarray, resolution: float, origin: tuple[float, float]
) -> tuple[float, float, float, float]:
    rows, cols = occupancy.shape
    return (origin[0], origin[0] + cols * resolution, origin[1], origin[1] + rows * resolution)


def plot_overlay(
    episode: Episode,
    out_path: Path,
    *,
    occupancy: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
    d_min: float,
) -> Path:
    """图 1：参考轨迹 vs 实际执行轨迹 + 动态障碍足迹。"""
    _setup_fonts()
    figure, axis = plt.subplots(figsize=(9.5, 8), constrained_layout=True)
    extent = _resolve_world_bounds(episode, occupancy, resolution, origin)
    axis.imshow(
        occupancy.astype(float),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#ffffff", "#3b3b3b"]),
        interpolation="nearest",
    )
    clearance = astar.clearance_field(occupancy, resolution)
    overlay = np.zeros((*occupancy.shape, 4), dtype=float)
    free = ~occupancy
    overlay[free & (clearance + 1e-9 < d_min)] = (0.95, 0.65, 0.15, 0.32)
    axis.imshow(overlay, origin="lower", extent=extent, interpolation="nearest")

    axis.plot(
        episode.reference[:, 0],
        episode.reference[:, 1],
        "-",
        color=COLOR_REF,
        linewidth=1.6,
        alpha=0.85,
        label=f"参考轨迹（B 样条）{episode.reference_length_m:.2f} m",
    )
    if episode.actual.shape[0] > 1:
        axis.plot(
            episode.actual[:, 0],
            episode.actual[:, 1],
            "-",
            color=COLOR_ACTUAL,
            linewidth=2.2,
            label=f"实际执行 {episode.path_length_m:.2f} m",
        )
    if episode.obstacle_trace.shape[0] > 1:
        axis.plot(
            episode.obstacle_trace[:, 0],
            episode.obstacle_trace[:, 1],
            "--",
            color=COLOR_OBSTACLE,
            linewidth=1.4,
            label=f"动态障碍（{episode.obstacle_label}）",
        )
    axis.plot([], [], "s", color=(0.95, 0.65, 0.15, 0.55), label=f"d_min {d_min:.2f} m")
    axis.plot(*episode.reference[0], "o", color="#ff7f0e", markersize=9, label="起点")
    axis.plot(
        *episode.reference[-1], "*", color="#9467bd", markersize=16, label="终点"
    )

    info = (
        f"档位 = {episode.grade}   障碍 = {episode.obstacle_label}\n"
        f"结果 = {'到达' if episode.reached_goal else '未到达'}（{episode.reason}）\n"
        f"周期 = {episode.periods}   车身穿墙帧 = {episode.collision_frames}   "
        f"动态接触帧 = {episode.contact.frames}\n"
        f"  主动碰撞（本车在动）= {episode.contact.moving_frames}   "
        f"被动接触（车已停）= {episode.contact.stopped_frames}\n"
        f"  瓶颈 W = {episode.bottleneck_width_m:.3f} m / 会车需 "
        f"2(body_radius+r_o) = {episode.meeting_width_required_m:.2f} m\n"
        f"最小离障 = {episode.min_clearance_m:.3f} m\n"
        f"圆心距（实际）= {episode.min_gap_realized_m:.3f} m / 不接触门槛 "
        f"body_radius+r_o = {episode.contact_gap_m:.3f} m\n"
        f"圆心距（预测）= {episode.min_gap_dynamic_m:.3f} m   "
        f"到达时间 = {episode.completion_time_s:.2f} s   能耗 = {episode.energy_j:.1f} J"
    )
    axis.text(
        0.02,
        0.02,
        info,
        transform=axis.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.6"),
    )
    axis.set_title(
        f"{episode.scenario_id}：参考轨迹 vs DWA 执行轨迹（{episode.obstacle_label}）\n"
        f"hash {episode.config_hash[:8]}",
        fontsize=11,
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal")
    axis.legend(loc="upper right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


def plot_time_series(episode: Episode, out_path: Path) -> Path:
    """图 2：速度 / 角速度 / 横向加速度 a_y 时间曲线，含硬约束限值线。"""
    _setup_fonts()
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)

    if episode.times.size == 0:
        for axis in axes:
            axis.text(
                0.5, 0.5, "无可执行轨迹", ha="center", va="center",
                transform=axis.transAxes,
            )
    else:
        axes[0].plot(episode.times, episode.speeds, "-", color=COLOR_ACTUAL, linewidth=1.8)
        axes[0].axhline(
            episode.speeds.max() if episode.speeds.size else 0.0,
            color="0.6", linestyle=":", linewidth=1.0,
        )
        axes[0].set_title(f"线速度 v(t)   均值 {episode.v_mean:.2f} m/s", fontsize=10)
        axes[0].set_ylabel("v [m/s]")

        axes[1].plot(
            episode.times, episode.yaw_rates, "-", color="#9467bd", linewidth=1.6
        )
        axes[1].axhline(0.0, color="0.6", linewidth=0.8)
        axes[1].set_title(
            f"角速度 ω(t)   平均绝对值 {episode.w_abs_mean:.3f} rad/s", fontsize=10
        )
        axes[1].set_ylabel("ω [rad/s]")

        axes[2].plot(
            episode.times, episode.a_y_series, "-", color=COLOR_OBSTACLE, linewidth=1.6,
            label=f"a_y = v·ω   最大 {episode.a_y_max:.3f} m/s²",
        )
        axes[2].axhline(
            episode.a_y_limit, color=COLOR_LIMIT, linestyle="--", linewidth=1.8,
            label=f"硬约束 a_max = {episode.a_y_limit:.2f} m/s²",
        )
        axes[2].fill_between(
            episode.times, 0.0, episode.a_y_series, color=COLOR_OBSTACLE, alpha=0.15
        )
        axes[2].set_title(
            "横向加速度 a_y(t)（**硬约束**：候选直接剔除）", fontsize=10
        )
        axes[2].set_ylabel("a_y [m/s²]")
        axes[2].legend(loc="upper right", fontsize=8)
        for axis in axes:
            axis.grid(alpha=0.25)
        axes[2].set_xlabel("t [s]")

    figure.suptitle(
        f"{episode.scenario_id}: 速度 / 角速度 / 横向加速度（{episode.obstacle_label}）\n"
        f"违反率（按周期）= {episode.violation_rate_per_period:.3f} "
        f"[{episode.violation_periods}/{episode.periods}]  "
        f"（按任务）= {episode.violation_rate_per_task:.1f}\n"
        f"物理碰撞 {episode.collision_frames} 帧  |  "
        f"策略裕度不足 {episode.policy_violation_periods} 周期  |  "
        f"jerk RMS {episode.jerk_rms:.2f} m/s^3",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# 指标表
# ---------------------------------------------------------------------------

METRIC_COLUMNS: tuple[str, ...] = (
    "scenario_id",
    "grade",
    "obstacle_kind",
    "obstacle_label",
    "trackable",
    "smooth_status",
    "task_ok",
    "reached_goal",
    "reason",
    "periods",
    # ---- 两级安全约束（本轮验收阻断 1）----
    "body_radius_m",
    "d_min_m",
    "physical_collision_frames",
    "physical_violation_periods",
    "policy_violation_periods",
    "dynamic_violation_periods",
    "min_clearance_m",
    "min_gap_dynamic_m",
    # ---- 圆心距的执行口径 + 动态接触（2026-10-30 修正）----
    "obstacle_radius_m",
    "contact_gap_m",
    "min_gap_realized_m",
    "dynamic_contact_periods",
    # ---- 接触帧的二维分解（2026-10-31 口径）----
    "bottleneck_width_m",
    "meeting_width_required_m",
    "contact_moving_frames",
    "contact_stopped_frames",
    "contact_moving_narrow",
    "contact_moving_wide",
    "contact_stopped_narrow",
    "contact_stopped_wide",
    # ---- 超出幅度（本轮验收补充 2）----
    "max_clearance_deficit_m",
    "max_physical_deficit_m",
    "max_gap_deficit_m",
    "max_a_y_deficit_mps2",
    # ---- 其余指标 ----
    "completion_time_s",
    "path_length_m",
    "reference_length_m",
    "energy_j",
    "a_y_max_mps2",
    "a_y_limit",
    "jerk_rms_mps3",
    "v_mean_mps",
    "w_abs_mean_radps",
    "exit_distance_m",
    "fell_back_periods",
    "violation_periods",
    "violation_rate_per_period",
    "violation_rate_per_task",
    "config_hash",
)


def _row(episode: Episode) -> dict[str, Any]:
    return {
        "scenario_id": episode.scenario_id,
        "grade": episode.grade,
        "obstacle_kind": episode.obstacle_kind,
        "obstacle_label": episode.obstacle_label,
        "trackable": int(episode.trackable),
        "smooth_status": episode.smooth_status,
        "task_ok": int(episode.task_ok),
        "reached_goal": int(episode.reached_goal),
        "reason": episode.reason,
        "periods": episode.periods,
        "body_radius_m": f"{episode.body_radius:.3f}",
        "d_min_m": f"{episode.d_min:.3f}",
        "physical_collision_frames": episode.collision_frames,
        "physical_violation_periods": episode.physical_violation_periods,
        "policy_violation_periods": episode.policy_violation_periods,
        "dynamic_violation_periods": episode.dynamic_violation_periods,
        "min_clearance_m": f"{episode.min_clearance_m:.4f}",
        "min_gap_dynamic_m": f"{episode.min_gap_dynamic_m:.4f}",
        "obstacle_radius_m": f"{episode.obstacle_radius_m:.3f}",
        "contact_gap_m": f"{episode.contact_gap_m:.3f}",
        "min_gap_realized_m": (
            "inf" if not np.isfinite(episode.min_gap_realized_m)
            else f"{episode.min_gap_realized_m:.4f}"
        ),
        "dynamic_contact_periods": episode.dynamic_contact_periods,
        "bottleneck_width_m": (
            "nan" if not np.isfinite(episode.bottleneck_width_m)
            else f"{episode.bottleneck_width_m:.3f}"
        ),
        "meeting_width_required_m": f"{episode.meeting_width_required_m:.3f}",
        "contact_moving_frames": episode.contact.moving_frames,
        "contact_stopped_frames": episode.contact.stopped_frames,
        "contact_moving_narrow": episode.contact.moving_narrow,
        "contact_moving_wide": episode.contact.moving_wide,
        "contact_stopped_narrow": episode.contact.stopped_narrow,
        "contact_stopped_wide": episode.contact.stopped_wide,
        "max_clearance_deficit_m": f"{episode.max_clearance_deficit_m:.4f}",
        "max_physical_deficit_m": f"{episode.max_physical_deficit_m:.4f}",
        "max_gap_deficit_m": f"{episode.max_gap_deficit_m:.4f}",
        "max_a_y_deficit_mps2": f"{episode.max_a_y_deficit:.4f}",
        "completion_time_s": f"{episode.completion_time_s:.3f}",
        "path_length_m": f"{episode.path_length_m:.4f}",
        "reference_length_m": f"{episode.reference_length_m:.4f}",
        "energy_j": f"{episode.energy_j:.2f}",
        "a_y_max_mps2": f"{episode.a_y_max:.4f}",
        "a_y_limit": f"{episode.a_y_limit:.3f}",
        "jerk_rms_mps3": f"{episode.jerk_rms:.3f}",
        "v_mean_mps": f"{episode.v_mean:.4f}",
        "w_abs_mean_radps": f"{episode.w_abs_mean:.4f}",
        "exit_distance_m": f"{episode.exit_distance_m:.4f}",
        "fell_back_periods": episode.fell_back_periods,
        "violation_periods": episode.violation_periods,
        "violation_rate_per_period": f"{episode.violation_rate_per_period:.4f}",
        "violation_rate_per_task": f"{episode.violation_rate_per_task:.1f}",
        "config_hash": episode.config_hash,
    }


def write_metrics(episodes: Sequence[Episode], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for episode in episodes:
            writer.writerow(_row(episode))
    return path


# ---------------------------------------------------------------------------
# 圆心距最小值分布（2026-10-30 修正的验收口径）
# ---------------------------------------------------------------------------

GAP_DISTRIBUTION_CSV = PROJECT_ROOT / "docs" / "dwa_gap_distribution.csv"
GAP_PERCENTILES: tuple[int, ...] = (0, 10, 25, 50, 75, 100)
TIER_ORDER: tuple[str, ...] = ("easy", "medium", "hard")

GAP_COLUMNS: tuple[str, ...] = (
    "group",
    "obstacle_kind",
    "tasks",
    "task_ok",
    "periods",
    "fallback_periods",
    "physical_collision_frames",
    "dynamic_contact_periods",
    "contact_gap_m",
    "tasks_below_contact_gap",
    *tuple(f"gap_realized_p{p:02d}_m" for p in GAP_PERCENTILES),
    "gap_realized_min_m",
    "gap_pred_p50_m",
)


def gap_distribution_rows(episodes: Sequence[Episode]) -> list[dict[str, Any]]:
    """按档位 / 障碍类型汇总**圆心距最小值分布**（执行口径）。

    只统计 ``trackable`` 且带动态障碍的任务：无障碍任务的圆心距无定义
    （``inf``），不能混进分布里。
    """
    usable = [
        e for e in episodes if e.trackable and e.obstacle_kind != "none"
    ]
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, list[Episode]]] = []
    for grade in TIER_ORDER:
        subset = [e for e in usable if e.grade == grade]
        if subset:
            groups.append((grade, subset))
    groups.append(("all", list(usable)))

    for group_name, subset in groups:
        for kind in ("all", "head_on", "crossing", "same_direction"):
            bucket = (
                subset
                if kind == "all"
                else [e for e in subset if e.obstacle_kind == kind]
            )
            if not bucket:
                continue
            realized = np.array(
                [
                    e.min_gap_realized_m
                    for e in bucket
                    if np.isfinite(e.min_gap_realized_m)
                ],
                dtype=float,
            )
            predicted = np.array(
                [
                    e.min_gap_dynamic_m
                    for e in bucket
                    if np.isfinite(e.min_gap_dynamic_m)
                ],
                dtype=float,
            )
            contact = float(bucket[0].contact_gap_m)
            percentiles = (
                np.percentile(realized, GAP_PERCENTILES)
                if realized.size
                else np.full(len(GAP_PERCENTILES), np.nan)
            )
            row: dict[str, Any] = {
                "group": group_name,
                "obstacle_kind": kind,
                "tasks": len(bucket),
                "task_ok": sum(1 for e in bucket if e.task_ok),
                "periods": sum(e.periods for e in bucket),
                "fallback_periods": sum(e.fell_back_periods for e in bucket),
                "physical_collision_frames": sum(
                    e.collision_frames for e in bucket
                ),
                "dynamic_contact_periods": sum(
                    e.dynamic_contact_periods for e in bucket
                ),
                "contact_gap_m": f"{contact:.3f}",
                "tasks_below_contact_gap": sum(
                    1 for e in bucket if e.min_gap_realized_m < contact - 1e-9
                ),
                "gap_realized_min_m": (
                    f"{realized.min():.4f}" if realized.size else "nan"
                ),
                "gap_pred_p50_m": (
                    f"{float(np.median(predicted)):.4f}" if predicted.size else "nan"
                ),
            }
            for percentile, value in zip(GAP_PERCENTILES, percentiles):
                tinted = "nan" if not np.isfinite(value) else f"{float(value):.4f}"
                row[f"gap_realized_p{percentile:02d}_m"] = tinted
            rows.append(row)
    return rows


def write_gap_distribution(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GAP_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def format_gap_table(rows: Sequence[dict[str, Any]]) -> str:
    """把圆心距分布渲染成等宽表（分档 × 障碍类型）。"""
    header = (
        f"{'档位':<8}{'障碍':<16}{'任务':>4}{'成功':>5}{'兜底周期':>9}"
        f"{'车身穿墙帧':>11}{'动态接触帧':>11}{'低于碰撞门槛任务':>17}"
        f"{'圆心距最小 [m]':>15}{'p10':>8}{'p50':>8}"
    )
    lines = [header, "-" * len(header)]
    label = {
        "all": "全部",
        "head_on": "迎面",
        "crossing": "横穿",
        "same_direction": "同向慢速",
    }
    for row in rows:
        lines.append(
            f"{row['group']:<8}{label.get(str(row['obstacle_kind']), row['obstacle_kind']):<16}"
            f"{row['tasks']:>4}{row['task_ok']:>5}{row['fallback_periods']:>9}"
            f"{row['physical_collision_frames']:>11}{row['dynamic_contact_periods']:>11}"
            f"{row['tasks_below_contact_gap']:>17}"
            f"{row['gap_realized_min_m']:>15}"
            f"{row['gap_realized_p10_m']:>8}{row['gap_realized_p50_m']:>8}"
        )
    return "\n".join(lines)


def plot_gap_distribution(
    episodes: Sequence[Episode], out_path: Path, *, title_suffix: str = ""
) -> Path:
    """图 3：**圆心距最小值分布**（执行口径）按档位 + 障碍类型散点。

    横轴 = 每个任务的最小实际圆心距；竖线 = 两圆不接触门槛
    ``body_radius + r_o``（0.88 m）与修正后的同向圆心距阈值 1.13 m。
    点落在 0.88 m 左侧即**两圆重叠**（动态接触）。
    """
    _setup_fonts()
    usable = [e for e in episodes if e.trackable and e.obstacle_kind != "none"]
    records = [e for e in usable if np.isfinite(e.min_gap_realized_m)]
    figure, axis = plt.subplots(figsize=(10.5, 6.4), constrained_layout=True)
    if not records:
        axis.text(0.5, 0.5, "没有带动态障碍的可跟踪任务", ha="center", va="center",
                  transform=axis.transAxes)
    else:
        grade_order = [g for g in TIER_ORDER if any(e.grade == g for e in records)]
        kind_order = [k for k, _ in OBSTACLE_KINDS if k != "none"]
        colors = {"head_on": "#d62728", "crossing": "#9467bd",
                  "same_direction": "#2ca02c"}
        markers = {"head_on": "o", "crossing": "^", "same_direction": "s"}
        for grade_index, grade in enumerate(grade_order):
            for kind in kind_order:
                values = [
                    e.min_gap_realized_m
                    for e in records
                    if e.grade == grade and e.obstacle_kind == kind
                ]
                if not values:
                    continue
                axis.scatter(
                    values,
                    np.full(len(values), grade_index)
                    + np.linspace(-0.16, 0.16, len(values)),
                    s=48,
                    color=colors[kind],
                    marker=markers[kind],
                    alpha=0.85,
                    label=dict(OBSTACLE_KINDS)[kind],
                    zorder=3,
                )
        contact = float(records[0].contact_gap_m)
        axis.axvline(contact, color="#d62728", linestyle="--", linewidth=1.8,
                     label=f"两圆不接触门槛 body_radius + r_o = {contact:.2f} m")
        axis.axvspan(0.0, contact, color="#d62728", alpha=0.10)
        same_dir_threshold = (
            records[0].body_radius + records[0].obstacle_radius_m + 0.25
        )
        axis.axvline(
            same_dir_threshold,
            color="#ff7f0e",
            linestyle=":",
            linewidth=1.8,
            label=f"同向要求圆心距 = {same_dir_threshold:.2f} m",
        )
        axis.set_yticks(range(len(grade_order)))
        axis.set_yticklabels(grade_order)
        axis.set_xlabel("每个任务的最小实际圆心距 [m]（执行口径，逐周期实测）")
        axis.set_xlim(left=0.0)
        axis.grid(axis="x", alpha=0.25)
        # 去掉重复图例
        handles, labels = axis.get_legend_handles_labels()
        unique: dict[str, Any] = {}
        for handle, name in zip(handles, labels):
            unique.setdefault(name, handle)
        axis.legend(unique.values(), unique.keys(), loc="lower right", fontsize=8.5)
    axis.set_title(
        "DWA 圆心距最小值分布（红线左侧 = 两圆重叠 = 动态接触）" + title_suffix,
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# 接触帧二维分解 + 各档瓶颈宽度 vs 会车净宽（2026-10-31）
# ---------------------------------------------------------------------------

CONTACT_DECOMPOSITION_CSV = PROJECT_ROOT / "docs" / "dwa_contact_decomposition.csv"
BOTTLENECK_COMPARISON_CSV = PROJECT_ROOT / "docs" / "dwa_bottleneck_vs_meeting.csv"

CONTACT_COLUMNS: tuple[str, ...] = (
    "group",
    "obstacle_kind",
    "tasks",
    "executed_tasks",
    "bottleneck_min_m",
    "bottleneck_median_m",
    "meeting_width_required_m",
    "tasks_geometrically_impossible",
    "contact_frames",
    "moving_frames",
    "stopped_frames",
    "narrow_frames",
    "wide_frames",
    "moving_narrow",
    "moving_wide",
    "stopped_narrow",
    "stopped_wide",
    "stopped_narrow_share",
    "tasks_with_active_collision",
    "tasks_with_passive_contact",
    "tasks_with_avoidable_contact",
    "contact_v_le_001",
    "contact_v_001_010",
    "contact_v_010_030",
    "contact_v_gt_030",
)


def _contact_group_rows(
    group_name: str, kind: str, bucket: Sequence[Episode]
) -> dict[str, Any]:
    frames = sum(e.contact.frames for e in bucket)
    moving = sum(e.contact.moving_frames for e in bucket)
    stopped = sum(e.contact.stopped_frames for e in bucket)
    narrow = sum(e.contact.narrow_frames for e in bucket)
    wide = sum(e.contact.wide_frames for e in bucket)
    stopped_narrow = sum(e.contact.stopped_narrow for e in bucket)
    bins = [
        sum(e.contact.speed_bin_counts[i] for e in bucket) for i in range(4)
    ]
    widths = np.array(
        [e.bottleneck_width_m for e in bucket if np.isfinite(e.bottleneck_width_m)],
        dtype=float,
    )
    meeting = (
        float(bucket[0].meeting_width_required_m) if bucket else 0.0
    )
    return {
        "group": group_name,
        "obstacle_kind": kind,
        "tasks": len(bucket),
        "executed_tasks": sum(1 for e in bucket if e.periods > 0),
        "bottleneck_min_m": f"{widths.min():.3f}" if widths.size else "nan",
        "bottleneck_median_m": (
            f"{float(np.median(widths)):.3f}" if widths.size else "nan"
        ),
        "meeting_width_required_m": f"{meeting:.3f}",
        "tasks_geometrically_impossible": int(
            np.sum(widths < meeting - 1e-9)
        ) if widths.size else 0,
        "contact_frames": frames,
        "moving_frames": moving,
        "stopped_frames": stopped,
        "narrow_frames": narrow,
        "wide_frames": wide,
        "moving_narrow": sum(e.contact.moving_narrow for e in bucket),
        "moving_wide": sum(e.contact.moving_wide for e in bucket),
        "stopped_narrow": stopped_narrow,
        "stopped_wide": sum(e.contact.stopped_wide for e in bucket),
        "stopped_narrow_share": f"{(stopped_narrow / frames if frames else 0.0):.4f}",
        "tasks_with_active_collision": sum(
            1 for e in bucket if e.contact.has_active_collision
        ),
        "tasks_with_passive_contact": sum(
            1 for e in bucket if e.contact.stopped_frames > 0
        ),
        "tasks_with_avoidable_contact": sum(
            1 for e in bucket if e.contact.wide_frames > 0
        ),
        "contact_v_le_001": bins[0],
        "contact_v_001_010": bins[1],
        "contact_v_010_030": bins[2],
        "contact_v_gt_030": bins[3],
    }


def contact_decomposition_rows(
    episodes: Sequence[Episode],
) -> list[dict[str, Any]]:
    """按档位 / 障碍类型汇总**接触帧二维分解**（只统计执行过的动态任务）。"""
    ran = [
        e for e in episodes if e.trackable and e.obstacle_kind != "none" and e.periods > 0
    ]
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, list[Episode]]] = []
    for grade in TIER_ORDER:
        subset = [e for e in ran if e.grade == grade]
        if subset:
            groups.append((grade, subset))
    groups.append(("all", list(ran)))
    for group_name, subset in groups:
        for kind in ("all", "head_on", "crossing", "same_direction"):
            bucket = (
                subset if kind == "all" else [e for e in subset if e.obstacle_kind == kind]
            )
            if bucket:
                rows.append(_contact_group_rows(group_name, kind, bucket))
    return rows


def write_contact_decomposition(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTACT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def format_contact_tables(rows: Sequence[dict[str, Any]]) -> str:
    """两张表：① 二维分解矩阵（全档合计）；② 分档 × 障碍类型的明细。"""
    label = {
        "all": "全部",
        "head_on": "迎面",
        "crossing": "横穿",
        "same_direction": "同向慢速",
    }
    total = next((r for r in rows if r["group"] == "all" and r["obstacle_kind"] == "all"), None)
    lines: list[str] = []
    if total is not None:
        meeting = float(total["meeting_width_required_m"])
        lines.append(
            f"接触帧二维分解（全部执行过的动态任务；会车净宽判据 W >= 2(R + r_o) = {meeting:.2f} m）"
        )
        header = (
            f"{'':<16}{'几何不可避 W < %.2f' % meeting:>22}"
            f"{'规划器本可避免 W >= %.2f' % meeting:>24}{'合计':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        lines.append(
            f"{'本车在动（主动）':<16}{total['moving_narrow']:>22}"
            f"{total['moving_wide']:>24}{total['moving_frames']:>10}"
        )
        lines.append(
            f"{'本车已停（被动）':<16}{total['stopped_narrow']:>22}"
            f"{total['stopped_wide']:>24}{total['stopped_frames']:>10}"
        )
        lines.append(
            f"{'合计':<16}{total['narrow_frames']:>22}"
            f"{total['wide_frames']:>24}{total['contact_frames']:>10}"
        )
        lines.append(
            "|v| 分箱（接触帧）："
            + "  ".join(
                f"{name}={value}"
                for name, value in zip(
                    CONTACT_SPEED_BIN_LABELS, (
                        total["contact_v_le_001"],
                        total["contact_v_001_010"],
                        total["contact_v_010_030"],
                        total["contact_v_gt_030"],
                    )
                )
            )
            + f"   （「已停」判据 |v| <= {V_STOPPED_MPS} m/s）"
        )
        lines.append("")
    lines.append("分档 × 障碍类型明细（接触帧前/后各列均为**帧数**，任务列为任务数）")
    header2 = (
        f"{'档位':<8}{'障碍':<12}{'任务':>4}{'W最小':>8}{'W中位':>8}{'不可会车任务':>13}"
        f"{'接触帧':>8}{'主动':>6}{'被动':>6}{'不可避':>8}{'本可避免':>9}"
        f"{'停+窄':>7}{'停+窄占比':>11}"
    )
    lines.append(header2)
    lines.append("-" * len(header2))
    for row in rows:
        lines.append(
            f"{row['group']:<8}{label.get(str(row['obstacle_kind']), row['obstacle_kind']):<12}"
            f"{row['tasks']:>4}{row['bottleneck_min_m']:>8}{row['bottleneck_median_m']:>8}"
            f"{row['tasks_geometrically_impossible']:>13}"
            f"{row['contact_frames']:>8}{row['moving_frames']:>6}{row['stopped_frames']:>6}"
            f"{row['narrow_frames']:>8}{row['wide_frames']:>9}"
            f"{row['stopped_narrow']:>7}{float(row['stopped_narrow_share']):>11.3f}"
        )
    return "\n".join(lines)


def bottleneck_comparison_rows(
    *, config_root: Path = CONFIG_ROOT, meeting_width_m: float = 1.76
) -> list[dict[str, Any]]:
    """各档**任务瓶颈宽度**与会车净宽 1.76 m 的比较（与算法、与是否可跟踪都无关）。"""
    rows: list[dict[str, Any]] = []
    for pattern, forced_tier in (
        ("houseexpo_*.yaml", None),
        ("narrowdoor_*.yaml", "narrowdoor"),
    ):
        for path in sorted((config_root / "scenarios").glob(pattern)):
            doc = config_module.read_config(path)
            quality = doc.get("quality") or {}
            width = quality.get("bottleneck_width_m")
            if width is None:
                continue
            # 窄门组在场景 YAML 里 grade 也是 "hard"，但**不参与三档**
            # （README「场景分档统计」）：必须按文件名强制分到 narrowdoor，
            # 否则 0.75/0.95/1.15 m 会被并进 hard，把 hard 的最窄瓶颈从 1.46 拉到 0.75。
            tier = (
                forced_tier
                if forced_tier is not None
                else str(quality.get("grade", "-"))
            )
            rows.append(
                {
                    "scenario_id": str(doc.get("id", path.stem)),
                    "tier": tier,
                    "bottleneck_width_m": float(width),
                }
            )
    summaries: list[dict[str, Any]] = []
    for tier in (*TIER_ORDER, "narrowdoor"):
        subset = [r for r in rows if r["tier"] == tier]
        if not subset:
            continue
        widths = np.array([r["bottleneck_width_m"] for r in subset], dtype=float)
        summaries.append(
            {
                "tier": tier,
                "tasks": len(subset),
                "bottleneck_min_m": f"{widths.min():.3f}",
                "bottleneck_median_m": f"{float(np.median(widths)):.3f}",
                "bottleneck_max_m": f"{widths.max():.3f}",
                "meeting_width_required_m": f"{meeting_width_m:.3f}",
                "tasks_below_meeting_width": int(np.sum(widths < meeting_width_m - 1e-9)),
                "scenario_ids": " ".join(r["scenario_id"] for r in subset),
            }
        )
    return summaries


def write_bottleneck_comparison(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    columns = (
        "tier",
        "tasks",
        "bottleneck_min_m",
        "bottleneck_median_m",
        "bottleneck_max_m",
        "meeting_width_required_m",
        "tasks_below_meeting_width",
        "scenario_ids",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def format_bottleneck_table(rows: Sequence[dict[str, Any]]) -> str:
    meeting = float(rows[0]["meeting_width_required_m"]) if rows else 1.76
    header = (
        f"{'档位':<12}{'任务':>4}{'W 最小':>9}{'W 中位':>9}{'W 最大':>9}"
        f"{'W < %.2f 的任务' % meeting:>16}{'结论':>28}"
    )
    lines = [f"各档任务瓶颈宽度 vs 会车净宽 2(R + r_o) = {meeting:.2f} m", header, "-" * len(header)]
    for row in rows:
        below = int(row["tasks_below_meeting_width"])
        if below == 0:
            verdict = "本档全部任务可会车"
        elif below == int(row["tasks"]):
            verdict = "本档全部任务**几何上无法会车**"
        else:
            verdict = f"{below}/{row['tasks']} 个任务无法会车"
        lines.append(
            f"{row['tier']:<12}{row['tasks']:>4}{row['bottleneck_min_m']:>9}"
            f"{row['bottleneck_median_m']:>9}{row['bottleneck_max_m']:>9}"
            f"{below:>16}{verdict:>28}"
        )
    return "\n".join(lines)


def plot_contact_decomposition(
    episodes: Sequence[Episode], out_path: Path
) -> Path:
    """图 4：接触帧二维分解（按档位堆叠：主动/被动 × 几何不可避/本可避免）。"""
    _setup_fonts()
    ran = [
        e for e in episodes if e.trackable and e.obstacle_kind != "none" and e.periods > 0
    ]
    grades = [g for g in TIER_ORDER if any(e.grade == g for e in ran)]
    figure, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    colors = ("#d62728", "#ff7f0e", "#1f77b4", "#2ca02c")
    names = (
        "主动碰撞 + 几何不可避",
        "主动碰撞 + 本可避免",
        "被动接触 + 几何不可避",
        "被动接触 + 本可避免",
    )
    if not ran:
        axis.text(0.5, 0.5, "没有执行过的动态任务", ha="center", va="center",
                  transform=axis.transAxes)
    else:
        values = np.array(
            [
                [
                    sum(e.contact.moving_narrow for e in ran if e.grade == g),
                    sum(e.contact.moving_wide for e in ran if e.grade == g),
                    sum(e.contact.stopped_narrow for e in ran if e.grade == g),
                    sum(e.contact.stopped_wide for e in ran if e.grade == g),
                ]
                for g in grades
            ],
            dtype=float,
        )
        positions = np.arange(len(grades))
        bottoms = np.zeros(len(grades))
        for index in range(4):
            axis.barh(
                positions, values[:, index], left=bottoms, color=colors[index],
                label=names[index], height=0.55,
            )
            for row, (value, base) in enumerate(zip(values[:, index], bottoms)):
                if value > 0:
                    axis.text(
                        base + value / 2, positions[row], f"{int(value)}",
                        ha="center", va="center", fontsize=8.5, color="white",
                    )
            bottoms = bottoms + values[:, index]
        axis.set_yticks(positions)
        axis.set_yticklabels(grades)
        meeting = max(e.meeting_width_required_m for e in ran)
        axis.set_title(
            "接触帧二维分解（主动碰撞必须为 0；被动接触 = 车已停、障碍扫过它）\n"
            f"会车净宽判据 W >= 2(R + r_o) = {meeting:.2f} m；"
            f"「本车已停」判据 |v| <= {V_STOPPED_MPS} m/s",
            fontsize=10.5,
        )
        axis.set_xlabel("接触帧数")
        axis.grid(axis="x", alpha=0.25)
        axis.legend(fontsize=8.5, loc="lower right")
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
        prog="python -m analysis.dwa",
        description="DWA 局部规划：轨迹叠图 + 时间曲线 + 指标表",
    )
    parser.add_argument("--tier", default="all")
    parser.add_argument("--scenario", nargs="*", default=None)
    parser.add_argument("--algo", default=DEFAULT_ALGO)
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--metrics", default=str(METRICS_CSV))
    parser.add_argument(
        "--gap-csv",
        default=str(GAP_DISTRIBUTION_CSV),
        help="圆心距最小值分布（按档位 × 障碍类型）汇总表输出路径",
    )
    parser.add_argument(
        "--contact-csv",
        default=str(CONTACT_DECOMPOSITION_CSV),
        help="接触帧二维分解（成因 × 几何可行性）汇总表输出路径",
    )
    parser.add_argument(
        "--bottleneck-csv",
        default=str(BOTTLENECK_COMPARISON_CSV),
        help="各档任务瓶颈宽度 vs 会车净宽 1.76 m 的比较表输出路径",
    )
    parser.add_argument("--kinds", nargs="*", default=None,
                        help="障碍类型子集：none / head_on / crossing / same_direction")
    parser.add_argument("--review", action="store_true",
                        help="把需要人工核验的图复制到 docs/review/")
    parser.add_argument("--review-start", type=int, default=1,
                        help="docs/review/ 的起始编号（避免覆盖已有图）")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-gap-plots", action="store_true")
    parser.add_argument("--jobs", type=int, default=0,
                        help="并行进程数；0 = 自动（min(任务数, CPU)）")
    return parser


def _run_episode_task(
    scenario_path: str, kind: str, algo: str, config_root: str
) -> Episode:
    """进程池入口：必须是**顶层函数**才能被 pickle。"""
    return run_episode(
        Path(scenario_path), kind, algo=algo, config_root=Path(config_root)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_root = Path(args.config_root) if args.config_root else CONFIG_ROOT
    paths = (
        [Path(item) for item in args.scenario]
        if args.scenario
        else list_scenarios(args.tier, config_root=config_root)
    )
    if not paths:
        print(f"没有找到场景（tier={args.tier}）", file=sys.stderr)
        return 2

    kinds = (
        [k for k in OBSTACLE_KINDS if args.kinds is None or k[0] in args.kinds]
        if args.kinds is None
        else [k for k in OBSTACLE_KINDS if k[0] in args.kinds]
    )
    out_dir = Path(args.out_dir)

    # ---- 并行执行（不改算法语义，只改调度）----
    # 每个 (场景, 障碍类型) 是独立 episode，结果与顺序无关。
    # DWA 单场景 ~16 s，串行跑 8 场景 x 4 障碍要 8.5 分钟。
    tasks = [(path, kind) for path in paths for kind, _label in kinds]
    workers = int(args.jobs) if args.jobs and args.jobs > 0 else min(len(tasks), os.cpu_count() or 1)
    episodes: list[Episode] = []
    failures = 0
    results: list[Episode | None] = [None] * len(tasks)

    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_episode_task,
                    str(task_path),
                    kind,
                    str(args.algo),
                    str(config_root),
                ): index
                for index, (task_path, kind) in enumerate(tasks)
            }
            for future in as_completed(futures):
                index = futures[future]
                task_path, kind = tasks[index]
                try:
                    results[index] = future.result()
                except RuntimeError as exc:
                    failures += 1
                    print(f"FAIL {task_path} / {kind}: {exc}", file=sys.stderr)
    else:
        for index, (task_path, kind) in enumerate(tasks):
            try:
                results[index] = run_episode(
                    task_path, kind, algo=args.algo, config_root=args.config_root
                )
            except RuntimeError as exc:
                failures += 1
                print(f"FAIL {task_path} / {kind}: {exc}", file=sys.stderr)

    review_index = args.review_start

    for (path, kind), episode in zip(tasks, results):
        if episode is None:
            continue
        episodes.append(episode)

        if episode.trackable and not args.no_plots:
            resolved = config_module.load_resolved(
                args.algo, path, config_root=config_root
            )
            scene = experiment.build_scene(resolved.scenario)
            tag = "ff55" if "feedforward" in str(args.algo) else "base"
            overlay = out_dir / f"{episode.scenario_id}_{kind}_overlay.png"
            series = out_dir / f"{episode.scenario_id}_{kind}_timeseries.png"
            plot_overlay(
                episode,
                overlay,
                occupancy=scene.occupancy,
                resolution=scene.resolution,
                origin=scene.origin,
                d_min=episode.d_min,
            )
            plot_time_series(episode, series)
            print(f"OK   {overlay}")
            print(f"OK   {series}")
            if args.review:
                REVIEW_DIR.mkdir(parents=True, exist_ok=True)
                # 命名约定：NN_<场景>_<配置>_<障碍类型>_<图类型>.png
                # 必须带障碍类型，否则同一场景的 4 种障碍会互相覆盖
                # （实测 head_on/crossing/same_direction 都写成同名文件）。
                for source, suffix in ((overlay, "overlay"), (series, "timeseries")):
                    target = (
                        REVIEW_DIR
                        / f"{review_index:02d}_{episode.scenario_id}_{tag}"
                        f"_{kind}_{suffix}.png"
                    )
                    shutil.copyfile(source, target)
                    review_index += 1
                    print(f"REVIEW {target}")

        print(
            f"{episode.scenario_id} [{episode.grade}] {kind}: "
            f"trackable={int(episode.trackable)} "
            f"{'reached' if episode.reached_goal else 'NOT reached'}"
            f"({episode.reason}) per={episode.periods} "
            f"viol={episode.violation_periods} "
            f"a_y={episode.a_y_max:.3f}/{episode.a_y_limit:.2f} "
            f"minclr={episode.min_clearance_m:.3f}"
        )

    if episodes:
        print(f"metrics: {write_metrics(episodes, Path(args.metrics))}")
        gap_rows = gap_distribution_rows(episodes)
        contact_rows = contact_decomposition_rows(episodes)
        if gap_rows:
            print(f"圆心距分布: {write_gap_distribution(gap_rows, Path(args.gap_csv))}")
            if not args.no_plots and not args.no_gap_plots:
                gap_figure = out_dir / "gap_distribution.png"
                plot_gap_distribution(episodes, gap_figure)
                print(f"OK   {gap_figure}")
        if contact_rows:
            print(
                f"接触帧二维分解: "
                f"{write_contact_decomposition(contact_rows, Path(args.contact_csv))}"
            )
            if not args.no_plots and not args.no_gap_plots:
                contact_figure = out_dir / "contact_decomposition.png"
                plot_contact_decomposition(episodes, contact_figure)
                print(f"OK   {contact_figure}")
        bottleneck_rows = bottleneck_comparison_rows(config_root=config_root)
        if bottleneck_rows:
            print(
                f"瓶颈 vs 会车净宽: "
                f"{write_bottleneck_comparison(bottleneck_rows, Path(args.bottleneck_csv))}"
            )
        usable = [e for e in episodes if e.trackable]
        skipped = len(episodes) - len(usable)
        print(
            f"\n可用任务 {len(usable)}；因不可跟踪跳过 {skipped}（不计入指标分母）"
        )
        if usable:
            total_periods = sum(e.periods for e in usable)
            total_violation = sum(e.violation_periods for e in usable)
            total_physical = sum(e.physical_violation_periods for e in usable)
            total_policy = sum(e.policy_violation_periods for e in usable)
            total_dynamic = sum(e.dynamic_violation_periods for e in usable)
            reached = sum(1 for e in usable if e.reached_goal)
            task_ok = sum(1 for e in usable if e.task_ok)
            collision_tasks = sum(1 for e in usable if e.collision_frames > 0)
            timeouts = sum(1 for e in usable if e.reason == "timeout")
            dynamic_usable = [e for e in usable if e.obstacle_kind != "none"]
            active_collision_frames = sum(
                e.contact.moving_frames for e in dynamic_usable
            )
            passive_contact_frames = sum(
                e.contact.stopped_frames for e in dynamic_usable
            )
            passive_narrow_frames = sum(
                e.contact.stopped_narrow for e in dynamic_usable
            )
            active_collision_tasks = sum(
                1 for e in dynamic_usable if e.contact.has_active_collision
            )

            # **分母必须写出来**（本轮验收补充 3a）：
            # "违反率 8.1%" 是 144/1774（跨 8 个任务加总，含 4 个零违反任务），
            # 而 "58/247 = 23.5%" 是单个任务的比值 —— 两者都对，但分母不同，
            # 不写清楚就会被读成自相矛盾。
            #
            # **成功判据（2026-10-31）**：success = 到达 ∧ 无物理违反；
            # timeout **不算成功**，所以"任务成功"必须等于"到达"。
            print(
                f"任务成功 = {task_ok}/{len(usable)}"
                f"（= 到达 ∧ 无物理违反，成功率即到达率；"
                f"到达 = {reached}/{len(usable)}，超时 = {timeouts}，"
                f"因穿墙判失败 = {collision_tasks}）"
            )
            # **三类结论**（不再使用"零碰撞"这一说法）：
            #   ① 主动碰撞（本车仍在推进时发生重叠）= 0，不为 0 即 bug；
            #   ② 被动接触（车已停、障碍扫过它）；
            #   ③ 未完成（超时）。
            print(
                "结论三类：\n"
                f"  ① 主动碰撞帧 = {active_collision_frames}"
                f"（涉及任务 {active_collision_tasks}/{len(dynamic_usable)}）"
                "  —— 必须为 0"
                f"\n  ② 被动接触帧 = {passive_contact_frames}"
                f"（其中「本车已停 ∧ 瓶颈 < 1.76 m」= {passive_narrow_frames}，"
                f"占全部接触帧 {(passive_narrow_frames / max(1, active_collision_frames + passive_contact_frames)):.1%}）"
                f"\n  ③ 未完成（超时）= {timeouts} 个任务"
            )
            print(
                f"违反率（按周期，分母 = 全部任务周期数 {total_periods}）"
                f"= {total_violation / max(1, total_periods):.4f}"
                f"  [{total_violation}/{total_periods}]"
            )
            print(
                f"  分解：物理碰撞 {total_physical} + 策略裕度 {total_policy}"
                f" + 动态间距 {total_dynamic}（互斥，三者之和可能小于总违反）"
            )
            print(
                f"违反率（按任务，分母 = 任务数 {len(usable)}）= "
                f"{sum(1 for e in usable if e.violation_rate_per_task > 0)}"
                f"/{len(usable)}"
            )
            print(
                f"物理碰撞帧数合计（车身穿墙）= "
                f"{sum(e.collision_frames for e in usable)}   "
                f"最小离障 = {min(e.min_clearance_m for e in usable):.3f} m"
                f"（车身半径 {usable[0].body_radius:.2f} / 策略裕度 {usable[0].d_min:.2f}）"
            )
            print(
                f"动态接触帧数合计（实际圆心距 < body_radius + r_o）= "
                f"{sum(e.dynamic_contact_periods for e in usable)}   涉及任务 = "
                f"{sum(1 for e in usable if e.dynamic_contact_periods > 0)}"
                f"/{sum(1 for e in usable if e.obstacle_kind != 'none')}"
            )
            print(
                "超出幅度（最大值）："
                f"策略裕度不足 {max(e.max_clearance_deficit_m for e in usable):.4f} m   "
                f"物理穿墙 {max(e.max_physical_deficit_m for e in usable):.4f} m   "
                f"动态间距不足 {max(e.max_gap_deficit_m for e in usable):.4f} m   "
                f"a_y 超出 {max(e.max_a_y_deficit for e in usable):.6f} m/s^2"
            )
            print(
                "圆心距（执行口径）：最小 = "
                f"{min((e.min_gap_realized_m for e in usable if e.obstacle_kind != 'none'), default=float('nan')):.3f} m"
                "   （**圆心距**；判据 = body_radius + r_o + required_gap(δ)，"
                "不接触门槛 = body_radius + r_o = 0.88 m）"
            )
            print(
                "圆心距（预测口径）：最小 = "
                f"{min((e.min_gap_dynamic_m for e in usable if e.obstacle_kind != 'none'), default=float('nan')):.3f} m"
                "   （停车兜底周期记 inf，故可能高于执行口径）"
            )
            if gap_rows:
                print("\n圆心距最小值分布（执行口径，逐周期实测；按档位 × 障碍类型）")
                print(format_gap_table(gap_rows))
            if contact_rows:
                print()
                print(format_contact_tables(contact_rows))
            if bottleneck_rows:
                print()
                print(format_bottleneck_table(bottleneck_rows))
            print(
                f"a_y 最大 = {max(e.a_y_max for e in usable):.3f} / 限值 "
                f"{usable[0].a_y_limit:.2f}   "
                f"jerk RMS 最大 = {max(e.jerk_rms for e in usable):.2f}"
            )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
