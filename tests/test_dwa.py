"""core/dwa.py 的验收测试（交接文档 7.9 第 1~5 条 + 前置红线）。

覆盖：
* 前置红线：``trackable=False`` 必须被拒绝；
* 四个动态窗口（v / ω / a / α）；
* **a_y 是硬约束**（候选直接剔除），jerk 只是代价项；
* 绕行判据是**可行性**（布尔 + 间距阈值），不是加分；
* 三种确定性手工动态障碍：迎面 / 横穿 / 同向慢速；
* 同输入同输出。

全部用合成几何，不依赖 HouseExpo 数据，保证可离线复现。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core import astar, bspline, energy, dwa

PLATFORM = energy.EnergyParams(mass=100.0, eta=0.85, p_aux=15.0)
SPEED = 1.0


# ---------------------------------------------------------------------------
# 合成场景
# ---------------------------------------------------------------------------


def open_field(rows: int = 60, cols: int = 120, res: float = 0.1) -> np.ndarray:
    occupancy = np.ones((rows, cols), dtype=bool)
    occupancy[1:-1, 1:-1] = False
    return astar.clearance_field(occupancy, res)


def straight_track(length: float = 6.0, y: float = 3.0, res: float = 0.1) -> np.ndarray:
    count = max(8, int(round(length / 0.05)) + 1)
    x = np.linspace(0.6, 0.6 + length, count)
    return np.column_stack([x, np.full(count, y)])


def make_params(**overrides: object) -> dwa.DwaParams:
    values: dict[str, object] = {
        "v_max": 1.0,
        "w_max": 2.09,
        "a_max": 0.8,
        "alpha_max": 2.09,
        "horizon_s": 0.6,
        "dt_s": 0.1,
        "v_nominal": 1.0,
        # 合成测试用的小车体：物理半径 0.10、策略裕度 0.30。
        # 两级约束必须保持 body_radius <= d_min（DwaParams 会校验）；
        # 这里都用小值，才能在 0.1 m 分辨率的小场地上跑出可行路径。
        "body_radius": 0.10,
        "d_min": 0.30,
    }
    values.update(overrides)
    return dwa.DwaParams(**values)  # type: ignore[arg-type]


def make_reference(points: np.ndarray, **overrides: object) -> dwa.Reference:
    """用真实平滑链路生成可跟踪参考（保证与生产路径一致）。"""
    field = open_field()
    params = bspline.BSplineParams(
        kappa_max=1.5,
        d_min=0.3,
        e_budget=1e9,
        control_spacing_m=0.8,
    )
    smooth = bspline.smooth_path(
        points,
        params=params,
        clearance_field=field,
        resolution=0.1,
        speed=SPEED,
        energy_params=PLATFORM,
        c_rr=0.015,
        slope_rad=0.0,
        origin=(0.0, 0.0),
    )
    assert smooth.trackable, "测试前提：参考轨迹必须可跟踪"
    return dwa.reference_from_smooth(smooth)


def run_dwa(
    reference: dwa.Reference,
    *,
    params: dwa.DwaParams | None = None,
    obstacles: tuple[dwa.ObstacleTrack, ...] = (),
    clearance_field: np.ndarray | None = None,
) -> dwa.DwaResult:
    planner = dwa.DwaPlanner(
        reference=reference,
        params=params or make_params(),
        clearance_field=clearance_field if clearance_field is not None else open_field(),
        resolution=0.1,
        origin=(0.0, 0.0),
        start_xy=reference.xy[0].copy(),
    )
    return planner.simulate(obstacles)


# ---------------------------------------------------------------------------
# 前置红线：只接受 trackable=True
# ---------------------------------------------------------------------------


class _FakeSmooth:
    """伪造一个不可跟踪的平滑结果（回退折线）。"""

    def __init__(self, xy: np.ndarray, *, trackable: bool) -> None:
        self.xy = xy
        self.yaw = np.zeros(xy.shape[0])
        self.s = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        )
        self.curvature = (
            np.zeros(xy.shape[0])
            if trackable
            else np.concatenate([[0.0], np.full(xy.shape[0] - 2, np.inf), [0.0]])
        )
        self.length_m = float(self.s[-1])
        self.kappa_max = 1.5
        self.trackable = trackable


def test_untrackable_reference_is_rejected() -> None:
    """回退折线（trackable=False）不得进入局部规划——7.9 前置红线。"""
    points = straight_track()
    fake = _FakeSmooth(points, trackable=False)
    with pytest.raises(dwa.NotTrackableError):
        dwa.reference_from_smooth(fake)


def test_untrackable_curvature_is_rejected() -> None:
    """即使对方谎报 trackable=True，曲率无界也必须被拒。"""
    points = straight_track()
    fake = _FakeSmooth(points, trackable=False)
    fake.trackable = True                      # 谎报
    with pytest.raises(dwa.NotTrackableError):
        dwa.reference_from_smooth(fake)


def test_trackable_reference_is_accepted() -> None:
    reference = make_reference(straight_track())
    assert np.all(np.isfinite(reference.curvature))


# ---------------------------------------------------------------------------
# 动态窗口
# ---------------------------------------------------------------------------


def test_window_bounds_expand_within_limits() -> None:
    params = make_params()
    v_lo, v_hi, w_lo, w_hi = dwa._window_bounds((0.0, 0.0), params)
    assert v_lo >= -params.v_max
    assert v_hi <= params.v_max
    assert abs(v_hi - params.a_max * params.dt_s) < 1e-9
    assert abs(w_hi - params.alpha_max * params.dt_s) < 1e-9


def test_window_never_exceeds_platform_limits() -> None:
    params = make_params(v_max=1.0, w_max=2.09)
    v_lo, v_hi, w_lo, w_hi = dwa._window_bounds((1.0, 2.09), params)
    assert v_hi <= 1.0 + 1e-9
    assert w_hi <= 2.09 + 1e-9


def test_window_is_centred_on_previous_command() -> None:
    params = make_params()
    v_lo, v_hi, _, _ = dwa._window_bounds((0.5, 0.0), params)
    assert v_hi > 0.5
    assert v_lo < 0.5


# ---------------------------------------------------------------------------
# a_y 是硬约束；jerk 是代价项
# ---------------------------------------------------------------------------


def test_a_y_violation_rejects_candidate() -> None:
    """a_y = v*ω 超过 a_y_max 时，候选必须被**直接剔除**（不是扣分）。"""
    params = make_params(a_y_max=0.5)
    reference = make_reference(straight_track())
    field = open_field()
    candidate = dwa.evaluate_candidate(
        v=1.0,
        w=1.0,              # a_y = 1.0 > 0.5
        xy0=reference.xy[0],
        yaw0=float(reference.yaw[0]),
        reference=reference,
        params=params,
        clearance_field=field,
        resolution=0.1,
        origin=(0.0, 0.0),
        obstacle_positions=(),
        obstacle_velocities=(),
        obstacle_radii=(),
        previous_commands=(),
        t_now=0.0,
        goal_xy=reference.xy[-1],
    )
    assert candidate.admissable is False
    assert candidate.reject_reason == "a_y"
    assert candidate.cost == math.inf


def test_a_y_within_limit_is_admissible() -> None:
    params = make_params(a_y_max=0.8)
    reference = make_reference(straight_track())
    field = open_field()
    candidate = dwa.evaluate_candidate(
        v=0.5,
        w=0.5,              # a_y = 0.25 < 0.8
        xy0=reference.xy[0],
        yaw0=float(reference.yaw[0]),
        reference=reference,
        params=params,
        clearance_field=field,
        resolution=0.1,
        origin=(0.0, 0.0),
        obstacle_positions=(),
        obstacle_velocities=(),
        obstacle_radii=(),
        previous_commands=(),
        t_now=0.0,
        goal_xy=reference.xy[-1],
    )
    assert candidate.admissable is True
    assert candidate.a_y_max == pytest.approx(0.25)


def test_jerk_is_cost_not_constraint() -> None:
    """jerk 大只提高代价，不改变可行性——这是与 a_y 的本质区别。"""
    params = make_params(a_y_max=0.8)
    reference = make_reference(straight_track())
    field = open_field()

    def candidate_with_history(commands: tuple[dwa.Command, ...]):
        return dwa.evaluate_candidate(
            v=1.0,
            w=0.0,
            xy0=reference.xy[0],
            yaw0=float(reference.yaw[0]),
            reference=reference,
            params=params,
            clearance_field=field,
            resolution=0.1,
            origin=(0.0, 0.0),
            obstacle_positions=(),
            obstacle_velocities=(),
            obstacle_radii=(),
            previous_commands=commands,
            t_now=0.2,
            goal_xy=reference.xy[-1],
        )

    smooth_history = (
        dwa.Command(0.0, reference.xy[0], 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        dwa.Command(0.1, reference.xy[0], 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    jerky_history = (
        dwa.Command(0.0, reference.xy[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        dwa.Command(0.1, reference.xy[0], 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    smooth_candidate = candidate_with_history(smooth_history)
    jerky_candidate = candidate_with_history(jerky_history)

    assert smooth_candidate.admissable and jerky_candidate.admissable
    assert jerky_candidate.jerk_rms > smooth_candidate.jerk_rms
    assert jerky_candidate.cost > smooth_candidate.cost


def test_jerk_needs_two_history_periods() -> None:
    """只有 1 个历史周期时 jerk 记 0——否则会把正常起步加速当成 jerk 惩罚。"""
    params = make_params()
    reference = make_reference(straight_track())
    field = open_field()
    one_history = (
        dwa.Command(0.0, reference.xy[0], 0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    candidate = dwa.evaluate_candidate(
        v=0.16, w=0.0, xy0=reference.xy[0], yaw0=float(reference.yaw[0]),
        reference=reference, params=params, clearance_field=field,
        resolution=0.1, origin=(0.0, 0.0),
        obstacle_positions=(), obstacle_velocities=(), obstacle_radii=(),
        previous_commands=one_history, t_now=0.1, goal_xy=reference.xy[-1],
    )
    assert candidate.jerk_rms == 0.0


# ---------------------------------------------------------------------------
# 绕行判据是可行性，不是权重
# ---------------------------------------------------------------------------


def test_obstacle_and_body_radius_are_part_of_gap_threshold() -> None:
    """圆心距阈值必须是 ``body_radius + r_o + required_gap(δ)``（2026-10-30 修正）。

    旧实现写成 ``required_gap(δ) + r_o``，**漏掉车体半径**，同向档只有
    ``0.25 + 0.30 = 0.55 m``，比两圆不接触所需的 ``body_radius + r_o`` = 0.88 m
    还小 0.33 m —— 等于允许穿透。圆心距小于两半径之和即碰撞。
    """
    params = make_params(gap_overtake_m=0.25, gap_head_on_m=0.60)
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()

    track = dwa.build_obstacle_track(
        start_xy=(reference.xy[0][0] + 0.45, reference.xy[0][1]),
        velocity=(-0.5, 0.0),
        duration_s=3.0,
        radius_m=0.30,
        dt_s=0.1,
    )
    times = np.arange(params.steps + 1) * params.dt_s
    candidate = dwa.evaluate_candidate(
        v=0.5,
        w=0.0,
        xy0=reference.xy[0],
        yaw0=float(reference.yaw[0]),
        reference=reference,
        params=params,
        clearance_field=field,
        resolution=0.1,
        origin=(0.0, 0.0),
        obstacle_positions=(track.predict(0.0, times),),
        obstacle_velocities=(track.velocity_at(0.0),),
        obstacle_radii=(0.30,),
        previous_commands=(),
        t_now=0.0,
        goal_xy=reference.xy[-1],
    )
    assert candidate.admissable is False
    assert candidate.reject_reason == "dynamic_gap"
    # 完整公式：车体半径 + 障碍半径 + 表面间隙要求
    assert candidate.gap_required == pytest.approx(
        params.body_radius + 0.30 + dwa.required_gap(candidate.delta_rad, params)
    )
    assert candidate.gap_required >= params.body_radius + 0.30 - 1e-9
    assert candidate.contact_gap_dynamic == pytest.approx(params.body_radius + 0.30)


def test_center_gap_threshold_never_below_contact() -> None:
    """圆心距阈值**恒 ≥ body_radius + r_o**（任意 δ、任意障碍半径）。

    结构性不变量：表面间隙项 ``required_gap(δ) >= 0``，所以"两圆不接触"这个
    物理下限永远包含在阈值里，不会被任何 δ 或参数组合挖掉。
    """
    params = make_params(
        body_radius=0.58, d_min=0.70, gap_overtake_m=0.25, gap_head_on_m=0.60
    )
    for radius in (0.15, 0.30, 0.50):
        for delta in np.linspace(0.0, math.pi, 25):
            threshold = dwa.required_center_gap(float(delta), params, radius)
            assert threshold >= params.body_radius + radius - 1e-12
    # 显式数值：同向 0.58+0.30+0.25 = 1.13 m；迎面 0.58+0.30+0.60 = 1.48 m
    assert dwa.required_center_gap(math.pi, params, 0.30) == pytest.approx(1.13)
    assert dwa.required_center_gap(0.0, params, 0.30) == pytest.approx(1.48)
    # 旧公式（漏掉车体半径）在同向档只有 0.55 m，**低于不接触下限 0.88 m**
    assert dwa.required_gap(math.pi, params) + 0.30 == pytest.approx(0.55)
    assert dwa.required_gap(math.pi, params) + 0.30 < params.body_radius + 0.30


def test_center_gap_inside_penetration_band_is_rejected() -> None:
    """圆心距落在 ``(r_o, body_radius + r_o)`` 内 = 两圆重叠，必须判不可行。

    这一段正是旧判据**放过**的区间（旧同向阈值 0.55 m、碰撞门槛 0.88 m）：
    实测最小圆心距恰好饱和在 0.550 m，说明规划器贴着这个会穿透的阈值行驶。
    """
    radius = 0.30
    params = make_params(
        body_radius=0.58, d_min=0.70, gap_overtake_m=0.25, gap_head_on_m=0.60
    )
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()
    # 起点取路径中段：真实车体半径 0.58 在起点（距左墙 0.6 m）会先触发**策略离障**
    # 判据，掩盖动态间距判据（reject_reason 会变成 clearance）。
    index = int(np.argmin(np.abs(reference.xy[:, 0] - 3.0)))
    xy0 = reference.xy[index].copy()
    yaw0 = float(reference.yaw[index])
    # 同向慢速（0.30 m/s），车以 0.5 m/s 追近 → δ=π（最宽容档），
    # 圆心距最小值落在 (0.30, 0.88) 区间内
    track = dwa.build_obstacle_track(
        start_xy=(float(xy0[0]) + 0.82, float(xy0[1])),
        velocity=(0.30, 0.0),
        duration_s=4.0,
        radius_m=radius,
        dt_s=0.1,
    )
    times = np.arange(params.steps + 1) * params.dt_s
    candidate = dwa.evaluate_candidate(
        v=0.5,
        w=0.0,
        xy0=xy0,
        yaw0=yaw0,
        reference=reference,
        params=params,
        clearance_field=field,
        resolution=0.1,
        origin=(0.0, 0.0),
        obstacle_positions=(track.predict(0.0, times),),
        obstacle_velocities=(track.velocity_at(0.0),),
        obstacle_radii=(radius,),
        previous_commands=(),
        t_now=0.0,
        goal_xy=reference.xy[-1],
    )
    contact = params.body_radius + radius
    assert radius <= candidate.min_gap_dynamic < contact - 1e-9, (
        f"构造失败：圆心距 {candidate.min_gap_dynamic:.3f} m 必须落在 "
        f"[{radius}, {contact:.2f}) 的穿透区间内"
    )
    # 旧判据（required_gap(δ) + r_o = 0.55 m）会放过这条候选
    assert (
        dwa.required_gap(candidate.delta_rad, params) + radius
        <= candidate.min_gap_dynamic
    )
    # 新判据必须判其不可行，并把碰撞门槛与圆心距阈值都记录成圆心距口径
    assert candidate.admissable is False
    assert candidate.reject_reason == "dynamic_gap"
    assert candidate.contact_gap_dynamic == pytest.approx(contact)
    assert candidate.gap_required == pytest.approx(contact + params.gap_overtake_m)


def test_no_physical_penetration_with_radius() -> None:
    """实际执行中圆心距不得小于 ``body_radius + r_o``（否则两圆重叠）。"""
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()
    radius = 0.30
    params = make_params()
    track = dwa.make_scenario_obstacle(
        "crossing",
        start_xy=(3.0, 6.0),
        direction=(0.0, -1.0),
        speed=0.5,
        radius_m=radius,
        duration_s=14.0,
    )
    result = run_dwa(reference, params=params, obstacles=(track,), clearance_field=field)
    contact = params.body_radius + radius
    assert result.min_gap_dynamic_m >= contact - 1e-6, (
        f"圆心距 {result.min_gap_dynamic_m:.3f} m 小于 body_radius + r_o = "
        f"{contact:.2f} m：两圆重叠（穿透）"
    )


def test_center_gap_never_below_contact_in_execution() -> None:
    """三种手工动态障碍下，执行轨迹的圆心距最小值恒 ≥ ``body_radius + r_o``。

    修正前同向档阈值只有 0.55 m（< 0.88 m），实测 easy / hard 的最小圆心距
    恰好饱和在 **0.550 m** —— 规划器贴着一个会穿透的阈值行驶。
    """
    params = make_params()
    radius = 0.30
    contact = params.body_radius + radius
    cases = {
        "head_on": ((8.0, 3.0), (-1.0, 0.0), 0.5),
        "crossing": ((3.0, 6.0), (0.0, -1.0), 0.4),
        "same_direction": ((2.6, 3.0), (1.0, 0.0), 0.3),
    }
    for kind, (start, direction, speed) in cases.items():
        reference = make_reference(straight_track(length=6.0, y=3.0))
        track = dwa.make_scenario_obstacle(
            kind,
            start_xy=start,
            direction=direction,
            speed=speed,
            radius_m=radius,
        )
        result = run_dwa(reference, params=params, obstacles=(track,))
        assert result.periods > 0, f"{kind}: 没有任何周期"
        assert result.min_gap_dynamic_m >= contact - 1e-6, (
            f"{kind}: 圆心距最小值 {result.min_gap_dynamic_m:.3f} m < "
            f"body_radius + r_o = {contact:.2f} m"
        )


def test_required_gap_interpolates_between_extremes() -> None:
    params = make_params(gap_head_on_m=0.6, gap_overtake_m=0.25)
    assert dwa.required_gap(0.0, params) == pytest.approx(0.6)
    assert dwa.required_gap(math.pi, params) == pytest.approx(0.25)
    middle = dwa.required_gap(math.pi / 2, params)
    assert params.gap_overtake_m < middle < params.gap_head_on_m


def test_obstacle_too_close_makes_candidate_infeasible() -> None:
    """迎面障碍贴得过近时，候选必须是**不可行**，而不是"代价高但还能选"。"""
    params = make_params(gap_head_on_m=0.6, gap_overtake_m=0.25)
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()

    # 迎面障碍：正对本车前方 1.2 m
    track = dwa.build_obstacle_track(
        start_xy=(reference.xy[0][0] + 1.2, reference.xy[0][1]),
        velocity=(-1.0, 0.0),
        duration_s=5.0,
        radius_m=0.3,
        label="迎面",
        dt_s=0.1,
    )
    times = np.arange(params.steps + 1) * params.dt_s
    obstacle_now = (track.predict(0.0, times),)
    velocity = (track.velocity_at(0.0),)

    candidate = dwa.evaluate_candidate(
        v=1.0,
        w=0.0,
        xy0=reference.xy[0],
        yaw0=float(reference.yaw[0]),
        reference=reference,
        params=params,
        clearance_field=field,
        resolution=0.1,
        origin=(0.0, 0.0),
        obstacle_positions=obstacle_now,
        obstacle_velocities=velocity,
        obstacle_radii=(track.radius_m,),
        previous_commands=(),
        t_now=0.0,
        goal_xy=reference.xy[-1],
    )
    assert candidate.admissable is False
    assert candidate.reject_reason == "dynamic_gap"
    # 阈值来自 required_gap，不是任何权重加分的产物
    assert candidate.gap_required > 0


def test_relative_direction_geometry() -> None:
    """迎面 δ≈0（障碍朝我逼近），同向 δ≈π（障碍与我同向）。"""
    # 障碍在 (1,0)，朝 -x 走（向我逼近 → 迎面）
    delta_head_on = dwa.relative_direction(
        np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 0.0])
    )
    # 障碍在 (1,0)，朝 +x 走（远离我 → 同向）
    delta_same = dwa.relative_direction(
        np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 0.0])
    )
    assert delta_head_on == pytest.approx(0.0, abs=1e-6)
    assert delta_same == pytest.approx(math.pi, abs=1e-6)


# ---------------------------------------------------------------------------
# 三种确定性手工动态障碍
# ---------------------------------------------------------------------------


def test_head_on_obstacle_is_avoided_or_stopped() -> None:
    """迎面障碍：要么绕开、要么停住；**绝不能穿过去**。"""
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()
    track = dwa.make_scenario_obstacle(
        "head_on",
        start_xy=(8.0, 3.0),
        direction=(-1.0, 0.0),
        speed=0.5,
        radius_m=0.3,
    )
    result = run_dwa(reference, obstacles=(track,), clearance_field=field)
    assert result.min_gap_dynamic_m >= 0.0
    assert result.periods > 0


def test_crossing_obstacle_is_avoided() -> None:
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()
    track = dwa.make_scenario_obstacle(
        "crossing",
        start_xy=(3.0, 6.0),
        direction=(0.0, -1.0),
        speed=0.4,
        radius_m=0.3,
    )
    result = run_dwa(reference, obstacles=(track,), clearance_field=field)
    assert result.periods > 0
    assert result.constraint_violation_rate_per_task in (0.0, 1.0)


def test_same_direction_slow_obstacle_allows_closer_gap() -> None:
    """同向慢速障碍的**要求间距**更小（相对速度接近 0）。"""
    params = make_params(gap_head_on_m=0.6, gap_overtake_m=0.25)
    assert dwa.required_gap(math.pi, params) < dwa.required_gap(0.0, params)


def test_obstacle_track_out_of_range_extrapolates() -> None:
    track = dwa.build_obstacle_track(
        start_xy=(0.0, 0.0), velocity=(1.0, 0.0), duration_s=1.0, radius_m=0.3
    )
    position = track.position_at(np.array([2.0]))
    assert position[0, 0] == pytest.approx(1.0, abs=1e-9)


# -----------------------------------------------------------------------------
# 约束违反率口径
# ---------------------------------------------------------------------------


def test_violation_rate_reports_both_period_and_task() -> None:
    """必须同时提供按周期与按任务两种口径，论文里要声明用哪个。"""
    reference = make_reference(straight_track())
    result = run_dwa(reference)
    assert 0.0 <= result.constraint_violation_rate_per_period <= 1.0
    assert result.constraint_violation_rate_per_task in (0.0, 1.0)
    payload = result.as_dict()
    assert "violation_rate_per_period" in payload
    assert "violation_rate_per_task" in payload


# ---------------------------------------------------------------------------
# 确定性与基本行为
# ---------------------------------------------------------------------------


def test_same_input_same_output() -> None:
    reference = make_reference(straight_track())
    first = run_dwa(reference)
    second = run_dwa(reference)
    assert first.periods == second.periods
    assert first.reason == second.reason
    assert np.array_equal([c.v for c in first.commands], [c.v for c in second.commands])
    assert first.min_clearance_m == second.min_clearance_m


def test_straight_reference_is_completed() -> None:
    reference = make_reference(straight_track(length=6.0))
    result = run_dwa(reference)
    assert result.reached_goal is True
    assert result.reason == "reached_goal"
    assert result.violation_periods == 0


def test_clearance_constraint_respected_without_obstacles() -> None:
    reference = make_reference(straight_track(length=6.0))
    result = run_dwa(reference)
    assert result.min_clearance_m >= 0.3 - 1e-9


def test_inline_sampler_matches_bspline() -> None:
    """DWA 热路径的内联双线性采样必须与 ``core.bspline.sample_field`` 完全等价。

    内联版是为了性能（单场景 37800 次调用，占 4 s）。一旦两者语义漂移，
    离障判据就不再与 B 样条阶段同一个口径，整个约束链条失去可比性。
    这里用随机点覆盖**网格内外**、**边界**与**非整数格**三类情形。
    """
    from core.bspline import sample_field

    field = open_field(rows=40, cols=60, res=0.1)
    rng = np.random.default_rng(20261030)
    # 覆盖有理范围 + 网格外 + 边界
    points = np.vstack(
        [
            rng.uniform(-0.5, 6.5, size=(60, 2)),      # 含网格外
            np.array([[0.0, 0.0], [5.9, 3.9], [0.05, 0.05]]),  # 边界附近
            np.array([[0.123, 0.777], [1.234, 2.345], [4.56, 1.23]]),
        ]
    )
    expected = sample_field(field, points, resolution=0.1, origin=(0.0, 0.0))
    actual = dwa._sample_clearance(
        field, points, resolution=0.1, origin=(0.0, 0.0), ox=0.0, oy=0.0
    )
    assert np.allclose(actual, expected, atol=1e-12), (
        "内联采样器与 bspline.sample_field 不一致"
    )


def test_windowed_projection_matches_full_projection() -> None:
    """窗口化投影在窗口内必须与全量投影等价（性能优化不能改变语义）。"""
    reference = make_reference(straight_track(length=6.0, y=3.0))
    rng = np.random.default_rng(20261030)
    probes = rng.uniform(0.5, 6.5, size=(40, 2))
    probes[:, 1] = rng.uniform(2.4, 3.6, size=40)

    full_point, full_arc, full_lat = reference.project(probes)
    for arc_center in (0.5, 2.0, 4.0, 6.0):
        win_point, win_arc, win_lat = reference.project_windowed(
            probes, arc_center, 1.5
        )
        # 窗口内的探针必须与全量结果一致
        near = np.abs(full_arc - arc_center) <= 1.4
        if not np.any(near):
            continue
        assert np.allclose(win_arc[near], full_arc[near], atol=1e-9)
        assert np.allclose(win_lat[near], full_lat[near], atol=1e-9)


def test_reverse_is_never_cheaper_than_advancing() -> None:
    """在还能前进时，倒车**不得**比前进便宜。

    实测过的坑：倒车惩罚按 |v| 计时，低速倒车只罚 0.048，于是规划器在终点前
    0.379 m 处反复微倒车，耗尽 247 个周期而超时（houseexpo_07432f2d）。
    这条断言直接用代价大小关系表达该要求，不依赖内部分量拆解
    （拆解会被 ``remaining_overrun`` 等其他项串扰，实测无法稳定断言）。
    """
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()
    params = make_params()
    # 取参考路径中段，两侧都有余量
    mid = len(reference.xy) // 2
    xy0 = reference.xy[mid]
    yaw0 = float(reference.yaw[mid])

    common = dict(
        reference=reference, params=params, clearance_field=field,
        resolution=0.1, origin=(0.0, 0.0),
        obstacle_positions=(), obstacle_velocities=(), obstacle_radii=(),
        previous_commands=(), t_now=1.0, goal_xy=reference.xy[-1],
    )
    forward = dwa.evaluate_candidate(v=0.08, w=0.0, xy0=xy0, yaw0=yaw0, **common)
    backward = dwa.evaluate_candidate(v=-0.08, w=0.0, xy0=xy0, yaw0=yaw0, **common)
    assert forward.admissable and backward.admissable
    assert forward.cost < backward.cost, (
        f"前进 {forward.cost:.4f} 竟然比倒车 {backward.cost:.4f} 更贵："
        "会让规划器宁可在终点前反复微倒车"
    )


def test_stall_penalty_forbids_free_stop() -> None:
    """还有余量前进时，停车不应比前进便宜——否则车会停在终点前。

    实测 houseexpo_07432f2d：速度窗口上界仅 0.08 m/s，`|v-1|` 在 0 与 0.08
    之间只差 0.024，横向误差的微小差异就能让"停车"胜出，车停在终点前 0.379 m。
    """
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()
    params = make_params()
    # 当前位置：弧长上尚有一定距离
    xy0 = reference.xy[len(reference.xy) // 3]
    yaw0 = float(reference.yaw[len(reference.xy) // 3])
    stopped = dwa.evaluate_candidate(
        v=0.0, w=0.0, xy0=xy0, yaw0=yaw0, reference=reference, params=params,
        clearance_field=field, resolution=0.1, origin=(0.0, 0.0),
        obstacle_positions=(), obstacle_velocities=(), obstacle_radii=(),
        previous_commands=(), t_now=1.0, goal_xy=reference.xy[-1],
    )
    moving = dwa.evaluate_candidate(
        v=0.08, w=0.0, xy0=xy0, yaw0=yaw0, reference=reference, params=params,
        clearance_field=field, resolution=0.1, origin=(0.0, 0.0),
        obstacle_positions=(), obstacle_velocities=(), obstacle_radii=(),
        previous_commands=(), t_now=1.0, goal_xy=reference.xy[-1],
    )
    assert moving.admissable and stopped.admissable
    assert moving.cost < stopped.cost, (
        f"停车代价 {stopped.cost:.4f} 低于前进 {moving.cost:.4f}："
        "会停在终点前不再移动"
    )


def test_spin_gate_is_window_gated() -> None:
    """原地自转抑制只在速度窗口够得着门槛时才生效。

    若无条件生效：上一周期 ω 打满后 ω 窗口锁在 [1.881,2.09]，而 v 窗口只有
    [0,0.08]，全部候选同时违反 -> 车彻底锁死（实测 30a18345 卡 191 周期）。
    """
    params = make_params()
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()
    xy0 = reference.xy[0]
    yaw0 = float(reference.yaw[0])

    common = dict(
        reference=reference, params=params, clearance_field=field,
        resolution=0.1, origin=(0.0, 0.0),
        obstacle_positions=(), obstacle_velocities=(), obstacle_radii=(),
        previous_commands=(), t_now=0.0, goal_xy=reference.xy[-1],
    )
    # 高 ω + 低速；窗口够得着门槛 -> 判不可行
    gated = dwa.evaluate_candidate(
        v=0.0, w=2.0, xy0=xy0, yaw0=yaw0, spin_floor_active=True, **common
    )
    assert gated.admissable is False
    assert gated.reject_reason == "spin"
    # 同样动作，但窗口够不着门槛 -> 允许（否则锁死）
    ungated = dwa.evaluate_candidate(
        v=0.0, w=2.0, xy0=xy0, yaw0=yaw0, spin_floor_active=False, **common
    )
    assert ungated.admissable is True


def test_two_tier_clearance_constraints() -> None:
    """物理约束（body_radius）比策略约束（d_min）严格一级，且不可混同。

    阻断 1：``clearance >= body_radius`` 违反即碰撞、任务失败；
    ``clearance >= d_min`` 违反只是降级，必须记录幅度。
    """
    params = make_params(body_radius=0.10, d_min=0.30)
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()

    # 构造一条候选：轨迹终点在 d_min 与 body_radius 之间（策略降级但不碰撞）
    candidate = dwa.evaluate_candidate(
        v=0.08, w=0.0,
        xy0=reference.xy[0], yaw0=float(reference.yaw[0]),
        reference=reference, params=params, clearance_field=field,
        resolution=0.1, origin=(0.0, 0.0),
        obstacle_positions=(), obstacle_velocities=(), obstacle_radii=(),
        previous_commands=(), t_now=0.0, goal_xy=reference.xy[-1],
    )
    # 三个幅度字段必须独立存在且语义正确
    assert candidate.clearance_deficit >= 0.0
    assert candidate.physical_deficit >= 0.0
    assert candidate.a_y_deficit >= 0.0
    # 物理不足量永远不超过策略不足量（body_radius <= d_min）
    assert candidate.physical_deficit <= candidate.clearance_deficit + 1e-9


def test_physical_violation_fails_task() -> None:
    """物理碰撞必须让任务失败，不能记为"到达"。

    阻断 1：hard 档曾出现 min_clearance=0.371 < body_radius=0.58（车身穿墙）
    却仍计入"到达 8/8"。
    """
    params = make_params(body_radius=0.10, d_min=0.30)
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()
    result = run_dwa(reference, params=params)
    if result.collision_frames > 0:
        assert result.success is False, "发生物理碰撞后 success 必须为 False"
        assert result.reached_goal is False, "发生物理碰撞后不得记为到达"
        assert result.task_failed_by_collision is True


def test_collision_frames_only_count_physical() -> None:
    """碰撞帧数只统计物理违反，不把策略裕度不足算作碰撞。"""
    params = make_params(body_radius=0.10, d_min=0.30)
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()
    result = run_dwa(reference, params=params)
    assert result.collision_frames == result.physical_violation_periods
    # 策略违反与物理违反互斥
    assert result.policy_violation_periods <= result.violation_periods


def test_timeout_is_not_success() -> None:
    """**超时不算成功**（2026-10-31 口径修正）：success = 到达 ∧ 无物理违反。

    旧口径 ``reason in {reached_goal, timeout}`` 把"走不动"也算成功，
    于是 hard 档打印 "task_ok = 8/8" 掩盖了"1 个任务超时未完成"。
    成功率必须等于到达率。
    """
    reference = make_reference(straight_track(length=6.0, y=3.0))
    planner = dwa.DwaPlanner(
        reference=reference,
        params=make_params(),
        clearance_field=open_field(),
        resolution=0.1,
        origin=(0.0, 0.0),
        start_xy=reference.xy[0].copy(),
    )
    # 只给 1 个周期：必定走不完
    result = planner.simulate(obstacles=(), max_periods=1)
    assert result.reason == "timeout"
    assert result.reached_goal is False
    assert result.success is False, "超时必须判为不成功"
    assert result.periods > 0, "本用例要的是「走了但没走完」，不是「没有指令」"


def test_success_requires_reached_goal() -> None:
    """到达且无物理违反 -> success/ reached_goal 一致。"""
    reference = make_reference(straight_track(length=6.0))
    result = run_dwa(reference)
    assert result.reason == "reached_goal"
    assert result.reached_goal is True
    assert result.success is True


def test_fallback_stops_without_braking_violation() -> None:
    """兜底停车不得造出超过 a_max 的减速度。

    实测坑：直接置 v=0 会让 0.961 -> 0 一步完成（9.6 m/s²），
    远超 a_max=0.8，与"违反率 0"自相矛盾。
    """
    params = make_params(body_radius=0.10, d_min=0.30)
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()
    result = run_dwa(reference, params=params)
    assert result.max_accel_deficit <= 1e-6, (
        f"出现 {result.max_accel_deficit:.3f} m/s² 的加速度超限：兜底停车未受限"
    )


def test_a_y_boundary_equality_is_admissible() -> None:
    """a_y 恰好等于上限必须判**合规**（浮点等式不能被算成违反）。

    补充 2：easy 按任务违反率 50% 而 a_y 最大值恰为 0.800，
    疑似边界等式被计为违反。
    """
    params = make_params(a_y_max=0.8)
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()
    # v=1.0, w=0.8 -> a_y 恰为 0.800
    candidate = dwa.evaluate_candidate(
        v=1.0, w=0.8,
        xy0=reference.xy[0], yaw0=float(reference.yaw[0]),
        reference=reference, params=params, clearance_field=field,
        resolution=0.1, origin=(0.0, 0.0),
        obstacle_positions=(), obstacle_velocities=(), obstacle_radii=(),
        previous_commands=(), t_now=0.0, goal_xy=reference.xy[-1],
        spin_floor_active=False,
    )
    assert candidate.a_y_max == pytest.approx(0.8)
    assert candidate.a_y_deficit == pytest.approx(0.0)
    assert candidate.reject_reason != "a_y", "边界等式被误判为违反"


def test_deficits_are_magnitudes_not_booleans() -> None:
    """超出幅度必须是数值（非布尔），布尔率会掩盖 1 mm 与 30 cm 的差别。"""
    params = make_params(body_radius=0.10, d_min=0.30)
    reference = make_reference(straight_track(length=6.0, y=3.0))
    field = open_field()
    result = run_dwa(reference, params=params)
    for name in (
        "max_clearance_deficit_m",
        "max_physical_deficit_m",
        "max_a_y_deficit",
        "max_gap_deficit_m",
    ):
        value = getattr(result, name)
        assert isinstance(value, float), f"{name} 必须是 float 而非 bool"


def test_params_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        make_params(v_max=-1.0)
    with pytest.raises(ValueError):
        make_params(a_y_max=0.0)
    with pytest.raises(ValueError):
        make_params(v_nominal=1.5, v_max=1.0)
    with pytest.raises(ValueError):
        make_params(dt_s=0.7, horizon_s=0.6)
