"""core/bspline.py 的验收测试（交接文档 7.7 第 8 条）。

覆盖：端点重合、C² 连续（节点处位置/切向/曲率）、``|kappa| <= kappa_max``、
``clearance >= d_min``、``E <= e_budget``、同输入同输出、回退分支可触发。

几何用例都是**合成**的（不依赖 HouseExpo 数据），这样测试可离线复现；
真实场景上的行为由 ``analysis/smooth.py`` 的回退率统计报告。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from core import astar, bspline, energy

PLATFORM = energy.EnergyParams(mass=100.0, eta=0.85, p_aux=15.0)
SPEED = 1.0


# ---------------------------------------------------------------------------
# 合成场景工具
# ---------------------------------------------------------------------------


def open_field(rows: int = 60, cols: int = 60, res: float = 0.1) -> np.ndarray:
    """四周有墙、内部全自由的地图（墙在边界上，保证有限离障）。"""
    occupancy = np.zeros((rows, cols), dtype=bool)
    occupancy[0, :] = True
    occupancy[-1, :] = True
    occupancy[:, 0] = True
    occupancy[:, -1] = True
    return astar.clearance_field(occupancy, res)


def straight_line(length: float = 4.0, step: float = 0.1) -> np.ndarray:
    """一条水平栅格折线（含栅格阶梯：对角/直步交替，模拟真实 A* 输出）。"""
    count = int(round(length / step)) + 1
    x = np.linspace(0.2, 0.2 + (count - 1) * step, count)
    y = np.full(count, 3.0)
    return np.column_stack([x, y])


def right_angle_turn(arm: float = 2.0, step: float = 0.1) -> np.ndarray:
    """先沿 +x 再沿 +y 的 90° 折线，带栅格阶梯。"""
    count = int(round(arm / step)) + 1
    first = np.column_stack(
        [np.linspace(0.5, 0.5 + (count - 1) * step, count), np.full(count, 0.5)]
    )
    second = np.column_stack(
        [np.full(count, first[-1, 0]), np.linspace(0.5, 0.5 + (count - 1) * step, count)]
    )
    return np.vstack([first, second[1:]])


def corridor(field: np.ndarray, res: float = 0.1) -> tuple[np.ndarray, float]:
    return field, res


def default_params(**overrides: object) -> bspline.BSplineParams:
    values: dict[str, object] = {
        "kappa_max": 1.5,
        "d_min": 0.1,
        "e_budget": 100_000.0,
    }
    values.update(overrides)
    return bspline.BSplineParams(**values)  # type: ignore[arg-type]


def smooth(points: np.ndarray, field: np.ndarray, **overrides: object) -> bspline.SmoothPath:
    return bspline.smooth_path(
        points,
        params=default_params(**overrides),
        clearance_field=field,
        resolution=0.1,
        speed=SPEED,
        energy_params=PLATFORM,
        c_rr=0.015,
        slope_rad=0.0,
        origin=(0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# 端点重合（clamped）
# ---------------------------------------------------------------------------


def test_endpoints_coincide_exactly() -> None:
    """曲线必须精确经过起点与终点——不做 clamped 就是经典 bug。"""
    field = open_field()
    points = right_angle_turn()
    result = smooth(points, field)
    assert result.success
    assert result.status == "spline"
    assert np.allclose(result.xy[0], points[0], atol=1e-12)
    assert np.allclose(result.xy[-1], points[-1], atol=1e-12)


def test_clamped_knots_repeat_endpoints() -> None:
    knots = bspline._uniform_clamped_knots(6)
    assert np.allclose(knots[:4], 0.0)
    assert np.allclose(knots[-4:], 1.0)
    assert np.all(np.diff(knots) >= -1e-12)


# ---------------------------------------------------------------------------
# C² 连续：节点处位置 / 切向 / 曲率
# ---------------------------------------------------------------------------


def test_curve_is_c2_at_knots() -> None:
    """内部节点处位置、一阶导、二阶导左右极限一致（C²）。"""
    control = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]], dtype=float
    )
    curve = bspline.build_curve(control)
    interior = curve.knots[4:-4]
    assert interior.size > 0, "至少要有一个内部节点"
    for knot in interior:
        delta = 1e-7
        left = curve.at(float(knot) - delta)
        right = curve.at(float(knot) + delta)
        here = curve.at(float(knot))
        assert np.linalg.norm(left - here) < 1e-5
        assert np.linalg.norm(right - here) < 1e-5

    # 用解析导数在节点处比较左右极限
    for knot in interior:
        left = _derivatives(curve, float(knot) - 1e-6)
        right = _derivatives(curve, float(knot) + 1e-6)
        assert np.allclose(left[0], right[0], atol=1e-4), "一阶导不连续"
        assert np.allclose(left[1], right[1], atol=1e-3), "二阶导不连续"


def _derivatives(curve: bspline.SplineCurve, u: float) -> tuple[np.ndarray, np.ndarray]:
    value, first, second = bspline._basis_matrix(
        np.array([u]), curve.control_points.shape[0], curve.knots
    )
    d1 = (first @ curve.control_points)[0]
    d2 = (second @ curve.control_points)[0]
    return d1, d2


# ---------------------------------------------------------------------------
# 曲率约束
# ---------------------------------------------------------------------------


def test_curvature_within_limit_on_turn() -> None:
    """90° 转角经过平滑后，全程 |kappa| 不得超过与 A* 共用的 kappa_max。"""
    field = open_field()
    result = smooth(right_angle_turn(), field, kappa_max=1.5)
    assert result.status == "spline"
    assert result.max_abs_curvature <= 1.5 + 1e-9
    assert np.all(np.abs(result.curvature) <= 1.5 + 1e-9)


def test_tight_curvature_limit_falls_back_or_inserts() -> None:
    """把 kappa_max 压到极小时，要么插入控制点后满足，要么明确回退——绝不偷偷放行。"""
    field = open_field()
    result = smooth(right_angle_turn(), field, kappa_max=0.05)
    if result.status == "spline":
        assert result.max_abs_curvature <= 0.05 + 1e-9
    else:
        assert result.fell_back
        assert math.isinf(result.max_abs_curvature), "回退折线的曲率应如实记为无界"


def test_kappa_max_is_shared_not_duplicated() -> None:
    """平滑参数里的 kappa_max 必须与 A* 的 AStarParams 取同一个值。"""
    algo = {"id": "t", "astar": {"e_budget": 2000.0, "d_min": 0.7, "kappa_max": 1.5}}
    platform = {"mass": 100.0, "g": 9.81, "body_radius": 0.58, "v_max": 1.0,
                "w_max": 2.09, "a_max": 0.8, "alpha_max": 2.09, "eta": 0.85,
                "p_aux": 15.0, "battery_voltage": 24.0, "battery_resistance": 0.08}
    from runner import config as config_module

    astar_params = config_module.astar_effective_params(algo, platform)
    smooth_params = config_module.bspline_effective_params(algo, platform)
    assert smooth_params["kappa_max"] == astar_params["kappa_max"]
    assert smooth_params["d_min"] == astar_params["d_min"]
    assert smooth_params["e_budget"] == astar_params["e_budget"]


def test_bspline_layer_cannot_redeclare_shared_constraints() -> None:
    """algo.bspline 里重复声明 kappa_max / d_min / e_budget 必须被拒绝。"""
    from runner import config as config_module

    doc = {
        "id": "bad",
        "astar": {"e_budget": 2000.0, "d_min": 0.7, "kappa_max": 1.5},
        "bspline": {"kappa_max": 3.0},
    }
    with pytest.raises(config_module.ConfigError, match="共用|不得重复声明"):
        config_module.validate_algo(doc)


def test_planner_setup_shares_constraints_with_astar() -> None:
    """端到端装配后，平滑参数里的三个约束必须与 A* 逐字段相等（无前馈时）。"""
    from runner import config as config_module
    from runner import experiment

    config_root = Path(__file__).resolve().parents[1] / "config"
    resolved = config_module.load_resolved(
        "configs/frozen_A.yaml",
        "scenarios/houseexpo_8bb3c35d.yaml",
        config_root=config_root,
    )
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    assert setup.bspline_params.kappa_max == setup.astar_params.kappa_max
    assert setup.bspline_params.d_min == setup.astar_params.d_min
    assert setup.bspline_params.e_budget == setup.astar_params.e_budget
    # 无前馈时裕度为 0，A* 与平滑的 d_min 完全一致。
    assert setup.clearance_margin_m == 0.0


# ---------------------------------------------------------------------------
# 曲率裕度前馈
# ---------------------------------------------------------------------------


def test_clearance_margin_only_affects_global_layer() -> None:
    """前馈只抬高 A* 的离障门槛，**平滑侧仍守 d_min**。

    若平滑侧也跟着抬高，前馈就从"留出切角空间"变成了"偷偷抬高安全约束"，
    那会污染"平滑后 clearance >= d_min"这一口径。
    """
    from runner import config as config_module
    from runner import experiment

    config_root = Path(__file__).resolve().parents[1] / "config"
    scenario = "scenarios/houseexpo_8bb3c35d.yaml"
    baseline = config_module.load_resolved(
        "configs/frozen_A.yaml", scenario, config_root=config_root
    )
    feedforward = config_module.load_resolved(
        "configs/frozen_A_feedforward.yaml", scenario, config_root=config_root
    )
    base_setup = experiment.planner_setup(baseline.platform, baseline.algo)
    feed_setup = experiment.planner_setup(feedforward.platform, feedforward.algo)

    margin = 0.055
    assert feed_setup.clearance_margin_m == pytest.approx(margin)
    assert feed_setup.d_min_policy == pytest.approx(base_setup.d_min_policy)
    # A* 门槛抬高，平滑门槛不动
    assert feed_setup.astar_params.d_min == pytest.approx(
        base_setup.astar_params.d_min + margin
    )
    assert feed_setup.bspline_params.d_min == pytest.approx(
        base_setup.bspline_params.d_min
    )


def test_feedforward_margin_is_non_negative() -> None:
    """clearance_margin_m 不能为负（会反向侵蚀安全裕度）。"""
    from runner import config as config_module

    doc = {
        "id": "bad",
        "astar": {
            "e_budget": 2000.0,
            "d_min": 0.7,
            "kappa_max": 1.5,
            "clearance_margin_m": -0.03,
        },
    }
    with pytest.raises(config_module.ConfigError, match="clearance_margin_m"):
        config_module.validate_algo(doc)


def test_violation_gate_uses_policy_d_min_not_astar_gate() -> None:
    """轨迹违反判据必须用策略 d_min，不能用 A* 的前馈门槛。

    A* 门槛 = d_min + margin；若拿它判"违反"，会把合法但比前馈门槛贴墙的
    平滑路径误判成约束违反（实测 12f7c0b0 会报 25.3% 违反率）。
    """
    from runner import experiment

    config_root = Path(__file__).resolve().parents[1] / "config"
    from runner import config as config_module

    resolved = config_module.load_resolved(
        "configs/frozen_A_feedforward.yaml",
        "scenarios/houseexpo_12f7c0b0.yaml",
        config_root=config_root,
    )
    scene = experiment.build_scene(resolved.scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    result = experiment.plan_path(scene, setup)
    smooth = experiment.smooth_result(result, scene, setup)
    assert setup.astar_params.d_min > setup.bspline_params.d_min, "前馈应抬高 A* 门槛"

    trajectory = experiment.build_trajectory(
        result,
        dt=0.05,
        cruise_speed=setup.cruise_speed,
        w_max=float(resolved.platform["w_max"]),
        body_radius=float(resolved.platform["body_radius"]),
        d_min=setup.bspline_params.d_min,
        e_budget=float(setup.astar_params.e_budget),
        smooth=smooth,
    )
    assert trajectory.violation.sum() == 0, "合法平滑路径不应被判违反"
    assert smooth.min_clearance < setup.astar_params.d_min, (
        "本用例的前提：平滑离障低于 A* 门槛（否则测不到这个坑）"
    )


def test_curvature_infeasible_is_distinct_from_pipeline_success() -> None:
    """回退折线必须报告 curvature_infeasible，即使 success=True。

    这是 2026-10-29 的重新定性：回退不是"降级成功"，是曲率约束违反。
    """
    field = open_field()
    result = smooth(right_angle_turn(), field, e_budget=1.0)
    assert result.success is True, "几何链路确实产出了折线"
    assert result.fell_back is True
    assert result.curvature_ok is False, "折线拐点曲率无界 -> 约束违反"
    assert result.curvature_infeasible is True
    assert result.trackable is False, "回退产物不得交给局部规划器"
    report = result.report
    assert report.curvature_infeasible is True
    assert report.curvature_satisfied is False
    payload = result.as_dict()
    assert payload["success"] == 1 and payload["curvature_satisfied"] == 0


def test_spline_reports_curvature_satisfied() -> None:
    field = open_field()
    result = smooth(right_angle_turn(), field)
    assert result.status == "spline"
    assert result.curvature_ok is True
    assert result.curvature_infeasible is False
    assert result.trackable is True


def test_feasibility_stats_counts_infeasible_not_success() -> None:
    """统计只认 curvature_satisfied，不能用"产出了结果"掩盖违反。"""
    field = open_field()
    ok = smooth(right_angle_turn(), field)
    bad = smooth(right_angle_turn(), field, e_budget=1.0)
    stats = bspline.feasibility_stats([ok.report, bad.report])
    assert stats["tasks"] == 2
    assert stats["curvature_satisfied"] == 1
    assert stats["curvature_infeasible"] == 1
    assert stats["curvature_infeasible_rate"] == pytest.approx(0.5)
    # 两个任务都"产出了结果"，但其中一个是违反
    assert stats["spline"] == 1


# ---------------------------------------------------------------------------
# 离障约束
# ---------------------------------------------------------------------------


def test_clearance_respected_on_corridor() -> None:
    """在窄走廊里，平滑结果全程 clearance >= d_min；做不到就必须回退。

    走廊内最宽处 clearance = 0.65 m（1.4 m 宽走廊的一半），所以 d_min 必须 <= 0.65
    才存在可行路径；这里取 0.60，考察平滑后是否仍守住。
    """
    occupancy = np.ones((40, 120), dtype=bool)
    occupancy[12:26, 5:115] = False  # 1.4 m 高的走廊：y ∈ [1.25, 2.55]
    field = astar.clearance_field(occupancy, 0.1)

    points = np.column_stack([np.linspace(1.15, 10.85, 400), np.full(400, 1.9)])
    result = smooth(points, field, d_min=0.60)
    assert result.min_clearance >= 0.60 - 1e-9
    assert np.all(result.clearance >= 0.60 - 1e-9)


def test_clearance_violation_triggers_insertion_or_fallback() -> None:
    """贴着走廊边界走的折线，平滑切角会侵入边界：必须插入控制点或回退。"""
    occupancy = np.ones((40, 120), dtype=bool)
    occupancy[12:26, 5:115] = False
    field = astar.clearance_field(occupancy, 0.1)

    # 贴着走廊上边界：y = 2.45 时 clearance 只有 0.15 左右，
    # 平滑一旦切角就会跌破 d_min —— 这正是要触发的分支。
    points = np.column_stack([np.linspace(0.6, 11.4, 400), np.full(400, 2.45)])
    result = smooth(points, field, d_min=0.55)
    assert result.report.rounds >= 1
    if result.status == "spline":
        assert result.min_clearance >= 0.55 - 1e-9
    else:
        assert result.fell_back


# ---------------------------------------------------------------------------
# 能耗预算
# ---------------------------------------------------------------------------


def test_energy_within_budget() -> None:
    field = open_field()
    result = smooth(right_angle_turn(), field, e_budget=100_000.0)
    assert result.energy_j <= 100_000.0 + 1e-9


def test_tiny_energy_budget_forces_fallback() -> None:
    """预算小于任何可行路径的能耗时，平滑必须回退而不是"假装满足"。"""
    field = open_field()
    result = smooth(right_angle_turn(), field, e_budget=1.0)
    assert result.fell_back
    assert result.status == "polyline"
    assert result.report.reason in {"energy", "insert_exhausted"}


def test_energy_recheck_uses_geometry() -> None:
    """能耗按实际几何长度复算：曲线比折线短时，电能也应更少。"""
    field = open_field()
    points = right_angle_turn()
    result = smooth(points, field)
    assert result.status == "spline"
    polyline = bspline._polyline_energy(
        points,
        speed=SPEED,
        energy_params=PLATFORM,
        c_rr=0.015,
        slope_rad=0.0,
        resolution=0.1,
        origin=(0.0, 0.0),
    )
    # 切角会缩短长度 -> 电能下降（本用例的两臂等长，切角是纯收益）。
    assert result.length_m < bspline._polyline_path(points)[3]
    assert result.energy_j < polyline


# ---------------------------------------------------------------------------
# 确定性与回退分支
# ---------------------------------------------------------------------------


def test_same_input_same_output() -> None:
    field = open_field()
    points = right_angle_turn()
    first = smooth(points, field)
    second = smooth(points, field)
    assert first.status == second.status
    assert np.array_equal(first.xy, second.xy)
    assert np.array_equal(first.curvature, second.curvature)
    assert first.report.as_dict() == second.report.as_dict()
    assert first.energy_j == second.energy_j


def test_fallback_branch_is_reachable_and_reported() -> None:
    """回退分支必须能被触发，并且在 report 里如实记录原因。"""
    field = open_field()
    result = smooth(right_angle_turn(), field, e_budget=1.0)
    assert result.fell_back
    assert result.report.fell_back
    report = result.report.as_dict()
    assert report["status"] == "polyline"
    assert report["reason"] != "ok"


def test_fallback_rate_alias_matches_feasibility_stats() -> None:
    """废弃别名 fallback_rate 必须与 feasibility_stats 一致（单一事实来源）。"""
    field = open_field()
    reports = [
        smooth(right_angle_turn(), field).report,
        smooth(right_angle_turn(), field, e_budget=1.0).report,
    ]
    assert bspline.fallback_rate(reports) == bspline.feasibility_stats(reports)
    stats = bspline.feasibility_stats(reports)
    assert stats["tasks"] == 2
    assert stats["curvature_infeasible"] == 1
    assert stats["curvature_infeasible_rate"] == pytest.approx(0.5)


def test_empty_and_single_point_inputs() -> None:
    field = open_field()
    with pytest.raises(ValueError):
        smooth(np.zeros((0, 2)), field)
    single = smooth(np.array([[1.0, 1.0]]), field)
    assert single.success
    assert single.length_m == 0.0


def test_rdp_keeps_endpoints_and_simplifies() -> None:
    points = straight_line(length=4.0)
    kept = bspline.rdp_indices(points, 0.15)
    assert kept[0] == 0
    assert kept[-1] == points.shape[0] - 1
    assert kept.size < points.shape[0]


def test_rdp_epsilon_smaller_than_raster_sawtooth_is_dangerous() -> None:
    """阈值小于栅格锯齿振幅时，锯齿会被当成真实几何保留下来。

    这解释了默认 ``rdp_epsilon_m = 0.15``（≈1.5 格）不是随手取的数：
    8 邻域折线的"阶梯"振幅就是一格，阈值低于它，阶梯会被原样保留。
    """
    # 对角阶梯：每走一格 x 就抬一格 y，锯齿振幅 = 1 格 = 0.1 m
    count = 40
    stair = np.column_stack(
        [0.2 + np.repeat(np.arange(count), 2)[: count * 2] * 0.1,
         0.2 + np.arange(count * 2) * 0.05]
    )
    kept_default = bspline.rdp_indices(stair, 0.15)
    kept_tight = bspline.rdp_indices(stair, 0.01)
    assert kept_tight.size > kept_default.size
    assert kept_default.size <= 4, "默认阈值应把整段阶梯压成一条直线"
