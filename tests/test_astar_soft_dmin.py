"""第 3 章新增可选参数的测试：``enforce_dmin`` / ``soft_penalty_weight``。

锁三件事：

1. **默认行为逐字段不变**：``enforce_dmin=True``、``soft_penalty_weight=0.0``
   时，返回值与加入这两个字段**之前**抓取的基准值完全一致（含 expanded /
   labels_created 与 cells / xy / 逐段能耗的字节哈希）。
2. **关掉硬约束 = 软惩罚**：贴墙捷径被启用，违反率 > 0 且随 w 上升而下降；
   硬约束下违反率恒为 0。
3. **配置管道**：``algo.astar.enforce_dmin`` / ``soft_penalty_weight`` /
   ``collect_pareto`` 被正确校验与传递，非法值被拒。

另外锁定 Pareto 前沿机制（第 3 章 Pareto 图依赖它）：合成双通道地图上
``collect_pareto=True`` 必须返回 2 个非支配点（短-高耗 / 长-低耗）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from core import astar, energy
from runner import config as config_module
from runner import experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config"
PLATFORM = energy.EnergyParams(mass=100.0, g=9.81, eta=0.85, p_aux=15.0)

# ---------------------------------------------------------------------------
# 基准值：2026-11-07 在**受控材质因子**（c_rr 四档 0.015/0.03/0.04/0.06）重生成
# 场景之后重新抓取。
#
# 为什么必须重抓（口径说明，不是"改测试凑结果"）：
#   * 换材质只改能耗输入，不改几何与约束。``houseexpo_1241c62f`` 与
#     ``houseexpo_5427f9b0`` 的 cells 字节哈希**与换材质前完全一致**，
#     只有 ``total_energy_j`` 变（328.76 / 834.11 J）—— 这正是"默认行为不变"的证据；
#   * ``houseexpo_07432f2d`` 的路径**确实变了**（length 相同 7.904 m，
#     cells 哈希从 57eccbbe 变为 228c097b，最小离障 1.203 -> 0.872 m）。
#     原因是多目标标签搜索在 (长度, 能耗) 上取非支配最优：同一长度下存在多条
#     候选路线时，换材质会改变哪一条能耗最低，于是 tie-break 结果不同。
#     长度不变、只有 tie-break 改变，与"新开关默认值不改变搜索行为"不矛盾。
#
# 字段：length_m, total_energy_j, clearance.min(), expanded, labels_created,
#       sha256(cells.tobytes())[:16]
# ---------------------------------------------------------------------------
GOLDEN: dict[str, tuple[float, float, float, int, int, str]] = {
    "houseexpo_07432f2d": (
        7.904163056034262, 283.780870928592, 0.8719544457292887, 5323, 7954,
        "228c097b3e3b461e",
    ),
    "houseexpo_1241c62f": (
        9.374011537017763, 328.76243133211915, 0.8555385138137418, 9526, 13076,
        "ce7d5c9159d3c0b6",
    ),
    "houseexpo_5427f9b0": (
        19.04142135623731, 834.1061842187888, 0.711577310586391, 358, 622,
        "e35bbc052214ba4f",
    ),
}


def _plan_scenario(scenario_id: str) -> astar.PathResult:
    resolved = config_module.load_resolved(
        "configs/frozen_A.yaml", CONFIG_ROOT / "scenarios" / f"{scenario_id}.yaml",
        config_root=CONFIG_ROOT,
    )
    scene = experiment.build_scene(resolved.scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    return experiment.plan_path(scene, setup)


@pytest.mark.parametrize("scenario_id", sorted(GOLDEN))
def test_default_params_reproduce_pre_change_baseline(scenario_id: str) -> None:
    """默认参数下结果与加入新字段前**逐字段一致**（不是"近似一致"）。"""
    expected = GOLDEN[scenario_id]
    result = _plan_scenario(scenario_id)
    digest = hashlib.sha256(result.cells.tobytes()).hexdigest()[:16]
    assert result.success is True
    assert result.length_m == expected[0]
    assert result.total_energy_j == expected[1]
    assert float(result.clearance.min()) == expected[2]
    assert result.expanded == expected[3]
    assert result.labels_created == expected[4]
    assert digest == expected[5]
    # 新字段的默认值不引入任何软惩罚
    assert result.soft_penalty == 0.0
    assert astar.AStarParams(e_budget=1.0, d_min=0.1, kappa_max=1.0).enforce_dmin is True
    assert astar.AStarParams(
        e_budget=1.0, d_min=0.1, kappa_max=1.0
    ).soft_penalty_weight == 0.0


# ---------------------------------------------------------------------------
# 合成地图：一条"贴墙捷径" vs 一条"绕行合规路线"
# ---------------------------------------------------------------------------

ROWS, COLS = 40, 80
RES = 0.1
D_MIN = 0.30
START_XY = (0.5, 2.05)
GOAL_XY = (7.5, 2.05)


def shortcut_map() -> tuple[np.ndarray, np.ndarray]:
    """一堵竖墙把起终点隔开，只留两条通道：

    * **row 20 的 1 格窄缝**（clearance ≈ 0.05，违反 ``d_min = 0.30``）—— 几乎在
      直线上，走它只比直线多 0.8 m 左右；
    * **rows 30~38 的宽口**（clearance ≈ 0.35，合规）—— 要下移 10 多行再回来，
      多走约 2.4 m。

    两条通道的长度差足够大（≈2.4 m），所以"违反的边际收益"足够显著：
    w 从 0 加到 ~2 的过程中，规划器一直选窄缝；w 再大才切到宽口。
    """
    occ = np.zeros((ROWS, COLS), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    occ[1:39, 40] = True        # 整堵竖墙
    occ[20, 40] = False         # 窄缝（1 格）
    occ[30:39, 40] = False      # 宽口（9 格）
    clear = astar.clearance_field(occ, RES)
    return occ, clear


def _run(
    *,
    enforce_dmin: bool,
    weight: float,
    occ: np.ndarray,
    clear: np.ndarray,
    collect_pareto: bool = False,
) -> astar.PathResult:
    return astar.find_path(
        occ, RES, START_XY, GOAL_XY,
        astar.AStarParams(
            e_budget=1e6, d_min=D_MIN, kappa_max=1.5, kappa_window_m=0.6,
            enforce_dmin=enforce_dmin, soft_penalty_weight=weight,
            collect_pareto=collect_pareto,
        ),
        energy_params=PLATFORM, speed=1.0, c_rr=0.015, slope_rad=0.0,
        origin=(0.0, 0.0), clearance=clear,
    )


def _violation_length(result: astar.PathResult) -> float:
    steps = np.asarray(result.step_length, dtype=float)
    clear = np.asarray(result.clearance, dtype=float)
    mask = clear < D_MIN - 1e-6
    return float(steps[mask].sum())


def test_scene_probe_geometry_is_as_designed() -> None:
    """先确认地图确实同时提供"窄缝捷径"和"宽绕行"，否则后面的断言无意义。"""
    occ, clear = shortcut_map()
    assert clear[20, 40] < D_MIN, "row 20 的窄缝必须违反 d_min"
    assert clear[34, 40] >= D_MIN, "rows 30~38 的宽口必须满足 d_min"
    assert clear[20, 40] < clear[34, 40]


def test_hard_constraint_never_violates_and_takes_detour() -> None:
    occ, clear = shortcut_map()
    result = _run(enforce_dmin=True, weight=0.0, occ=occ, clear=clear)
    assert result.success is True
    assert result.soft_penalty == 0.0
    assert float(result.clearance.min()) >= D_MIN - 1e-6
    assert _violation_length(result) == 0.0


def test_soft_mode_with_zero_weight_takes_the_shortcut() -> None:
    """w=0 且关硬约束 = 完全无安全约束：走窄缝、违反率 > 0。"""
    occ, clear = shortcut_map()
    hard = _run(enforce_dmin=True, weight=0.0, occ=occ, clear=clear)
    soft = _run(enforce_dmin=False, weight=0.0, occ=occ, clear=clear)
    assert soft.success is True
    assert _violation_length(soft) > 0.0
    assert float(soft.clearance.min()) < D_MIN - 1e-6
    assert soft.length_m < hard.length_m, "捷径必须比绕行短，否则对照没有意义"
    assert soft.soft_penalty == 0.0, "w=0 时不累加任何惩罚"


def test_soft_penalty_reduces_violation_monotonically() -> None:
    """违反量随 w 单调不增；w 足够大时回到 0 违反的绕行路线。

    本数据集上的关键事实（第 3 章表）：**w 不需要很大就已经切到合规路线**，
    所以约束违反率在 w >= 1 时已经归零 —— 软惩罚并不是"永远留一点违反"，
    它只是**没有保证**；一旦 w 掉到阈值以下（这里是 0.5），违反立刻回来。
    """
    occ, clear = shortcut_map()
    violations = []
    lengths = []
    penalties = []
    for weight in (0.0, 0.5, 1.0, 2.0, 5.0):
        result = _run(enforce_dmin=False, weight=weight, occ=occ, clear=clear)
        assert result.success is True
        violations.append(_violation_length(result))
        lengths.append(result.length_m)
        penalties.append(result.soft_penalty)
    assert violations[0] > 0.0, "w=0 必须违反"
    assert all(
        later <= earlier + 1e-9 for earlier, later in zip(violations, violations[1:])
    ), f"违反长度必须随 w 单调不增，实际 {violations}"
    assert violations[-1] == 0.0, "w=5 时应当已经回到合规绕行"
    assert violations[1] > 0.0, "w=0.5 时仍应走在窄缝上（阈值在 0.5~1 之间）"
    assert penalties[1] > 0.0, "走在窄缝上必须累加惩罚"
    assert lengths[-1] > lengths[0], "代价项把路径推长"
    hard = _run(enforce_dmin=True, weight=0.0, occ=occ, clear=clear)
    assert lengths[-1] == pytest.approx(hard.length_m, abs=1e-9)


def test_soft_penalty_is_reported_separately_from_length() -> None:
    """软惩罚只进 ``soft_penalty``，不得混进几何长度与能耗。"""
    occ, clear = shortcut_map()
    soft = _run(enforce_dmin=False, weight=0.5, occ=occ, clear=clear)
    assert soft.soft_penalty > 0.0
    assert soft.length_m == pytest.approx(
        float(np.asarray(soft.step_length).sum()), abs=1e-9
    )
    assert soft.total_energy_j > 0.0


def test_soft_mode_allows_start_and_goal_below_d_min() -> None:
    """关掉硬约束后，起终点自身低于 d_min 也必须能规划（否则无法做对照）。"""
    occ, clear = shortcut_map()
    result = astar.find_path(
        occ, RES, START_XY, GOAL_XY,
        astar.AStarParams(e_budget=1e6, d_min=2.0, kappa_max=1.5,
                          kappa_window_m=0.6, enforce_dmin=False,
                          soft_penalty_weight=1.0),
        energy_params=PLATFORM, speed=1.0, c_rr=0.015, slope_rad=0.0,
        origin=(0.0, 0.0), clearance=clear,
    )
    assert result.success is True
    # 同一张图、同一个 d_min 在硬约束下必须判"起点违反离障"
    blocked = astar.find_path(
        occ, RES, START_XY, GOAL_XY,
        astar.AStarParams(e_budget=1e6, d_min=2.0, kappa_max=1.5,
                          kappa_window_m=0.6),
        energy_params=PLATFORM, speed=1.0, c_rr=0.015, slope_rad=0.0,
        origin=(0.0, 0.0), clearance=clear,
    )
    assert blocked.success is False
    assert blocked.reason == "start_violates_clearance"


def test_params_validation_rejects_negative_weight() -> None:
    with pytest.raises(ValueError):
        astar.AStarParams(e_budget=1.0, d_min=0.1, kappa_max=1.0,
                          soft_penalty_weight=-0.1)


# ---------------------------------------------------------------------------
# Pareto 前沿机制（第 3 章 Pareto 图依赖）
# ---------------------------------------------------------------------------


def test_collect_pareto_returns_two_points_on_two_route_map() -> None:
    """两条通道（短-高阻 / 长-低阻）必须给出 2 个非支配点。

    如果这里退化成 1 点，第 3 章的 Pareto 图就没有意义 —— 而真实户型场景
    的前沿**确实是** 1 点（见 docs/chapter3_pareto_points.csv），
    所以这条测试同时说明"退化是数据性质，不是代码缺陷"。
    """
    rows, cols = 40, 80
    occ = np.zeros((rows, cols), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    occ[:, 40] = True
    occ[15:18, 40] = False      # 短通道（高阻）
    occ[30:36, 40] = False      # 长通道（低阻）
    clear = astar.clearance_field(occ, RES)
    c_rr = np.full((rows, cols), 0.015)
    c_rr[:20, :] = 0.06
    c_rr[20:30, :] = 0.03
    result = astar.find_path(
        occ, RES, (0.4, 2.0), (7.6, 2.0),
        astar.AStarParams(e_budget=1e9, d_min=0.0, kappa_max=1.5,
                          kappa_window_m=0.6, collect_pareto=True),
        energy_params=PLATFORM, speed=1.0, c_rr=c_rr, slope_rad=0.0,
        origin=(0.0, 0.0), clearance=clear,
    )
    assert result.pareto_length_m.size == 2
    short_l, long_l = result.pareto_length_m
    short_e, long_e = result.pareto_energy_j
    assert short_l < long_l
    assert short_e > long_e, "短路线必须更耗电，否则不是权衡关系"


# ---------------------------------------------------------------------------
# 配置管道
# ---------------------------------------------------------------------------


def _algo_doc(**astar_overrides: object) -> dict[str, object]:
    astar_section: dict[str, object] = {
        "e_budget": 2000.0, "d_min": 0.70, "kappa_max": 1.5,
    }
    astar_section.update(astar_overrides)
    return {"id": "tmp_soft", "astar": astar_section}


def test_validate_algo_accepts_soft_switch() -> None:
    doc = config_module.validate_algo(
        _algo_doc(enforce_dmin=False, soft_penalty_weight=2.0, collect_pareto=True)
    )
    assert doc["astar"]["enforce_dmin"] is False
    assert doc["astar"]["soft_penalty_weight"] == pytest.approx(2.0)
    assert doc["astar"]["collect_pareto"] is True


def test_effective_params_default_to_hard_constraint() -> None:
    effective = config_module.astar_effective_params(
        _algo_doc(), {"v_max": 1.0}
    )
    assert effective["enforce_dmin"] is True
    assert effective["soft_penalty_weight"] == 0.0
    assert effective["collect_pareto"] is False


def test_validate_algo_rejects_bad_soft_values() -> None:
    with pytest.raises(config_module.ConfigError):
        config_module.validate_algo(_algo_doc(soft_penalty_weight=-1.0))
    with pytest.raises(config_module.ConfigError):
        config_module.validate_algo(_algo_doc(enforce_dmin="no"))
    with pytest.raises(config_module.ConfigError):
        config_module.validate_algo(_algo_doc(collect_pareto=1))


def test_chapter3_configs_are_loadable() -> None:
    """第 3 章用到的配置必须能通过三层校验（含跨层 d_min >= body_radius）。"""
    scenario = CONFIG_ROOT / "scenarios" / "houseexpo_07432f2d.yaml"
    for algo in (
        "configs/chapter3/baseline_length_a.yaml",
        "configs/chapter3/baseline_esa_a.yaml",
        "configs/chapter3/soft_w000.yaml",
        "configs/chapter3/soft_w050.yaml",
        "configs/chapter3/soft_w100.yaml",
        "configs/chapter3/soft_w200.yaml",
        "configs/chapter3/soft_w500.yaml",
    ):
        resolved = config_module.load_resolved(algo, scenario, config_root=CONFIG_ROOT)
        params = experiment.planner_setup(resolved.platform, resolved.algo).astar_params
        if "baseline" in algo:
            assert params.d_min == pytest.approx(0.58)
            assert params.enforce_dmin is True
        else:
            assert params.d_min == pytest.approx(0.70)
            assert params.enforce_dmin is False
        assert params.soft_penalty_weight >= 0.0
