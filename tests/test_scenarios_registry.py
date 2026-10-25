"""已固化 HouseExpo 场景的集成测试：可加载、哈希可信、起终点自由且有解。"""

from pathlib import Path

import numpy as np
import pytest

from core import astar
from runner import config as config_module
from runner import experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config"
SCENARIOS = sorted((CONFIG_ROOT / "scenarios").glob("houseexpo_*.yaml"))

PLATFORM = config_module.validate_platform(
    config_module.read_config(CONFIG_ROOT / "platform.yaml")
)


def test_houseexpo_subset_is_present() -> None:
    # 方案要求 10~20 个户型
    assert 10 <= len(SCENARIOS) <= 20


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda item: item.stem)
def test_scenario_is_free_and_solvable(path: Path) -> None:
    resolved = config_module.load_resolved("configs/frozen_A.yaml", path)
    scenario = resolved.scenario
    assert scenario["content_hash"] == config_module.compute_scenario_hash(scenario)
    assert scenario["map"]["inflation_m"] == pytest.approx(
        float(PLATFORM["body_radius"]) + float(PLATFORM["inflation"])
    )

    scene = experiment.build_scene(scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    d_min = setup.astar_params.d_min

    # 起终点必须落在自由区且满足安全半径，并且 world <-> grid 往返一致
    for xy, cell in ((scene.start_xy, scene.start_cell), (scene.goal_xy, scene.goal_cell)):
        assert not scene.occupancy[cell]
        assert scene.clearance[cell] >= d_min - 1e-9
        assert astar.world_to_cell(xy, scene.resolution, scene.origin) == cell

    result = experiment.plan_path(scene, setup)
    assert result.success, f"{path.name} 无解: {result.reason}"
    assert result.clearance.min() >= d_min - 1e-9
    assert result.total_energy_j <= setup.astar_params.e_budget + 1e-9
    assert result.length_m > 0.0
    assert result.pareto_length_m.size >= 1


def test_scenarios_have_distinct_content_hashes() -> None:
    hashes = {
        config_module.read_config(path)["content_hash"] for path in SCENARIOS
    }
    assert len(hashes) == len(SCENARIOS)


def test_polygon_area_is_consistent_with_rasterization() -> None:
    """栅格化面积应与轮廓 shoelace 面积一致（扫描线填充的正确性检查）。"""
    for path in SCENARIOS[:4]:
        scenario = config_module.read_config(path)
        polygon = np.asarray(scenario["map"]["polygons"][0], dtype=float)
        x, y = polygon[:, 0], polygon[:, 1]
        shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        scene = experiment.build_scene(scenario)
        free_area = float((~scene.occupancy).sum()) * scene.resolution**2
        assert free_area == pytest.approx(shoelace, rel=0.05)
