"""已固化场景的集成测试：可加载、哈希可信、起终点自由且有解、难度分档真实。"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from core import astar
from env import scene_quality
from runner import config as config_module
from runner import experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config"
SCENARIOS = sorted((CONFIG_ROOT / "scenarios").glob("houseexpo_*.yaml"))
NARROW = sorted((CONFIG_ROOT / "scenarios").glob("narrowdoor_*.yaml"))
METADATA = ROOT / "docs" / "scene_metadata.csv"

PLATFORM = config_module.validate_platform(
    config_module.read_config(CONFIG_ROOT / "platform.yaml")
)


def test_houseexpo_subset_is_present() -> None:
    # 方案要求 10~20 个户型场景，另有 3 个窄门场景
    assert 10 <= len(SCENARIOS) <= 20
    assert len(NARROW) == 3


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda item: item.stem)
def test_scenario_is_free_and_solvable(path: Path) -> None:
    resolved = config_module.load_resolved("configs/frozen_A.yaml", path)
    scenario = resolved.scenario
    assert scenario["content_hash"] == config_module.compute_scenario_hash(scenario)
    # 解耦后场景只声明物理半径（body_radius），策略裕度在 algo.astar.d_min
    assert scenario["map"]["inflation_m"] == pytest.approx(float(PLATFORM["body_radius"]))

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

    # 质量档案必须能独立复算，且与场景里记录的跨门次数一致
    quality = scenario["quality"]
    assert quality["grade"] == quality["tier_declared"]
    segmentation = scene_quality.segment_rooms(scene.clearance, core_clearance_m=0.9)
    recomputed = scene_quality.build_quality(
        path_len_m=result.length_m,
        cells=result.cells,
        labels=segmentation.labels,
        clearance=scene.clearance,
        free_mask=scene.clearance + 1e-9 >= d_min,
        class_grid=None,
        default_class="concrete",
    )
    assert recomputed.room_label_changes == quality["room_label_changes"]
    assert recomputed.rooms_visited == quality["rooms_visited"]
    assert recomputed.grade == quality["grade"]
    assert recomputed.path_len_m == pytest.approx(quality["path_len_m"], abs=1e-3)


def test_all_three_difficulty_tiers_are_represented() -> None:
    grades = Counter(
        config_module.read_config(path)["quality"]["grade"] for path in SCENARIOS
    )
    assert grades[scene_quality.EASY] >= 3
    assert grades[scene_quality.MEDIUM] >= 3
    assert grades[scene_quality.HARD] >= 3
    assert sum(grades.values()) == len(SCENARIOS)


def test_hard_scenarios_have_three_or_more_room_label_changes() -> None:
    for path in SCENARIOS:
        quality = config_module.read_config(path)["quality"]
        if quality["grade"] == scene_quality.HARD:
            assert quality["room_label_changes"] >= 3
            assert quality["path_len_m"] >= 8.0
            assert quality["bottleneck_width_m"] > 1.2  # 分割无关的硬指标也要能复算


def test_scenarios_have_distinct_content_hashes() -> None:
    hashes = {config_module.read_config(path)["content_hash"] for path in SCENARIOS}
    assert len(hashes) == len(SCENARIOS)


def test_metadata_csv_matches_scenarios() -> None:
    with METADATA.open(encoding="utf-8") as handle:
        rows = {row["id"]: row for row in csv.DictReader(handle)}
    assert len(rows) == len(SCENARIOS) + len(NARROW)
    for path in SCENARIOS + NARROW:
        scenario = config_module.read_config(path)
        row = rows[scenario["id"]]
        assert row["content_hash"] == scenario["content_hash"]
        assert row["grade"] == scenario["quality"]["grade"]
        assert float(row["path_len_m"]) == pytest.approx(
            float(scenario["quality"]["path_len_m"]), abs=1e-3
        )


def test_narrow_door_scenarios_have_two_feasibility_flags() -> None:
    body_radius = float(PLATFORM["body_radius"])
    physically_ok = 0
    for path in NARROW:
        resolved = config_module.load_resolved("configs/frozen_A.yaml", path)
        quality = resolved.scenario["quality"]
        width = float(quality["narrow_door_width_m"])
        assert quality["feasible_at_physical_radius"] == (width >= 2.0 * body_radius)
        assert quality["feasible_at_base_d_min"] is False
        physically_ok += int(quality["feasible_at_physical_radius"])
        scene = experiment.build_scene(resolved.scenario)
        setup = experiment.planner_setup(resolved.platform, resolved.algo)
        assert not experiment.plan_path(scene, setup).success
    # 1.2 m 门洞对物理半径（1.16 m）已够宽、对基准 d_min（1.40 m）不够：
    # 这正是通过性实验要测的边界
    assert physically_ok == 1


def test_platform_has_no_policy_margin_and_scenarios_use_physical_radius() -> None:
    platform = config_module.validate_platform(
        config_module.read_config(CONFIG_ROOT / "platform.yaml")
    )
    assert "inflation" not in platform  # 策略裕度已移到 algo.astar.d_min
    for path in SCENARIOS:
        scenario = config_module.read_config(path)
        assert scenario["map"]["inflation_m"] == pytest.approx(
            float(platform["body_radius"])
        )
        assert scenario["quality"]["physical_radius_m"] == pytest.approx(
            float(platform["body_radius"])
        )
        assert scenario["quality"]["base_d_min_m"] == pytest.approx(0.70)


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
