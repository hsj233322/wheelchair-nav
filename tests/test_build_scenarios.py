"""env/build_scenarios.py 的测试：确定性、可行性、膨胀来源、地形一致性。"""

import json
from pathlib import Path

import numpy as np
import pytest

from core import astar
from env import build_scenarios, floorplan
from runner import config as config_module
from runner import experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config"

HOUSE = {
    "id": "synthetic0001",
    "room_num": 3,
    "bbox": {"min": [0.0, 0.0], "max": [8.0, 6.0]},
    "verts": [[0, 0], [8, 0], [8, 6], [0, 6]],
    "room_category": {
        "Kitchen": [[0.0, 0.0, 3.0, 3.0]],
        "Bedroom": [[3.0, 0.0, 8.0, 3.0]],
        "Hallway": [[0.0, 3.0, 8.0, 6.0]],
    },
}

NARROW = {
    "id": "narrow0001",
    "room_num": 1,
    "bbox": {"min": [0.0, 0.0], "max": [1.0, 6.0]},
    "verts": [[0, 0], [1, 0], [1, 6], [0, 6]],
    "room_category": {"Room": [[0.0, 0.0, 1.0, 6.0]]},
}


def write_plan(tmp_path: Path, data: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def load_configs() -> tuple[dict, dict, build_scenarios.TerrainRules]:
    platform = config_module.validate_platform(
        config_module.read_config(CONFIG_ROOT / "platform.yaml")
    )
    algo = config_module.validate_algo(
        config_module.read_config(CONFIG_ROOT / "configs" / "frozen_A.yaml")
    )
    rules = build_scenarios.load_terrain_rules(CONFIG_ROOT / "terrain_rules.yaml")
    return platform, algo, rules


def build(tmp_path: Path, data: dict, *, seed: int = 7, name: str = "plan.json"):
    platform, algo, rules = load_configs()
    source = write_plan(tmp_path, data, name)
    return build_scenarios.build_one(
        source,
        resolution=0.1,
        platform=platform,
        algo=algo,
        rules=rules,
        seed=seed,
        min_start_goal_m=3.0,
        max_attempts=20,
    )


def test_build_one_produces_valid_solvable_scenario(tmp_path: Path) -> None:
    scenario, info = build(tmp_path, HOUSE)
    assert scenario["id"] == "houseexpo_syntheti"
    config_module.validate_scenario(scenario)  # 同时校验 content_hash

    platform, algo, _ = load_configs()
    expected_inflation = float(platform["body_radius"]) + float(platform["inflation"])
    assert scenario["map"]["inflation_m"] == pytest.approx(expected_inflation)
    assert scenario["terrain"]["note"].startswith("基于房间类型的假设")

    resolved = config_module.load_resolved(
        "configs/frozen_A.yaml", _write(tmp_path, scenario), config_root=CONFIG_ROOT
    )
    scene = experiment.build_scene(resolved.scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)

    d_min = setup.astar_params.d_min
    assert scene.clearance[scene.start_cell] >= d_min - 1e-9
    assert scene.clearance[scene.goal_cell] >= d_min - 1e-9
    result = experiment.plan_path(scene, setup)
    assert result.success
    assert result.total_energy_j <= setup.astar_params.e_budget + 1e-9
    assert result.clearance.min() >= d_min - 1e-9
    assert info["path_length_m"] == pytest.approx(result.length_m, abs=1e-3)

    start_cell = astar.world_to_cell(scene.start_xy, scene.resolution, scene.origin)
    assert start_cell == scene.start_cell
    goal_cell = astar.world_to_cell(scene.goal_xy, scene.resolution, scene.origin)
    assert goal_cell == scene.goal_cell


def _write(tmp_path: Path, scenario: dict) -> Path:
    path = tmp_path / "scenario.yaml"
    build_scenarios.write_scenario(scenario, path)
    return path


def test_same_seed_is_byte_identical(tmp_path: Path) -> None:
    # 同一个源文件、同一个 seed -> 场景 dict 与 YAML 字节都一致
    first, _ = build(tmp_path, HOUSE, seed=11)
    second, _ = build(tmp_path, HOUSE, seed=11)
    assert first == second
    assert first["content_hash"] == second["content_hash"]
    path_a = tmp_path / "out_a.yaml"
    path_b = tmp_path / "out_b.yaml"
    build_scenarios.write_scenario(first, path_a)
    build_scenarios.write_scenario(second, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_different_seed_moves_start_and_goal(tmp_path: Path) -> None:
    first, _ = build(tmp_path, HOUSE, seed=1)
    second, _ = build(tmp_path, HOUSE, seed=2)
    assert (first["start"], first["goal"]) != (second["start"], second["goal"])
    # 换种子只改起终点，几何与地形不变
    assert first["map"] == second["map"]
    assert first["terrain"] == second["terrain"]
    assert first["content_hash"] != second["content_hash"]


def test_narrow_floorplan_is_skipped(tmp_path: Path) -> None:
    with pytest.raises(build_scenarios.SkipScene):
        build(tmp_path, NARROW)


def test_inflation_larger_than_d_min_is_rejected(tmp_path: Path) -> None:
    platform, _, rules = load_configs()
    tight_algo = config_module.validate_algo(
        {
            "id": "tight",
            "astar": {"e_budget": 2000.0, "d_min": 0.6, "kappa_max": 1.5, "v_cruise": 1.0},
        }
    )
    source = write_plan(tmp_path, HOUSE, "tight.json")
    with pytest.raises(config_module.ConfigError, match="inflation"):
        build_scenarios.build_one(
            source,
            resolution=0.1,
            platform=platform,
            algo=tight_algo,
            rules=rules,
            seed=3,
            min_start_goal_m=3.0,
            max_attempts=5,
        )


def test_terrain_classes_match_runner_reconstruction(tmp_path: Path) -> None:
    scenario, _ = build(tmp_path, HOUSE)
    _, _, rules = load_configs()
    plan = floorplan.load_floorplan(tmp_path / "plan.json")
    rows, cols = scenario["map"]["shape"]
    grid = floorplan.terrain_class_grid(
        plan.rooms,
        (rows, cols),
        scenario["map"]["resolution"],
        scenario["map"]["origin"],
        rules.room_type_to_class,
        default_class=rules.default_class,
        generic_types=rules.generic_types,
    )
    scene = experiment.build_scene(scenario)
    for class_name, props in rules.classes.items():
        mask = grid == class_name
        if mask.any():
            assert np.allclose(scene.c_rr[mask], props["c_rr"])
    # 至少覆盖到 tile（厨房）与 wood（卧室）
    assert (grid == "tile").any()
    assert (grid == "wood").any()


def test_cli_writes_loadable_scenarios(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_plan(data_dir, HOUSE, "synthetic0001.json")
    out_dir = tmp_path / "out"
    exit_code = build_scenarios.main(
        [
            "--data-dir", str(data_dir),
            "--out-dir", str(out_dir),
            "--seed", "5",
            "--max-scenes", "3",
        ]
    )
    assert exit_code == 0
    written = sorted(out_dir.glob("*.yaml"))
    assert len(written) == 1
    scenario = config_module.validate_scenario(
        config_module.read_config(written[0])
    )
    assert scenario["source"]["dataset"] == "HouseExpo"
    assert "generated 1 scenes" in capsys.readouterr().out
