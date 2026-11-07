"""env/build_scenarios.py 的测试：分档采样、可行性、确定性、窄门组、CLI。"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from core import astar
from env import build_scenarios, floorplan, scene_quality
from runner import config as config_module
from runner import experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config"

# 单一房间：easy 档
ROOM = {
    "id": "room0001",
    "room_num": 3,
    "bbox": {"min": [0.0, 0.0], "max": [12.0, 8.0]},
    "verts": [[0, 0], [12, 0], [12, 8], [0, 8]],
    "room_category": {
        "Kitchen": [[0.0, 0.0, 4.0, 4.0]],
        "Bedroom": [[4.0, 0.0, 12.0, 4.0]],
        "Hallway": [[0.0, 4.0, 12.0, 8.0]],
    },
}

# 两个 8x6 房间 + 1.6 m 走廊：medium 档（走廊宽度卡在 0.7 < w/2 < 0.9）
DUMBBELL = {
    "id": "dumbbell0001",
    "room_num": 3,
    "bbox": {"min": [0.0, 0.0], "max": [26.0, 6.0]},
    "verts": [
        [0, 0], [8, 0], [8, 2.2], [18, 2.2], [18, 0], [26, 0],
        [26, 6], [18, 6], [18, 3.8], [8, 3.8], [8, 6], [0, 6],
    ],
    "room_category": {
        "Living_Room": [[0.0, 0.0, 8.0, 6.0]],
        "Kitchen": [[18.0, 0.0, 26.0, 6.0]],
        "Hallway": [[8.0, 2.2, 18.0, 3.8]],
    },
}

# 1 m 宽过道：任何安全半径 0.7 m 的规划都不可行
NARROW = {
    "id": "narrow0001",
    "room_num": 1,
    "bbox": {"min": [0.0, 0.0], "max": [1.0, 6.0]},
    "verts": [[0, 0], [1, 0], [1, 6], [0, 6]],
    "room_category": {"Room": [[0.0, 0.0, 1.0, 6.0]]},
}


def write_plan(tmp_path: Path, data: dict, name: str = "plan.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def load_configs():
    platform = config_module.validate_platform(
        config_module.read_config(CONFIG_ROOT / "platform.yaml")
    )
    algo = config_module.validate_algo(
        config_module.read_config(CONFIG_ROOT / "configs" / "frozen_A.yaml")
    )
    rules = build_scenarios.load_terrain_rules(CONFIG_ROOT / "terrain_rules.yaml")
    return platform, algo, rules


def make_selection(min_path_len_m: float = 8.0, easy_min_path_len_m: float = 4.0):
    return build_scenarios.Selection(
        min_path_len_m=min_path_len_m,
        easy_min_path_len_m=easy_min_path_len_m,
        tiers={},
    )


def build(tmp_path: Path, data: dict, *, tier: str, seed: int = 7, algorithm: dict | None = None):
    platform, algo, rules = load_configs()
    source = write_plan(tmp_path, data)
    return build_scenarios.build_one(
        source,
        tier=tier,
        resolution=0.1,
        platform=platform,
        algo=algorithm or algo,
        rules=rules,
        selection=make_selection(),
        seed=seed,
        core_clearance_m=0.9,
        max_attempts=24,
        min_start_goal_m=4.0,
    )


def _write_scenario(tmp_path: Path, scenario: dict, name: str = "scenario.yaml") -> Path:
    path = tmp_path / name
    build_scenarios.write_scenario(scenario, path)
    return path


def test_build_easy_scenario(tmp_path: Path) -> None:
    scenario, info = build(tmp_path, ROOM, tier=scene_quality.EASY)
    config_module.validate_scenario(scenario)
    quality = scenario["quality"]
    assert quality["grade"] == scene_quality.EASY
    assert quality["bottleneck_width_m"] >= scene_quality.BOTTLENECK_EASY_M
    assert quality["room_label_changes"] == 0
    assert quality["rooms_visited"] == 1
    assert quality["path_len_m"] >= 4.0
    assert quality["group"] == "houseexpo"
    assert quality["tier_declared"] == scene_quality.EASY
    # 场景只声明物理半径；策略裕度在 algo.astar.d_min
    assert scenario["map"]["inflation_m"] == pytest.approx(0.58)
    assert quality["base_d_min_m"] == pytest.approx(0.70)
    assert info["tier"] == scene_quality.EASY


def test_build_hard_scenario_crosses_a_narrow_door(tmp_path: Path) -> None:
    scenario, _ = build(tmp_path, DUMBBELL, tier=scene_quality.HARD)
    config_module.validate_scenario(scenario)
    quality = scenario["quality"]
    assert quality["grade"] == scene_quality.HARD
    assert quality["room_label_changes"] == 1
    assert quality["rooms_visited"] == 2
    assert quality["path_len_m"] >= 8.0
    assert quality["min_interface_width_m"] > 1.4
    # 保守 clearance 定义（减半个单元）让 1.6 m 走廊实测为 1.5 m
    assert quality["bottleneck_width_m"] == pytest.approx(1.5, abs=0.1)


def test_declared_tier_is_enforced(tmp_path: Path) -> None:
    # 该户型的必经通道只有 1.5 m（hard 档），声明 medium 必须跳过而不是硬贴标签
    with pytest.raises(build_scenarios.SkipScene):
        build(tmp_path, DUMBBELL, tier=scene_quality.MEDIUM)


def test_scenario_is_solvable_and_quality_matches_replay(tmp_path: Path) -> None:
    scenario, _ = build(tmp_path, DUMBBELL, tier=scene_quality.HARD)
    resolved = config_module.load_resolved(
        "configs/frozen_A.yaml", _write_scenario(tmp_path, scenario), config_root=CONFIG_ROOT
    )
    scene = experiment.build_scene(resolved.scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    result = experiment.plan_path(scene, setup)
    assert result.success
    assert result.clearance.min() >= setup.astar_params.d_min - 1e-9
    assert result.total_energy_j <= setup.astar_params.e_budget + 1e-9

    # 质量档案可以独立复算：同输入同输出
    segmentation = scene_quality.segment_rooms(scene.clearance, core_clearance_m=0.9)
    required = max(
        float(scenario["map"]["inflation_m"]), setup.astar_params.d_min
    )
    free = scene.clearance + 1e-9 >= required
    quality = scene_quality.build_quality(
        path_len_m=result.length_m,
        cells=result.cells,
        labels=segmentation.labels,
        clearance=scene.clearance,
        free_mask=free,
        class_grid=None,
        default_class="concrete",
    )
    assert quality.to_dict()["room_label_changes"] == scenario["quality"]["room_label_changes"]
    assert quality.to_dict()["path_len_m"] == pytest.approx(
        scenario["quality"]["path_len_m"], abs=1e-3
    )
    assert quality.to_dict()["grade"] == scenario["quality"]["grade"]


def test_same_seed_is_byte_identical(tmp_path: Path) -> None:
    first, _ = build(tmp_path, DUMBBELL, tier=scene_quality.HARD, seed=11)
    second, _ = build(tmp_path, DUMBBELL, tier=scene_quality.HARD, seed=11)
    # 同一 seed：几何、起终点与质量档案（跨门次数/房间数/难度/门洞宽度）逐字段一致
    assert first["quality"] == second["quality"]
    assert first == second
    assert first["content_hash"] == second["content_hash"]
    path_a = tmp_path / "a.yaml"
    path_b = tmp_path / "b.yaml"
    build_scenarios.write_scenario(first, path_a)
    build_scenarios.write_scenario(second, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_different_seed_moves_start_and_goal(tmp_path: Path) -> None:
    first, _ = build(tmp_path, DUMBBELL, tier=scene_quality.HARD, seed=1)
    second, _ = build(tmp_path, DUMBBELL, tier=scene_quality.HARD, seed=2)
    assert (first["start"], first["goal"]) != (second["start"], second["goal"])
    assert first["map"] == second["map"]
    assert first["terrain"] == second["terrain"]
    assert first["content_hash"] != second["content_hash"]


def test_narrow_floorplan_is_skipped(tmp_path: Path) -> None:
    with pytest.raises(build_scenarios.SkipScene):
        build(tmp_path, NARROW, tier=scene_quality.EASY)
    with pytest.raises(build_scenarios.SkipScene):
        build(tmp_path, NARROW, tier=scene_quality.MEDIUM)


def test_inflation_larger_than_d_min_is_rejected(tmp_path: Path) -> None:
    # d_min 低于物理半径 body_radius=0.58 必须被拒（解耦后这是唯一的物理下界）
    tight_algo = config_module.validate_algo(
        {
            "id": "tight",
            "astar": {"e_budget": 2000.0, "d_min": 0.5, "kappa_max": 1.5, "v_cruise": 1.0},
        }
    )
    with pytest.raises(config_module.ConfigError, match="body_radius"):
        build(tmp_path, ROOM, tier=scene_quality.EASY, algorithm=tight_algo)


def test_terrain_classes_match_runner_reconstruction(tmp_path: Path) -> None:
    """受控材质分区：``material_class_grid`` 与场景 ``terrain.patches`` 必须逐单元一致。

    这是"同一份材质不会被两处代码各算一遍"的锁定：生成器用
    ``env/terrain_partition.py`` 出分区，runner 用 ``terrain.patches`` 重建 ``c_rr``，
    两者如果采样规则不一致，路线上的地形占比就会是错的。
    """
    scenario, _ = build(tmp_path, ROOM, tier=scene_quality.EASY)
    scene = experiment.build_scene(scenario)
    mapping = {
        str(name): float(props["c_rr"])
        for name, props in (scenario["terrain"]["partition"]["classes"]).items()
    }
    grid = build_scenarios.terrain_partition_grid_from_scenario(scenario, scene)
    assert grid.shape == scene.c_rr.shape
    for class_name, c_rr in mapping.items():
        mask = grid == class_name
        if mask.any():
            assert np.allclose(scene.c_rr[mask], c_rr)
    # 四档材质都要在自由区里出现（受控因子覆盖论文材质表）
    free = ~scene.occupancy
    present = {float(value) for value in np.unique(scene.c_rr[free])}
    assert present == {0.015, 0.030, 0.040, 0.060}
    # 实际占比与目标占比的偏差必须很小（整块不可分带来的量化误差）
    realized = {
        name: float((grid[free] == name).mean()) for name in mapping
    }
    for name, target in scenario["terrain"]["partition"]["target_shares"].items():
        assert realized[name] == pytest.approx(float(target), abs=0.06)

    # 同一户型同一材质档两次生成必须逐字段一致（确定性）
    again, _ = build(tmp_path, ROOM, tier=scene_quality.EASY)
    assert again["terrain"] == scenario["terrain"]
    assert again["content_hash"] == scenario["content_hash"]


# ---------------------------------------------------------------------------
# 窄门组
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [0.8, 1.0, 1.2])
def test_narrow_door_scenarios(tmp_path: Path, width: float) -> None:
    platform, algo, rules = load_configs()
    resolution = 0.05
    scenario, info = build_scenarios.build_narrow_door(
        width,
        resolution=resolution,
        platform=platform,
        algo=algo,
        rules=rules,
        seed=5,
    )
    config_module.validate_scenario(scenario)
    quality = scenario["quality"]
    assert quality["group"] == "narrow_door"
    assert quality["narrow_door_width_m"] == pytest.approx(width)
    # 实测门洞宽度 = 设计宽度 - 一个单元（clearance 定义本身偏保守半格）
    assert quality["measured_door_width_m"] == pytest.approx(width - resolution, abs=resolution)
    assert quality["verified_clearance_m"] == pytest.approx(width / 2 - resolution)
    # 解耦后两个口径分开记录：物理半径 0.58 m、基准 d_min 0.70 m
    assert quality["physical_radius_m"] == pytest.approx(0.58)
    assert quality["base_d_min_m"] == pytest.approx(0.70)
    assert quality["feasible_at_physical_radius"] == (width >= 2 * 0.58)
    assert quality["feasible_at_base_d_min"] is False  # 三档都小于 2 x 0.70 m

    # 用验证口径规划必须成功；用 frozen_A 的 d_min 必须失败
    resolved = config_module.load_resolved(
        "configs/frozen_A.yaml", _write_scenario(tmp_path, scenario), config_root=CONFIG_ROOT
    )
    scene = experiment.build_scene(resolved.scenario)
    setup = experiment.planner_setup(resolved.platform, resolved.algo)
    assert not experiment.plan_path(scene, setup).success
    from dataclasses import replace

    narrow_setup = replace(
        setup,
        astar_params=replace(
            setup.astar_params, d_min=float(quality["verified_clearance_m"])
        ),
    )
    assert experiment.plan_path(scene, narrow_setup).success


def test_narrow_door_group_is_separate_from_houseexpo() -> None:
    narrow = sorted((CONFIG_ROOT / "scenarios").glob("narrowdoor_*.yaml"))
    assert len(narrow) == 3
    for path in narrow:
        scenario = config_module.read_config(path)
        assert scenario["quality"]["group"] == "narrow_door"


# ---------------------------------------------------------------------------
# 选取文件与 CLI
# ---------------------------------------------------------------------------


def test_repository_selection_file_is_valid() -> None:
    selection = build_scenarios.load_selection(CONFIG_ROOT / "scenario_selection.yaml")
    assert selection.min_path_len_m == pytest.approx(8.0)
    assert set(selection.tiers) == {
        scene_quality.EASY,
        scene_quality.MEDIUM,
        scene_quality.HARD,
    }
    for ids in selection.tiers.values():
        assert ids
        for plan_id in ids:
            assert len(plan_id) == 32


def test_cli_writes_scenarios_and_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_plan(data_dir, DUMBBELL, "dumbbell0001.json")
    selection_path = tmp_path / "selection.yaml"
    selection_path.write_text(
        "version: 1\n"
        "min_path_len_m: 8.0\n"
        "easy_min_path_len_m: 4.0\n"
        "tiers:\n"
        "  hard:\n"
        "    - dumbbell0001\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    metadata = tmp_path / "metadata.csv"
    exit_code = build_scenarios.main(
        [
            "--data-dir", str(data_dir),
            "--out-dir", str(out_dir),
            "--selection", str(selection_path),
            "--metadata-out", str(metadata),
            "--only", "hard",
        ]
    )
    assert exit_code == 0
    written = sorted(out_dir.glob("*.yaml"))
    assert len(written) == 1
    scenario = config_module.validate_scenario(config_module.read_config(written[0]))
    assert scenario["quality"]["grade"] == scene_quality.HARD

    with metadata.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["id"] == scenario["id"]
    assert rows[0]["grade"] == scene_quality.HARD
    assert float(rows[0]["bottleneck_width_m"]) < scene_quality.BOTTLENECK_HARD_M
    assert "generated 1 scenes" in capsys.readouterr().out
