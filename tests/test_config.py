from pathlib import Path

import pytest

from runner import config as config_module
from runner.config import ConfigError

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config"
ALGO = "configs/frozen_A.yaml"
SCENARIO = "scenarios/house_01.yaml"


def load(**kwargs: object) -> config_module.ResolvedConfig:
    return config_module.load_resolved(
        ALGO, SCENARIO, config_root=CONFIG_ROOT, **kwargs  # type: ignore[arg-type]
    )


def test_three_layers_are_loaded() -> None:
    resolved = load()
    assert resolved.platform["mass"] == 100.0
    assert resolved.platform["battery_voltage"] == 24.0
    assert resolved.scenario["id"] == "house_01"
    assert resolved.scenario["map"]["resolution"] == 0.2
    assert resolved.algo["id"] == "frozen_A"
    assert resolved.algo["astar"]["e_budget"] == 2000.0
    assert resolved.run["seed"] == 42
    assert len(resolved.config_hash) == 64


def test_hash_is_stable_and_seed_is_not_part_of_it() -> None:
    first = load()
    second = load(seed=7, repeats=3)
    assert first.config_hash == second.config_hash
    assert second.run["seed"] == 7
    assert second.run["repeats"] == 3


def test_hash_changes_when_model_layers_change() -> None:
    resolved = load()
    changed_platform = dict(resolved.platform, mass=120.0)
    changed_algo = {
        **resolved.algo,
        "astar": {**resolved.algo["astar"], "e_budget": 1500.0},
    }
    assert (
        config_module.compute_config_hash(
            changed_platform, resolved.scenario, resolved.algo
        )
        != resolved.config_hash
    )
    assert (
        config_module.compute_config_hash(
            resolved.platform, resolved.scenario, changed_algo
        )
        != resolved.config_hash
    )


def test_snapshot_matches_resolved_layers() -> None:
    resolved = load()
    snapshot = resolved.snapshot(git_commit="deadbeef", git_dirty=False)
    assert snapshot["platform"] == resolved.platform
    assert snapshot["scenario"] == resolved.scenario
    assert snapshot["algo"] == resolved.algo
    assert snapshot["run"] == resolved.run
    assert snapshot["config_hash"] == resolved.config_hash
    assert snapshot["git"] == {"commit": "deadbeef", "dirty": False}


def test_relative_inputs_resolve_against_config_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resolved_path = config_module.resolve_input(ALGO, CONFIG_ROOT)
    assert resolved_path == (CONFIG_ROOT / ALGO).resolve()


def test_missing_config_file_is_reported() -> None:
    with pytest.raises(ConfigError, match="找不到配置文件"):
        config_module.load_resolved(
            "configs/does_not_exist.yaml", SCENARIO, config_root=CONFIG_ROOT
        )


def test_algo_layer_cannot_set_platform_parameters(tmp_path: Path) -> None:
    bad = tmp_path / "bad_algo.yaml"
    bad.write_text(
        "id: bad\n"
        "mass: 120.0\n"
        "astar:\n"
        "  e_budget: 1000.0\n"
        "  d_min: 0.7\n"
        "  kappa_max: 1.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="未知字段"):
        config_module.load_resolved(
            bad,
            SCENARIO,
            config_root=CONFIG_ROOT,
        )


def test_unknown_platform_field_is_rejected(tmp_path: Path) -> None:
    text = (CONFIG_ROOT / "platform.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "platform.yaml"
    bad.write_text(text + "wheels: 4\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="wheels"):
        config_module.load_resolved(
            ALGO,
            SCENARIO,
            platform_path=bad,
            config_root=CONFIG_ROOT,
        )


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "id: first\n"
        "id: second\n"
        "astar:\n"
        "  e_budget: 1000.0\n"
        "  d_min: 0.7\n"
        "  kappa_max: 1.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="重复的配置键"):
        config_module.load_resolved(bad, SCENARIO, config_root=CONFIG_ROOT)


def test_d_min_below_body_radius_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad_dmin.yaml"
    bad.write_text(
        "id: bad\n"
        "astar:\n"
        "  e_budget: 1000.0\n"
        "  d_min: 0.1\n"
        "  kappa_max: 1.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="body_radius"):
        config_module.load_resolved(bad, SCENARIO, config_root=CONFIG_ROOT)


def test_cruise_speed_above_platform_limit_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad_speed.yaml"
    bad.write_text(
        "id: bad\n"
        "astar:\n"
        "  e_budget: 1000.0\n"
        "  d_min: 0.7\n"
        "  kappa_max: 1.5\n"
        "  v_cruise: 2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="v_max"):
        config_module.load_resolved(bad, SCENARIO, config_root=CONFIG_ROOT)


def test_astar_effective_params_fill_defaults() -> None:
    resolved = load()
    effective = config_module.astar_effective_params(resolved.algo, resolved.platform)
    assert effective["v_cruise"] == 1.0
    assert effective["kappa_window_m"] == 0.6
    assert effective["max_labels_per_state"] == 64

    bare = {
        "id": "bare",
        "astar": {"e_budget": 100.0, "d_min": 0.7, "kappa_max": 1.0},
    }
    defaults = config_module.astar_effective_params(bare, resolved.platform)
    assert defaults["v_cruise"] == resolved.platform["v_max"]
    assert defaults["kappa_window_m"] == 0.6


# ---------------------------------------------------------------------------
# HouseExpo 场景：轮廓多边形、场景内容哈希、膨胀半径
# ---------------------------------------------------------------------------

HOUSEEXPO_SCENARIO = CONFIG_ROOT / "scenarios" / "houseexpo_a24e5d6b.yaml"


def test_polygon_scenario_loads_with_inflation() -> None:
    resolved = config_module.load_resolved(
        ALGO, HOUSEEXPO_SCENARIO, config_root=CONFIG_ROOT
    )
    assert "polygons" in resolved.scenario["map"]
    assert "walls" not in resolved.scenario["map"]
    assert resolved.scenario["map"]["inflation_m"] == 0.7
    assert resolved.scenario["source"]["dataset"] == "HouseExpo"
    assert resolved.scenario["content_hash"]


def test_tampered_scenario_is_detected_by_content_hash() -> None:
    doc = config_module.read_config(HOUSEEXPO_SCENARIO)
    doc["start"] = [doc["start"][0] + 0.5, doc["start"][1]]
    with pytest.raises(ConfigError, match="content_hash"):
        config_module.validate_scenario(doc)


def test_content_hash_ignores_the_hash_field_itself() -> None:
    doc = config_module.read_config(HOUSEEXPO_SCENARIO)
    expected = config_module.compute_scenario_hash(doc)
    doc["content_hash"] = "whatever"
    assert config_module.compute_scenario_hash(doc) == expected


def test_walls_and_polygons_are_mutually_exclusive() -> None:
    base = config_module.read_config(CONFIG_ROOT / "scenarios" / "house_01.yaml")
    base["map"]["polygons"] = [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]
    with pytest.raises(ConfigError, match="walls"):
        config_module.validate_scenario(base)

    without_walls = config_module.read_config(CONFIG_ROOT / "scenarios" / "house_01.yaml")
    del without_walls["map"]["walls"]
    with pytest.raises(ConfigError, match="walls"):
        config_module.validate_scenario(without_walls)


def test_degenerate_polygon_is_rejected() -> None:
    base = config_module.read_config(HOUSEEXPO_SCENARIO)
    base["map"]["polygons"] = [[[0.0, 0.0], [1.0, 1.0]]]
    base["content_hash"] = config_module.compute_scenario_hash(base)
    with pytest.raises(ConfigError, match="顶点"):
        config_module.validate_scenario(base)


def test_inflation_above_d_min_is_rejected_by_cross_check() -> None:
    platform = config_module.validate_platform(
        config_module.read_config(CONFIG_ROOT / "platform.yaml")
    )
    algo = config_module.validate_algo(
        config_module.read_config(CONFIG_ROOT / "configs" / "frozen_A.yaml")
    )
    scenario = config_module.read_config(HOUSEEXPO_SCENARIO)
    scenario["map"]["inflation_m"] = 0.9  # > algo.astar.d_min = 0.7
    scenario["content_hash"] = config_module.compute_scenario_hash(scenario)
    with pytest.raises(ConfigError, match="inflation_m"):
        config_module.validate_cross(platform, scenario, algo)
