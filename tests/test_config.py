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
