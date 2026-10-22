import csv
import json
from pathlib import Path

import numpy as np
import pytest

from runner import config as config_module
from runner import experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config"
ALGO = "configs/frozen_A.yaml"
SCENARIO = "scenarios/house_01.yaml"


def read_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_scenario(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_run_writes_artifacts_and_succeeds(tmp_path: Path) -> None:
    result = experiment.run_experiment(
        ALGO, SCENARIO, config_root=CONFIG_ROOT, out_root=tmp_path
    )
    assert result.success
    run_dir = result.run_dir
    for name in ("config.json", "metrics.csv", "trajectory.npz"):
        assert (run_dir / name).is_file(), name

    rows = read_rows(run_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["config_hash"] == result.config_hash
    assert row["success"] == "1"
    assert row["reason"] == "ok"
    assert float(row["constraint_violation_rate"]) == 0.0
    assert float(row["min_clearance_m"]) >= 0.7 - 1e-9
    assert float(row["energy_j"]) <= 2000.0 + 1e-9
    assert float(row["path_length_m"]) > 5.0

    trajectory = np.load(run_dir / "trajectory.npz")
    assert trajectory["xy"].shape[0] > 50
    assert trajectory["t"].shape == trajectory["xy"].shape[:1]
    assert np.all(np.diff(trajectory["t"]) > 0)
    assert np.all(np.diff(trajectory["energy_j"]) >= -1e-9)


def test_config_snapshot_matches_the_config_used(tmp_path: Path) -> None:
    resolved = config_module.load_resolved(
        ALGO, SCENARIO, config_root=CONFIG_ROOT
    )
    result = experiment.run_experiment(
        ALGO, SCENARIO, config_root=CONFIG_ROOT, out_root=tmp_path
    )
    snapshot = json.loads((result.run_dir / "config.json").read_text(encoding="utf-8"))
    assert snapshot["platform"] == resolved.platform
    assert snapshot["scenario"] == resolved.scenario
    assert snapshot["algo"] == resolved.algo
    assert snapshot["run"] == resolved.run
    assert snapshot["config_hash"] == resolved.config_hash
    assert snapshot["run"]["seed"] == result.seed


def test_same_seed_reproduces_metrics_exactly(tmp_path: Path) -> None:
    first = experiment.run_experiment(
        ALGO, SCENARIO, config_root=CONFIG_ROOT, out_root=tmp_path / "first"
    )
    second = experiment.run_experiment(
        ALGO, SCENARIO, config_root=CONFIG_ROOT, out_root=tmp_path / "second"
    )
    first_row = read_rows(first.run_dir)[0]
    second_row = read_rows(second.run_dir)[0]
    for key in first_row:
        if key == "run_id":
            continue
        assert first_row[key] == second_row[key], key

    left = np.load(first.run_dir / "trajectory.npz")
    right = np.load(second.run_dir / "trajectory.npz")
    for name in left.files:
        assert np.array_equal(left[name], right[name]), name


def test_unsolvable_scene_still_writes_a_row(tmp_path: Path) -> None:
    sealed = write_scenario(
        tmp_path / "sealed.yaml",
        "id: sealed\n"
        "description: 一道贯通墙，保证无解\n"
        "map:\n"
        "  resolution: 0.5\n"
        "  shape: [12, 12]\n"
        "  walls:\n"
        "    - [3.0, 0.0, 3.5, 6.0]\n"
        "start: [1.0, 3.0]\n"
        "goal: [4.5, 3.0]\n"
        "terrain:\n"
        "  default:\n"
        "    c_rr: 0.015\n"
        "    slope_deg: 0.0\n"
        "  patches: []\n",
    )
    result = experiment.run_experiment(
        ALGO, sealed, config_root=CONFIG_ROOT, out_root=tmp_path / "out"
    )
    assert not result.success
    row = read_rows(result.run_dir)[0]
    assert row["success"] == "0"
    assert row["reason"] == "no_path"
    assert float(row["path_length_m"]) == 0.0
    trajectory = np.load(result.run_dir / "trajectory.npz")
    assert trajectory["xy"].shape == (0, 2)


def test_start_inside_wall_is_a_config_error(tmp_path: Path) -> None:
    broken = write_scenario(
        tmp_path / "broken.yaml",
        "id: broken\n"
        "map:\n"
        "  resolution: 0.5\n"
        "  shape: [10, 10]\n"
        "  walls:\n"
        "    - [0.0, 0.0, 1.0, 1.0]\n"
        "start: [0.5, 0.5]\n"
        "goal: [4.0, 4.0]\n"
        "terrain:\n"
        "  default:\n"
        "    c_rr: 0.015\n"
        "    slope_deg: 0.0\n",
    )
    with pytest.raises(config_module.ConfigError, match="障碍物"):
        experiment.run_experiment(
            ALGO, broken, config_root=CONFIG_ROOT, out_root=tmp_path / "out"
        )


def test_cli_accepts_paths_relative_to_config_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = experiment.main(
        [ALGO, SCENARIO, "--out", str(tmp_path / "cli")]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "config_hash:" in output
    assert len(list((tmp_path / "cli").glob("*/metrics.csv"))) == 1


def test_repeats_use_consecutive_seeds(tmp_path: Path) -> None:
    result = experiment.run_experiment(
        ALGO,
        SCENARIO,
        config_root=CONFIG_ROOT,
        out_root=tmp_path,
        seed=100,
        repeats=2,
    )
    rows = read_rows(result.run_dir)
    assert [int(row["seed"]) for row in rows] == [100, 101]
    assert (result.run_dir / "trajectory_001.npz").is_file()
