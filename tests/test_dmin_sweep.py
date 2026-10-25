"""analysis/dmin_sweep.py 的测试：d_min 档位扫描与汇总表。"""

import csv
from pathlib import Path

import pytest

from analysis import dmin_sweep

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config"
SWEEP_DIR = CONFIG_ROOT / "configs" / "dmin_sweep"
EASY_SCENE = CONFIG_ROOT / "scenarios" / "houseexpo_0d78ffe6.yaml"
HARD_SCENE = CONFIG_ROOT / "scenarios" / "houseexpo_5427f9b0.yaml"
SCENES = [EASY_SCENE, HARD_SCENE]


def test_sweep_covers_four_levels_and_reports_feasibility() -> None:
    rows = dmin_sweep.run_sweep(
        config_dir=SWEEP_DIR, scenario_paths=SCENES, config_root=CONFIG_ROOT
    )
    assert len(rows) == 4 * len(SCENES)
    assert sorted({row["d_min_m"] for row in rows}) == [0.58, 0.70, 0.90, 1.10]

    hard = {
        row["d_min_m"]: row for row in rows if row["scenario"] == "houseexpo_5427f9b0"
    }
    # 该场景的必经走廊宽 1.5 m：d_min <= 0.70 可行，>= 0.90 不可行
    assert hard[0.58]["feasible"] == 1
    assert hard[0.70]["feasible"] == 1
    assert hard[0.90]["feasible"] == 0
    assert hard[1.10]["feasible"] == 0
    assert hard[0.90]["reason"] == "start_violates_clearance"

    # 场景层只声明物理半径；d_min 放松后路径贴边程度下降
    assert hard[0.58]["clearance_min_m"] < hard[0.70]["clearance_min_m"]
    # 路径会贴着可行边界走，但 0.1 m 栅格 + 8 邻域限制了贴边精度：
    # d_min=0.58 时实测最小离障约 0.65 m（< 走廊半宽 0.75 m）
    assert 0.58 - 1e-9 <= hard[0.58]["clearance_min_m"] <= 0.75
    assert hard[0.58]["bottleneck_width_m"] == pytest.approx(1.5, abs=0.1)


def test_sweep_summary_is_monotone_in_feasibility() -> None:
    rows = dmin_sweep.run_sweep(
        config_dir=SWEEP_DIR, scenario_paths=SCENES, config_root=CONFIG_ROOT
    )
    summary = dmin_sweep.summarize(rows)
    assert [item["d_min_m"] for item in summary] == [0.58, 0.70, 0.90, 1.10]
    counts = [item["n_feasible"] for item in summary]
    assert counts == sorted(counts, reverse=True)
    table = dmin_sweep.format_table(summary)
    assert "0.58" in table and "1.10" in table and "可行场景" in table


def test_sweep_cli_writes_csv(tmp_path: Path) -> None:
    out_path = tmp_path / "sweep.csv"
    exit_code = dmin_sweep.main(
        [
            "--configs", str(SWEEP_DIR),
            "--scenarios", str(EASY_SCENE),
            "--out", str(out_path),
        ]
    )
    assert exit_code == 0
    with out_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert set(rows[0]) == set(dmin_sweep.CSV_COLUMNS)
    assert all(int(row["feasible"]) == 1 for row in rows)
