"""analysis/dmin_sweep.py 的测试：固定起终点 vs 按档位重新采样。"""

import csv
import shutil
from pathlib import Path

from analysis import dmin_sweep

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config"
SWEEP_DIR = CONFIG_ROOT / "configs" / "dmin_sweep"
# 中档场景（必经通道 1.70 m）：0.90 档下冻结端点不合法，但同分量内仍有中档任务
MEDIUM_SCENE = CONFIG_ROOT / "scenarios" / "houseexpo_0df8d098.yaml"
# 难档场景（必经通道 1.50 m）：0.90 档需要 1.80 m，物理上不存在该档任务
HARD_SCENE = CONFIG_ROOT / "scenarios" / "houseexpo_5427f9b0.yaml"


def _two_level_configs(tmp_path: Path) -> Path:
    directory = tmp_path / "sweep"
    directory.mkdir()
    for name in ("dmin_070.yaml", "dmin_090.yaml"):
        shutil.copy(SWEEP_DIR / name, directory / name)
    return directory


def test_resampling_recovers_tasks_hidden_by_frozen_endpoints(tmp_path: Path) -> None:
    rows = dmin_sweep.run_sweep(
        config_dir=_two_level_configs(tmp_path),
        scenario_paths=[MEDIUM_SCENE],
        config_root=CONFIG_ROOT,
        max_samples=12,
    )
    by_level = {float(row["d_min_m"]): row for row in rows}
    assert set(by_level) == {0.70, 0.90}

    # 基准档：固定起终点与重采样都可行
    assert by_level[0.70]["frozen_feasible"] == 1
    assert by_level[0.70]["resampled_feasible"] == 1
    assert by_level[0.70]["resample_tier_ok"] == 1

    # 0.90 档：冻结任务不可用（端点或路径不满足新档），
    # 同一分量内重新采样后仍有中档任务（必经通道 >= 2 x 0.90），说明失败来自冻结任务
    assert by_level[0.90]["frozen_feasible"] == 0
    assert by_level[0.90]["frozen_reason"] in {
        "start_violates_clearance",
        "goal_violates_clearance",
        "no_path",
    }
    assert by_level[0.90]["resample_tier_ok"] == 1
    assert by_level[0.90]["resampled_feasible"] == 1
    assert by_level[0.90]["resampled_bottleneck_width_m"] >= 2 * 0.90 - 0.1


def test_hard_scene_has_no_task_at_090(tmp_path: Path) -> None:
    rows = dmin_sweep.run_sweep(
        config_dir=_two_level_configs(tmp_path),
        scenario_paths=[HARD_SCENE],
        config_root=CONFIG_ROOT,
        max_samples=12,
    )
    by_level = {float(row["d_min_m"]): row for row in rows}
    assert by_level[0.70]["resampled_feasible"] == 1
    # 1.50 m 的必经通道 < 2 x 0.90 m：该档物理上不存在"难"任务
    assert by_level[0.90]["frozen_feasible"] == 0
    assert by_level[0.90]["resample_tier_ok"] == 0
    assert by_level[0.90]["resampled_feasible"] == 0
    assert by_level[0.90]["resampled_bottleneck_width_m"] == 0.0


def test_summary_reports_both_modes(tmp_path: Path) -> None:
    rows = dmin_sweep.run_sweep(
        config_dir=_two_level_configs(tmp_path),
        scenario_paths=[MEDIUM_SCENE],
        config_root=CONFIG_ROOT,
        max_samples=6,
    )
    summary = dmin_sweep.summarize(rows)
    assert [item["d_min_m"] for item in summary] == [0.70, 0.90]
    for item in summary:
        assert item["n_total"] == 1
        assert 0 <= item["frozen_feasible"] <= 1
        assert 0 <= item["resampled_feasible"] <= 1
        assert item["resampled_feasible"] <= item["resample_tier_ok"]
    table = dmin_sweep.format_table(summary)
    assert "固定:可行" in table and "重采样:成功率" in table


def test_sweep_cli_writes_wide_csv(tmp_path: Path) -> None:
    out_path = tmp_path / "sweep.csv"
    exit_code = dmin_sweep.main(
        [
            "--configs", str(_two_level_configs(tmp_path)),
            "--scenarios", str(MEDIUM_SCENE),
            "--out", str(out_path),
            "--max-samples", "6",
        ]
    )
    assert exit_code == 0
    with out_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert set(rows[0]) == set(dmin_sweep.CSV_COLUMNS)
    # 两套列都在：固定起终点 + 按档位重新采样
    assert {"frozen_feasible", "frozen_path_len_m"} <= set(rows[0])
    assert {"resample_tier_ok", "resampled_feasible", "resampled_start"} <= set(rows[0])
    assert all(row["d_min_m"] in {"0.70", "0.90"} for row in rows)
