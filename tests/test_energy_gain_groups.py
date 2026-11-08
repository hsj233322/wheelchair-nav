"""``analysis/energy_gain_groups.py`` 的纯函数测试：分组边界、符号约定、汇总口径。"""

from __future__ import annotations

import numpy as np
import pytest

from analysis import energy_gain_groups as egg


@pytest.mark.parametrize(
    "ratio,expected",
    [
        (0.0, "low"),
        (0.05, "low"),
        (0.0999, "low"),
        (0.10, "mid"),        # 左闭
        (0.25, "mid"),
        (0.3999, "mid"),
        (0.40, "high"),       # 左闭
        (1.0, "high"),        # 右端必须落在最后一组
    ],
)
def test_group_of_boundaries(ratio: float, expected: str) -> None:
    assert egg.group_of(ratio) == expected


def _gain(
    *,
    high_ratio: float,
    len_energy: float,
    esa_energy: float,
    len_length: float = 10.0,
    esa_length: float = 10.0,
) -> egg.TaskGain:
    delta_energy = esa_energy - len_energy
    return egg.TaskGain(
        scenario_id="s",
        grade="hard",
        terrain_tier="mixed",
        port="frozen",
        high_ratio=high_ratio,
        length_m=10.0,
        energy_j=0.0,
        esa_length_m=esa_length,
        esa_energy_j=esa_energy,
        len_length_m=len_length,
        len_energy_j=len_energy,
        delta_length_m=esa_length - len_length,
        delta_energy_j=delta_energy,
        delta_energy_pct=(100.0 * delta_energy / len_energy if len_energy else 0.0),
        c_rr_hist={},
    )


def test_gain_sign_convention() -> None:
    """收益 = 仅长度能耗 − ESA* 能耗：ESA* 更省电时收益为正。"""
    gain = _gain(high_ratio=0.9, len_energy=500.0, esa_energy=480.0)
    assert gain.gain_j == pytest.approx(20.0)
    assert gain.delta_energy_j == pytest.approx(-20.0)
    assert gain.delta_energy_pct < 0
    assert gain.group == "high"


def test_summarise_reports_median_and_total() -> None:
    """中位数与总和都要给，且**两者可以不同向** —— 这正是论文里要说明的点。

    构造的数据刻意复制真实数据的形态：低阻组有**一个很大的 tie-break 收益**加上
    一堆 0（中位 0、总和很大），高阻组则是**普遍的中等收益**（中位高、总和较小）。
    """
    gains = [
        # 低阻组：1 个大任务 + 4 个 0 收益 -> 中位 0，总和最大
        _gain(high_ratio=0.02, len_energy=1000.0, esa_energy=800.0),   # 收益 200 J
        _gain(high_ratio=0.03, len_energy=200.0, esa_energy=200.0),
        _gain(high_ratio=0.04, len_energy=200.0, esa_energy=200.0),
        _gain(high_ratio=0.06, len_energy=200.0, esa_energy=200.0),
        _gain(high_ratio=0.08, len_energy=200.0, esa_energy=200.0),
        # 中阻组
        _gain(high_ratio=0.20, len_energy=300.0, esa_energy=290.0),    # 收益 10 J
        _gain(high_ratio=0.30, len_energy=300.0, esa_energy=290.0),    # 收益 10 J
        # 高阻组：普遍 20 J
        _gain(high_ratio=0.50, len_energy=400.0, esa_energy=380.0),
        _gain(high_ratio=0.70, len_energy=400.0, esa_energy=380.0),
        _gain(high_ratio=0.95, len_energy=400.0, esa_energy=380.0),
    ]
    summaries = {row.group: row for row in egg.summarise(gains)}

    low = summaries["low"]
    assert low.tasks == 5
    assert low.gain_median_j == pytest.approx(0.0)
    assert low.gain_total_j == pytest.approx(200.0)
    assert low.high_ratio_min == pytest.approx(0.02)
    assert low.high_ratio_max == pytest.approx(0.08)
    assert low.improved_tasks == 1

    mid = summaries["mid"]
    assert mid.tasks == 2
    assert mid.gain_median_j == pytest.approx(10.0)
    assert mid.gain_total_j == pytest.approx(20.0)

    high = summaries["high"]
    assert high.tasks == 3
    assert high.gain_median_j == pytest.approx(20.0)
    assert high.gain_total_j == pytest.approx(60.0)
    assert high.high_ratio_min == pytest.approx(0.50)
    assert high.high_ratio_max == pytest.approx(0.95)

    # 中位收益随高阻占比单调上升；总和却不是（低阻组总和最大）
    medians = [summaries[name].gain_median_j for name in ("low", "mid", "high")]
    totals = [summaries[name].gain_total_j for name in ("low", "mid", "high")]
    assert medians == sorted(medians) and medians[0] < medians[-1], "中位收益应单调上升"
    assert totals[0] == max(totals), "总和被单个大任务带偏：低阻组反而最大"


def test_summarise_empty_group_is_reported_as_zero() -> None:
    summaries = {row.group: row for row in egg.summarise([_gain(high_ratio=0.5, len_energy=10.0, esa_energy=9.0)])}
    assert set(summaries) == {"low", "mid", "high"}
    assert summaries["low"].tasks == 0
    assert summaries["low"].gain_total_j == 0.0
    assert summaries["high"].tasks == 1


def test_row_for_contains_all_columns() -> None:
    gain = _gain(high_ratio=0.5, len_energy=100.0, esa_energy=95.0)
    row = egg.row_for(gain)
    for column in egg.CSV_COLUMNS:
        assert column in row, f"缺少列 {column}"
    assert row["group"] == "high"
    assert float(row["gain_j"]) == pytest.approx(5.0)
    assert float(row["hist_c_rr_0.060"]) == 0.0


def test_summary_row_and_format_table() -> None:
    summaries = egg.summarise([_gain(high_ratio=0.5, len_energy=100.0, esa_energy=95.0)])
    row = egg.summary_row(summaries[2])
    assert row["group"] == "high"
    table = egg.format_table(summaries, tasks=1, label="P_aux = 15 W")
    assert "高（> 40%）" in table
    assert "P_aux = 15 W" in table


def test_group_edges_cover_full_range() -> None:
    """三组必须覆盖 [0, 1] 且互不重叠。"""
    edges = egg.GROUP_EDGES
    assert edges[0][1] == 0.0
    assert edges[-1][2] > 1.0
    for (name_a, _lo_a, hi_a), (name_b, lo_b, _hi_b) in zip(edges, edges[1:]):
        assert hi_a == pytest.approx(lo_b), f"{name_a} 与 {name_b} 之间有缝隙或重叠"
    assert egg.HIGH_C_RR == pytest.approx(0.04)


def test_delta_percent_matches_definition() -> None:
    gain = _gain(high_ratio=0.5, len_energy=200.0, esa_energy=190.0)
    assert gain.delta_energy_pct == pytest.approx(-5.0)
    assert np.isclose(gain.gain_j, 10.0)
