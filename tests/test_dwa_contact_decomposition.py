"""接触帧二维分解与会车净宽判据的测试（2026-10-31 口径）。

覆盖：

* ``decompose_contact`` 的 2×2 分类（成因 × 几何可行性）与 |v| 分箱；
* 会车净宽闭式判据 ``W_required = 2(R + r_o)`` = 1.76 m，且**与算法无关**；
* 各档瓶颈宽度与 1.76 m 的比较（真实场景文件）；
* 汇总表的列与"主动碰撞必须为 0"的可判读性。
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis import dwa as analysis_dwa

MEETING_WIDTH = 1.76
CONTACT_GAP = 0.88


# ---------------------------------------------------------------------------
# 纯函数：二维分解
# ---------------------------------------------------------------------------


def test_decompose_contact_splits_by_motion_and_geometry() -> None:
    """接触帧必须同时按「本车是否在动」与「瓶颈是否允许会车」分开计数。"""
    # 5 个周期：接触 4 帧（第 0 帧不接触），其中 2 帧在动、2 帧已停
    gaps = np.array([1.20, 0.50, 0.60, 0.40, 0.70])
    speeds = np.array([0.50, 0.50, 0.00, 0.00, 0.08])  # 第 4 帧 0.08 > 0.01 = 在动
    breakdown = analysis_dwa.decompose_contact(
        gaps,
        speeds,
        contact_gap_m=CONTACT_GAP,
        bottleneck_width_m=1.50,  # < 1.76 -> 几何不可会车
        meeting_width_required_m=MEETING_WIDTH,
    )
    # 接触帧：index 1,2,3,4 -> 在动：1(0.50)、4(0.08)；已停：2,3
    assert breakdown.frames == 4
    assert breakdown.moving_frames == 2
    assert breakdown.stopped_frames == 2
    assert breakdown.narrow_frames == 4
    assert breakdown.wide_frames == 0
    assert breakdown.moving_narrow == 2
    assert breakdown.stopped_narrow == 2
    assert breakdown.moving_wide == 0
    assert breakdown.stopped_wide == 0
    assert breakdown.has_active_collision is True
    assert breakdown.stopped_narrow_share == pytest.approx(0.5)
    # |v| 分箱：<=0.01 -> 2 帧（v=0）；0.01~0.10 -> 1 帧（0.08）；>0.30 -> 1 帧（0.50）
    # **v 恰为 0 的帧必须落在第一箱**（用 v > 0 做左端会把它们整帧漏掉）
    assert breakdown.speed_bin_counts == (2, 1, 0, 1)
    assert sum(breakdown.speed_bin_counts) == breakdown.frames


def test_decompose_contact_wide_corridor_is_planner_avoidable() -> None:
    """瓶颈 >= 1.76 m 时，接触一律归入「规划器本可避免」。"""
    gaps = np.array([0.80, 0.80])
    speeds = np.array([0.00, 0.30])
    breakdown = analysis_dwa.decompose_contact(
        gaps,
        speeds,
        contact_gap_m=CONTACT_GAP,
        bottleneck_width_m=2.40,  # >= 1.76 -> 可以会车
        meeting_width_required_m=MEETING_WIDTH,
    )
    assert breakdown.frames == 2
    assert breakdown.narrow_frames == 0
    assert breakdown.wide_frames == 2
    assert breakdown.stopped_wide == 1
    assert breakdown.moving_wide == 1
    assert breakdown.stopped_frames == 1


def test_decompose_contact_empty_and_missing_width() -> None:
    """空轨迹返回全 0；瓶颈未知（nan）不得被当成"几何不可避"。"""
    empty = analysis_dwa.decompose_contact(
        np.zeros(0),
        np.zeros(0),
        contact_gap_m=CONTACT_GAP,
        bottleneck_width_m=1.20,
        meeting_width_required_m=MEETING_WIDTH,
    )
    assert empty.frames == 0 and empty.speed_bin_counts == (0, 0, 0, 0)
    unknown = analysis_dwa.decompose_contact(
        np.array([0.50]),
        np.array([0.00]),
        contact_gap_m=CONTACT_GAP,
        bottleneck_width_m=float("nan"),
        meeting_width_required_m=MEETING_WIDTH,
    )
    assert unknown.frames == 1
    assert unknown.narrow_frames == 0, "瓶颈未知时不得声称几何不可避"
    assert unknown.wide_frames == 1


def test_moving_definition_excludes_standstill_only() -> None:
    """「已停」判据是 |v| <= 0.01 m/s（标称速度的 1%），不是严格 0。

    实测中 DWA 候选最低速度是 ``a_max * dt = 0.08 m/s``：若把 0.08 也算"在动"，
    蠕动帧会被记成主动碰撞；若把阈值抬到 0.1，真正在动的低速帧又会被算成被动。
    """
    assert analysis_dwa.V_STOPPED_MPS == pytest.approx(0.01)
    for speed, expected_moving in ((0.0, False), (0.01, False), (0.08, True)):
        breakdown = analysis_dwa.decompose_contact(
            np.array([0.50]),
            np.array([speed]),
            contact_gap_m=CONTACT_GAP,
            bottleneck_width_m=2.0,
            meeting_width_required_m=MEETING_WIDTH,
        )
        assert breakdown.moving_frames == (1 if expected_moving else 0)


# ---------------------------------------------------------------------------
# 会车净宽判据与各档瓶颈宽度
# ---------------------------------------------------------------------------


def test_meeting_width_formula() -> None:
    """``W_required = 2 * (body_radius + r_o)`` = 2 × 0.88 = 1.76 m。"""
    assert analysis_dwa.MEETING_WIDTH_FACTOR == pytest.approx(2.0)
    assert analysis_dwa.MEETING_WIDTH_FACTOR * (0.58 + 0.30) == pytest.approx(
        MEETING_WIDTH
    )


def test_bottleneck_comparison_matches_tiers() -> None:
    """各档瓶颈与会车净宽 1.76 m 的比较（真实场景文件，与算法无关）。

    * easy：全部 >= 1.76（最窄 2.405 m），可会车；
    * medium：全部 < 1.76（1.620~1.711 m），**几何上无法会车**；
    * hard：全部 < 1.76（1.46~1.50 m），**几何上无法会车**；
    * 窄门组单独成档，不得并进 hard（其 grade 字段也是 "hard"）。
    """
    rows = analysis_dwa.bottleneck_comparison_rows()
    by_tier = {row["tier"]: row for row in rows}
    assert {"easy", "medium", "hard", "narrowdoor"} <= set(by_tier)

    easy = by_tier["easy"]
    assert int(easy["tasks"]) == 8
    assert float(easy["bottleneck_min_m"]) == pytest.approx(2.405, abs=1e-3)
    assert int(easy["tasks_below_meeting_width"]) == 0

    for tier, low, high in (("medium", 1.62, 1.712), ("hard", 1.45, 1.51)):
        row = by_tier[tier]
        assert int(row["tasks_below_meeting_width"]) == int(row["tasks"]), (
            f"{tier} 档应当**全部**任务都无法会车"
        )
        assert low <= float(row["bottleneck_min_m"]) <= float(row["bottleneck_max_m"]) <= high
        # 会车净宽确实是判据本身，不是从算法参数推的
        assert float(row["meeting_width_required_m"]) == pytest.approx(MEETING_WIDTH)

    narrowdoor = by_tier["narrowdoor"]
    assert int(narrowdoor["tasks"]) == 3
    assert int(narrowdoor["tasks_below_meeting_width"]) == 3


def test_contact_table_reports_active_collision_separately() -> None:
    """汇总表必须把主动碰撞与被动接触分列，并给出「停 + 窄」占比。"""

    class _FakeEpisode:
        def __init__(self, grade: str, kind: str, breakdown) -> None:
            self.grade = grade
            self.obstacle_kind = kind
            self.trackable = True
            self.periods = 10
            self.bottleneck_width_m = 1.50
            self.meeting_width_required_m = MEETING_WIDTH
            self.contact = breakdown

    narrow = analysis_dwa.ContactBreakdown(
        frames=6, moving_narrow=0, moving_wide=0, stopped_narrow=6,
        stopped_wide=0, speed_bin_counts=(6, 0, 0, 0),
    )
    wide = analysis_dwa.ContactBreakdown(
        frames=2, moving_narrow=0, moving_wide=2, stopped_narrow=0,
        stopped_wide=0, speed_bin_counts=(0, 0, 2, 0),
    )
    episodes = [
        _FakeEpisode("hard", "head_on", narrow),
        _FakeEpisode("easy", "crossing", wide),
    ]
    rows = analysis_dwa.contact_decomposition_rows(episodes)  # type: ignore[arg-type]
    overall = next(
        row for row in rows if row["group"] == "all" and row["obstacle_kind"] == "all"
    )
    assert overall["contact_frames"] == 8
    assert overall["moving_frames"] == 2
    assert overall["stopped_frames"] == 6
    assert overall["stopped_narrow"] == 6
    assert overall["tasks_with_active_collision"] == 1
    assert overall["tasks_with_passive_contact"] == 1
    table = analysis_dwa.format_contact_tables(rows)
    assert "主动" in table and "被动" in table
    assert "1.76" in table, "表里必须写出会车净宽判据的数值"
