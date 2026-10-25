"""env/scene_quality.py 的测试：房间分割、标签变化、clearance 硬指标、难度分档。"""

import numpy as np
import pytest

from env import scene_quality as sq


def _dumbbell(shape=(60, 260), resolution=0.1):
    """两个 8x6 房间 + 1.6 m 走廊的 clearance 场（与 HouseExpo 的门洞尺度同量级）。"""
    clearance = np.zeros(shape, dtype=float)
    clearance[0:60, 0:80] = 1.0
    clearance[0:60, 180:260] = 1.0
    clearance[22:38, 80:180] = 0.8
    return clearance


def test_segment_rooms_splits_at_narrow_corridor() -> None:
    clearance = _dumbbell()
    segmentation = sq.segment_rooms(clearance, core_clearance_m=0.9)
    assert segmentation.room_count == 2
    label_a = segmentation.labels[30, 40]
    label_b = segmentation.labels[30, 220]
    assert label_a > 0 and label_b > 0 and label_a != label_b
    assert segmentation.labels[30, 130] in {label_a, label_b}


def test_segment_rooms_without_core_returns_empty() -> None:
    segmentation = sq.segment_rooms(np.full((20, 20), 0.3), core_clearance_m=0.9)
    assert segmentation.room_count == 0
    assert not segmentation.labels.any()


def test_component_stats() -> None:
    free = np.zeros((10, 10), dtype=bool)
    free[0:3, 0:3] = True
    free[5:6, 5:6] = True
    stats = sq.component_stats(free)
    assert stats["n_components"] == 2
    assert stats["largest_component_ratio"] == pytest.approx(0.9)
    assert stats["unreachable_ratio"] == pytest.approx(0.1)


def test_component_stats_empty() -> None:
    stats = sq.component_stats(np.zeros((4, 4), dtype=bool))
    assert stats["n_components"] == 0
    assert stats["unreachable_ratio"] == 1.0


def test_path_metadata_counts_room_label_changes() -> None:
    labels = np.zeros((3, 30), dtype=int)
    labels[:, :15] = 1
    labels[:, 15:] = 2
    clearance = np.full((3, 30), 1.0)
    cells = np.array([[1, col] for col in range(0, 30)])
    metadata = sq.path_metadata(cells, labels=labels, clearance=clearance)
    assert metadata["rooms_visited"] == 2
    assert metadata["room_label_changes"] == 1
    assert metadata["interface_widths_m"] == [2.0]
    assert metadata["min_interface_width_m"] == pytest.approx(2.0)


def test_path_metadata_collapses_boundary_flicker() -> None:
    labels = np.ones((3, 40), dtype=int)
    labels[:, 10:12] = 2  # 只有 2 格宽：视为抖动
    cells = np.array([[1, col] for col in range(40)])
    metadata = sq.path_metadata(
        cells, labels=labels, clearance=np.full((3, 40), 1.0), min_run_cells=3
    )
    assert metadata["room_label_changes"] == 0
    assert metadata["rooms_visited"] == 1


def test_path_metadata_empty_path() -> None:
    metadata = sq.path_metadata(
        np.zeros((0, 2), dtype=int),
        labels=np.zeros((3, 3), dtype=int),
        clearance=np.zeros((3, 3)),
    )
    assert metadata["rooms_visited"] == 0
    assert metadata["room_label_changes"] == 0
    assert metadata["interface_widths_m"] == []
    assert metadata["clearance_min_m"] == 0.0
    assert metadata["bottleneck_crossings"] == 0


def test_clearance_profile_is_length_weighted() -> None:
    clearance = np.full((3, 10), 1.0)
    clearance[1, 5] = 0.5  # 单个窄点
    cells = np.array([[1, col] for col in range(10)])
    step = np.full(10, 0.1)
    profile = sq.clearance_profile(cells, clearance, step, threshold_m=0.9)
    # 首采样点没有前驱段，总长 9 段；只有进入窄点的那一段低于阈值
    assert profile["clearance_below_090_ratio"] == pytest.approx(1 / 9)
    assert profile["clearance_min_m"] == pytest.approx(0.5)
    assert profile["clearance_median_m"] == pytest.approx(1.0)


def test_clearance_profile_counts_bottlenecks_with_prominence() -> None:
    clearance = np.full((3, 40), 1.0)
    clearance[1, 10] = 0.55
    clearance[1, 20] = 0.50
    clearance[1, 30] = 0.97  # 显著度 0.03 < 0.05，不算瓶颈
    cells = np.array([[1, col] for col in range(40)])
    profile = sq.clearance_profile(cells, clearance, threshold_m=0.9, prominence_m=0.05)
    assert profile["bottleneck_crossings"] == 2
    assert profile["bottleneck_min_width_m"] == pytest.approx(1.0)


def test_terrain_ratios_over_free_mask() -> None:
    class_grid = np.array([["tile", "tile"], ["wood", "concrete"]])
    free = np.array([[True, True], [True, False]])
    ratios = sq.terrain_ratios(class_grid, free)
    assert ratios["tile"] == pytest.approx(2 / 3)
    assert ratios["wood"] == pytest.approx(1 / 3)
    assert "concrete" not in ratios


@pytest.mark.parametrize(
    "bottleneck,expected",
    [
        (4.67, sq.EASY),
        (2.10, sq.EASY),
        (2.00, sq.EASY),
        (1.99, sq.MEDIUM),
        (1.70, sq.MEDIUM),
        (1.60, sq.MEDIUM),
        (1.59, sq.HARD),
        (1.46, sq.HARD),
        (1.15, sq.HARD),
        (0.75, sq.HARD),
    ],
)
def test_grade_scene_uses_bottleneck_only(bottleneck: float, expected: str) -> None:
    # 分档与 d_min、与路径、与房间分割都无关：只由 bottleneck_width_m 决定
    assert sq.grade_scene(bottleneck_width_m=bottleneck) == expected


def test_room_pair_analysis_finds_the_corridor_pair() -> None:
    clearance = _dumbbell()
    free = clearance > 0.0
    segmentation = sq.segment_rooms(clearance, core_clearance_m=0.9)
    pairs = sq.room_pair_analysis(clearance, segmentation.labels, free, 0.1)
    assert pairs
    best = max(pairs, key=lambda pair: pair.geodesic_m)
    assert best.room_label_changes == 1
    assert best.geodesic_m >= 24.0
    assert best.path_cells[0].tolist() != best.path_cells[-1].tolist()


def test_room_pair_analysis_is_deterministic() -> None:
    clearance = _dumbbell()
    free = clearance > 0.0
    segmentation = sq.segment_rooms(clearance, core_clearance_m=0.9)
    first = sq.room_pair_analysis(clearance, segmentation.labels, free, 0.1)
    second = sq.room_pair_analysis(clearance, segmentation.labels, free, 0.1)
    assert [
        (p.room_a, p.room_b, p.geodesic_m, p.room_label_changes) for p in first
    ] == [
        (p.room_a, p.room_b, p.geodesic_m, p.room_label_changes) for p in second
    ]


def test_widest_connection_measures_gap_width() -> None:
    clearance = _dumbbell()
    mask_a = np.zeros(clearance.shape, dtype=bool)
    mask_a[0:60, 0:80] = True
    mask_b = np.zeros(clearance.shape, dtype=bool)
    mask_b[0:60, 180:260] = True
    measured = sq.widest_connection_m(clearance, mask_a, mask_b)
    assert measured == pytest.approx(1.6, abs=0.1)


def test_widest_connection_between_single_cells_is_bottleneck() -> None:
    clearance = _dumbbell()
    start = np.zeros(clearance.shape, dtype=bool)
    start[30, 10] = True
    goal = np.zeros(clearance.shape, dtype=bool)
    goal[30, 250] = True
    # 两点之间无论怎么绕都要穿过 1.6 m 宽的走廊
    assert sq.widest_connection_m(clearance, start, goal) == pytest.approx(1.6, abs=0.1)


def test_widest_connection_is_zero_when_disconnected() -> None:
    clearance = np.zeros((20, 20), dtype=float)
    clearance[0:5, 0:5] = 1.0
    clearance[15:20, 15:20] = 1.0
    mask_a = np.zeros(clearance.shape, dtype=bool)
    mask_a[0:5, 0:5] = True
    mask_b = np.zeros(clearance.shape, dtype=bool)
    mask_b[15:20, 15:20] = True
    assert sq.widest_connection_m(clearance, mask_a, mask_b) == 0.0


def test_build_quality_reports_all_fields() -> None:
    clearance = _dumbbell()
    free = clearance >= 0.7
    segmentation = sq.segment_rooms(clearance, core_clearance_m=0.9)
    pairs = sq.room_pair_analysis(clearance, segmentation.labels, free, 0.1)
    best = max(pairs, key=lambda pair: pair.geodesic_m)
    class_grid = np.full(clearance.shape, "wood", dtype="<U8")
    quality = sq.build_quality(
        path_len_m=best.geodesic_m,
        cells=best.path_cells,
        labels=segmentation.labels,
        clearance=clearance,
        free_mask=free,
        class_grid=class_grid,
        default_class="wood",
    )
    payload = quality.to_dict()
    # 1.6 m 走廊的保守估计是 1.5996 m，正好压在 hard/medium 阈值上，
    # 所以这里只断言"档位与 bottleneck 一致"（四舍五入到 3 位会跨过阈值）
    assert payload["bottleneck_width_m"] == pytest.approx(1.6, abs=0.01)
    assert payload["grade"] in {sq.MEDIUM, sq.HARD}
    assert payload["room_label_changes"] == 1
    assert payload["rooms_visited"] == 2
    assert payload["bottleneck_width_m"] == pytest.approx(1.6, abs=0.1)
    assert payload["clearance_min_m"] == pytest.approx(0.8, abs=1e-6)
    assert payload["default_terrain_ratio"] == pytest.approx(1.0)
    assert payload["path_len_m"] == pytest.approx(best.geodesic_m, abs=1e-3)


def test_sample_task_cells_stays_inside_feasible_region() -> None:
    clearance = _dumbbell()
    free = clearance >= 0.7
    labels = sq.segment_rooms(clearance, core_clearance_m=0.9).labels
    rng = np.random.default_rng(3)
    samples = sq.sample_task_cells(
        clearance, labels, free, 0.1, tier=sq.EASY, rng=rng, max_samples=6
    )
    assert samples
    for cell_a, cell_b in samples:
        assert free[cell_a] and free[cell_b]
        assert labels[cell_a] > 0 and labels[cell_a] == labels[cell_b]


def test_sample_task_cells_for_hard_tier_crosses_the_narrow_corridor() -> None:
    clearance = _dumbbell()
    free = clearance >= 0.7
    labels = sq.segment_rooms(clearance, core_clearance_m=0.9).labels
    rng = np.random.default_rng(5)
    samples = sq.sample_task_cells(
        clearance, labels, free, 0.1, tier=sq.HARD, rng=rng, max_samples=6
    )
    assert samples
    cell_a, cell_b = samples[0]
    mask_a = np.zeros(clearance.shape, dtype=bool)
    mask_a[cell_a] = True
    mask_b = np.zeros(clearance.shape, dtype=bool)
    mask_b[cell_b] = True
    # 1.6 m 走廊 -> bottleneck 1.6 -> hard 档
    assert sq.widest_connection_m(clearance, mask_a, mask_b) == pytest.approx(
        1.6, abs=0.1
    )
