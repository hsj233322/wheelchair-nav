"""env/scene_quality.py 的测试：房间分割、跨门计数、难度分档、门洞宽度。"""

import numpy as np
import pytest

from env import scene_quality as sq


def _dumbbell(shape=(60, 260), resolution=0.1):
    """两个 8x6 房间 + 1.6 m 走廊的 clearance 场（与 HouseExpo 的门洞尺度同量级）。"""
    clearance = np.zeros(shape, dtype=float)
    clearance[0:60, 0:80] = 1.0          # A 房间（用较小的值即可，够成核心）
    clearance[0:60, 180:260] = 1.0       # B 房间
    clearance[22:38, 80:180] = 0.8       # 1.6 m 走廊：可通行（>0.7）但不是核心（<0.9）
    return clearance


def test_segment_rooms_splits_at_narrow_corridor() -> None:
    clearance = _dumbbell()
    segmentation = sq.segment_rooms(clearance, core_clearance_m=0.9)
    assert segmentation.room_count == 2
    label_a = segmentation.labels[30, 40]
    label_b = segmentation.labels[30, 220]
    assert label_a > 0 and label_b > 0 and label_a != label_b
    # 走廊中点会被分给其中一个房间（watershed 分界落在走廊内）
    assert segmentation.labels[30, 130] in {label_a, label_b}


def test_segment_rooms_without_core_returns_empty() -> None:
    segmentation = sq.segment_rooms(np.full((20, 20), 0.3), core_clearance_m=0.9)
    assert segmentation.room_count == 0
    assert not segmentation.labels.any()


def test_component_stats() -> None:
    free = np.zeros((10, 10), dtype=bool)
    free[0:3, 0:3] = True   # 9 格
    free[5:6, 5:6] = True   # 1 格
    stats = sq.component_stats(free)
    assert stats["n_components"] == 2
    assert stats["largest_component_ratio"] == pytest.approx(0.9)
    assert stats["unreachable_ratio"] == pytest.approx(0.1)


def test_component_stats_empty() -> None:
    stats = sq.component_stats(np.zeros((4, 4), dtype=bool))
    assert stats["n_components"] == 0
    assert stats["unreachable_ratio"] == 1.0


def test_path_metadata_counts_door_crossings() -> None:
    labels = np.zeros((3, 30), dtype=int)
    labels[:, :15] = 1
    labels[:, 15:] = 2
    clearance = np.full((3, 30), 1.0)
    cells = np.array([[1, col] for col in range(0, 30)])
    metadata = sq.path_metadata(cells, labels=labels, clearance=clearance)
    assert metadata["rooms_visited"] == 2
    assert metadata["door_crossings"] == 1
    assert metadata["door_widths_m"] == [2.0]
    assert metadata["min_passage_width_m"] == pytest.approx(2.0)


def test_path_metadata_collapses_boundary_flicker() -> None:
    labels = np.ones((3, 40), dtype=int)
    labels[:, 10:12] = 2  # 只有 2 格宽的另一房间标签：视为抖动
    cells = np.array([[1, col] for col in range(40)])
    metadata = sq.path_metadata(
        cells, labels=labels, clearance=np.full((3, 40), 1.0), min_run_cells=3
    )
    assert metadata["door_crossings"] == 0
    assert metadata["rooms_visited"] == 1


def test_path_metadata_empty_path() -> None:
    metadata = sq.path_metadata(
        np.zeros((0, 2), dtype=int),
        labels=np.zeros((3, 3), dtype=int),
        clearance=np.zeros((3, 3)),
    )
    assert metadata == {
        "rooms_visited": 0,
        "door_crossings": 0,
        "door_widths_m": [],
        "min_door_width_m": 0.0,
        "min_passage_width_m": 0.0,
    }


def test_terrain_ratios_over_free_mask() -> None:
    class_grid = np.array([["tile", "tile"], ["wood", "concrete"]])
    free = np.array([[True, True], [True, False]])
    ratios = sq.terrain_ratios(class_grid, free)
    assert ratios["tile"] == pytest.approx(2 / 3)
    assert ratios["wood"] == pytest.approx(1 / 3)
    assert "concrete" not in ratios


@pytest.mark.parametrize(
    "rooms_visited,door_crossings,min_passage,expected",
    [
        (1, 0, 2.0, sq.EASY),
        (2, 1, 1.6, sq.MEDIUM),
        (3, 2, 1.5, sq.MEDIUM),
        (4, 3, 1.5, sq.HARD),
        (6, 5, 1.5, sq.HARD),
        (1, 0, 1.2, sq.HARD),   # 单房间但必经窄通道
        (1, 0, 1.0, sq.HARD),
    ],
)
def test_grade_scene(
    rooms_visited: int, door_crossings: int, min_passage: float, expected: str
) -> None:
    assert (
        sq.grade_scene(
            rooms_visited=rooms_visited,
            door_crossings=door_crossings,
            min_passage_width_m=min_passage,
        )
        == expected
    )


def test_room_pair_analysis_finds_the_corridor_pair() -> None:
    clearance = _dumbbell()
    free = clearance > 0.0
    segmentation = sq.segment_rooms(clearance, core_clearance_m=0.9)
    pairs = sq.room_pair_analysis(clearance, segmentation.labels, free, 0.1)
    assert pairs
    best = max(pairs, key=lambda pair: pair.geodesic_m)
    assert best.crossings == 1
    # 锚点取房间极点：A 的远角 -> B 的远角，约 26 m
    assert best.geodesic_m >= 24.0
    assert best.path_cells[0].tolist() != best.path_cells[-1].tolist()
    assert best.path_cells.shape[1] == 2


def test_room_pair_analysis_is_deterministic() -> None:
    clearance = _dumbbell()
    free = clearance > 0.0
    segmentation = sq.segment_rooms(clearance, core_clearance_m=0.9)
    first = sq.room_pair_analysis(clearance, segmentation.labels, free, 0.1)
    second = sq.room_pair_analysis(clearance, segmentation.labels, free, 0.1)
    assert [(p.room_a, p.room_b, p.geodesic_m, p.crossings) for p in first] == [
        (p.room_a, p.room_b, p.geodesic_m, p.crossings) for p in second
    ]


def test_widest_connection_measures_gap_width() -> None:
    clearance = _dumbbell()
    mask_a = np.zeros(clearance.shape, dtype=bool)
    mask_a[0:60, 0:80] = True
    mask_b = np.zeros(clearance.shape, dtype=bool)
    mask_b[0:60, 180:260] = True
    measured = sq.widest_connection_m(clearance, mask_a, mask_b)
    assert measured == pytest.approx(1.6, abs=0.1)


def test_widest_connection_is_zero_when_disconnected() -> None:
    clearance = np.zeros((20, 20), dtype=float)
    clearance[0:5, 0:5] = 1.0
    clearance[15:20, 15:20] = 1.0
    mask_a = clearance > 0
    mask_b = np.zeros_like(mask_a)
    mask_b[15:20, 15:20] = True
    mask_a_only = np.zeros_like(mask_a)
    mask_a_only[0:5, 0:5] = True
    assert sq.widest_connection_m(clearance, mask_a_only, mask_b) == 0.0


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
    assert payload["grade"] == sq.MEDIUM
    assert payload["door_crossings"] == 1
    assert payload["rooms_visited"] == 2
    assert payload["default_terrain_ratio"] == pytest.approx(1.0)
    assert payload["path_len_m"] == pytest.approx(best.geodesic_m, abs=1e-3)
