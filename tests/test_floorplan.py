"""env/floorplan.py 的纯函数测试：解析、栅格化、地形标签、坐标往返。"""

import json
from pathlib import Path

import numpy as np
import pytest

from core import astar
from env import floorplan

L_SHAPE = {
    "id": "lshape",
    "room_num": 2,
    "bbox": {"min": [0.0, 0.0], "max": [6.0, 6.0]},
    "verts": [[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]],
    "room_category": {
        "Kitchen": [[0.0, 0.0, 2.0, 2.0]],
        "Room": [[0.0, 0.0, 2.0, 2.0]],
        "Bedroom": [[2.0, 2.0, 6.0, 2.0]],  # 退化的盒子，应被跳过
        "Toilet": [[2.0, 4.0, 6.0, 6.0]],
    },
}


def write_plan(tmp_path: Path, data: dict, name: str = "plan.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def shoelace_area(polygon: list[list[float]]) -> float:
    points = np.asarray(polygon, dtype=float)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_load_floorplan_parses_polygon_and_rooms(tmp_path: Path) -> None:
    plan = floorplan.load_floorplan(write_plan(tmp_path, L_SHAPE))
    assert plan.id == "lshape"
    assert len(plan.polygons) == 1
    assert len(plan.polygons[0]) == 6
    assert plan.bbox == (0.0, 0.0, 6.0, 6.0)
    # Toilet -> bathroom（与官方 jsonReader 一致），退化盒子被跳过并计数
    assert "bathroom" in plan.categories
    assert "bedroom" not in plan.categories
    assert plan.skipped_rooms == 1
    assert len(plan.rooms) == 3


def test_closed_polygon_duplicate_point_is_removed(tmp_path: Path) -> None:
    data = dict(L_SHAPE, verts=[[0, 0], [6, 0], [6, 2], [0, 2], [0, 0]])
    plan = floorplan.load_floorplan(write_plan(tmp_path, data))
    assert len(plan.polygons[0]) == 4


def test_rasterize_l_shape_matches_shoelace_area() -> None:
    polygons = (tuple((float(x), float(y)) for x, y in L_SHAPE["verts"]),)
    shape = (60, 60)
    occupancy = floorplan.rasterize_polygons(polygons, 0.1, shape, (0.0, 0.0))
    free_area = float((~occupancy).sum()) * 0.1 * 0.1
    assert free_area == pytest.approx(shoelace_area(L_SHAPE["verts"]), rel=0.03)
    # 凹口内部（4, 4）在多边形外 -> 障碍；L 形内部（1, 5）-> 自由
    assert occupancy[40, 40]
    assert not occupancy[50, 10]


def test_rasterize_is_deterministic_and_convention_consistent() -> None:
    polygons = (tuple((float(x), float(y)) for x, y in L_SHAPE["verts"]),)
    first = floorplan.rasterize_polygons(polygons, 0.2, (30, 30), (0.0, 0.0))
    second = floorplan.rasterize_polygons(polygons, 0.2, (30, 30), (0.0, 0.0))
    assert np.array_equal(first, second)
    # 单元 (0, 0) 的中心是 (0.1, 0.1)，在多边形内
    assert not first[0, 0]
    # 单元 (29, 29) 的中心是 (5.9, 5.9)，落在 L 形凹口里 -> 障碍
    assert first[29, 29]


def test_world_grid_round_trip() -> None:
    resolution = 0.2
    origin = (0.1, -0.4)
    for row in range(15):
        for col in range(17):
            x, y = astar.cell_to_world(row, col, resolution, origin)
            assert astar.world_to_cell((x, y), resolution, origin) == (row, col)


def test_ordered_room_boxes_puts_generic_types_first() -> None:
    rooms = (
        floorplan.RoomBox("kitchen", (0.0, 0.0, 2.0, 2.0), 0),
        floorplan.RoomBox("room", (0.0, 0.0, 6.0, 6.0), 1),
    )
    ordered = floorplan.ordered_room_boxes(
        rooms,
        {"kitchen": "tile", "room": "concrete"},
        default_class="concrete",
        generic_types=("room",),
    )
    # 泛化类别先画，具体类别后画 -> 厨房区间赢
    assert ordered == (("concrete", (0.0, 0.0, 6.0, 6.0)),
                       ("tile", (0.0, 0.0, 2.0, 2.0)))


def test_terrain_class_grid_is_deterministic_and_specific_wins() -> None:
    rooms = (
        floorplan.RoomBox("room", (0.0, 0.0, 6.0, 6.0), 0),
        floorplan.RoomBox("kitchen", (0.0, 0.0, 2.0, 2.0), 1),
        floorplan.RoomBox("bathroom", (4.0, 4.0, 6.0, 6.0), 2),
    )
    mapping = {"kitchen": "tile", "bathroom": "tile", "room": "concrete"}
    arguments = dict(
        shape=(60, 60),
        resolution=0.1,
        origin=(0.0, 0.0),
        room_type_to_class=mapping,
        default_class="concrete",
        generic_types=("room",),
    )
    first = floorplan.terrain_class_grid(rooms, **arguments)
    second = floorplan.terrain_class_grid(rooms, **arguments)
    assert np.array_equal(first, second)
    assert first[10, 10] == "tile"       # 厨房覆盖泛化 Room
    assert first[50, 50] == "tile"       # 卫生间
    assert first[30, 30] == "concrete"   # 泛化 Room 之外的默认（这里仍是 Room 区域）
    assert first.dtype.kind == "U"


def test_grid_shape_and_bounds_cover_extent() -> None:
    rows, cols, origin = floorplan.grid_shape_for_bounds((0.1, 0.1, 15.15, 13.89), 0.1)
    assert (rows, cols) == (138, 151)
    assert origin == (0.1, 0.1)
    assert origin[0] + cols * 0.1 >= 15.15
    assert origin[1] + rows * 0.1 >= 13.89


def test_invalid_polygon_is_rejected() -> None:
    with pytest.raises(ValueError):
        floorplan.rasterize_polygons((((0.0, 0.0), (1.0, 1.0))), 0.1, (10, 10), (0.0, 0.0))
