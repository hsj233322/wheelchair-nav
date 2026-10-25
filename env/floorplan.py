"""HouseExpo 平面图 -> 占据栅格与地形标签（纯函数，不读配置文件、不写文件、不生成随机数）。

数据来源
--------
HouseExpo (IROS 2020, ``TeaganLi/HouseExpo``, MIT) 每个户型一个 JSON：

* ``verts``          : 户型外轮廓多边形顶点 ``(x, y)``，单位 m，世界坐标；
* ``room_category``  : ``{房间类型: [[x1, y1, x2, y2], ...]}`` 房间**包围盒**；
* ``bbox`` / ``room_num`` / ``id``。

注意 HouseExpo 只给外轮廓 + 房间包围盒，**没有墙厚、没有门洞几何、没有地面材质**。

约定（与 README「关键约定」一致）
--------------------------------
* 世界单位 m；数组第 0 行对应 y 最小侧，第 0 列对应 x 最小侧；
* 单元中心 = ``origin + (i + 0.5) * resolution``；
* 外轮廓多边形用**单元中心是否落在多边形内**判定自由/障碍。矩形 ``walls`` 的
  「与单元有任何重叠即障碍」规则针对有厚度的墙，不适用于零厚度的轮廓线。

地形标签
--------
HouseExpo 没有地面材质，本模块只做**基于房间类型的假设映射**（厨房/卫生间 -> 瓷砖、
卧室/客厅 -> 地板、走廊 -> 水泥），规则由调用方以字典形式传入（见
``config/terrain_rules.yaml``）。这不是实测数据，不得写成实测结论。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# 与 HouseExpo 官方 jsonReader 保持一致的类别归一化（toilet 并入 bathroom 等）。
CATEGORY_ALIASES: dict[str, str] = {
    "toilet": "bathroom",
    "guest_room": "bedroom",
}

Rect = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class RoomBox:
    """一个房间的包围盒与其（归一化）类别。``index`` 是文件内的稳定出现序号。"""

    category: str
    rect: Rect
    index: int


@dataclass(frozen=True)
class FloorPlan:
    """一个 HouseExpo 户型的几何与房间标注。"""

    id: str
    polygons: tuple[tuple[Point, ...], ...]
    rooms: tuple[RoomBox, ...]
    room_num: int
    bbox: Rect
    skipped_rooms: int = 0

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({room.category for room in self.rooms}))


def normalize_category(name: str) -> str:
    """类别名归一化：去空白、转小写、空格/连字符转下划线，并套用别名表。"""
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return CATEGORY_ALIASES.get(normalized, normalized)


def load_floorplan(path: str | Path) -> FloorPlan:
    """读取一个 HouseExpo JSON。几何与标签之外的内容一律忽略。"""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)

    for key in ("verts", "room_category"):
        if key not in data:
            raise ValueError(f"{source}: HouseExpo JSON 缺少字段 {key!r}")

    vertices = np.asarray(data["verts"], dtype=float)
    if vertices.ndim == 2:
        raw_polygons = [vertices]
    elif vertices.ndim == 3:
        raw_polygons = [polygon for polygon in vertices]
    else:
        raise ValueError(f"{source}: verts 形状 {vertices.shape} 无法解释")

    polygons: list[tuple[Point, ...]] = []
    for polygon in raw_polygons:
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError(f"{source}: 多边形形状 {polygon.shape} 非法")
        points = [(float(x), float(y)) for x, y in polygon]
        if len(points) >= 2 and points[0] == points[-1]:
            points = points[:-1]  # 去掉重复的闭合点
        if len(points) < 3:
            raise ValueError(f"{source}: 多边形顶点少于 3 个")
        polygons.append(tuple(points))

    rooms: list[RoomBox] = []
    skipped = 0
    index = 0
    for raw_category in sorted(data["room_category"]):
        category = normalize_category(raw_category)
        boxes = data["room_category"][raw_category]
        for box in boxes:
            if len(box) != 4:
                raise ValueError(f"{source}: 房间包围盒必须是 4 个数，实际为 {box!r}")
            x0, y0, x1, y1 = (float(value) for value in box)
            if x1 <= x0 or y1 <= y0:
                skipped += 1
                index += 1
                continue
            rooms.append(RoomBox(category=category, rect=(x0, y0, x1, y1), index=index))
            index += 1

    flat = np.concatenate([np.asarray(polygon, dtype=float) for polygon in polygons], axis=0)
    bbox = (
        float(flat[:, 0].min()),
        float(flat[:, 1].min()),
        float(flat[:, 0].max()),
        float(flat[:, 1].max()),
    )
    return FloorPlan(
        id=str(data.get("id", source.stem)),
        polygons=tuple(polygons),
        rooms=tuple(rooms),
        room_num=int(data.get("room_num", len(rooms))),
        bbox=bbox,
        skipped_rooms=skipped,
    )


def polygon_bounds(polygons: Sequence[Sequence[Point]]) -> Rect:
    """所有多边形顶点的包围盒 ``(xmin, ymin, xmax, ymax)``。"""
    points = np.concatenate(
        [np.asarray(polygon, dtype=float) for polygon in polygons], axis=0
    )
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def grid_shape_for_bounds(
    bounds: Rect, resolution: float
) -> tuple[int, int, tuple[float, float]]:
    """由包围盒确定栅格 ``(rows, cols, origin)``；origin 取包围盒左下角。"""
    xmin, ymin, xmax, ymax = bounds
    res = float(resolution)
    if not res > 0:
        raise ValueError(f"resolution 必须 > 0，实际为 {resolution}")
    cols = max(1, int(math.ceil((xmax - xmin) / res)))
    rows = max(1, int(math.ceil((ymax - ymin) / res)))
    return rows, cols, (xmin, ymin)


def cell_centers(
    shape: tuple[int, int], resolution: float, origin: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(xs, ys)``：列/行方向的单元中心坐标。"""
    rows, cols = shape
    res = float(resolution)
    xs = float(origin[0]) + (np.arange(cols) + 0.5) * res
    ys = float(origin[1]) + (np.arange(rows) + 0.5) * res
    return xs, ys


def rasterize_polygons(
    polygons: Sequence[Sequence[Point]],
    resolution: float,
    shape: tuple[int, int],
    origin: Sequence[float],
) -> np.ndarray:
    """轮廓多边形 -> 占据栅格（``True`` 为障碍）。

    扫描线填充：逐行求多边形与水平线的交点，成对填充区间，等价于
    「单元中心在多边形内即自由」，但复杂度是 O(行数 x 边数)。
    多边形之外（含地图边界）为障碍，与 ``clearance_field`` 把地图边界
    当作墙体一致。
    """
    rows, cols = int(shape[0]), int(shape[1])
    res = float(resolution)
    xs, ys = cell_centers((rows, cols), res, origin)
    ox = float(origin[0])
    occupancy = np.ones((rows, cols), dtype=bool)

    for polygon in polygons:
        points = np.asarray(polygon, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            raise ValueError(f"多边形必须至少 3 个 (x, y) 顶点，实际形状 {points.shape}")
        x1 = points[:, 0]
        y1 = points[:, 1]
        x2 = np.roll(x1, -1)
        y2 = np.roll(y1, -1)
        crossing = y1 != y2

        for row, y in enumerate(ys):
            selected = crossing & ((y1 > y) != (y2 > y))
            if not selected.any():
                continue
            xa, xb = x1[selected], x2[selected]
            ya, yb = y1[selected], y2[selected]
            intersections = np.sort(xa + (y - ya) * (xb - xa) / (yb - ya))
            for k in range(0, intersections.size - 1, 2):
                low = (intersections[k] - ox) / res - 0.5
                high = (intersections[k + 1] - ox) / res - 0.5
                first = max(0, int(math.ceil(low - 1e-9)))
                last = min(cols - 1, int(math.floor(high + 1e-9)))
                if last >= first:
                    occupancy[row, first : last + 1] = False
    return occupancy


def ordered_room_boxes(
    rooms: Sequence[RoomBox],
    room_type_to_class: Mapping[str, str],
    *,
    default_class: str,
    generic_types: Sequence[str] = (),
) -> tuple[tuple[str, Rect], ...]:
    """把房间包围盒按**确定性的绘制顺序**排好，返回 ``(地形类别, rect)``。

    重叠处理：泛化类别（如 SUNCG 的 ``Room``）先画，具体类别后画，
    所以具体标注赢；同优先级按 ``(类别, 文件内序号)`` 排序。
    未出现在映射表里的类别取 ``default_class``。
    """
    generic = set(generic_types)
    resolved = [
        (
            0 if room.category in generic else 1,
            room.category,
            room.index,
            room_type_to_class.get(room.category, default_class),
            room.rect,
        )
        for room in rooms
    ]
    resolved.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple((item[3], item[4]) for item in resolved)


def terrain_class_grid(
    rooms: Sequence[RoomBox],
    shape: tuple[int, int],
    resolution: float,
    origin: Sequence[float],
    room_type_to_class: Mapping[str, str],
    *,
    default_class: str,
    generic_types: Sequence[str] = (),
) -> np.ndarray:
    """逐单元的地形类别名网格（``dtype='<U16'``）。

    绘制顺序与 :func:`ordered_room_boxes` 一致，采样规则与 runner 的
    ``terrain.patches`` 一致（单元中心落在矩形内即命中，后画的覆盖先画的），
    因此本函数的结果与场景 YAML 经 runner 重建出的分区逐单元相同。
    """
    rows, cols = int(shape[0]), int(shape[1])
    xs, ys = cell_centers((rows, cols), resolution, origin)
    grid = np.full((rows, cols), default_class, dtype="<U16")
    for class_name, (x0, y0, x1, y1) in ordered_room_boxes(
        rooms,
        room_type_to_class,
        default_class=default_class,
        generic_types=generic_types,
    ):
        mask = ((xs >= x0) & (xs <= x1))[None, :] & ((ys >= y0) & (ys <= y1))[:, None]
        grid[mask] = class_name
    return grid


def summarize_floorplan(plan: FloorPlan) -> dict[str, Any]:
    """给日志/场景 source 字段用的一行摘要（纯数据，无副作用）。"""
    xmin, ymin, xmax, ymax = plan.bbox
    return {
        "id": plan.id,
        "room_num": plan.room_num,
        "rooms_labelled": len(plan.rooms),
        "rooms_skipped": plan.skipped_rooms,
        "categories": list(plan.categories),
        "extent_m": [round(xmax - xmin, 3), round(ymax - ymin, 3)],
        "polygon_vertices": [len(polygon) for polygon in plan.polygons],
    }
