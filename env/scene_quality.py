"""场景质量评估（纯函数）：房间分割、门洞计数、连通性、地形占比、难度分档。

这里所有的量都是**从几何推断**的，不是数据里现成的字段：

* HouseExpo 的 ``verts`` 是自由区轮廓，内墙表现为轮廓里宽约 0.12 m 的狭缝，
  门洞是狭缝中断处（实测见 ``data/README.md``）。数据本身没有「墙/门」对象列表，
  所以门洞只能从几何推断。
* 房间分割：对 clearance 场做阈值化得到「房间核心」，再把每个自由单元分配给
  最近的核心（等价于对距离场做 watershed）。门洞比房间窄，因此会在核心之间断开。
* ``door_crossings``：沿路径相邻单元的房间标签变化次数（带抖动合并）。
* 通道宽度按 ``2 * clearance`` 估计（单元中心到最近障碍物表面的距离的两倍）。

参数（``core_clearance_m`` 等）都必须由调用方传入，模块本身不读配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"

_STRUCTURE = np.ones((3, 3), dtype=int)


@dataclass(frozen=True)
class RoomSegmentation:
    labels: np.ndarray  # 0 = 障碍，>0 = 房间编号
    room_count: int
    core_clearance_m: float


def segment_rooms(
    clearance: np.ndarray,
    *,
    core_clearance_m: float = 0.9,
    min_core_cells: int = 6,
) -> RoomSegmentation:
    """按 clearance 场把自由区切成房间。

    ``clearance >= core_clearance_m`` 的单元构成房间核心（小于 ``min_core_cells``
    的核心会被丢弃，避免噪声单元自成一间）；其余自由单元归属最近的核心。
    """
    clearance = np.asarray(clearance, dtype=float)
    if clearance.ndim != 2:
        raise ValueError("clearance 必须是二维数组")
    cores = clearance >= float(core_clearance_m)
    if not cores.any():
        return RoomSegmentation(
            labels=np.zeros(clearance.shape, dtype=int),
            room_count=0,
            core_clearance_m=float(core_clearance_m),
        )

    labels, _ = ndimage.label(cores, structure=_STRUCTURE)
    if min_core_cells > 1:
        sizes = np.bincount(labels.ravel())
        small = np.flatnonzero((sizes < min_core_cells))
        small = small[small != 0]
        if small.size:
            cores = cores & ~np.isin(labels, small)
            labels, _ = ndimage.label(cores, structure=_STRUCTURE)

    if not cores.any():
        return RoomSegmentation(
            labels=np.zeros(clearance.shape, dtype=int),
            room_count=0,
            core_clearance_m=float(core_clearance_m),
        )

    _, indices = ndimage.distance_transform_edt(~cores, return_indices=True)
    room_labels = labels[tuple(indices)].astype(int)
    room_labels[clearance <= 0.0] = 0  # 障碍单元不属于任何房间
    room_count = int(np.unique(room_labels[room_labels > 0]).size)
    return RoomSegmentation(
        labels=room_labels,
        room_count=room_count,
        core_clearance_m=float(core_clearance_m),
    )


def component_stats(free_mask: np.ndarray) -> dict[str, float]:
    """连通性：分量个数、最大分量占比、不可达自由区占比。"""
    free_mask = np.asarray(free_mask, dtype=bool)
    if not free_mask.any():
        return {
            "n_components": 0,
            "largest_component_ratio": 0.0,
            "unreachable_ratio": 1.0,
        }
    labels, count = ndimage.label(free_mask, structure=_STRUCTURE)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    total = float(sizes.sum())
    largest = float(sizes.max())
    return {
        "n_components": int(count),
        "largest_component_ratio": largest / total if total else 0.0,
        "unreachable_ratio": 1.0 - (largest / total if total else 1.0),
    }


def _free_graph(free: np.ndarray):
    """自由单元的 8 邻接稀疏图（边长按单元数计），返回 (graph, compact_index)。"""
    from scipy.sparse import csr_matrix

    shape = free.shape
    index = -np.ones(shape, dtype=int)
    index[free] = np.arange(int(free.sum()))
    count = int(free.sum())
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r0, r1 = max(0, dr), shape[0] + min(0, dr)
            c0, c1 = max(0, dc), shape[1] + min(0, dc)
            src = index[r0:r1, c0:c1]
            dst = index[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
            valid = (src >= 0) & (dst >= 0)
            rows.append(src[valid])
            cols.append(dst[valid])
            data.append(np.full(int(valid.sum()), float(np.hypot(dr, dc))))
    graph = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(count, count),
    )
    return graph, index


@dataclass(frozen=True)
class RoomPair:
    """同一连通分量内两个房间之间的测地连通信息。"""

    room_a: int
    room_b: int
    geodesic_m: float  # 近似最长跨房间测地距离（锚点取房间极点）
    crossings: int
    anchor_a: tuple[int, int]
    anchor_b: tuple[int, int]
    path_cells: np.ndarray  # (N, 2) 单元索引


def room_pair_analysis(
    clearance: np.ndarray,
    labels: np.ndarray,
    free_mask: np.ndarray,
    resolution: float,
    *,
    max_components: int = 3,
) -> list[RoomPair]:
    """逐一评估「同一连通分量内的房间对」的测地距离与跨门洞次数。

    锚点取房间的极点：先在 A 区里挑一个离 B 区质心最远的单元，再从它出发做
    单源 dijkstra，取 B 区里可达最远的单元。这样得到的是**近似最长**的跨房间
    路径（A、B 都取极点），用作"这个房间对能做出多长的任务"的估计；
    路径上的房间标签变化次数即跨门洞次数。

    只在同一 ``free_mask`` 连通分量内计算，因此结果一定可达。
    返回按 ``(crossings, geodesic_m)`` 降序排列的房间对。
    """
    from scipy.sparse.csgraph import dijkstra

    clearance = np.asarray(clearance, dtype=float)
    labels = np.asarray(labels)
    free_mask = np.asarray(free_mask, dtype=bool)
    res = float(resolution)
    if not free_mask.any():
        return []

    graph, compact_index = _free_graph(free_mask)
    flat_index = compact_index.ravel()
    free_positions = flat_index >= 0
    flat_positions = np.flatnonzero(free_positions)
    compact_labels = labels.ravel()[free_positions]
    component_labels, _ = ndimage.label(free_mask, structure=_STRUCTURE)
    compact_component = component_labels.ravel()[free_positions]
    coordinates = np.column_stack(np.unravel_index(flat_positions, free_mask.shape))

    pairs: list[RoomPair] = []
    sizes = np.bincount(compact_component)
    for component_id in np.argsort(sizes)[::-1][:max_components]:
        if component_id == 0:
            continue
        in_component = compact_component == component_id
        rooms = sorted(
            int(value) for value in np.unique(compact_labels[in_component]) if value > 0
        )
        if len(rooms) < 2:
            continue
        cells_of = {
            room: np.flatnonzero(in_component & (compact_labels == room))
            for room in rooms
        }
        for position, room_a in enumerate(rooms):
            cells_a = cells_of[room_a]
            if cells_a.size == 0:
                continue
            for room_b in rooms[position + 1 :]:
                cells_b = cells_of[room_b]
                if cells_b.size == 0:
                    continue
                centroid_b = coordinates[cells_b].mean(axis=0)
                anchor_a = int(
                    cells_a[
                        int(
                            np.argmax(
                                np.linalg.norm(coordinates[cells_a] - centroid_b, axis=1)
                            )
                        )
                    ]
                )
                distance, predecessors = dijkstra(
                    graph, indices=int(anchor_a), return_predecessors=True
                )
                distance = np.asarray(distance).reshape(-1)
                predecessors = np.asarray(predecessors).reshape(-1)
                reachable = cells_b[np.isfinite(distance[cells_b])]
                if reachable.size == 0:
                    continue
                anchor_b = int(reachable[int(np.argmax(distance[reachable]))])
                chain: list[int] = []
                cursor = anchor_b
                while cursor >= 0:
                    chain.append(cursor)
                    cursor = int(predecessors[cursor])
                chain.reverse()
                path = flat_positions[np.asarray(chain)]
                labels_on_path = compact_labels[np.asarray(chain)]
                crossings = int(np.count_nonzero(np.diff(labels_on_path) != 0))
                pairs.append(
                    RoomPair(
                        room_a=room_a,
                        room_b=room_b,
                        geodesic_m=float(distance[anchor_b]) * res,
                        crossings=crossings,
                        anchor_a=tuple(int(v) for v in coordinates[anchor_a]),
                        anchor_b=tuple(int(v) for v in coordinates[anchor_b]),
                        path_cells=np.column_stack(
                            np.unravel_index(path, free_mask.shape)
                        ),
                    )
                )
    pairs.sort(key=lambda pair: (pair.crossings, pair.geodesic_m), reverse=True)
    return pairs


def widest_connection_m(
    clearance: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    iterations: int = 12,
) -> float:
    """二分 clearance，求两个区域之间「最宽连通」的通道宽度 [m]。

    这是门洞宽度的几何估计：只要 A 区某个单元与 B 区某个单元在
    ``clearance >= t`` 里同属一个连通分量，就说明存在宽度 ``2t`` 的通道。
    小于 ``2 * 安全半径`` 时，两个区域在当前安全半径下不可达。
    """
    clearance = np.asarray(clearance, dtype=float)
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    cells_a = np.flatnonzero(mask_a.ravel())
    cells_b = np.flatnonzero(mask_b.ravel())
    if cells_a.size == 0 or cells_b.size == 0:
        return 0.0
    low, high = 0.0, float(clearance.max())
    best = 0.0
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        component, _ = ndimage.label(clearance >= mid, structure=_STRUCTURE)
        flat = component.ravel()
        set_a = set(np.unique(flat[cells_a])) - {0}
        set_b = set(np.unique(flat[cells_b])) - {0}
        if set_a & set_b:
            best = mid
            low = mid
        else:
            high = mid
    return 2.0 * best


def _collapse_short_runs(labels_along: Sequence[int], min_run: int) -> list[int]:
    """把路径上短于 ``min_run`` 的标签游程并回前一段，抑制分界线抖动。"""
    collapsed: list[int] = []
    run_lengths: list[int] = []
    for label in labels_along:
        if collapsed and collapsed[-1] == label:
            run_lengths[-1] += 1
            continue
        collapsed.append(int(label))
        run_lengths.append(1)
    # 反复合并过短的中间游程，直到稳定
    changed = True
    while changed and len(collapsed) > 1:
        changed = False
        for index in range(1, len(collapsed) - 1):
            if run_lengths[index] < min_run:
                left = collapsed[index - 1]
                right = collapsed[index + 1]
                collapsed[index] = left if left == right else left
                run_lengths[index - 1] += run_lengths[index]
                del collapsed[index]
                del run_lengths[index]
                changed = True
                break
    return collapsed


def path_metadata(
    cells: np.ndarray,
    *,
    labels: np.ndarray,
    clearance: np.ndarray,
    free_mask: np.ndarray | None = None,
    min_run_cells: int = 3,
) -> dict[str, Any]:
    """路径的房间/门洞/通道宽度元数据。

    * ``door_crossings``：路径上房间标签的变化次数（合并短游程后的）；
    * ``door_widths_m``：每次跨越处**分界面上**的 ``2 * max(clearance)``，
      这是门洞宽度的几何估计（不取路径点的 clearance，避免"贴墙走"造成假窄门）；
    * ``min_passage_width_m``：整条路径上最窄处的 ``2 * clearance``（路径属性，
      通常贴近 ``2 * d_min``，因为最短路径会贴着安全边界走）。
    """
    cells = np.asarray(cells, dtype=int)
    if cells.size == 0:
        return {
            "rooms_visited": 0,
            "door_crossings": 0,
            "door_widths_m": [],
            "min_door_width_m": 0.0,
            "min_passage_width_m": 0.0,
        }
    rows = cells[:, 0]
    cols = cells[:, 1]
    labels_along = [int(value) for value in labels[rows, cols]]
    closed = _collapse_short_runs(labels_along, min_run_cells)

    component: np.ndarray | None = None
    if free_mask is not None:
        free_mask = np.asarray(free_mask, dtype=bool)
        component_labels, _ = ndimage.label(free_mask, structure=_STRUCTURE)
        component = component_labels == component_labels[rows[0], cols[0]]

    door_widths: list[float] = []
    previous = closed[0]
    for index in range(1, len(closed)):
        current = closed[index]
        if current == previous:
            continue
        mask_previous = labels == previous
        mask_current = labels == current
        if component is not None:
            mask_previous = mask_previous & component
            mask_current = mask_current & component
        boundary = mask_previous & ndimage.binary_dilation(
            mask_current, structure=_STRUCTURE
        )
        if boundary.any():
            door_widths.append(2.0 * float(clearance[boundary].max()))
        previous = current

    path_clearance = clearance[rows, cols]
    return {
        "rooms_visited": int(len({value for value in closed if value > 0})),
        "door_crossings": int(len(door_widths)),
        "door_widths_m": door_widths,
        "min_door_width_m": min(door_widths) if door_widths else 0.0,
        "min_passage_width_m": float(2.0 * path_clearance.min()),
    }


def terrain_ratios(class_grid: np.ndarray, free_mask: np.ndarray) -> dict[str, float]:
    """自由区（通常是侵蚀后的可行区）里各材质类别的占比。"""
    class_grid = np.asarray(class_grid)
    free_mask = np.asarray(free_mask, dtype=bool)
    total = int(free_mask.sum())
    if total == 0:
        return {}
    values = class_grid[free_mask]
    names, counts = np.unique(values, return_counts=True)
    return {str(name): float(count) / total for name, count in zip(names, counts)}


def grade_scene(
    *,
    rooms_visited: int,
    door_crossings: int,
    min_passage_width_m: float,
    narrow_passage_width_m: float = 1.2,
    hard_door_crossings: int = 3,
) -> str:
    """难度分档：易＝单房间；中＝跨 1~2 个门洞；难＝跨 ≥3 个门洞或过窄通道。"""
    if door_crossings >= hard_door_crossings:
        return HARD
    if min_passage_width_m <= narrow_passage_width_m:
        return HARD
    if door_crossings >= 1:
        return MEDIUM
    return EASY


@dataclass(frozen=True)
class SceneQuality:
    path_len_m: float
    door_crossings: int
    rooms_visited: int
    n_components: int
    unreachable_ratio: float
    largest_component_ratio: float
    min_passage_width_m: float
    min_door_width_m: float
    door_widths_m: tuple[float, ...]
    terrain_ratios: Mapping[str, float]
    default_terrain_ratio: float
    grade: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_len_m": round(float(self.path_len_m), 3),
            "door_crossings": int(self.door_crossings),
            "rooms_visited": int(self.rooms_visited),
            "n_components": int(self.n_components),
            "unreachable_ratio": round(float(self.unreachable_ratio), 4),
            "largest_component_ratio": round(float(self.largest_component_ratio), 4),
            "min_passage_width_m": round(float(self.min_passage_width_m), 3),
            "min_door_width_m": round(float(self.min_door_width_m), 3),
            "door_widths_m": [round(value, 3) for value in self.door_widths_m],
            "terrain_ratios": {
                key: round(float(value), 4) for key, value in self.terrain_ratios.items()
            },
            "default_terrain_ratio": round(float(self.default_terrain_ratio), 4),
            "grade": self.grade,
        }


def build_quality(
    *,
    path_len_m: float,
    cells: np.ndarray,
    labels: np.ndarray,
    clearance: np.ndarray,
    free_mask: np.ndarray,
    class_grid: np.ndarray | None = None,
    default_class: str = "default",
    narrow_passage_width_m: float = 1.2,
    hard_door_crossings: int = 3,
) -> SceneQuality:
    """把几何量汇总成一个场景的质量档案。"""
    rooms = path_metadata(
        cells, labels=labels, clearance=clearance, free_mask=free_mask
    )
    components = component_stats(free_mask)
    ratios = (
        terrain_ratios(class_grid, free_mask) if class_grid is not None else {}
    )
    door_min = float(rooms["min_door_width_m"])
    path_min = float(rooms["min_passage_width_m"])
    # 判定"窄通道"时同时看门洞几何宽度与路径最窄处
    narrow_width = min(path_min, door_min) if door_min > 0 else path_min
    grade = grade_scene(
        rooms_visited=int(rooms["rooms_visited"]),
        door_crossings=int(rooms["door_crossings"]),
        min_passage_width_m=narrow_width,
        narrow_passage_width_m=narrow_passage_width_m,
        hard_door_crossings=hard_door_crossings,
    )
    return SceneQuality(
        path_len_m=float(path_len_m),
        door_crossings=int(rooms["door_crossings"]),
        rooms_visited=int(rooms["rooms_visited"]),
        n_components=int(components["n_components"]),
        unreachable_ratio=float(components["unreachable_ratio"]),
        largest_component_ratio=float(components["largest_component_ratio"]),
        min_passage_width_m=float(rooms["min_passage_width_m"]),
        min_door_width_m=float(rooms["min_door_width_m"]),
        door_widths_m=tuple(float(value) for value in rooms["door_widths_m"]),
        terrain_ratios=ratios,
        default_terrain_ratio=float(ratios.get(default_class, 0.0)),
        grade=grade,
    )
