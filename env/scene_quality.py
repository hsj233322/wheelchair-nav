"""场景质量评估（纯函数）：房间分割、跨门/瓶颈、连通性、地形占比、难度分档。

两类指标要分清：

**依赖分割的（推断，口径必须写清）**

* ``room_label_changes``：路径上**房间标签**变化次数。房间由 clearance 阈值化出的
  「房间核心」再做最近核心归属得到（等价于对距离场做 watershed）。注意：
  **这不是"门洞个数"**。走廊若比 ``core_clearance_m`` 窄，走廊内没有核心，
  会被按最近核心切成若干段，于是沿走廊直行也会产生若干次标签变化
  （实测 ``houseexpo_5427f9b0``：7 次标签变化里 5 次发生在同一条 1.50 m 宽的走廊上）。
* ``rooms_visited``：路径经过的不同房间标签数（同上受分割影响）。
* ``interface_widths_m`` / ``min_interface_width_m``：每次标签变化处**分界面**上的
  ``2 x max(clearance)``；同样可能包含走廊内的切段界面，只作参考。

**不依赖分割的（几何硬指标，推荐用于论文）**

* ``clearance_min_m`` / ``clearance_p10_m`` / ``clearance_median_m``：路径上 clearance
  的最小值 / 10% 分位 / 中位数 [m]；
* ``clearance_below_090_ratio``：路径上 ``clearance < clearance_threshold_m``（默认 0.9 m）
  的**路段长度占比**（按单元间长度加权，不是采样点计数）；
* ``bottleneck_width_m``：起点到终点的**最宽瓶颈**宽度 [m]（二分 clearance，最大连通阈值 x2）。
  这是"无论如何都要挤过的最窄通道"的宽度，与 A* 具体走哪条路、与房间分割都无关。
* ``bottleneck_crossings`` / ``bottleneck_min_width_m``：沿所选路径的 clearance 局部极小
  个数（``scipy.signal.find_peaks``，显著度 >= ``prominence_m``）与其中最窄的宽度。

HouseExpo 的墙/门来自轮廓几何（见 data/README），不需要推断；下面所有量都是
在此基础上做的几何推断，参数必须由调用方传入，模块本身不读配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"

DEFAULT_CLEARANCE_THRESHOLD_M = 0.9
DEFAULT_PROMINENCE_M = 0.05

# 难度分档阈值：只看 bottleneck_width_m（起终点之间的最宽瓶颈宽度，与路径、与 d_min 无关）
#   1.16 m = 2 x platform.body_radius(0.58)  —— 物理最小值
#   1.40 m = 2 x base d_min(0.70)            —— 基准策略的最小可通行宽度
#   1.60 m = 1.40 + 0.20（两个栅格）          —— 余量不足 0.2 m 视为"难"
#   2.00 m                                   —— 余量 >= 0.6 m（约 1.5 x 车身半径）视为"易"
BOTTLENECK_HARD_M = 1.6
BOTTLENECK_EASY_M = 2.0
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
    """按 clearance 场把自由区切成房间（watershed）。

    注意走廊比 ``core_clearance_m`` 窄时走廊内没有核心，会被按最近核心切成段；
    这会影响 ``room_label_changes`` 的口径，但不影响 clearance 系列硬指标。
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
        small = np.flatnonzero(sizes < min_core_cells)
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
    room_label_changes: int
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
    """逐一评估「同一连通分量内的房间对」的测地距离与标签变化次数。

    锚点取房间极点：先在 A 区里挑一个离 B 区质心最远的单元，再从它出发做单源
    dijkstra，取 B 区里可达最远的单元。这样得到的是**近似最长**的跨房间路径。
    返回按 ``(room_label_changes, geodesic_m)`` 降序排列的房间对。
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
                changes = int(np.count_nonzero(np.diff(labels_on_path) != 0))
                pairs.append(
                    RoomPair(
                        room_a=room_a,
                        room_b=room_b,
                        geodesic_m=float(distance[anchor_b]) * res,
                        room_label_changes=changes,
                        anchor_a=tuple(int(v) for v in coordinates[anchor_a]),
                        anchor_b=tuple(int(v) for v in coordinates[anchor_b]),
                        path_cells=np.column_stack(
                            np.unravel_index(path, free_mask.shape)
                        ),
                    )
                )
    pairs.sort(
        key=lambda pair: (pair.room_label_changes, pair.geodesic_m), reverse=True
    )
    return pairs


def widest_connection_m(
    clearance: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    iterations: int = 12,
) -> float:
    """二分 clearance，求两个区域之间「最宽连通」的通道宽度 [m]。

    只要 A 区某个单元与 B 区某个单元在 ``clearance >= t`` 里同属一个连通分量，
    就说明存在宽度 ``2t`` 的通道。取 A、B 各为一个单元时，得到的就是
    **最宽瓶颈宽度**：无论怎么绕，都必须挤过的最窄通道有多宽。
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
    changed = True
    while changed and len(collapsed) > 1:
        changed = False
        for index in range(1, len(collapsed) - 1):
            if run_lengths[index] < min_run:
                left = collapsed[index - 1]
                collapsed[index] = left
                run_lengths[index - 1] += run_lengths[index]
                del collapsed[index]
                del run_lengths[index]
                changed = True
                break
    return collapsed


def clearance_profile(
    cells: np.ndarray,
    clearance: np.ndarray,
    step_length: np.ndarray | None = None,
    *,
    threshold_m: float = DEFAULT_CLEARANCE_THRESHOLD_M,
    prominence_m: float = DEFAULT_PROMINENCE_M,
) -> dict[str, Any]:
    """路径 clearance 的分布与瓶颈（**不依赖房间分割**）。

    * ``clearance_below_090_ratio`` 按单元间长度加权（不是采样点计数）；
    * ``bottleneck_crossings`` 是 clearance 沿路径的局部极小个数
      （``scipy.signal.find_peaks`` 的 prominence 判定），是"挤过几个窄口"的
      分割无关版本。
    """
    cells = np.asarray(cells, dtype=int)
    if cells.size == 0:
        return {
            "clearance_min_m": 0.0,
            "clearance_p10_m": 0.0,
            "clearance_median_m": 0.0,
            "clearance_below_090_ratio": 0.0,
            "clearance_threshold_m": float(threshold_m),
            "bottleneck_crossings": 0,
            "bottleneck_min_width_m": 0.0,
            "bottleneck_widths_m": [],
        }
    values = np.asarray(clearance[cells[:, 0], cells[:, 1]], dtype=float)
    if step_length is None:
        lengths = np.ones(values.size, dtype=float)
    else:
        lengths = np.asarray(step_length, dtype=float).reshape(-1)
    lengths[0] = 0.0
    total = float(lengths.sum())
    below = float(lengths[values < threshold_m].sum())
    valleys, _ = find_peaks(-values, prominence=float(prominence_m))
    widths = [2.0 * float(values[index]) for index in valleys]
    return {
        "clearance_min_m": float(values.min()),
        "clearance_p10_m": float(np.percentile(values, 10)),
        "clearance_median_m": float(np.median(values)),
        "clearance_below_090_ratio": (below / total) if total > 0 else 0.0,
        "clearance_threshold_m": float(threshold_m),
        "bottleneck_crossings": int(valleys.size),
        "bottleneck_min_width_m": min(widths) if widths else 0.0,
        "bottleneck_widths_m": widths,
    }


def path_metadata(
    cells: np.ndarray,
    *,
    labels: np.ndarray,
    clearance: np.ndarray,
    step_length: np.ndarray | None = None,
    free_mask: np.ndarray | None = None,
    min_run_cells: int = 3,
    clearance_threshold_m: float = DEFAULT_CLEARANCE_THRESHOLD_M,
    prominence_m: float = DEFAULT_PROMINENCE_M,
) -> dict[str, Any]:
    """路径的房间标签变化与 clearance 硬指标。"""
    cells = np.asarray(cells, dtype=int)
    profile = clearance_profile(
        cells,
        clearance,
        step_length,
        threshold_m=clearance_threshold_m,
        prominence_m=prominence_m,
    )
    if cells.size == 0:
        profile.update(
            {
                "rooms_visited": 0,
                "room_label_changes": 0,
                "interface_widths_m": [],
                "min_interface_width_m": 0.0,
            }
        )
        return profile

    rows = cells[:, 0]
    cols = cells[:, 1]
    labels_along = [int(value) for value in labels[rows, cols]]
    closed = _collapse_short_runs(labels_along, min_run_cells)

    component: np.ndarray | None = None
    if free_mask is not None:
        free_mask = np.asarray(free_mask, dtype=bool)
        component_labels, _ = ndimage.label(free_mask, structure=_STRUCTURE)
        component = component_labels == component_labels[rows[0], cols[0]]

    interface_widths: list[float] = []
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
            interface_widths.append(2.0 * float(clearance[boundary].max()))
        previous = current

    profile.update(
        {
            "rooms_visited": int(len({value for value in closed if value > 0})),
            "room_label_changes": int(len(interface_widths)),
            "interface_widths_m": interface_widths,
            "min_interface_width_m": min(interface_widths) if interface_widths else 0.0,
        }
    )
    return profile


def pairs_for_tier(
    pairs: Sequence[RoomPair],
    tier: str,
    clearance: np.ndarray,
    labels: np.ndarray,
    *,
    max_pairs: int = 6,
) -> list[RoomPair]:
    """按难度档挑候选房间对。

    难度档只看 ``bottleneck_width_m``，所以先用**区域级**瓶颈宽度（两房间之间的
    最宽连通宽度）估计该房间对属于哪一档，把同档的排在前面，再按测地距离降序。
    区域级宽度 >= 单元级宽度，因此这只是排序偏好；最终以起终点算出的
    ``bottleneck_width_m`` 为准。
    """
    by_geodesic = sorted(pairs, key=lambda pair: pair.geodesic_m, reverse=True)
    scored: list[tuple[bool, float, RoomPair]] = []
    for pair in by_geodesic[: max_pairs * 3]:
        width = widest_connection_m(
            clearance, labels == pair.room_a, labels == pair.room_b
        )
        scored.append((grade_scene(bottleneck_width_m=width) == tier, pair.geodesic_m, pair))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    result: list[RoomPair] = []
    seen: set[tuple[int, int]] = set()
    for _matched, _geodesic, pair in scored:
        key = (pair.room_a, pair.room_b)
        if key in seen:
            continue
        seen.add(key)
        result.append(pair)
        if len(result) >= max_pairs:
            break
    return result


def sample_task_cells(
    clearance: np.ndarray,
    labels: np.ndarray,
    free_mask: np.ndarray,
    resolution: float,
    *,
    tier: str,
    rng,
    min_start_goal_m: float = 4.0,
    max_samples: int = 12,
    max_pairs: int = 4,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """在**给定档位的可行域内**采样起终点（两端点在同一连通分量、保持该档结构）。

    用途：d_min 扫描时不能沿用上一档冻结的起终点（那些点在更严的档位下本身就不合法），
    必须按该档的 ``free_mask = clearance >= d_min`` 重新采样，才能区分
    "该档不存在可行任务"与"冻结端点在新档不合法"。

    返回候选列表（不跑 A*，由调用方验证）；空列表表示该档不存在该档结构的任务。
    """
    clearance = np.asarray(clearance, dtype=float)
    labels = np.asarray(labels)
    free_mask = np.asarray(free_mask, dtype=bool)
    res = float(resolution)
    if not free_mask.any():
        return []
    components, _ = ndimage.label(free_mask, structure=_STRUCTURE)
    samples: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if tier == EASY:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        component = components == int(sizes.argmax())
        room_ids, counts = np.unique(labels[component], return_counts=True)
        valid = room_ids > 0
        if not valid.any():
            return []
        room = int(room_ids[valid][int(np.argmax(counts[valid]))])
        cells = np.argwhere(component & (labels == room))
        if cells.shape[0] < 2:
            return []
        span = cells.max(axis=0) - cells.min(axis=0)
        min_distance = min(min_start_goal_m, 0.5 * float(np.hypot(*span)) * res)
        while len(samples) < max_samples:
            first = int(rng.integers(cells.shape[0]))
            second = int(rng.integers(cells.shape[0]))
            if first == second:
                continue
            cell_a = cells[first]
            cell_b = cells[second]
            if (
                float(np.hypot(*(cell_b - cell_a))) * res < min_distance
            ):
                continue
            samples.append(
                ((int(cell_a[0]), int(cell_a[1])), (int(cell_b[0]), int(cell_b[1])))
            )
        return samples

    pairs = room_pair_analysis(clearance, labels, free_mask, res)
    for pair in pairs_for_tier(pairs, tier, clearance, labels, max_pairs=max_pairs):
        component_id = int(components[tuple(pair.path_cells[0])])
        component = components == component_id
        cells_a = np.argwhere(component & (labels == pair.room_a))
        cells_b = np.argwhere(component & (labels == pair.room_b))
        if cells_a.shape[0] == 0 or cells_b.shape[0] == 0:
            continue
        min_distance = min(min_start_goal_m, 0.6 * pair.geodesic_m)
        per_pair = max(1, max_samples // max(1, max_pairs))
        produced = 0
        guard = 0
        while produced < per_pair and guard < per_pair * 20:
            guard += 1
            cell_a = cells_a[int(rng.integers(cells_a.shape[0]))]
            cell_b = cells_b[int(rng.integers(cells_b.shape[0]))]
            if float(np.hypot(*(cell_b - cell_a))) * res < min_distance:
                continue
            samples.append(
                ((int(cell_a[0]), int(cell_a[1])), (int(cell_b[0]), int(cell_b[1])))
            )
            produced += 1
            if len(samples) >= max_samples:
                return samples

    # 补充"同房间极值对"：难度只看必经瓶颈宽度，同一房间内部的窄颈（例如
    # L 形房间里 1.8 m 的过道）同样可以是 medium/hard 任务，不能只靠跨房间对覆盖。
    if samples:
        return samples
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    component = components == int(sizes.argmax())
    room_ids, counts = np.unique(labels[component], return_counts=True)
    valid = room_ids > 0
    if not valid.any():
        return samples
    room = int(room_ids[valid][int(np.argmax(counts[valid]))])
    cells = np.argwhere(component & (labels == room))
    if cells.shape[0] < 2:
        return samples
    span = cells.max(axis=0) - cells.min(axis=0)
    min_distance = min(min_start_goal_m, 0.5 * float(np.hypot(*span)) * res)
    guard = 0
    while len(samples) < max_samples and guard < max_samples * 20:
        guard += 1
        first = int(rng.integers(cells.shape[0]))
        second = int(rng.integers(cells.shape[0]))
        if first == second:
            continue
        cell_a = cells[first]
        cell_b = cells[second]
        if float(np.hypot(*(cell_b - cell_a))) * res < min_distance:
            continue
        samples.append(
            ((int(cell_a[0]), int(cell_a[1])), (int(cell_b[0]), int(cell_b[1])))
        )
    return samples


def feasible_area_ratio(
    clearance: np.ndarray,
    anchor_cell: Sequence[int],
    *,
    d_min: float,
    free_area_mask: np.ndarray | None = None,
) -> float:
    """**确定性**可行面积占比（不依赖采样次数与采样预算）。

    定义：场景自由区中满足 ``clearance >= d_min``、且与 ``anchor_cell``（起终点之一）
    同属一个连通分量的单元面积占比。单元等面积，所以单元数之比即面积之比。

    * 分母 = 自由区（默认 ``clearance > 0``，即轮廓内的可通行区域）；
    * 锚点自身不满足门槛时返回 0.0（表示"这个任务在该档不可用"）；
    * 该量适合写进论文；"可行任务数"仍保留，但要标注采样预算与二项置信区间。
    """
    clearance = np.asarray(clearance, dtype=float)
    free = (
        clearance > 0.0
        if free_area_mask is None
        else np.asarray(free_area_mask, dtype=bool)
    )
    total = float(free.sum())
    if total <= 0:
        return 0.0
    row, col = int(anchor_cell[0]), int(anchor_cell[1])
    feasible = clearance + 1e-9 >= float(d_min)
    if not feasible[row, col]:
        return 0.0
    components, _ = ndimage.label(feasible, structure=_STRUCTURE)
    component_id = int(components[row, col])
    if component_id == 0:
        return 0.0
    return float((components == component_id).sum()) / total


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
    bottleneck_width_m: float,
    hard_max_m: float = BOTTLENECK_HARD_M,
    easy_min_m: float = BOTTLENECK_EASY_M,
) -> str:
    """难度分档：**只依赖 bottleneck_width_m**（起终点之间的最宽瓶颈宽度）。

    该量与 A* 具体走哪条路无关、与 algo.astar.d_min 无关，所以同一个场景在任何
    d_min 档位下的难度档都一致，场景表不会自相矛盾。

    * ``< hard_max_m``（默认 1.6 m）：最窄必经通道距离基准要求 1.40 m 的余量 < 0.2 m；
    * ``< easy_min_m``（默认 2.0 m）：中间档；
    * ``>= easy_min_m``：余量 >= 0.6 m，通道基本不构成约束。

    ``room_label_changes`` / ``bottleneck_crossings`` 只作描述性统计（路径相关），
    不参与分档。
    """
    if bottleneck_width_m < hard_max_m:
        return HARD
    if bottleneck_width_m < easy_min_m:
        return MEDIUM
    return EASY


@dataclass(frozen=True)
class SceneQuality:
    path_len_m: float
    room_label_changes: int
    rooms_visited: int
    n_components: int
    unreachable_ratio: float
    largest_component_ratio: float
    clearance_min_m: float
    clearance_p10_m: float
    clearance_median_m: float
    clearance_below_090_ratio: float
    clearance_threshold_m: float
    bottleneck_width_m: float
    bottleneck_crossings: int
    bottleneck_min_width_m: float
    interface_widths_m: tuple[float, ...]
    min_interface_width_m: float
    terrain_ratios: Mapping[str, float]
    default_terrain_ratio: float
    grade: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_len_m": round(float(self.path_len_m), 3),
            "room_label_changes": int(self.room_label_changes),
            "rooms_visited": int(self.rooms_visited),
            "n_components": int(self.n_components),
            "unreachable_ratio": round(float(self.unreachable_ratio), 4),
            "largest_component_ratio": round(float(self.largest_component_ratio), 4),
            "clearance_min_m": round(float(self.clearance_min_m), 3),
            "clearance_p10_m": round(float(self.clearance_p10_m), 3),
            "clearance_median_m": round(float(self.clearance_median_m), 3),
            "clearance_below_090_ratio": round(float(self.clearance_below_090_ratio), 4),
            "clearance_threshold_m": round(float(self.clearance_threshold_m), 3),
            "bottleneck_width_m": round(float(self.bottleneck_width_m), 3),
            "bottleneck_crossings": int(self.bottleneck_crossings),
            "bottleneck_min_width_m": round(float(self.bottleneck_min_width_m), 3),
            "interface_widths_m": [round(value, 3) for value in self.interface_widths_m],
            "min_interface_width_m": round(float(self.min_interface_width_m), 3),
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
    step_length: np.ndarray | None = None,
    class_grid: np.ndarray | None = None,
    default_class: str = "default",
    clearance_threshold_m: float = DEFAULT_CLEARANCE_THRESHOLD_M,
) -> SceneQuality:
    """把几何量汇总成一个场景的质量档案。

    分档只看 ``bottleneck_width_m``（起终点之间的最宽瓶颈宽度）；其余量是描述性统计。
    """
    rooms = path_metadata(
        cells,
        labels=labels,
        clearance=clearance,
        step_length=step_length,
        free_mask=free_mask,
        clearance_threshold_m=clearance_threshold_m,
    )
    components = component_stats(free_mask)
    ratios = terrain_ratios(class_grid, free_mask) if class_grid is not None else {}

    # 最宽瓶颈：起点与终点之间的分割无关几何量
    cells = np.asarray(cells, dtype=int)
    bottleneck = 0.0
    if cells.size >= 2:
        start_mask = np.zeros(clearance.shape, dtype=bool)
        start_mask[cells[0, 0], cells[0, 1]] = True
        goal_mask = np.zeros(clearance.shape, dtype=bool)
        goal_mask[cells[-1, 0], cells[-1, 1]] = True
        bottleneck = widest_connection_m(clearance, start_mask, goal_mask)

    grade = grade_scene(bottleneck_width_m=bottleneck)
    return SceneQuality(
        path_len_m=float(path_len_m),
        room_label_changes=int(rooms["room_label_changes"]),
        rooms_visited=int(rooms["rooms_visited"]),
        n_components=int(components["n_components"]),
        unreachable_ratio=float(components["unreachable_ratio"]),
        largest_component_ratio=float(components["largest_component_ratio"]),
        clearance_min_m=float(rooms["clearance_min_m"]),
        clearance_p10_m=float(rooms["clearance_p10_m"]),
        clearance_median_m=float(rooms["clearance_median_m"]),
        clearance_below_090_ratio=float(rooms["clearance_below_090_ratio"]),
        clearance_threshold_m=float(rooms["clearance_threshold_m"]),
        bottleneck_width_m=float(bottleneck),
        bottleneck_crossings=int(rooms["bottleneck_crossings"]),
        bottleneck_min_width_m=float(rooms["bottleneck_min_width_m"]),
        interface_widths_m=tuple(float(v) for v in rooms["interface_widths_m"]),
        min_interface_width_m=float(rooms["min_interface_width_m"]),
        terrain_ratios=ratios,
        default_terrain_ratio=float(ratios.get(default_class, 0.0)),
        grade=grade,
    )
