"""约束化 A*：把载人特性写成硬约束，而不是代价函数里的权重。

硬约束（不满足即不可行，而不是扣分）：

* ``d_min``：车身中心到最近障碍物的距离下限 [m]；
* ``e_budget``：累计电能预算 [J]；
* ``kappa_max``：曲率上限 [1/m]。

曲率在栅格搜索里的处理
----------------------
8 邻域路径的相邻线段转角只能是 45° 的整数倍。直接对 0.2 m 步长套
``Δθ <= kappa_max * ds`` 会把所有转弯都判为不可行（1.5 * 0.2 = 0.3 rad < 45°），
所以这里采用**曲率窗口**语义：在长度 ``kappa_window_m`` 的窗口上，
允许的单步转角为 ``max_turn = kappa_max * kappa_window_m``。这是搜索侧的
代理约束；路径的精确曲率由后续 core/bspline.py 在平滑阶段保证。两者共用
同一个 ``kappa_max``，参数不会分裂成两套。

搜索与最优性
------------
状态是 ``(行, 列, 朝向)``。每个状态保留若干 (长度, 能耗) 互不支配的标签
（多目标标签设置），因此：

* 返回的是**满足全部硬约束的最短路径**（在栅格离散化意义下）；
* ``collect_pareto=True`` 时把搜索跑完，返回终点处完整的 (长度, 能耗)
  Pareto 集，供第 3 章前沿分析使用。

单位统一：g 与 h 都是米，不存在旧代码里 “g 累计 J、h 返回 m” 的量纲错位。
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from core import energy

EPS = 1e-9

# (dr, dc)：第 0 行对应 y 最小的一侧；世界方向向量为 (dc, dr)。
_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
_HEADINGS: tuple[float, ...] = tuple(math.atan2(dr, dc) for dr, dc in _DIRECTIONS)


@dataclass(frozen=True)
class AStarParams:
    e_budget: float
    d_min: float
    kappa_max: float
    kappa_window_m: float = 0.6
    max_labels_per_state: int = 64
    max_expansions: int = 2_000_000
    collect_pareto: bool = False

    def __post_init__(self) -> None:
        if self.e_budget < 0:
            raise ValueError(f"e_budget 必须 >= 0，实际为 {self.e_budget}")
        if self.d_min < 0:
            raise ValueError(f"d_min 必须 >= 0，实际为 {self.d_min}")
        if not self.kappa_max > 0:
            raise ValueError(f"kappa_max 必须 > 0，实际为 {self.kappa_max}")
        if not self.kappa_window_m > 0:
            raise ValueError(f"kappa_window_m 必须 > 0，实际为 {self.kappa_window_m}")
        if self.max_labels_per_state < 1:
            raise ValueError("max_labels_per_state 必须 >= 1")
        if self.max_expansions < 1:
            raise ValueError("max_expansions 必须 >= 1")


@dataclass(frozen=True)
class PathResult:
    """规划结果。失败时数组为空，``reason`` 给出可机读的失败原因。"""

    success: bool
    reason: str
    cells: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    step_length: np.ndarray
    step_energy: np.ndarray
    energy_j: np.ndarray
    clearance: np.ndarray
    length_m: float
    total_energy_j: float
    expanded: int
    labels_created: int
    pareto_length_m: np.ndarray
    pareto_energy_j: np.ndarray


@dataclass(slots=True)
class _Label:
    parent: int
    row: int
    col: int
    heading: int
    length: float
    energy_j: float
    active: bool = field(default=True)


# ---------------------------------------------------------------------------
# 坐标与距离场
# ---------------------------------------------------------------------------


def world_to_cell(
    xy: Sequence[float], resolution: float, origin: Sequence[float] = (0.0, 0.0)
) -> tuple[int, int]:
    """世界坐标 (x, y) -> (row, col)。"""
    res = float(resolution)
    if not res > 0:
        raise ValueError(f"resolution 必须 > 0，实际为 {resolution}")
    x, y = float(xy[0]), float(xy[1])
    row = int(math.floor((y - float(origin[1])) / res))
    col = int(math.floor((x - float(origin[0])) / res))
    return row, col


def cell_to_world(
    row: int, col: int, resolution: float, origin: Sequence[float] = (0.0, 0.0)
) -> tuple[float, float]:
    """(row, col) 的单元中心 -> 世界坐标 (x, y)。"""
    res = float(resolution)
    x = float(origin[0]) + (int(col) + 0.5) * res
    y = float(origin[1]) + (int(row) + 0.5) * res
    return x, y


def clearance_field(occupancy: np.ndarray, resolution: float) -> np.ndarray:
    """每个单元中心到最近障碍物表面的距离 [m]。

    约定：
    * 地图边界之外视为障碍（轮椅不能驶出地图）；
    * 障碍单元与障碍单元中心相距 1 个像素时距离为 0.5 * resolution，
      减掉半个像素后与真实墙面距离一致；
    * 障碍单元本身为 0。
    """
    occ = np.asarray(occupancy, dtype=bool)
    if occ.ndim != 2 or occ.size == 0:
        raise ValueError("occupancy 必须是二维非空数组")
    res = float(resolution)
    if not res > 0:
        raise ValueError(f"resolution 必须 > 0，实际为 {resolution}")
    padded = np.pad(occ, 1, mode="constant", constant_values=True)
    distance = ndimage.distance_transform_edt(~padded)[1:-1, 1:-1]
    return np.maximum(distance - 0.5, 0.0) * res


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _field_at(field: np.ndarray | float, row: int, col: int) -> float:
    if np.ndim(field) == 0:
        return float(field)
    return float(np.asarray(field)[row, col])


def _empty_result(reason: str, *, expanded: int = 0, labels_created: int = 0) -> PathResult:
    return PathResult(
        success=False,
        reason=reason,
        cells=np.zeros((0, 2), dtype=np.int64),
        xy=np.zeros((0, 2), dtype=float),
        yaw=np.zeros(0, dtype=float),
        step_length=np.zeros(0, dtype=float),
        step_energy=np.zeros(0, dtype=float),
        energy_j=np.zeros(0, dtype=float),
        clearance=np.zeros(0, dtype=float),
        length_m=0.0,
        total_energy_j=0.0,
        expanded=expanded,
        labels_created=labels_created,
        pareto_length_m=np.zeros(0, dtype=float),
        pareto_energy_j=np.zeros(0, dtype=float),
    )


def _dominated(
    labels: list[_Label], bucket: list[int], length: float, energy_j: float
) -> bool:
    for index in bucket:
        other = labels[index]
        if other.length <= length + EPS and other.energy_j <= energy_j + EPS:
            return True
    return False


# ---------------------------------------------------------------------------
# 规划
# ---------------------------------------------------------------------------


def find_path(
    occupancy: np.ndarray,
    resolution: float,
    start_xy: Sequence[float],
    goal_xy: Sequence[float],
    params: AStarParams,
    *,
    energy_params: energy.EnergyParams,
    speed: float,
    c_rr: np.ndarray | float = 0.015,
    slope_rad: np.ndarray | float = 0.0,
    origin: Sequence[float] = (0.0, 0.0),
    clearance: np.ndarray | None = None,
) -> PathResult:
    """在占据栅格上求满足硬约束的最短路径。

    Parameters
    ----------
    occupancy:
        bool 数组，True 为障碍。
    start_xy, goal_xy:
        世界坐标 [m]。会被吸附到所在单元的中心。
    c_rr, slope_rad:
        标量或与 occupancy 同形状的数组；由 runner 从 scenario 栅格化后传入。
    clearance:
        可选的预计算距离场（``clearance_field`` 的输出），避免重复计算。
    """
    occ = np.asarray(occupancy, dtype=bool)
    if occ.ndim != 2 or occ.size == 0:
        raise ValueError("occupancy 必须是二维非空数组")
    res = float(resolution)
    if not res > 0:
        raise ValueError(f"resolution 必须 > 0，实际为 {resolution}")
    speed = float(speed)
    if not speed > 0:
        raise ValueError(f"speed 必须 > 0，实际为 {speed}")
    if params.d_min < 0 or params.e_budget < 0:
        raise ValueError("d_min 与 e_budget 必须非负")

    rows, cols = occ.shape
    clear = (
        clearance_field(occ, res)
        if clearance is None
        else np.asarray(clearance, dtype=float)
    )
    if clear.shape != occ.shape:
        raise ValueError(
            f"clearance 形状 {clear.shape} 与 occupancy 形状 {occ.shape} 不一致"
        )
    if np.any(clear < 0):
        raise ValueError("clearance 不能为负")

    origin = (float(origin[0]), float(origin[1]))
    start_row, start_col = world_to_cell(start_xy, res, origin)
    goal_row, goal_col = world_to_cell(goal_xy, res, origin)

    for name, row, col in (("start", start_row, start_col), ("goal", goal_row, goal_col)):
        if not (0 <= row < rows and 0 <= col < cols):
            return _empty_result(f"{name}_out_of_bounds")
        if occ[row, col]:
            return _empty_result(f"{name}_blocked")
        if clear[row, col] + EPS < params.d_min:
            return _empty_result(f"{name}_violates_clearance")

    if (start_row, start_col) == (goal_row, goal_col):
        x, y = cell_to_world(start_row, start_col, res, origin)
        return PathResult(
            success=True,
            reason="ok",
            cells=np.array([[start_row, start_col]], dtype=np.int64),
            xy=np.array([[x, y]], dtype=float),
            yaw=np.zeros(1, dtype=float),
            step_length=np.zeros(1, dtype=float),
            step_energy=np.zeros(1, dtype=float),
            energy_j=np.zeros(1, dtype=float),
            clearance=np.array([clear[start_row, start_col]], dtype=float),
            length_m=0.0,
            total_energy_j=0.0,
            expanded=0,
            labels_created=1,
            pareto_length_m=np.zeros(1, dtype=float),
            pareto_energy_j=np.zeros(1, dtype=float),
        )

    goal_x, goal_y = cell_to_world(goal_row, goal_col, res, origin)
    xs = origin[0] + (np.arange(cols) + 0.5) * res
    ys = origin[1] + (np.arange(rows) + 0.5) * res
    heuristic = np.hypot(xs[None, :] - goal_x, ys[:, None] - goal_y)

    max_turn = params.kappa_max * params.kappa_window_m
    labels: list[_Label] = [
        _Label(parent=-1, row=start_row, col=start_col, heading=-1, length=0.0, energy_j=0.0)
    ]
    state_labels: dict[tuple[int, int, int], list[int]] = {
        (start_row, start_col, -1): [0]
    }
    queue: list[tuple[float, int, int]] = [(float(heuristic[start_row, start_col]), 0, 0)]
    counter = 1
    goal_indices: list[int] = []
    best_index: int | None = None
    expanded = 0

    while queue:
        _, _, index = heapq.heappop(queue)
        label = labels[index]
        if not label.active:
            continue
        expanded += 1
        if expanded > params.max_expansions:
            if best_index is None:
                return _empty_result(
                    "expansion_limit", expanded=expanded, labels_created=len(labels)
                )
            break

        if label.row == goal_row and label.col == goal_col:
            if best_index is None or label.length < labels[best_index].length - EPS:
                best_index = index
            if not params.collect_pareto:
                break
            continue

        for direction, (dr, dc) in enumerate(_DIRECTIONS):
            if label.heading >= 0:
                turn = _wrap_angle(_HEADINGS[direction] - _HEADINGS[label.heading])
                if abs(turn) > max_turn + EPS:
                    continue
            next_row = label.row + dr
            next_col = label.col + dc
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            if occ[next_row, next_col]:
                continue
            diagonal = dr != 0 and dc != 0
            if diagonal and (
                occ[label.row, next_col] or occ[next_row, label.col]
            ):
                # 禁止贴着障碍角斜穿
                continue
            if clear[next_row, next_col] + EPS < params.d_min:
                continue

            distance = res * (math.sqrt(2.0) if diagonal else 1.0)
            direction_x = dc / math.hypot(dr, dc)
            step_energy = energy.step_energy(
                distance,
                speed,
                c_rr=_field_at(c_rr, next_row, next_col),
                slope_rad=_field_at(slope_rad, next_row, next_col),
                direction_x=direction_x,
                params=energy_params,
            )
            new_length = label.length + distance
            new_energy = label.energy_j + step_energy
            if new_energy > params.e_budget + EPS:
                continue

            key = (next_row, next_col, direction)
            bucket = state_labels.get(key)
            if bucket is not None and _dominated(labels, bucket, new_length, new_energy):
                continue
            if bucket is None:
                bucket = []
                state_labels[key] = bucket

            if len(bucket) >= params.max_labels_per_state:
                worst = max(
                    bucket, key=lambda i: (labels[i].length, labels[i].energy_j)
                )
                if (labels[worst].length, labels[worst].energy_j) <= (
                    new_length,
                    new_energy,
                ):
                    continue
                labels[worst].active = False
                bucket.remove(worst)

            kept: list[int] = []
            for other_index in bucket:
                other = labels[other_index]
                if (
                    other.length >= new_length - EPS
                    and other.energy_j >= new_energy - EPS
                ):
                    other.active = False
                else:
                    kept.append(other_index)
            bucket[:] = kept

            new_index = len(labels)
            labels.append(
                _Label(
                    parent=index,
                    row=next_row,
                    col=next_col,
                    heading=direction,
                    length=new_length,
                    energy_j=new_energy,
                )
            )
            bucket.append(new_index)
            heapq.heappush(
                queue,
                (new_length + float(heuristic[next_row, next_col]), counter, new_index),
            )
            counter += 1
            if next_row == goal_row and next_col == goal_col:
                goal_indices.append(new_index)

    if params.collect_pareto:
        active_goals = [i for i in goal_indices if labels[i].active]
        if active_goals:
            best_index = min(
                active_goals, key=lambda i: (labels[i].length, labels[i].energy_j)
            )

    if best_index is None:
        return _empty_result("no_path", expanded=expanded, labels_created=len(labels))

    front_pairs: list[tuple[float, float]] = []
    for index in goal_indices:
        goal_label = labels[index]
        if goal_label.active:
            front_pairs.append((goal_label.length, goal_label.energy_j))
    front_pairs.sort()
    front: list[tuple[float, float]] = []
    for length, energy_j in front_pairs:
        if not front or energy_j < front[-1][1] - EPS:
            front.append((length, energy_j))
    pareto_length = np.array([pair[0] for pair in front], dtype=float)
    pareto_energy = np.array([pair[1] for pair in front], dtype=float)

    chain: list[int] = []
    cursor = best_index
    while cursor != -1:
        chain.append(cursor)
        cursor = labels[cursor].parent
    chain.reverse()

    count = len(chain)
    cells = np.zeros((count, 2), dtype=np.int64)
    xy = np.zeros((count, 2), dtype=float)
    yaw = np.zeros(count, dtype=float)
    step_length = np.zeros(count, dtype=float)
    step_energy = np.zeros(count, dtype=float)
    energy_j = np.zeros(count, dtype=float)
    clearance_path = np.zeros(count, dtype=float)

    for position, index in enumerate(chain):
        label = labels[index]
        cells[position] = (label.row, label.col)
        xy[position] = cell_to_world(label.row, label.col, res, origin)
        energy_j[position] = label.energy_j
        clearance_path[position] = clear[label.row, label.col]
        if position > 0:
            dx = xy[position, 0] - xy[position - 1, 0]
            dy = xy[position, 1] - xy[position - 1, 1]
            step_length[position] = math.hypot(dx, dy)
            step_energy[position] = label.energy_j - labels[chain[position - 1]].energy_j
            yaw[position] = math.atan2(dy, dx)
    if count > 1:
        yaw[0] = yaw[1]

    return PathResult(
        success=True,
        reason="ok",
        cells=cells,
        xy=xy,
        yaw=yaw,
        step_length=step_length,
        step_energy=step_energy,
        energy_j=energy_j,
        clearance=clearance_path,
        length_m=float(step_length.sum()),
        total_energy_j=float(energy_j[-1]),
        expanded=expanded,
        labels_created=len(labels),
        pareto_length_m=pareto_length,
        pareto_energy_j=pareto_energy,
    )
