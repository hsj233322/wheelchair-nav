"""受控材质因子：在真实户型几何上叠加材质分区（**纯函数，不读配置、不写文件**）。

为什么需要这一层
----------------
HouseExpo 只提供户型轮廓与房间包围盒，**没有地面材质**。早期实现按房间类型
映射材质（厨房/卫生间 -> 瓷砖、卧室/客厅 -> 木地板、走廊 -> 水泥），
只有 ``c_rr = 0.015 / 0.020 / 0.025`` 三档、**跨度 1.67 倍**；而论文第 2 章
的材质表是水泥 0.015 / 草地 0.03 / 石子路 0.04 / 坑洼 0.06（**跨度 4 倍**）。
跨度不足的直接后果：高阻绕行永远被支配，``(长度, 能耗)`` Pareto 前沿退化为单点、
能耗加权 A* 与仅长度 A* 输出完全相同（见交接文档 7.13/7.14）。

本模块把**材质空间分布定义为受控实验因子**，而不是"实测地面材质"：

1. 几何仍来自真实 HouseExpo 户型（轮廓、房间、门洞都不动）；
2. 自由区按目标尺度 ``zone_size_m`` 递归二分成**连通矩形分区**；
3. 分区按目标**面积占比**分配材质：``low_resistance``（低阻为主）/
   ``mixed``（混合）/ ``high_resistance``（高阻为主）；
4. 每一档都覆盖论文材质表的四档 ``c_rr``（0.015 / 0.03 / 0.04 / 0.06），
   高阻材质是**地毯、防滑垫、地面格栅**这类室内真实存在的做法，
   **不是**把房间标成 0.06 的坑洼地貌。

论文与 README 必须声明：**这是受控实验因子，不是实测材质分布**；
``c_rr`` 的取值来自公开文献报道范围，需配合 ±30% 敏感性分析使用。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

# 三档材质分布的固定 id（论文与 README 引用这套名字）
LOW_RESISTANCE = "low_resistance"
MIXED = "mixed"
HIGH_RESISTANCE = "high_resistance"
TIER_IDS: tuple[str, ...] = (LOW_RESISTANCE, MIXED, HIGH_RESISTANCE)

TIER_LABELS_ZH: dict[str, str] = {
    LOW_RESISTANCE: "低阻为主",
    MIXED: "混合",
    HIGH_RESISTANCE: "高阻为主",
}


class TerrainPartitionError(ValueError):
    """材质分区配置/几何不合法。"""


@dataclass(frozen=True)
class MaterialClass:
    """一种地面材质（假设取值，非实测）。"""

    name: str
    c_rr: float
    slope_deg: float = 0.0
    label_zh: str = ""


@dataclass(frozen=True)
class TerrainTier:
    """一档受控材质分布：class 名 -> 自由区面积占比目标（和为 1）。"""

    name: str
    shares: Mapping[str, float]

    def ordered(self) -> list[tuple[str, float]]:
        """按占比升序；**最大占比放最后**（它是兜底类别，见 :func:`assign_materials`）。"""
        return sorted(self.shares.items(), key=lambda item: (item[1], item[0]))


@dataclass(frozen=True)
class PartitionConfig:
    """材质分区参数（来自 ``config/terrain_rules.yaml`` 的 ``material_partition`` 段）。"""

    zone_size_m: float
    default_tier: str
    classes: Mapping[str, MaterialClass]
    tiers: Mapping[str, TerrainTier]
    default_class: str
    min_free_cells: int = 4
    note: str = ""

    def tier(self, name: str | None = None) -> TerrainTier:
        key = str(name or self.default_tier)
        if key not in self.tiers:
            raise TerrainPartitionError(
                f"未知材质分布档 {key!r}；可选：{', '.join(sorted(self.tiers))}"
            )
        return self.tiers[key]

    def material(self, name: str) -> MaterialClass:
        if name not in self.classes:
            raise TerrainPartitionError(f"未知材质 {name!r}")
        return self.classes[name]


@dataclass(frozen=True)
class MaterialZone:
    """一个矩形材质分区（世界坐标，单元中心落在矩形内即命中）。"""

    index: int
    rect: tuple[float, float, float, float]
    class_name: str


# ---------------------------------------------------------------------------
# 1. 自由区 -> 连通矩形分区
# ---------------------------------------------------------------------------


def quadtree_zones(
    free_mask: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    *,
    zone_size_m: float,
    min_free_cells: int = 4,
) -> list[tuple[float, float, float, float]]:
    """把自由区递归二分成边长不超过 ``zone_size_m`` 的连通矩形分区。

    * 切分方向取当前区域较长的一边，两个子区域各自至少保住一个自由单元；
    因此每个分区都是**单一连通矩形**（词义上的 zone，不是随机噪声）；
    * 只保留自由单元数 ``>= min_free_cells`` 的分区，避免产生大量微分区；
    * 输出按 ``(y0, x0)`` 排序，保证确定性。
    """
    free = np.asarray(free_mask, dtype=bool)
    if free.ndim != 2:
        raise TerrainPartitionError("free_mask 必须是二维数组")
    res = float(resolution)
    if not res > 0:
        raise TerrainPartitionError(f"resolution 必须 > 0，实际为 {resolution}")
    size = float(zone_size_m)
    if not size > 0:
        raise TerrainPartitionError(f"zone_size_m 必须 > 0，实际为 {zone_size_m}")
    if not free.any():
        return []
    rows, cols = free.shape
    ox, oy = float(origin[0]), float(origin[1])
    size_cells = max(1.0, size / res)

    results: list[tuple[float, float, float, float]] = []

    def subdivide(r0: int, r1: int, c0: int, c1: int) -> None:
        height_cells = r1 - r0
        width_cells = c1 - c0
        if max(height_cells, width_cells) <= size_cells:
            results.append(
                (
                    ox + c0 * res,
                    oy + r0 * res,
                    ox + c1 * res,
                    oy + r1 * res,
                )
            )
            return
        if height_cells >= width_cells:
            mid = _best_split(free, r0, r1, c0, c1, axis=0)
            subdivide(r0, mid, c0, c1)
            subdivide(mid, r1, c0, c1)
        else:
            mid = _best_split(free, r0, r1, c0, c1, axis=1)
            subdivide(r0, r1, c0, mid)
            subdivide(r0, r1, mid, c1)

    subdivide(0, rows, 0, cols)

    kept: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1 in results:
        c0 = int(round((x0 - ox) / res))
        c1 = int(round((x1 - ox) / res))
        r0 = int(round((y0 - oy) / res))
        r1 = int(round((y1 - oy) / res))
        if int(free[r0:r1, c0:c1].sum()) >= max(1, int(min_free_cells)):
            kept.append((x0, y0, x1, y1))
    kept.sort(key=lambda rect: (round(rect[1], 6), round(rect[0], 6)))
    return kept


def _best_split(
    free: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    *,
    axis: int,
) -> int:
    """在 ``(r0, r1)`` 或 ``(c0, c1)`` 内挑一个切分位置：等分优先，且两侧都非空。

    切分沿**较长边**进行，所以孩子区域的较长边至多是一半（<= target），另一条边
    本来就不超过 target；于是递归一定终止，且每个 zone 的边长都 <= ``zone_size_m``。
    """
    lo, hi = (r0, r1) if axis == 0 else (c0, c1)
    if hi - lo < 2:
        raise TerrainPartitionError("无法继续切分：区域只剩一个单元")
    projection = (
        free[r0:r1, c0:c1].sum(axis=1)
        if axis == 0
        else free[r0:r1, c0:c1].sum(axis=0)
    )
    mid = lo + (hi - lo + 1) // 2
    for candidate in _split_candidates(mid, lo, hi):
        if int(projection[: candidate - lo].sum()) > 0 and int(
            projection[candidate - lo :].sum()
        ) > 0:
            return candidate
    return mid


def _split_candidates(mid: int, lo: int, hi: int) -> list[int]:
    """切分候选：中点优先，然后向两侧逐步退让（用于避开把自由区切成空侧）。"""
    candidates: list[int] = []
    for offset in range(0, hi - lo):
        for candidate in (mid - offset, mid + offset):
            if lo < candidate < hi and candidate not in candidates:
                candidates.append(candidate)
    return candidates


# ---------------------------------------------------------------------------
# 2. 分区 -> 材质分配
# ---------------------------------------------------------------------------


def assign_materials(
    zones: Sequence[tuple[float, float, float, float]],
    free_mask: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    tier: TerrainTier,
    *,
    default_class: str,
) -> list[MaterialZone]:
    """按目标面积占比给分区分配材质。

    算法（确定性）：按占比**升序**处理除兜底类别（占比最大者）以外的材质，
    逐个把尚未分配的分区划给它，直到其自由单元累计数达到目标；剩余分区全部给
    兜底类别。因此实际占比与目标占比的偏差只来自"整块不可分"，不会系统性偏小。
    """
    free = np.asarray(free_mask, dtype=bool)
    res = float(resolution)
    ox, oy = float(origin[0]), float(origin[1])
    rows, cols = free.shape

    counts: list[int] = []
    valid: list[tuple[int, tuple[float, float, float, float]]] = []
    for index, rect in enumerate(zones):
        x0, y0, x1, y1 = rect
        c0 = max(0, int(math.floor((x0 - ox) / res)))
        c1 = min(cols, int(math.ceil((x1 - ox) / res)))
        r0 = max(0, int(math.floor((y0 - oy) / res)))
        r1 = min(rows, int(math.ceil((y1 - oy) / res)))
        count = int(free[r0:r1, c0:c1].sum()) if c1 > c0 and r1 > r0 else 0
        if count > 0:
            valid.append((index, rect))
            counts.append(count)

    total = int(sum(counts))
    if total == 0:
        return []

    ordered = tier.ordered()
    if not ordered:
        raise TerrainPartitionError(f"材质分布档 {tier.name!r} 没有材质")
    fallback_class = ordered[-1][0]
    if fallback_class not in tier.shares:
        raise TerrainPartitionError(f"兜底材质 {fallback_class!r} 不在该档里")

    assigned: list[str | None] = [None] * len(valid)
    remaining = list(range(len(valid)))
    for class_name, share in ordered[:-1]:
        target = float(share) * total
        taken = 0
        while remaining and taken < target:
            position = remaining.pop(0)
            assigned[position] = class_name
            taken += counts[position]

    result: list[MaterialZone] = []
    for position, (index, rect) in enumerate(valid):
        class_name = assigned[position] or fallback_class
        if class_name not in tier.shares:
            class_name = default_class
        result.append(MaterialZone(index=index, rect=rect, class_name=class_name))
    return result


def material_class_grid(
    zones: Sequence[MaterialZone],
    shape: tuple[int, int],
    resolution: float,
    origin: Sequence[float],
    *,
    default_class: str,
) -> np.ndarray:
    """分区 -> 逐单元材质名网格（与场景 ``terrain.patches`` 的重建**逐单元一致**）。

    采样规则与 :func:`runner.experiment.build_scene` 相同：单元中心落在矩形内即命中，
    后画覆盖先画。
    """
    rows, cols = int(shape[0]), int(shape[1])
    res = float(resolution)
    ox, oy = float(origin[0]), float(origin[1])
    xs = ox + (np.arange(cols) + 0.5) * res
    ys = oy + (np.arange(rows) + 0.5) * res
    grid = np.full((rows, cols), default_class, dtype="<U16")
    for zone in zones:
        x0, y0, x1, y1 = zone.rect
        mask = ((xs >= x0) & (xs <= x1))[None, :] & ((ys >= y0) & (ys <= y1))[:, None]
        grid[mask] = zone.class_name
    return grid


def zones_to_patches(
    zones: Sequence[MaterialZone],
    classes: Mapping[str, MaterialClass],
    *,
    bounds: tuple[float, float, float, float] | None = None,
) -> list[dict[str, Any]]:
    """分区 -> 场景 ``terrain.patches``（确定性顺序，后画覆盖先画）。"""
    patches: list[dict[str, Any]] = []
    for zone in zones:
        x0, y0, x1, y1 = zone.rect
        if bounds is not None:
            bx0, by0, bx1, by1 = bounds
            x0, y0 = max(x0, bx0), max(y0, by0)
            x1, y1 = min(x1, bx1), min(y1, by1)
        if x1 <= x0 or y1 <= y0:
            continue
        material = classes[zone.class_name]
        patches.append(
            {
                "id": zone.class_name,
                "rect": [round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)],
                "c_rr": float(material.c_rr),
                "slope_deg": float(material.slope_deg),
            }
        )
    return patches


def partition_free_area(
    free_mask: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    config: PartitionConfig,
    *,
    tier: str | None = None,
    zone_size_m: float | None = None,
    min_free_cells: int = 4,
) -> tuple[list[MaterialZone], np.ndarray]:
    """一步产出 ``(材质分区, 逐单元材质名网格)``。

    网格覆盖**全部自由区**（不只是被安全半径侵蚀后的可行区），因此它可以直接与
    场景 YAML 的 ``terrain.patches`` 做逐单元比对（``tests/test_build_scenarios.py``）。
    """
    size = float(zone_size_m if zone_size_m is not None else config.zone_size_m)
    rects = quadtree_zones(
        free_mask,
        resolution,
        origin,
        zone_size_m=size,
        min_free_cells=min_free_cells,
    )
    zones = assign_materials(
        rects,
        free_mask,
        resolution,
        origin,
        config.tier(tier),
        default_class=config.default_class,
    )
    grid = material_class_grid(
        zones,
        (int(free_mask.shape[0]), int(free_mask.shape[1])),
        resolution,
        origin,
        default_class=config.default_class,
    )
    return zones, grid


# ---------------------------------------------------------------------------
# 3. 档位分配与统计
# ---------------------------------------------------------------------------


def stable_seed(token: str, salt: str = "") -> int:
    """由字符串确定性地派生一个非负整数种子（同输入同输出）。"""
    digest = hashlib.sha256(f"{token}|{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def tier_for_plan(plan_id: str, tiers: Sequence[str] = TIER_IDS, salt: str = "") -> str:
    """给户型确定性地指定一档材质分布（保证跨 20 个场景**均衡**、可复现）。

    做法：把 sha256(plan_id) 的头 8 字节当成无符号整数，对档数取模。
    它是 `plan_id` 的纯函数，所以同一户型永远落在同一档；不同 seed 重跑生成器
    也不会改变档位（材质档位是**场景属性**，不该随采样随机数漂移）。
    """
    if not tiers:
        raise TerrainPartitionError("tiers 不能为空")
    digest = hashlib.sha256(f"{plan_id}|{salt}".encode("utf-8")).hexdigest()
    return str(tiers[int(digest[:8], 16) % len(tiers)])


def c_rr_histogram(
    grid: np.ndarray,
    c_rr_by_class: Mapping[str, float],
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[float, float]:
    """``c_rr`` 分布直方图：返回 ``{c_rr: 占比}``。

    * ``mask`` 限定统计范围（默认全部单元；报告里用自由区或可行区）；
    * ``weights`` 给逐单元权重（报告"候选路线上的占比"时用步长加权）。
    """
    grid = np.asarray(grid)
    values = np.zeros(grid.shape, dtype=float)
    for name, value in c_rr_by_class.items():
        values[grid == name] = float(value)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
    else:
        selected = np.ones(grid.shape, dtype=bool)
    flat_values = values[selected]
    if flat_values.size == 0:
        return {}
    if weights is None:
        flat_weights = np.ones(flat_values.shape, dtype=float)
    else:
        flat_weights = np.asarray(weights, dtype=float)[selected]
        if flat_weights.shape != flat_values.shape:
            raise TerrainPartitionError("weights 与 grid 形状不一致")
    total = float(flat_weights.sum())
    if total <= 0:
        return {}
    result: dict[float, float] = {}
    for value in np.unique(flat_values):
        share = float(flat_weights[flat_values == value].sum()) / total
        result[float(value)] = share
    return dict(sorted(result.items()))


def path_c_rr_histogram(
    cells: np.ndarray,
    step_length: np.ndarray,
    grid: np.ndarray,
    c_rr_by_class: Mapping[str, float],
) -> dict[float, float]:
    """一条路径上的 ``c_rr`` 分布（按**路段长度**加权）。

    这是"高阻地形是否真的落在候选路线上"的直接证据：报告里同时给
    「自由区面积占比」与「路径长度加权占比」两套数。
    """
    cells = np.asarray(cells, dtype=int)
    if cells.size == 0:
        return {}
    steps = np.asarray(step_length, dtype=float).reshape(-1).copy()
    if steps.size != cells.shape[0]:
        raise TerrainPartitionError("step_length 与 cells 长度不一致")
    steps[0] = 0.0
    names = grid[cells[:, 0], cells[:, 1]]
    values = np.array([float(c_rr_by_class[str(name)]) for name in names], dtype=float)
    total = float(steps.sum())
    if total <= 0:
        return {}
    result: dict[float, float] = {}
    for value in np.unique(values):
        result[float(value)] = float(steps[values == value].sum()) / total
    return dict(sorted(result.items()))


def expected_c_rr(tier: TerrainTier, classes: Mapping[str, MaterialClass]) -> float:
    """该档的**面积加权**期望 ``c_rr``（报告里用来给三档排序与对照）。"""
    return float(
        sum(float(share) * float(classes[name].c_rr) for name, share in tier.shares.items())
    )
