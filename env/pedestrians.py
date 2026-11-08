"""ETH/UCY 行人轨迹 -> 场景世界坐标下的逐帧动态障碍（纯函数，不读配置文件）。

数据来源
--------
ETH Walking Pedestrians (BIWI) 与 UCY 两个公开行人数据集，共 8 个序列的原始
``obsmat`` 风格文本，每行 ``frame id x y``（空白分隔，无表头）：

| 文件 | 序列 | 场地 |
|---|---|---|
| ``eth.txt`` | ETH | ETH 校园（BIWI 标注） |
| ``hotel.txt`` | Hotel | ETH 酒店大堂 |
| ``zara01/02/03.txt`` | Zara | 苏黎世火车站前街道 |
| ``students001/003.txt`` | Students | UCY 校园 |
| ``uni_examples.txt`` | Univ | UCY 大学 |

* 帧号步长恒为 10，采样率 **2.5 fps（0.4 s）**；
* 坐标为**局部场地坐标** [m]，原点与朝向是各场地自己的，**与户型无关**。

因此本模块把"数据集局部坐标 -> 场景世界坐标"显式写成

    xy_world = scale * R(rotation_deg) @ xy_local + origin_xy

这三个量（``scale`` / ``rotation_deg`` / ``origin_xy``）就是**锚点映射**，
来自 ``config/pedestrian_sequences.yaml::anchor``（给定期望场景），并在运行时
由 :func:`fit_anchor` 依据**场景几何**确定到一个具体值；任务表里逐 (场景, 序列)
记录该值，所以可复现。

时间对齐
--------
``dt = 0.1 s`` 线性插值重采样（数据集 0.4 s -> 0.1 s 是**加密**，不是抽稀）。
速度用**重采样后的相邻帧中心差分 + 滑动平均**得到：直接对 0.4 s 原始帧做差分会把
标注噪声放大到 4 倍（``Δx/0.4``），重采样后再差分并平滑才是稳定的做法。

可用性筛选（三条，逐条报比例）
------------------------------
1. **出界**：轨迹点落在栅格之外（或障碍单元内）—— 裁掉该段；
2. **穿墙**：行人圆心到墙的距离 ``< body_radius`` 的样本 —— 剔除；
3. **连续穿墙**：连续穿墙样本数超过 ``max_consecutive_wall_crossings`` ——
   整条行人轨迹丢弃（说明该行人整体落在墙里，逐点剔除没有意义）。

输出的 :class:`PedestrianTrack` 直接构造 ``core.dwa.ObstacleTrack``（DWA 侧不需要
任何改动），半径取 ``config.pedestrian_sequences.yaml::pedestrian_radius_m``（0.30 m）。

口径（论文必须写）：**真人不避让机器人**。接入后的被动接触会大幅上升，这是外部
效度的必然代价；主动碰撞必须仍为 0。不得为了指标好看去修正行人轨迹或放宽判据。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# 数据集帧步长恒为 10、2.5 fps；这里显式写出来，便于核对与测试
DATASET_FRAME_STEP = 10
DATASET_FPS = 2.5
DATASET_DT_S = 1.0 / DATASET_FPS
DEFAULT_PEDESTRIAN_RADIUS_M = 0.30


class PedestrianDataError(ValueError):
    """数据缺失或格式不合法（不是"该序列不可用"，是硬错误）。"""


# ---------------------------------------------------------------------------
# 1. 读取原始序列
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawSequence:
    """一个原始序列（局部坐标，未做任何时间重采样）。"""

    name: str
    source: Path
    frame: np.ndarray
    """原始帧号（int），严格递增的采样时刻，单位 frame。"""
    ids: np.ndarray
    """行人编号（int）。"""
    xy_local: np.ndarray
    """局部场地坐标 [m]，形状 (N, 2)。"""

    @property
    def times_s(self) -> np.ndarray:
        """帧号 -> 时间 [s]（``frame / 25``；帧步长 10 即 0.4 s）。"""
        return np.asarray(self.frame, dtype=float) / (DATASET_FPS * DATASET_FRAME_STEP)

    @property
    def pedestrian_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in np.unique(self.ids))

    @property
    def duration_s(self) -> float:
        times = self.times_s
        return float(times.max() - times.min()) if times.size else 0.0

    @property
    def extent_m(self) -> tuple[float, float]:
        if self.xy_local.size == 0:
            return (0.0, 0.0)
        span = self.xy_local.max(axis=0) - self.xy_local.min(axis=0)
        return (float(span[0]), float(span[1]))


def load_sequence(path: str | Path, *, name: str | None = None) -> RawSequence:
    """读取一个 ``frame id x y`` 文本（空白分隔，允许任意行尾）。

    只做格式校验，不做任何坐标/时间变换 —— 变换是调用方的显式责任。

    **带进程内缓存**：同一个序列文件会被 20 个场景各读一次，解析 6 万行
    重复 20 次是纯浪费；缓存键是 ``(绝对路径, mtime_ns, name)``，
    文件被替换时缓存自动失效。
    """
    source = Path(path)
    if not source.is_file():
        raise PedestrianDataError(
            f"行人序列文件不存在: {source}；"
            "数据获取方式见 data/README.md（data/eth_ucy/ 不进 git）"
        )
    key = (str(source.resolve()), source.stat().st_mtime_ns, str(name or source.stem))
    cached = _SEQUENCE_CACHE.get(key)
    if cached is not None:
        return cached
    parsed = _parse_sequence(source, name=name)
    _SEQUENCE_CACHE[key] = parsed
    return parsed


_SEQUENCE_CACHE: dict[tuple[str, int, str], RawSequence] = {}


def _parse_sequence(source: Path, *, name: str | None) -> RawSequence:
    frames: list[float] = []
    ids: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    with source.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 4:
                raise PedestrianDataError(
                    f"{source}:{number} 需要至少 4 列（frame id x y），实际 {len(parts)} 列"
                )
            try:
                frames.append(float(parts[0]))
                ids.append(float(parts[1]))
                xs.append(float(parts[2]))
                ys.append(float(parts[3]))
            except ValueError as exc:
                raise PedestrianDataError(f"{source}:{number} 含非数值字段") from exc
    if not frames:
        raise PedestrianDataError(f"{source}: 没有任何数据行")
    frame = np.asarray(frames, dtype=float)
    identifier = np.asarray(ids, dtype=float)
    xy = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    if not np.isfinite(xy).all():
        raise PedestrianDataError(f"{source}: 坐标含 NaN/Inf")
    order = np.lexsort((xy[:, 0], xy[:, 1], frame, identifier))
    return RawSequence(
        name=str(name or source.stem),
        source=source,
        frame=frame[order],
        ids=identifier[order],
        xy_local=xy[order],
    )


# ---------------------------------------------------------------------------
# 2. 锚点映射（局部场地坐标 -> 场景世界坐标）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorMapping:
    """``xy_world = scale * R(rotation_deg) @ xy_local + origin_xy``。

    * ``scale`` 无量纲；**默认 1.0**（数据本身就是米，缩放会连带改变步行速度，
      所以只有在有独立依据时才允许 != 1）；
    * ``rotation_deg`` **逆时针**旋转角 [deg]；
    * ``origin_xy`` 是局部原点的世界坐标 [m]。
    """

    origin_xy: tuple[float, float]
    rotation_deg: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite([self.origin_xy[0], self.origin_xy[1], self.rotation_deg, self.scale]).all():
            raise PedestrianDataError("锚点映射参数必须是有限数值")
        if not self.scale > 0:
            raise PedestrianDataError(f"scale 必须 > 0，实际为 {self.scale}")

    @property
    def rotation_rad(self) -> float:
        return math.radians(float(self.rotation_deg))

    def matrix(self) -> np.ndarray:
        """2x2 线性部分 ``scale * R``。"""
        theta = self.rotation_rad
        cos, sin = math.cos(theta), math.sin(theta)
        return float(self.scale) * np.array([[cos, -sin], [sin, cos]], dtype=float)

    def apply(self, xy_local: np.ndarray) -> np.ndarray:
        """局部坐标 -> 世界坐标。"""
        points = np.asarray(xy_local, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise PedestrianDataError("xy_local 必须是 (N, 2)")
        return (self.matrix() @ points.T).T + np.asarray(self.origin_xy, dtype=float)

    def inverse(self, xy_world: np.ndarray) -> np.ndarray:
        """世界坐标 -> 局部坐标（坐标往返一致性的另一半）。"""
        points = np.asarray(xy_world, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise PedestrianDataError("xy_world 必须是 (N, 2)")
        shift = points - np.asarray(self.origin_xy, dtype=float)
        return (np.linalg.inv(self.matrix()) @ shift.T).T

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_xy": [round(float(self.origin_xy[0]), 6), round(float(self.origin_xy[1]), 6)],
            "rotation_deg": round(float(self.rotation_deg), 4),
            "scale": round(float(self.scale), 6),
        }


def largest_free_rectangle(
    free_mask: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    *,
    stride: int = 4,
) -> tuple[float, float, float, float]:
    """自由掩码内**最大全 1 轴对齐矩形**的世界坐标 ``(x0, y0, x1, y1)``。

    用途：给行人数据找一块"一定放得下"的落脚区域。它是纯几何量（与算法参数无关），
    用积分图 + 分块扩张求解，栅格尺度下足够快。
    """
    free = np.asarray(free_mask, dtype=bool)
    if free.ndim != 2 or not free.any():
        raise PedestrianDataError("free_mask 必须是非空二维数组")
    res = float(resolution)
    rows, cols = free.shape
    integral = np.pad(
        np.cumsum(np.cumsum(free.astype(np.int64), axis=0), axis=1), ((1, 0), (1, 0))
    )
    step = max(1, int(stride))
    best = (0, 0, 0, 0, 0)

    def area(r0: int, c0: int, r1: int, c1: int) -> int:
        return int(integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0])

    for r0 in range(0, rows, step):
        for r1 in range(r0 + 2 * step, min(rows, r0 + 600) + 1, step):
            height = r1 - r0
            c0 = 0
            while c0 + step <= cols:
                if area(r0, c0, r1, c0 + step) < height * step:
                    c0 += step
                    continue
                c1 = c0
                while (
                    c1 + step <= cols
                    and area(r0, c0, r1, c1 + step) == height * (c1 + step - c0)
                ):
                    c1 += step
                if height * (c1 - c0) > best[0]:
                    best = (height * (c1 - c0), r0, c0, r1, c1)
                c0 = c1 + step
    _, r0, c0, r1, c1 = best
    if best[0] == 0:
        raise PedestrianDataError("自由区里找不到任何合法矩形（free_mask 为空）")
    return (
        float(origin[0]) + c0 * res,
        float(origin[1]) + r0 * res,
        float(origin[0]) + c1 * res,
        float(origin[1]) + r1 * res,
    )


def wall_clearance(
    xy_world: np.ndarray,
    scene_clearance: np.ndarray,
    *,
    resolution: float,
    origin: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """世界坐标 -> ``(clearance, in_bounds)``。

    越界点 ``clearance = -1``、``in_bounds = False``（越界与贴墙是两种失败，
    统计上必须分开报）。
    """
    points = np.asarray(xy_world, dtype=float)
    field = np.asarray(scene_clearance, dtype=float)
    rows, cols = field.shape
    res = float(resolution)
    col = np.floor((points[:, 0] - float(origin[0])) / res).astype(int)
    row = np.floor((points[:, 1] - float(origin[1])) / res).astype(int)
    in_bounds = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    clearance = np.full(points.shape[0], -1.0, dtype=float)
    clearance[in_bounds] = field[row[in_bounds], col[in_bounds]]
    return clearance, in_bounds


def fit_anchor(
    raw: RawSequence,
    *,
    clearance_field: np.ndarray,
    occupancy: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    free_clearance_m: float,
    body_radius: float,
    rotation_step_deg: float = 5.0,
    offset_half_range_m: float = 4.0,
    offset_step_m: float = 1.0,
    scale: float = 1.0,
) -> tuple[AnchorMapping, float]:
    """为 (场景, 序列) 找一个**几何最优**的锚点映射，返回 ``(映射, 越界或贴墙比例)``。

    目标：把 ``scale = 1.0``（保真实尺度、保真实步行速度）下的行人样本尽量放进
    场景自由区。搜索空间 = ``rotation_deg`` 网格 × 目标中心附近的平移网格；
    目标函数 = **行人圆（半径 ``body_radius``）与墙冲突的样本比例**（越小越好）。

    锚点的三个量因此是**几何决定的**、可复算的，而不是手调的：给定 (序列, 场景)
    与上面的搜索参数，结果唯一。若最小冲突比例仍然很高，说明该序列与这个户型
    几何上放不下 —— 这本身就是应当报告的结果，而不是把 scale 调小去凑。

    **性能**：内层必须向量化。朴素实现把整条序列（``students001`` 有 18 353 行）
    在每个 (角度, 平移) 上重新旋转+查表，单组合就是 72×81×18 353 ≈ **1 亿次**逐点
    运算、几十秒。这里改成：

    * 每个角度只做**一次**旋转，并把查询点压成**一维线性下标**（``flat_index``）；
    * 平移只改变整数格点偏移，于是内层变成"标量加 + 一次 ``np.take`` + 一次布尔求和"，
      没有二维花式索引；
    * 判定场 ``bad_flat`` 预先算好，越界用 ``out_of_bounds`` 下标兜底。

    实测同一组合从数十秒降到亚秒量级；结果与朴素实现**完全一致**（同为目标函数
    最小，且并列时取先遇到的网格点，网格遍历顺序相同）。
    """
    free = np.asarray(clearance_field, dtype=float) + 1e-9 >= float(free_clearance_m)
    x0, y0, x1, y1 = largest_free_rectangle(free, resolution, origin)
    target = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=float)

    points = np.asarray(raw.xy_local, dtype=float)
    occupancy_array = np.asarray(occupancy)
    rows, cols = occupancy_array.shape
    res = float(resolution)
    ox, oy = float(origin[0]), float(origin[1])
    total = float(points.shape[0])

    # 判定场：界内且 clearance >= body_radius 才算"好"，其余（含越界）都算冲突
    in_bounds = np.zeros((rows, cols), dtype=bool)
    in_bounds[1:-1, 1:-1] = True
    bad_field = ~(
        in_bounds
        & (np.asarray(clearance_field, dtype=float) >= float(body_radius))
    )
    bad_flat = bad_field.ravel()
    # 越界下标：指向一个恒为 True 的哨兵位置
    sentinel = bad_flat.size
    bad_flat_ext = np.concatenate([bad_flat, np.array([True])])

    shifts = np.arange(
        -offset_half_range_m, offset_half_range_m + 1e-9, offset_step_m
    )
    shift_cells = np.round(shifts / res).astype(int)
    shift_cells = np.unique(shift_cells)

    best_ratio = float("inf")
    best_mapping = AnchorMapping(origin_xy=(0.0, 0.0), rotation_deg=0.0, scale=scale)
    angle = 0.0
    while angle < 360.0:
        theta = math.radians(angle)
        cos, sin = math.cos(theta) * scale, math.sin(theta) * scale
        centre = np.array(
            [
                points[:, 0].mean() * cos - points[:, 1].mean() * sin,
                points[:, 0].mean() * sin + points[:, 1].mean() * cos,
            ]
        )
        base = target - centre  # 把旋转后的重心搬到目标点
        col = np.floor((points[:, 0] * cos - points[:, 1] * sin + base[0] - ox) / res)
        row = np.floor((points[:, 0] * sin + points[:, 1] * cos + base[1] - oy) / res)
        col_i = col.astype(np.int64)
        row_i = row.astype(np.int64)
        for d_col in shift_cells:
            cols_shifted = col_i + d_col
            valid_col = (cols_shifted >= 0) & (cols_shifted < cols)
            for d_row in shift_cells:
                rows_shifted = row_i + d_row
                valid = valid_col & (rows_shifted >= 0) & (rows_shifted < rows)
                flat = np.where(
                    valid, rows_shifted * cols + cols_shifted, sentinel
                )
                ratio = float(bad_flat_ext[flat].sum()) / total
                if ratio < best_ratio - 1e-12:
                    best_ratio = ratio
                    best_mapping = AnchorMapping(
                        origin_xy=(
                            float(base[0] + d_col * res),
                            float(base[1] + d_row * res),
                        ),
                        rotation_deg=angle,
                        scale=scale,
                    )
        angle += float(rotation_step_deg)
    return best_mapping, float(best_ratio)


def _conflict_ratio(
    xy_world: np.ndarray,
    clearance_field: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    body_radius: float,
) -> float:
    """越界或"行人圆心离墙 < body_radius"的样本比例（拟合阶段的目标函数）。"""
    values, in_bounds = wall_clearance(
        xy_world, clearance_field, resolution=resolution, origin=origin
    )
    bad = (~in_bounds) | (values < float(body_radius))
    return float(np.mean(bad))


# ---------------------------------------------------------------------------
# 3. 时间对齐：重采样 + 速度平滑
# ---------------------------------------------------------------------------


def resample_track(
    times_s: np.ndarray,
    xy: np.ndarray,
    *,
    dt_s: float,
    t_start_s: float | None = None,
    t_end_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """把一条轨迹线性插值到 ``dt_s`` 等间隔时间基上。

    返回 ``(times, xy)``，``times`` 从 ``t_start`` 起、步长严格为 ``dt_s``，
    最后一个采样点不晚于 ``t_end``（默认取原轨迹的时间范围）。
    帧数口径：``n = floor((t_end - t_start) / dt_s) + 1``，因此**重采样只会加密**
    （数据集 0.4 s -> 0.1 s），不会丢信息。
    """
    t = np.asarray(times_s, dtype=float)
    points = np.asarray(xy, dtype=float)
    if t.ndim != 1 or points.ndim != 2 or points.shape[1] != 2:
        raise PedestrianDataError("times_s 必须是 (N,)，xy 必须是 (N,2)")
    if t.shape[0] != points.shape[0]:
        raise PedestrianDataError("times_s 与 xy 长度必须一致")
    if t.shape[0] < 2:
        raise PedestrianDataError("重采样至少需要 2 个样本")
    if not float(dt_s) > 0:
        raise PedestrianDataError(f"dt_s 必须 > 0，实际为 {dt_s}")
    order = np.argsort(t, kind="stable")
    t = t[order]
    points = points[order]
    if np.any(np.diff(t) <= 0):
        raise PedestrianDataError("times_s 必须严格递增（同一行人同一时刻只能有一个位置）")
    start = float(t[0] if t_start_s is None else t_start_s)
    end = float(t[-1] if t_end_s is None else t_end_s)
    if end < start:
        raise PedestrianDataError("t_end_s 不能早于 t_start_s")
    count = int(math.floor((end - start) / float(dt_s) + 1e-9)) + 1
    times = start + np.arange(count, dtype=float) * float(dt_s)
    out = np.zeros((count, 2), dtype=float)
    out[:, 0] = np.interp(times, t, points[:, 0])
    out[:, 1] = np.interp(times, t, points[:, 1])
    return times, out


def observation_runs(
    times_s: np.ndarray, *, max_gap_s: float, min_samples: int = 2
) -> list[tuple[int, int]]:
    """把一条行人的**观测序列**切成连续段（相邻样本间隔 ``<= max_gap_s``）。

    为什么必须切：用线性插值跨过观测空洞，等于**凭空造出**一段行人从未走过的
    直线运动（实测：ETH 的 ``eth.txt`` 帧号有 60/70/90 的大步长，跨空洞插值会让
    行人"瞬移"穿过墙，再被穿墙筛选剔掉）。切段之后，插值只发生在真实相邻观测之间，
    可用性统计与轨迹都不会被凭空生成的样本污染。

    ``max_gap_s`` 取 ``2 x dt_s``：数据集帧步长 10（0.4 s），插值到 0.1 s 后
    相邻真实观测之间最多 4 个周期，取 2 个周期就足以区分"正常采样间隔"与"空洞"。
    """
    t = np.asarray(times_s, dtype=float)
    if t.shape[0] < int(min_samples):
        return []
    breaks = np.flatnonzero(np.diff(t) > float(max_gap_s)) + 1
    bounds = [0, *breaks.tolist(), t.shape[0]]
    runs: list[tuple[int, int]] = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        if high - low >= int(min_samples):
            runs.append((int(low), int(high)))
    return runs


def smooth_velocity(
    xy: np.ndarray, dt_s: float, *, window: int = 5
) -> np.ndarray:
    """中心差分 + 滑动平均的速度 [m/s]。

    为什么要平滑：``dt = 0.1 s`` 下标注噪声 ``σ`` 会让差分速度带上 ``σ / 0.1``
    的抖动；DWA 的绕行判据用相对运动方向 δ，速度方向抖动会直接把 δ 打散，
    于是判据在相邻周期之间反复翻转。滑动平均窗口 ``window``（奇数）默认 5 个周期
    = 0.5 s，与 DWA 的预测时域同量级。
    """
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise PedestrianDataError("xy 必须是 (N,2)")
    if not float(dt_s) > 0:
        raise PedestrianDataError(f"dt_s 必须 > 0，实际为 {dt_s}")
    count = points.shape[0]
    if count < 2:
        return np.zeros_like(points)
    velocity = np.zeros_like(points)
    velocity[1:-1] = (points[2:] - points[:-2]) / (2.0 * float(dt_s))
    velocity[0] = (points[1] - points[0]) / float(dt_s)
    velocity[-1] = (points[-1] - points[-2]) / float(dt_s)
    size = int(window)
    if size <= 1:
        return velocity
    if size % 2 == 0:
        size += 1
    if size > count:
        size = count if count % 2 == 1 else count - 1
    if size <= 1:
        return velocity
    pad = size // 2
    kernel = np.ones(size, dtype=float) / size
    padded = np.pad(velocity, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.column_stack(
        [
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        ]
    )
    return smoothed


# ---------------------------------------------------------------------------
# 4. 可用性筛选
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsabilityReport:
    """一个 (场景, 序列) 的可用性统计（**每条都要报**）。

    **样本口径必须声明**（本轮踩过坑，必须写清）：三条比例用**同一个分母** ——
    连续观测段内按 ``dt_s`` 重采样后的样本数（``samples_resampled``）。
    * 原始数据是 0.4 s 一个样本，重采样到 0.1 s 会把它**加密约 4 倍**，
      所以"重采样样本数 / 原始行数"≈ 4 是正常的，**不是**重复计数；
      原始行数单独记在 ``samples_raw`` 里，两者不混着算比例。
    * ``pedestrians_usable_ratio`` 是行人级的可用比例（有多少行人的轨迹真的能进
      episode），与样本级比例是两个不同的分母，分列。
    """

    sequence: str
    pedestrians_total: int
    pedestrians_used: int
    samples_raw: int
    samples_resampled: int
    samples_out_of_bounds: int
    samples_wall_crossing: int
    samples_used: int
    dropped_consecutive_wall_crossing: int
    dropped_too_few_samples: int
    anchor_conflict_ratio: float
    duration_s: float
    usable_pedestrians: float
    out_of_bounds_ratio: float
    wall_crossing_ratio: float
    sample_usable_ratio: float
    wiped_windows: int = 0
    """被"连续穿墙"规则整段丢弃的观测段个数。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "pedestrians_total": int(self.pedestrians_total),
            "pedestrians_used": int(self.pedestrians_used),
            "pedestrians_usable_ratio": round(float(self.usable_pedestrians), 4),
            "samples_raw": int(self.samples_raw),
            "samples_resampled": int(self.samples_resampled),
            "samples_out_of_bounds": int(self.samples_out_of_bounds),
            "samples_wall_crossing": int(self.samples_wall_crossing),
            "samples_used": int(self.samples_used),
            "out_of_bounds_ratio": round(float(self.out_of_bounds_ratio), 4),
            "wall_crossing_ratio": round(float(self.wall_crossing_ratio), 4),
            "sample_usable_ratio": round(float(self.sample_usable_ratio), 4),
            "dropped_consecutive_wall_crossing": int(
                self.dropped_consecutive_wall_crossing
            ),
            "dropped_too_few_samples": int(self.dropped_too_few_samples),
            "wiped_windows": int(self.wiped_windows),
            "anchor_conflict_ratio": round(float(self.anchor_conflict_ratio), 4),
            "duration_s": round(float(self.duration_s), 3),
        }


@dataclass(frozen=True)
class PedestrianTrack:
    """筛选后的行人轨迹（**场景世界坐标**，等间隔 ``dt_s``）。"""

    pedestrian_id: int
    times_s: np.ndarray
    xy: np.ndarray
    velocity: np.ndarray
    radius_m: float

    def __post_init__(self) -> None:
        if self.times_s.shape != (self.xy.shape[0],):
            raise PedestrianDataError("times_s 与 xy 长度不一致")

    @property
    def label(self) -> str:
        return f"pedestrian_{self.pedestrian_id}"


@dataclass(frozen=True)
class SequenceTracks:
    """一个序列在一个场景里的全部可用轨迹 + 可用性统计。"""

    sequence: str
    anchor: AnchorMapping
    tracks: tuple[PedestrianTrack, ...]
    report: UsabilityReport
    dt_s: float


def _trim_out_of_bounds(
    xy: np.ndarray, field: np.ndarray, resolution: float, origin: Sequence[float]
) -> np.ndarray:
    """出界（栅格之外）的样本记为"出界"；障碍单元由 clearance 判定覆盖。"""
    values, in_bounds = wall_clearance(
        xy, field, resolution=resolution, origin=origin
    )
    del values
    return ~in_bounds


def _wall_crossing_mask(
    xy: np.ndarray,
    clearance_field: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    body_radius: float,
) -> np.ndarray:
    """行人圆心离墙 ``< body_radius`` 的样本（"行人身体压在墙上"）。"""
    values, in_bounds = wall_clearance(
        xy, clearance_field, resolution=resolution, origin=origin
    )
    return in_bounds & (values < float(body_radius))


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def build_sequence_tracks(
    raw: RawSequence,
    *,
    scene_occupancy: np.ndarray,
    clearance_field: np.ndarray,
    resolution: float,
    origin: Sequence[float],
    anchor: AnchorMapping,
    dt_s: float,
    body_radius: float,
    pedestrian_radius_m: float = DEFAULT_PEDESTRIAN_RADIUS_M,
    max_consecutive_wall_crossings: int = 3,
    max_observation_gap_s: float = 1.5 * DATASET_DT_S,
    min_samples: int = 10,
    velocity_window: int = 5,
) -> SequenceTracks:
    """原始序列 + 锚点 -> 逐帧动态障碍 + 可用性统计（筛选规则见模块说明）。

    统计口径（**必须互斥、不得重复计数**）：

    * 分母 = **连续观测段内按 ``dt_s`` 重采样后的样本数**（``samples_resampled``）；
      原始行数单独记在 ``samples_raw``，两者不混着算比例（0.4 s -> 0.1 s 会加密
      约 4 倍，这是加密不是重复）；
    * ``samples_out_of_bounds`` + ``samples_wall_crossing`` = 被剔除的重采样样本
      （两者互斥：先判出界，再在界内样本里判贴墙）；
    * ``samples_used`` = 实际进入 ``ObstacleTrack`` 的样本数。
      因此 ``used + out_of_bounds + wall_crossing == samples_resampled`` **恒成立**，
      ``sample_usable_ratio <= 1``。被"连续穿墙"整段丢弃的样本计入 ``dropped_*``，
      **不计入** ``samples_used``。
    * 同一行人被切成多段时，段与段不重叠，所以 ``used`` 不会重复计数。
    """
    world = anchor.apply(raw.xy_local)
    times_raw = raw.times_s
    ids = np.asarray(raw.ids, dtype=int)
    samples_raw = int(world.shape[0])

    out_of_bounds_total = 0
    wall_total = 0
    resampled_total = 0
    dropped_runs = 0
    dropped_few = 0
    wiped_windows = 0
    tracks: list[PedestrianTrack] = []
    used_ids: set[int] = set()

    for identifier in np.unique(ids):
        selection = ids == identifier
        track_xy = world[selection]
        track_t = times_raw[selection]
        if track_xy.shape[0] < 2:
            dropped_few += 1
            continue
        # 只在**连续观测段**内重采样，绝不跨观测空洞插值。
        # 阈值必须相对**数据集自身采样间隔**（0.4 s）取，不能相对 dt：
        # 本轮踩过一次坑 —— 取 2*dt = 0.2 s < 0.4 s，于是每个样本都被判成
        # "空洞"，所有轨迹被切成单点段、可用率全 0。
        runs = observation_runs(track_t, max_gap_s=float(max_observation_gap_s))
        if not runs:
            dropped_few += 1
            continue
        produced_any = False
        for run_low, run_high in runs:
            try:
                times, resampled = resample_track(
                    track_t[run_low:run_high], track_xy[run_low:run_high], dt_s=float(dt_s)
                )
            except PedestrianDataError:
                continue
            if times.shape[0] < 2:
                continue
            resampled_total += int(times.shape[0])
            out_of_bounds = _trim_out_of_bounds(
                resampled, clearance_field, resolution, origin
            )
            wall = _wall_crossing_mask(
                resampled, clearance_field, resolution, origin, body_radius
            )
            out_of_bounds_total += int(out_of_bounds.sum())
            wall_total += int(wall.sum())

            # 连续穿墙超过阈值：该段丢弃（说明这段整体落在墙里）。
            # 这类样本**不计入** samples_used —— 否则"可用率"会超过 1。
            longest_wall_run = _longest_run(wall)
            if longest_wall_run > int(max_consecutive_wall_crossings):
                dropped_runs += 1
                wiped_windows += int(
                    math.ceil(
                        longest_wall_run / max(1, int(max_consecutive_wall_crossings))
                    )
                )
                continue

            keep = ~out_of_bounds & ~wall
            # 逐点剔除后按"连续可用段"切分：时间轴出现空洞就必须断开，
            # 否则插值会把行人瞬移穿过被剔除的墙段（这会让指标虚假变好）。
            segments: list[tuple[int, int]] = []
            start = None
            for index, flag in enumerate(keep):
                if flag and start is None:
                    start = index
                elif not flag and start is not None:
                    segments.append((start, index))
                    start = None
            if start is not None:
                segments.append((start, keep.shape[0]))
            for low, high in segments:
                if high - low < int(min_samples):
                    continue
                segment_xy = resampled[low:high]
                segment_t = times[low:high]
                velocity = smooth_velocity(
                    segment_xy, float(dt_s), window=velocity_window
                )
                tracks.append(
                    PedestrianTrack(
                        pedestrian_id=int(identifier),
                        times_s=segment_t,
                        xy=segment_xy,
                        velocity=velocity,
                        radius_m=float(pedestrian_radius_m),
                    )
                )
                produced_any = True
        if produced_any:
            used_ids.add(int(identifier))

    samples_used = int(sum(track.xy.shape[0] for track in tracks))
    denominator = max(1, resampled_total)
    report = UsabilityReport(
        sequence=raw.name,
        pedestrians_total=int(np.unique(ids).size),
        pedestrians_used=len(used_ids),
        samples_raw=samples_raw,
        samples_resampled=resampled_total,
        samples_out_of_bounds=out_of_bounds_total,
        samples_wall_crossing=wall_total,
        samples_used=samples_used,
        dropped_consecutive_wall_crossing=dropped_runs,
        dropped_too_few_samples=dropped_few,
        anchor_conflict_ratio=float("nan"),
        duration_s=raw.duration_s,
        usable_pedestrians=(
            len(used_ids) / float(np.unique(ids).size) if np.unique(ids).size else 0.0
        ),
        out_of_bounds_ratio=out_of_bounds_total / denominator,
        wall_crossing_ratio=wall_total / denominator,
        sample_usable_ratio=samples_used / denominator,
        wiped_windows=wiped_windows,
    )
    return SequenceTracks(
        sequence=raw.name,
        anchor=anchor,
        tracks=tuple(tracks),
        report=report,
        dt_s=float(dt_s),
    )


def window_usability(
    tracks: Sequence[PedestrianTrack],
    *,
    window_start_s: float,
    window_length_s: float,
    dt_s: float,
    body_radius: float,
) -> tuple[UsabilityReport, tuple[PedestrianTrack, ...]]:
    """**观察窗内**的可用性统计（与全序列统计分列，两者不是同一个分母）。

    为什么要分列：全序列可用率回答"这份数据有多少能用"，而 episode 只跑一个
    ``window_s`` 长的观察窗；窗口内的可用率可能更好或更差，
    混在一起报会让"episode 的输入质量"无从判断。

    窗口内的可用样本数按"该行人在窗口内被观测到的周期数"计，
    因此 ``样本可用率 <= 1`` 恒成立。
    """
    start = float(window_start_s)
    end = start + float(window_length_s)
    grid = np.arange(start, end + 1e-9, float(dt_s))
    if grid.size == 0:
        grid = np.array([start], dtype=float)
    covered = np.zeros(grid.shape[0], dtype=bool)
    kept: list[PedestrianTrack] = []
    for track in tracks:
        times = np.asarray(track.times_s, dtype=float)
        xy = np.asarray(track.xy, dtype=float)
        keep = (times >= start) & (times <= end)
        if int(keep.sum()) < 2:
            continue
        index = np.clip(
            np.searchsorted(grid, times[keep], side="left"), 0, grid.shape[0] - 1
        )
        covered[index] = True
        kept.append(
            PedestrianTrack(
                pedestrian_id=track.pedestrian_id,
                times_s=times[keep],
                xy=xy[keep],
                velocity=np.asarray(track.velocity, dtype=float)[keep],
                radius_m=float(track.radius_m),
            )
        )
    samples_total = int(grid.shape[0])
    samples_used = int(covered.sum())
    report = UsabilityReport(
        sequence="window",
        pedestrians_total=len(tracks),
        pedestrians_used=len(kept),
        samples_raw=samples_total,
        samples_resampled=samples_total,
        samples_out_of_bounds=0,
        samples_wall_crossing=0,
        samples_used=samples_used,
        dropped_consecutive_wall_crossing=0,
        dropped_too_few_samples=max(0, len(tracks) - len(kept)),
        anchor_conflict_ratio=float("nan"),
        duration_s=float(window_length_s),
        usable_pedestrians=(len(kept) / len(tracks) if tracks else 0.0),
        out_of_bounds_ratio=0.0,
        wall_crossing_ratio=0.0,
        sample_usable_ratio=(samples_used / samples_total if samples_total else 0.0),
    )
    return report, tuple(kept)


def with_conflict_ratio(
    tracks: SequenceTracks, conflict_ratio: float
) -> SequenceTracks:
    """回填拟合阶段的冲突比例（拟合与筛选分成两步，报告里必须都留）。"""
    report = tracks.report
    updated = UsabilityReport(
        sequence=report.sequence,
        pedestrians_total=report.pedestrians_total,
        pedestrians_used=report.pedestrians_used,
        samples_raw=report.samples_raw,
        samples_resampled=report.samples_resampled,
        samples_out_of_bounds=report.samples_out_of_bounds,
        samples_wall_crossing=report.samples_wall_crossing,
        samples_used=report.samples_used,
        dropped_consecutive_wall_crossing=report.dropped_consecutive_wall_crossing,
        dropped_too_few_samples=report.dropped_too_few_samples,
        anchor_conflict_ratio=float(conflict_ratio),
        duration_s=report.duration_s,
        usable_pedestrians=report.usable_pedestrians,
        out_of_bounds_ratio=report.out_of_bounds_ratio,
        wall_crossing_ratio=report.wall_crossing_ratio,
        sample_usable_ratio=report.sample_usable_ratio,
        wiped_windows=report.wiped_windows,
    )
    return SequenceTracks(
        sequence=tracks.sequence,
        anchor=tracks.anchor,
        tracks=tracks.tracks,
        report=updated,
        dt_s=tracks.dt_s,
    )


# ---------------------------------------------------------------------------
# 5. 与 DWA 对接
# ---------------------------------------------------------------------------


def to_obstacle_tracks(
    tracks: Sequence[PedestrianTrack], *, t_offset_s: float = 0.0
) -> tuple[Any, ...]:
    """``PedestrianTrack`` -> ``core.dwa.ObstacleTrack``（DWA 侧零改动）。

    ``t_offset_s`` 把行人时间轴平移到 episode 的机器人时间轴上（同一任务里
    行人的起始时刻由 seed 固定并计入哈希）。
    """
    from core import dwa as dwa_module

    built = []
    for track in tracks:
        built.append(
            dwa_module.ObstacleTrack(
                times=np.asarray(track.times_s, dtype=float) + float(t_offset_s),
                xy=np.asarray(track.xy, dtype=float),
                radius_m=float(track.radius_m),
                label=track.label,
            )
        )
    return tuple(built)


def episode_time_range(
    tracks: Sequence[PedestrianTrack], *, dt_s: float | None = None
) -> tuple[float, float]:
    """所有轨迹的公共时间范围（用于把行人时间轴对齐到 episode）。

    ``dt_s`` 只用于自检（相邻采样步长是否等于重采样步长），不传则跳过。
    """
    if not tracks:
        return (0.0, 0.0)
    starts = [float(np.asarray(track.times_s)[0]) for track in tracks]
    ends = [float(np.asarray(track.times_s)[-1]) for track in tracks]
    if dt_s is not None and float(dt_s) > 0:
        for track in tracks:
            steps = np.diff(np.asarray(track.times_s, dtype=float))
            if steps.size and np.any(np.abs(steps - float(dt_s)) > 1e-6):
                raise PedestrianDataError("轨迹时间基不是等间隔 dt_s（重采样有误）")
    return (float(min(starts)), float(max(ends)))


def common_window(
    tracks: Sequence[PedestrianTrack], *, window_s: float, t_offset_s: float = 0.0
) -> tuple[tuple[Any, ...], float, float]:
    """截取一段**公共时间窗**，保持原有时间基（不重设 t=0）。

    为什么要公共窗：一个序列里不同行人的时间范围不同，DWA 的每条 ``ObstacleTrack``
    都按绝对时间查询；不截公共窗的话，早退出的行人会被"最近端点外推"钉在原地，
    变成一个永久静止障碍物，把走廊堵死。

    **时间基不重设**：整条序列（含全部行人）已经共享同一个公共时间栅格，
    截窗只做裁切，所有轨迹继续对齐；重设 t=0 会让"本车 ↔ 行人"的逐周期对齐
    又多一层偏移。
    """
    if not tracks:
        return ((), 0.0, 0.0)
    start, end = episode_time_range(tracks)
    start = max(start, float(t_offset_s))
    end = min(end, start + float(window_s))
    if end <= start:
        return ((), start, 0.0)
    clipped: list[PedestrianTrack] = []
    for track in tracks:
        times = np.asarray(track.times_s, dtype=float)
        xy = np.asarray(track.xy, dtype=float)
        keep = (times >= start) & (times <= end)
        if int(keep.sum()) < 2:
            continue
        clipped.append(
            PedestrianTrack(
                pedestrian_id=track.pedestrian_id,
                times_s=times[keep],
                xy=xy[keep],
                velocity=np.asarray(track.velocity, dtype=float)[keep],
                radius_m=float(track.radius_m),
            )
        )
    return (tuple(clipped), start, end - start)


def stable_seed(token: str, salt: str = "") -> int:
    """由字符串确定性地派生一个非负整数种子（同输入同输出）。

    用途：行人时间窗起点、任务级随机量都必须是 ``(场景, 序列, seed)`` 的纯函数，
    这样"随机"这件事才可复现、可写进哈希。
    """
    digest = hashlib.sha256(f"{token}|{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def track_summary(tracks: Sequence[PedestrianTrack]) -> dict[str, float]:
    """轨迹规模摘要（报告与测试用）。"""
    if not tracks:
        return {"pedestrians": 0.0, "samples": 0.0, "mean_speed_mps": 0.0, "max_speed_mps": 0.0}
    speeds = np.concatenate([np.linalg.norm(track.velocity, axis=1) for track in tracks])
    return {
        "pedestrians": float(len(tracks)),
        "samples": float(sum(track.xy.shape[0] for track in tracks)),
        "mean_speed_mps": float(np.mean(speeds)),
        "max_speed_mps": float(np.max(speeds)),
    }


def sequence_config(
    doc: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    """从 ``config/pedestrian_sequences.yaml`` 取出一个序列声明。"""
    sequences = doc.get("sequences")
    if not isinstance(sequences, Mapping) or name not in sequences:
        raise PedestrianDataError(f"pedestrian_sequences.yaml 里没有序列 {name!r}")
    entry = sequences[name]
    if not isinstance(entry, Mapping):
        raise PedestrianDataError(f"序列 {name!r} 的声明必须是映射")
    return entry
