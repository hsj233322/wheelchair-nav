"""RDP 简化 + 三次均匀 B 样条平滑（纯函数，无全局状态，不 import config）。

链路里的位置
------------
A* 给出的是**栅格折线**：8 邻域步长 0.1~0.141 m，转角只能是 45° 的整数倍，
拐点处曲率是尖峰。本模块把折线变成 ``位置 / 切向 / 曲率随弧长`` 的参数曲线，
但**不放松任何硬约束**——平滑只是"在可行域内换一种写法"，不是"让不可行的变可行"。

四道关口
--------
1. **端点重合（clamped）**：三次均匀 B 样条用 clamped 节点向量，首末节点各重复
   ``p+1`` 次，曲线严格经过首末控制点；而首末控制点就是 A* 的起点与终点。
   不做这一步会得到"曲线不经过起点"的经典 bug。
2. **曲率约束**：``|kappa(s)| <= kappa_max`` 在**密集采样**上逐点检查并报告
   ``max kappa``。这里的 ``kappa_max`` 与 ``core/astar.py`` 的曲率窗口是**同一个
   参数**（config 的 ``astar.kappa_max``），不分裂成两套：A* 用它做搜索侧代理约束，
   本模块用它做平滑侧的精确校验。
3. **离障约束**：平滑后全程 ``clearance >= d_min``，用**距离场逐点检查**
   （双线性插值），而不是"采样点是否落在膨胀栅格里"——后者在栅格离散误差下会
   放过贴墙的曲线。
4. **能耗预算复检**：平滑会改变几何长度，必须重新验证 ``E <= e_budget``。超出时
   **优先插入控制点**（把曲线往原始折线上拉，缩短长度），仍不满足才回退。

控制点调整策略（形式化）
-----------------------
三条约束对**控制点间距** ``h`` 的几何敏感性方向相反，因此调整方向必须按失败关口分开：

| 调整动作 | 对离障 clearance | 对曲率 kappa 峰值 |
|---|---|---|
| **加密**（h↓） | **升**（更贴折线，切角小） | **升**（间距越小，转角越"尖"） |
| **放稀**（h↑） | **降**（转角被切掉更多） | **降**（曲线更接近低曲率光滑弧） |

由此得到唯一自洽的调整规则：

* **离障失败 → 加密**：要的就是 clearance 回升；
* **曲率失败 → 放稀**：要的就是 ``|kappa|`` 下降；
* **曲率失败但已到 4 个控制点下限** → 只能**加密**：此时曲率过大的原因不是
  "间距太小"，而是"控制点太少、圆化半径太大，把两个拐角连成了一个大弯"。
  加密会把大弯拆回小弯，曲率随之下降。这条分支由 ``houseexpo_e888f399`` 实测暴露
  （放稀到 4 点后 κ=2.14 仍超限，加密 1 点后降到 0.66）；
* **能耗失败 → 放稀**：转角切得更多 → 几何长度与电能下降。

递归深度 <= ``max_insert_depth``（默认 3）。

回退的定性（不是"降级成功"，是**约束违反**）
-------------------------------------------
深度用尽仍失败则回退到 RDP 折线。**回退产物违反曲率约束**：折线在拐点处不可导，
曲率无界（``max_abs_curvature = inf``），所以它**不是可跟踪的参考轨迹，
不得交给局部规划器**。它只有一个用途——作为"该任务在当前安全裕度下无法生成
满足曲率约束的平滑路径"的证据，以及给论文提供``curvature_infeasible`` 的计数。

因此 ``SmoothPath`` 区分两个独立概念：

* ``success``：**几何链路是否产出结果**（回退也算 True，因为它确实产出了折线）；
* ``curvature_ok`` / ``curvature_infeasible``：**曲率硬约束是否真的成立**。
  回退恒为 ``curvature_infeasible = True``。

**任何统计都不允许用 `success` 冒充 `curvature_ok`**，见 ``report_dict`` 的字段命名。
回退是**正确行为**而非缺陷——hard 场景瓶颈 1.46~1.50 m、``d_min = 0.70`` 时可用带宽
只有约 0.10 m，平滑自由度本来就接近 0；但要如实记为"约束不可满足"。

RDP 阈值为什么不能用"拍脑袋的一个米数"
--------------------------------------
A* 折线上有大量栅格阶梯（对角步与直步交替），阶梯振幅是**一格**。若 RDP 阈值小于
该振幅，锯齿会被当成真实几何保留下来，被三次曲线忠实穿过后曲率会爆掉
（0.1 m 内转 45° ≈ 7.8 1/m）。所以 ``rdp_epsilon_m`` 必须大于一格，默认 0.15 m
（≈1.5 × resolution）——这是**相对栅格分辨率**的量，不是为了让图好看而调的数。

单位：m, rad, 1/m, J。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core import energy

EPS = 1e-9
_DEGREE = 3
_DEFAULT_SAMPLE_STEP_M = 0.02
_DEFAULT_RDP_EPSILON_M = 0.15


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BSplineParams:
    """平滑参数。全部由调用方显式传入（core 不读配置）。

    ``kappa_max`` / ``d_min`` / ``e_budget`` 必须与 ``core.astar.AStarParams``
    里的同名参数**取同一个值**，否则曲率窗口与平滑侧校验会分裂成两套口径。
    """

    kappa_max: float
    d_min: float
    e_budget: float
    control_spacing_m: float = 0.8
    rdp_epsilon_m: float = _DEFAULT_RDP_EPSILON_M
    sample_step_m: float = _DEFAULT_SAMPLE_STEP_M
    max_insert_depth: int = 3
    densify_factor: float = 2.0

    def __post_init__(self) -> None:
        if not self.kappa_max > 0:
            raise ValueError(f"kappa_max 必须 > 0，实际为 {self.kappa_max}")
        if self.d_min < 0:
            raise ValueError(f"d_min 必须 >= 0，实际为 {self.d_min}")
        if self.e_budget < 0:
            raise ValueError(f"e_budget 必须 >= 0，实际为 {self.e_budget}")
        if not self.rdp_epsilon_m > 0:
            raise ValueError(f"rdp_epsilon_m 必须 > 0，实际为 {self.rdp_epsilon_m}")
        if not self.control_spacing_m > 0:
            raise ValueError(
                f"control_spacing_m 必须 > 0，实际为 {self.control_spacing_m}"
            )
        if not self.sample_step_m > 0:
            raise ValueError(f"sample_step_m 必须 > 0，实际为 {self.sample_step_m}")
        if not self.densify_factor > 1.0:
            raise ValueError(
                f"densify_factor 必须 > 1，实际为 {self.densify_factor}"
            )
        if not 0 <= self.max_insert_depth <= 8:
            raise ValueError(
                f"max_insert_depth 必须在 [0, 8] 内，实际为 {self.max_insert_depth}"
            )


# ---------------------------------------------------------------------------
# 控制点构造：弧长等距重采样
# ---------------------------------------------------------------------------


def _arc_lengths(points: np.ndarray) -> np.ndarray:
    """折线各顶点处的累计弧长。"""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def resample_by_arclength(points: np.ndarray, spacing_m: float) -> np.ndarray:
    """沿折线按**等弧长**重采样，得到曲率有界的控制多边形。

    为什么必须等距：三次均匀 B 样条的曲率尺度由**控制点间距**决定。A* 的折线
    顶点间距极不均匀（直线段一段 9 m、拐角处相邻 0.1 m），直接当控制点会在
    密集处产生尖峰曲率。等弧长重采样把"转角的圆化半径"锁定为约 ``spacing_m``，
    从而让 ``|kappa| <= kappa_max`` 有可控的几何上界（经验上
    ``kappa_max ~ 1/spacing_m``，见 ``docs`` 里的间距扫描）。

    首末点精确保留（端点重合），采样点落在原折线**线段上**——即已验证可行的区域。
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("折线至少需要 2 个顶点")
    spacing = float(spacing_m)
    if not spacing > 0:
        raise ValueError(f"spacing_m 必须 > 0，实际为 {spacing_m}")
    s = _arc_lengths(pts)
    total = float(s[-1])
    if not total > EPS:
        return pts.copy()
    count = max(_DEGREE + 1, int(round(total / spacing)) + 1)
    query = np.linspace(0.0, total, count)
    resampled = np.column_stack(
        [
            np.interp(query, s, pts[:, 0]),
            np.interp(query, s, pts[:, 1]),
        ]
    )
    # 逐点插值在两端与原点完全一致；这里显式钉住，避免浮点插值带来的 1e-16 漂移。
    resampled[0] = pts[0]
    resampled[-1] = pts[-1]
    return resampled


# ---------------------------------------------------------------------------
# RDP 简化
# ---------------------------------------------------------------------------


def rdp_indices(points: np.ndarray, epsilon_m: float) -> np.ndarray:
    """RDP 简化，返回**保留点在原数组里的下标**（升序，含首末）。

    返回下标而不是点，是因为后续"插入控制点"必须从**原始 A* 节点**里挑，
    用下标才能保证控制点始终是原始路径的子序列。
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points 必须是 (N, 2) 数组，实际形状为 {pts.shape}")
    eps = float(epsilon_m)
    if not eps > 0:
        raise ValueError(f"epsilon_m 必须 > 0，实际为 {epsilon_m}")
    count = pts.shape[0]
    if count < 3:
        return np.arange(count, dtype=int)

    keep = np.zeros(count, dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, count - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        anchor, target = pts[first], pts[last]
        span = target - anchor
        span_len = float(np.hypot(span[0], span[1]))
        middle = pts[first + 1 : last]
        if span_len < EPS:
            deviations = np.linalg.norm(middle - anchor, axis=1)
        else:
            normal = np.array([-span[1], span[0]], dtype=float) / span_len
            deviations = np.abs((middle - anchor) @ normal)
        worst = int(np.argmax(deviations))
        if deviations[worst] > eps:
            split = first + 1 + worst
            keep[split] = True
            stack.append((first, split))
            stack.append((split, last))
    return np.nonzero(keep)[0]


def rdp_simplify(points: np.ndarray, epsilon_m: float) -> np.ndarray:
    """RDP 折线简化，保留首末点（``rdp_indices`` 的取值版本）。"""
    pts = np.asarray(points, dtype=float)
    return pts[rdp_indices(pts, epsilon_m)].copy()


# ---------------------------------------------------------------------------
# 三次均匀 B 样条
# ---------------------------------------------------------------------------


def _uniform_clamped_knots(control_count: int, degree: int = _DEGREE) -> np.ndarray:
    """三次**均匀** B 样条的 clamped 节点向量。

    节点在 ``[0, 1]`` 上**等距**，首末各重复 ``degree + 1`` 次。这就是"三次均匀
    B 样条"的字面含义：内部节点间距全相等，只有两端为了经过首末控制点而重节点。

    重节点保证 ``C(0) = P_0``、``C(1) = P_n``，即**端点重合**——不做这一步是经典 bug。
    内部每个节点只重复一次，因此插值曲线在节点处 ``C^{degree-1} = C^2`` 连续。
    """
    if control_count < degree + 1:
        raise ValueError(
            f"控制点数 {control_count} 少于 degree+1 = {degree + 1}，无法构造 B 样条"
        )
    inner_count = control_count - degree - 1
    inner = np.arange(1, inner_count + 1, dtype=float) / (inner_count + 1)
    return np.concatenate([[0.0] * (degree + 1), inner, [1.0] * (degree + 1)])


def _basis(index: int, degree: int, u: float, knots: np.ndarray) -> float:
    """Cox-de Boor 递推的单项基函数 ``N_{index, degree}(u)``。"""
    if degree == 0:
        lower, upper = knots[index], knots[index + 1]
        if (lower <= u < upper) or (u == knots[-1] and lower <= u <= upper):
            return 1.0
        return 0.0
    left = 0.0
    denominator = knots[index + degree] - knots[index]
    if denominator > EPS:
        left = (u - knots[index]) / denominator * _basis(index, degree - 1, u, knots)
    right = 0.0
    denominator = knots[index + degree + 1] - knots[index + 1]
    if denominator > EPS:
        right = (
            (knots[index + degree + 1] - u)
            / denominator
            * _basis(index + 1, degree - 1, u, knots)
        )
    return left + right


def _basis_derivative(index: int, degree: int, u: float, knots: np.ndarray) -> float:
    """一阶导 ``N'_{index, degree}(u)``。"""
    left = 0.0
    denominator = knots[index + degree] - knots[index]
    if denominator > EPS:
        left = degree / denominator * _basis(index, degree - 1, u, knots)
    right = 0.0
    denominator = knots[index + degree + 1] - knots[index + 1]
    if denominator > EPS:
        right = degree / denominator * _basis(index + 1, degree - 1, u, knots)
    return left - right


def _basis_second_derivative(
    index: int, degree: int, u: float, knots: np.ndarray
) -> float:
    """二阶导 ``N''_{index, degree}(u)``。"""
    left = 0.0
    denominator = knots[index + degree] - knots[index]
    if denominator > EPS:
        left = degree / denominator * _basis_derivative(index, degree - 1, u, knots)
    right = 0.0
    denominator = knots[index + degree + 1] - knots[index + 1]
    if denominator > EPS:
        right = degree / denominator * _basis_derivative(index + 1, degree - 1, u, knots)
    return left - right


def _basis_matrix(
    params: np.ndarray, control_count: int, knots: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基函数及其一、二阶导的矩阵，形状均为 ``(len(params), control_count)``。

    三次 B 样条在每个参数处只有 4 个非零基函数（局部支撑），所以这里按
    "每个参数只算它所属的 4 个支撑区间"来填，复杂度 O(N) 而不是 O(N*M)。
    """
    size = params.shape[0]
    count = control_count
    value = np.zeros((size, count), dtype=float)
    first = np.zeros_like(value)
    second = np.zeros_like(value)
    for row, u in enumerate(params):
        span = _find_span(u, knots, count, _DEGREE)
        low = span - _DEGREE
        high = span
        for index in range(max(0, low), min(count, high + 1)):
            value[row, index] = _basis(index, _DEGREE, u, knots)
            first[row, index] = _basis_derivative(index, _DEGREE, u, knots)
            second[row, index] = _basis_second_derivative(index, _DEGREE, u, knots)
    return value, first, second


def _find_span(u: float, knots: np.ndarray, count: int, degree: int) -> int:
    """找到 ``u`` 所在的节点区间下标（clamped 边界处取 ``count - 1``）。"""
    if u >= knots[count]:
        return count - 1
    low, high = degree, count
    while high - low > 1:
        middle = (low + high) // 2
        if u < knots[middle]:
            high = middle
        else:
            low = middle
    return low


def _curvature(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """有符号曲率 ``(x'y'' - y'x'') / |r'|^3``。"""
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    speed = np.linalg.norm(d1, axis=1)
    return cross / np.maximum(speed, EPS) ** 3


@dataclass(frozen=True)
class SplineCurve:
    """一条参数曲线：位置 / 一阶导 / 二阶导 / 曲率，均随归一化参数 u。"""

    data_points: np.ndarray
    control_points: np.ndarray
    knots: np.ndarray
    u: np.ndarray
    xy: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    curvature: np.ndarray
    s: np.ndarray
    length_m: float
    yaw: np.ndarray

    @property
    def max_abs_curvature(self) -> float:
        return float(np.max(np.abs(self.curvature))) if self.curvature.size else 0.0

    def at(self, u_value: float) -> np.ndarray:
        """在任意参数处求位置（测试与绘图用，不参与规划）。"""
        value, _, _ = _basis_matrix(
            np.array([float(u_value)]), self.control_points.shape[0], self.knots
        )
        return value @ self.control_points


def build_curve(
    points: np.ndarray, *, sample_step_m: float = _DEFAULT_SAMPLE_STEP_M
) -> SplineCurve:
    """由控制点生成三次**均匀** clamped B 样条并**按弧长步长密集采样**。

    控制点就是传入的点（RDP 顶点 / 插入后的原始节点）。均匀节点意味着这是一条
    **逼近**曲线而不是插值曲线：它不保证逐个穿过中间控制点，但保证
    * 首末控制点被精确经过（clamped 端点重合）；
    * 每个内部节点处位置、一阶、二阶导连续（C²）；
    * 整条曲线落在控制多边形的凸包内，因此"控制点可行"是"曲线可行"的充分条件——
      这正是后续离障检查能收敛的前提。

    逼近而非插值还避免了按弦长参数化求解线性系统的病态问题（间距不均时会奇异）。
    """
    data = np.asarray(points, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"控制点必须是 (N, 2) 数组，实际形状为 {data.shape}")
    if data.shape[0] < _DEGREE + 1:
        raise ValueError(
            f"三次 B 样条至少需要 {_DEGREE + 1} 个控制点，实际为 {data.shape[0]}"
        )
    knots = _uniform_clamped_knots(data.shape[0])

    coarse = np.linspace(0.0, 1.0, 257)
    rough_value, _, _ = _basis_matrix(coarse, data.shape[0], knots)
    rough = float(np.sum(np.linalg.norm(np.diff(rough_value @ data, axis=0), axis=1)))
    count = min(max(64, int(math.ceil(rough / float(sample_step_m))) + 1), 50_000)
    params = np.linspace(0.0, 1.0, count)

    value, first, second = _basis_matrix(params, data.shape[0], knots)
    xy = value @ data
    d1 = first @ data
    d2 = second @ data
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    return SplineCurve(
        data_points=data,
        control_points=data,
        knots=knots,
        u=params,
        xy=xy,
        d1=d1,
        d2=d2,
        curvature=_curvature(d1, d2),
        s=s,
        length_m=float(s[-1]),
        yaw=np.arctan2(d1[:, 1], d1[:, 0]),
    )


# ---------------------------------------------------------------------------
# 距离场采样（离障检查）
# ---------------------------------------------------------------------------


def sample_field(
    field: np.ndarray,
    xy: np.ndarray,
    *,
    resolution: float,
    origin: tuple[float, float] | np.ndarray = (0.0, 0.0),
) -> np.ndarray:
    """在单元中心距离场上做**双线性插值**（逐点取离障距离 / 路面参数）。

    约定与 ``core.astar`` 一致：第 0 行是 y 最小侧，单元中心为
    ``origin + (i + 0.5) * resolution``。双线性插值比"取最近单元"更保守地反映
    墙面真实距离，避免曲线恰好落在两个障碍单元之间时被高估 clearance。
    落在网格外的点按 0 处理（地图外视为障碍）。
    """
    grid = np.asarray(field, dtype=float)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("field 必须是二维非空数组")
    pts = np.asarray(xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"xy 必须是 (N, 2) 数组，实际形状为 {pts.shape}")
    res = float(resolution)
    if not res > 0:
        raise ValueError(f"resolution 必须 > 0，实际为 {resolution}")
    origin_array = np.asarray(origin, dtype=float)
    ox, oy = float(origin_array[0]), float(origin_array[1])

    rows, cols = grid.shape
    if rows < 2 or cols < 2:
        raise ValueError(f"距离场至少需要 2x2，实际为 {grid.shape}")
    gx = (pts[:, 0] - ox) / res - 0.5
    gy = (pts[:, 1] - oy) / res - 0.5
    inside = (gx >= 0) & (gx <= cols - 1) & (gy >= 0) & (gy <= rows - 1)
    x0 = np.clip(np.floor(gx).astype(int), 0, cols - 2)
    y0 = np.clip(np.floor(gy).astype(int), 0, rows - 2)
    fx = np.clip(gx - x0, 0.0, 1.0)
    fy = np.clip(gy - y0, 0.0, 1.0)
    values = (
        grid[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + grid[y0, x0 + 1] * fx * (1.0 - fy)
        + grid[y0 + 1, x0] * (1.0 - fx) * fy
        + grid[y0 + 1, x0 + 1] * fx * fy
    )
    return np.where(inside, values, 0.0)


def _field_at_points(
    values: np.ndarray | float,
    xy: np.ndarray,
    *,
    resolution: float,
    origin: tuple[float, float] | np.ndarray,
) -> np.ndarray:
    """标量参数直接广播，数组参数按距离场双线性插值。"""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(xy.shape[0], float(array))
    return sample_field(array, xy, resolution=resolution, origin=origin)


# ---------------------------------------------------------------------------
# 能耗复检
# ---------------------------------------------------------------------------


def curve_energy(
    curve: SplineCurve,
    *,
    speed: float,
    energy_params: energy.EnergyParams,
    c_rr: np.ndarray | float,
    slope_rad: np.ndarray | float,
    resolution: float,
    origin: tuple[float, float] | np.ndarray = (0.0, 0.0),
) -> float:
    """按**实际几何长度**重新估算这条曲线的电能 [J]。

    记账口径与 ``core.astar`` 保持一致：稳态巡航（``accel=0``）、逐段
    ``step_energy(distance, speed, ...)``、坡度按该段运动方向的 x 分量计。
    采样步长默认 0.02 m，远小于路面分区尺度，逐段求和与弧长积分等价。
    """
    pts = curve.xy
    if pts.shape[0] < 2:
        return 0.0
    deltas = np.diff(pts, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    midpoints = 0.5 * (pts[:-1] + pts[1:])
    c_rr_values = _field_at_points(
        c_rr, midpoints, resolution=resolution, origin=origin
    )
    slope_values = _field_at_points(
        slope_rad, midpoints, resolution=resolution, origin=origin
    )
    direction_x = deltas[:, 0] / np.maximum(distances, EPS)

    usable = distances > EPS
    if not np.any(usable):
        return 0.0
    total = 0.0
    for index in np.nonzero(usable)[0]:
        total += energy.step_energy(
            float(distances[index]),
            float(speed),
            c_rr=float(c_rr_values[index]),
            slope_rad=float(slope_values[index]),
            direction_x=float(direction_x[index]),
            params=energy_params,
        )
    return float(total)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothingReport:
    """一次平滑的完整过程记录，供**曲率不可满足率统计**与论文口径使用。

    注意 ``status`` 表达的是"几何链路产出了什么"，**不是**"约束是否满足"：

    * ``status == "spline"``：产出了 B 样条曲线，三道关口全部通过；
    * ``status == "polyline"``：回退到 RDP 折线。**这不是降级成功，是约束违反**——
      折线拐点处曲率无界，``curvature_infeasible`` 恒为 True。
    """

    status: str
    """``spline``：产出且约束满足；``polyline``：回退（曲率约束被违反）。"""

    reason: str
    """``ok`` / ``curvature`` / ``clearance`` / ``energy`` / ``insert_exhausted``。"""

    inserted_points: int
    """跨全部递归层累加插入的控制点数。"""

    removed_points: int
    """因曲率 / 能耗失败而被"放稀"掉的控制点数。"""

    depth: int
    """实际用到的调整递归深度。"""

    rounds: int
    """尝试生成曲线的轮数（含首轮）。"""

    control_points: int
    """最终（或最后一次尝试时）的控制点个数。"""

    max_abs_curvature: float
    """``inf`` 表示曲率无界（回退折线），即曲率约束**被违反**。"""

    min_clearance: float
    energy_j: float
    length_m: float
    rdp_vertices: int

    @property
    def fell_back(self) -> bool:
        return self.status == "polyline"

    @property
    def curvature_satisfied(self) -> bool:
        """曲率硬约束是否**真的**成立（回退恒为 False）。"""
        return self.status == "spline" and math.isfinite(self.max_abs_curvature)

    @property
    def curvature_infeasible(self) -> bool:
        """曲率约束被违反。**这是报告曲率口径的唯一入口**，不要用 ``fell_back`` 代替。"""
        return not self.curvature_satisfied

    @property
    def used_insertion(self) -> bool:
        return self.inserted_points > 0

    @property
    def adjusted(self) -> bool:
        """是否触发过控制点调整（插入或放稀）。"""
        return self.inserted_points > 0 or self.removed_points > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "produced_spline": int(self.status == "spline"),
            "curvature_satisfied": int(self.curvature_satisfied),
            "curvature_infeasible": int(self.curvature_infeasible),
            "inserted_points": self.inserted_points,
            "removed_points": self.removed_points,
            "depth": self.depth,
            "rounds": self.rounds,
            "control_points": self.control_points,
            "max_abs_curvature": self.max_abs_curvature,
            "min_clearance": self.min_clearance,
            "energy_j": self.energy_j,
            "length_m": self.length_m,
            "rdp_vertices": self.rdp_vertices,
        }


@dataclass(frozen=True)
class SmoothPath:
    """平滑结果：统一成"位置 / 切向 / 曲率随弧长"的接口。

    无论走 B 样条还是回退折线，字段含义一致，调用方不需要分支。
    **但两者语义不同**：回退折线违反曲率约束，不是可跟踪的参考轨迹。
    """

    success: bool
    """几何链路是否产出结果。**回退也算 True**，它不代表约束满足。"""

    status: str
    xy: np.ndarray
    yaw: np.ndarray
    s: np.ndarray
    curvature: np.ndarray
    length_m: float
    energy_j: float
    clearance: np.ndarray
    min_clearance: float
    max_abs_curvature: float
    kappa_max: float
    curve: SplineCurve | None
    polyline: np.ndarray
    report: SmoothingReport

    @property
    def fell_back(self) -> bool:
        return self.status == "polyline"

    @property
    def is_spline(self) -> bool:
        return self.status == "spline"

    @property
    def curvature_ok(self) -> bool:
        """曲率硬约束是否成立。回退恒为 False。"""
        return math.isfinite(self.max_abs_curvature) and (
            self.max_abs_curvature <= self.kappa_max + EPS
        )

    @property
    def curvature_infeasible(self) -> bool:
        return not self.curvature_ok

    @property
    def trackable(self) -> bool:
        """能否作为**可跟踪的参考轨迹**交给局部规划器。

        回退折线在拐点处曲率无界，任何有限转向能力的底盘都跟不动，故为 False。
        """
        return self.curvature_ok and not self.fell_back

    def as_dict(self) -> dict[str, object]:
        return {
            "success": int(self.success),
            "status": self.status,
            "fell_back": int(self.fell_back),
            "curvature_satisfied": int(self.curvature_ok),
            "curvature_infeasible": int(self.curvature_infeasible),
            "trackable": int(self.trackable),
            "used_insertion": int(self.report.used_insertion),
            "inserted_points": self.report.inserted_points,
            "removed_points": self.report.removed_points,
            "depth": self.report.depth,
            "rounds": self.report.rounds,
            "control_points": self.report.control_points,
            "reason": self.report.reason,
            "length_m": self.length_m,
            "energy_j": self.energy_j,
            "min_clearance_m": self.min_clearance,
            "max_abs_curvature": self.max_abs_curvature,
            "kappa_max": self.kappa_max,
            "rdp_vertices": self.report.rdp_vertices,
        }


# ---------------------------------------------------------------------------
# 折线参数化（回退分支）
# ---------------------------------------------------------------------------


def _polyline_path(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """把折线按弧长参数化，返回 ``(yaw, s, curvature, length)``。

    折线在顶点处不可导，曲率是**无界**的尖峰；这里用 ``inf`` 如实表示，
    而不是填一个有限值假装它是平滑的。
    """
    pts = np.asarray(vertices, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("折线至少需要 2 个顶点")
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    yaw = np.zeros(pts.shape[0], dtype=float)
    yaw[:-1] = np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0]))
    yaw[-1] = yaw[-2]
    curvature = np.zeros(pts.shape[0], dtype=float)
    curvature[1:-1] = np.inf
    return yaw, s, curvature, float(s[-1])


# ---------------------------------------------------------------------------
# 控制点插入策略
# ---------------------------------------------------------------------------


def _dedupe(points: np.ndarray) -> np.ndarray:
    """去掉**连续重复**的点（零长度段会让弦长参数化出现重复参数）。"""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return pts.copy()
    keep = np.ones(pts.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > EPS
    return pts[keep]


def _assemble(
    control: np.ndarray,
    *,
    params: BSplineParams,
    clearance_field: np.ndarray,
    resolution: float,
    origin: tuple[float, float] | np.ndarray,
    speed: float,
    energy_params: energy.EnergyParams,
    c_rr: np.ndarray | float,
    slope_rad: np.ndarray | float,
) -> tuple[SplineCurve, str, float, float, float, np.ndarray, np.ndarray]:
    """检查一条生成好的曲线：曲率 -> 离障 -> 能耗（顺序固定，报第一个失败）。

    返回 ``(curve, reason, max_kappa, min_clearance, energy_j, fail_mask, clearance)``。
    ``fail_mask`` 标出曲率或离障失败的采样点，供下一轮定位插入/放稀的区域。
    """
    curve = build_curve(control, sample_step_m=params.sample_step_m)

    kappa = np.abs(curve.curvature)
    max_kappa = float(kappa.max()) if kappa.size else 0.0
    curvature_fail = kappa > params.kappa_max + EPS

    clearance = sample_field(
        clearance_field, curve.xy, resolution=resolution, origin=origin
    )
    clearance_fail = clearance + EPS < params.d_min
    min_clearance = float(clearance.min()) if clearance.size else 0.0

    energy_j = curve_energy(
        curve,
        speed=speed,
        energy_params=energy_params,
        c_rr=c_rr,
        slope_rad=slope_rad,
        resolution=resolution,
        origin=origin,
    )

    if np.any(curvature_fail):
        return (
            curve, "curvature", max_kappa, min_clearance, energy_j,
            curvature_fail, clearance,
        )
    if np.any(clearance_fail):
        return (
            curve, "clearance", max_kappa, min_clearance, energy_j,
            clearance_fail, clearance,
        )
    if energy_j > params.e_budget + EPS:
        return (
            curve, "energy", max_kappa, min_clearance, energy_j,
            np.zeros(0, dtype=bool), clearance,
        )
    return (
        curve, "ok", max_kappa, min_clearance, energy_j,
        np.zeros(0, dtype=bool), clearance,
    )


def _polyline_at(vertices: np.ndarray, s: np.ndarray, query: np.ndarray) -> np.ndarray:
    """在折线上按弧长取样（线性插值）。"""
    pts = np.asarray(vertices, dtype=float)
    return np.column_stack(
        [np.interp(query, s, pts[:, 0]), np.interp(query, s, pts[:, 1])]
    )


def _polyline_energy(
    vertices: np.ndarray,
    *,
    speed: float,
    energy_params: energy.EnergyParams,
    c_rr: np.ndarray | float,
    slope_rad: np.ndarray | float,
    resolution: float,
    origin: tuple[float, float] | np.ndarray,
) -> float:
    """折线按实际段长估算电能，口径与 ``curve_energy`` 相同。"""
    pts = np.asarray(vertices, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    deltas = np.diff(pts, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    midpoints = 0.5 * (pts[:-1] + pts[1:])
    c_rr_values = _field_at_points(
        c_rr, midpoints, resolution=resolution, origin=origin
    )
    slope_values = _field_at_points(
        slope_rad, midpoints, resolution=resolution, origin=origin
    )
    direction_x = deltas[:, 0] / np.maximum(distances, EPS)
    total = 0.0
    for index in range(distances.shape[0]):
        if distances[index] <= EPS:
            continue
        total += energy.step_energy(
            float(distances[index]),
            float(speed),
            c_rr=float(c_rr_values[index]),
            slope_rad=float(slope_values[index]),
            direction_x=float(direction_x[index]),
            params=energy_params,
        )
    return float(total)


def _fallback(
    polyline: np.ndarray,
    *,
    params: BSplineParams,
    clearance_field: np.ndarray,
    resolution: float,
    origin: tuple[float, float] | np.ndarray,
    speed: float,
    energy_params: energy.EnergyParams,
    c_rr: np.ndarray | float,
    slope_rad: np.ndarray | float,
    reason: str,
    inserted: int,
    removed: int,
    depth: int,
    rounds: int,
    control_points: int,
) -> SmoothPath:
    """回退到 RDP 折线：A* 的输出天然满足离障与能耗预算，只是曲率有尖峰。"""
    vertices = np.asarray(polyline, dtype=float)
    if vertices.shape[0] < 2:
        raise ValueError("回退折线至少需要 2 个顶点")
    # 回退的折线按采样步长**密采样**后再对外返回：只返回顶点的话，下游
    # （轨迹采样、clearance 序列）会漏掉段中间的离障极小值。顶点本身仍完整保留在
    # 分开的 ``vertices`` 里供绘图使用。
    step = params.sample_step_m
    corners_s = _arc_lengths(vertices)
    dense_query = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, corners_s[-1], max(2, int(math.ceil(corners_s[-1] / step)) + 1)),
                corners_s,
            ]
        )
    )
    xy = _polyline_at(vertices, corners_s, dense_query)
    s = dense_query
    length = float(corners_s[-1])
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    yaw = np.zeros(xy.shape[0], dtype=float)
    yaw[:-1] = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
    yaw[-1] = yaw[-2]
    curvature = np.zeros(xy.shape[0], dtype=float)
    clearance = sample_field(
        clearance_field, xy, resolution=resolution, origin=origin
    )
    min_clearance = float(clearance.min()) if clearance.size else 0.0
    energy_j = _polyline_energy(
        vertices,
        speed=speed,
        energy_params=energy_params,
        c_rr=c_rr,
        slope_rad=slope_rad,
        resolution=resolution,
        origin=origin,
    )
    report = SmoothingReport(
        status="polyline",
        reason=reason,
        inserted_points=inserted,
        removed_points=removed,
        depth=depth,
        rounds=rounds,
        control_points=control_points,
        max_abs_curvature=float("inf"),
        min_clearance=min_clearance,
        energy_j=energy_j,
        length_m=length,
        rdp_vertices=int(vertices.shape[0]),
    )
    return SmoothPath(
        success=True,
        status="polyline",
        xy=xy,
        yaw=yaw,
        s=s,
        curvature=curvature,
        length_m=length,
        energy_j=energy_j,
        clearance=clearance,
        min_clearance=min_clearance,
        max_abs_curvature=float("inf"),
        kappa_max=params.kappa_max,
        curve=None,
        polyline=vertices,
        report=report,
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def select_indices_by_arclength(points: np.ndarray, spacing_m: float) -> np.ndarray:
    """在**原始节点**里按等弧长间隔挑下标，作为控制点的子序列。

    这样每个控制点都是 A* 的原始节点（7.7 第 5 条要求"从原始 A* 路径取中间节点"），
    同时控制点间距近似相等，曲率尺度可控。首末节点必选，保证端点重合。
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("路径至少需要 2 个节点")
    spacing = float(spacing_m)
    if not spacing > 0:
        raise ValueError(f"spacing_m 必须 > 0，实际为 {spacing_m}")
    s = _arc_lengths(pts)
    total = float(s[-1])
    if not total > EPS:
        return np.array([0, pts.shape[0] - 1], dtype=int)

def select_indices_by_arclength(points: np.ndarray, spacing_m: float) -> np.ndarray:
    """在**原始节点**里按等弧长间隔挑下标，作为控制点的子序列。

    这样每个控制点都是 A* 的原始节点（7.7 第 5 条要求"从原始 A* 路径取中间节点"），
    同时控制点间距近似相等，曲率尺度可控。首末节点必选，保证端点重合。

    返回的个数可能少于 4（短路径 + 大间距）：调用方应改用
    ``control_points_for``，它在节点不足时退化为"在线段上等弧长补点"。
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("路径至少需要 2 个节点")
    spacing = float(spacing_m)
    if not spacing > 0:
        raise ValueError(f"spacing_m 必须 > 0，实际为 {spacing_m}")
    s = _arc_lengths(pts)
    total = float(s[-1])
    if not total > EPS:
        return np.array([0, pts.shape[0] - 1], dtype=int)

    count = max(_DEGREE + 1, int(round(total / spacing)) + 1)
    targets = np.linspace(0.0, total, count)
    # 每个目标弧长处取最近的原始节点；首末目标固定为 0 与 total。
    chosen = np.searchsorted(s, targets, side="left")
    previous = np.clip(chosen - 1, 0, pts.shape[0] - 1)
    nearer = np.where(
        np.abs(s[previous] - targets) <= np.abs(s[chosen] - targets),
        previous,
        chosen,
    )
    nearer[0] = 0
    nearer[-1] = pts.shape[0] - 1
    return np.unique(nearer).astype(int)


def control_points_for(points: np.ndarray, spacing_m: float) -> np.ndarray:
    """挑控制点：先用原始节点；节点不足 4 个时在**已有线段上**等弧长补点。

    补出来的点落在已验证可行的原始线段内部（不是凭空造点），因此不会引入
    新的不可行区域。极端情况：直线任务（RDP 后只剩 2 个顶点）走的就是这条路径。
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        raise ValueError("路径至少需要 2 个节点")
    selected = select_indices_by_arclength(pts, spacing_m)
    if selected.shape[0] >= _DEGREE + 1:
        return pts[selected]
    # 节点不够：沿折线等弧长重采样（点仍落在原折线线段上）。
    return resample_by_arclength(pts, spacing_m)


def densify_control_points(
    control: np.ndarray, *, fail_xy: np.ndarray | None = None
) -> np.ndarray:
    """插入控制点：在**相邻控制点之间**补中点，把曲线拉回原始折线。

    这是 7.7 第 5 条"从原始 A* 路径取中间节点插入控制点"的落地：新点落在两个
    已验证可行的控制点之间（也就是原始路径附近），所以插入本身不会制造不可行区域；
    而控制点加密会让曲线更贴合折线，**切角变小、clearance 回升**。

    有失败定位时只在离失败点最近的若干个间隙插入，避免无谓地加密整条路径。
    """
    pts = np.asarray(control, dtype=float)
    if pts.shape[0] < 2:
        return pts
    midpoints = 0.5 * (pts[:-1] + pts[1:])
    if fail_xy is not None and np.asarray(fail_xy).size:
        reference = np.asarray(fail_xy, dtype=float)
        distances = np.min(
            np.linalg.norm(midpoints[:, None, :] - reference[None, :, :], axis=2),
            axis=1,
        )
        order = np.argsort(distances, kind="stable")
        keep_count = max(1, order.size // 2)
        keep_set = {int(index) for index in order[:keep_count]}
    else:
        keep_set = set(range(midpoints.shape[0]))

    pieces: list[np.ndarray] = []
    for index in range(pts.shape[0] - 1):
        pieces.append(pts[index][None, :])
        if index in keep_set:
            pieces.append(midpoints[index][None, :])
    pieces.append(pts[-1][None, :])
    return np.vstack(pieces)


def thin_control_points(control: np.ndarray) -> np.ndarray:
    """放稀控制点（隔一个取一个），让曲线更平滑、转角切得更多、长度更短。

    曲率与能耗失败都走这一步：控制点间距越大，曲线越接近低曲率的光滑弧。
    """
    pts = np.asarray(control, dtype=float)
    if pts.shape[0] <= _DEGREE + 1:
        return pts
    keep = np.zeros(pts.shape[0], dtype=bool)
    keep[::2] = True
    keep[-1] = True
    thinned = pts[keep]
    if thinned.shape[0] < _DEGREE + 1:
        return pts
    return thinned


def smooth_path(
    xy: np.ndarray,
    *,
    params: BSplineParams,
    clearance_field: np.ndarray,
    resolution: float,
    speed: float,
    energy_params: energy.EnergyParams,
    c_rr: np.ndarray | float = 0.015,
    slope_rad: np.ndarray | float = 0.0,
    origin: tuple[float, float] | np.ndarray = (0.0, 0.0),
) -> SmoothPath:
    """把 A* 折线平滑成满足全部硬约束的 B 样条曲线；失败则回退 RDP 折线。

    流程：RDP 简化 -> 在原始节点上按等弧长选控制点 -> 三次均匀 clamped B 样条
    -> 按弧长密集采样 -> 曲率 / 离障 / 能耗三道关口 -> 失败就调整控制点重来
    （深度 <= ``max_insert_depth``）-> 仍失败则回退折线，并在 ``report`` 里
    记录原因、插入与放稀的次数。

    调整策略（三条约束的几何敏感性方向相反，必须分开处理）：

    * **曲率失败 -> 放稀控制点**：间距越大，曲线越平滑、``max kappa`` 越低；
    * **能耗失败 -> 放稀控制点**：转角切得更多，几何长度与电能都会下降；
    * **离障失败 -> 插入中间节点**：间距越小，曲线越贴近已验证可行的折线，
      切角变小，clearance 回升。

    Parameters
    ----------
    xy:
        A* 的栅格折线顶点，世界坐标 (N, 2) [m]。
    params:
        平滑参数；``kappa_max`` / ``d_min`` / ``e_budget`` 必须与 ``AStarParams``
        取同一个值。
    clearance_field:
        ``core.astar.clearance_field`` 输出的距离场 [m]（逐点离障检查用）。
    """
    points = _dedupe(np.asarray(xy, dtype=float))
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"xy 必须是 (N, 2) 数组，实际形状为 {np.shape(xy)}")
    if points.shape[0] == 0:
        raise ValueError("xy 不能为空")
    clearance_grid = np.asarray(clearance_field, dtype=float)
    if clearance_grid.ndim != 2 or clearance_grid.size == 0:
        raise ValueError("clearance_field 必须是二维非空数组")
    if not float(resolution) > 0:
        raise ValueError(f"resolution 必须 > 0，实际为 {resolution}")

    # 单一节点（起点即终点）：长度为 0，无曲率，直接返回退化结果。
    if points.shape[0] == 1:
        clearance = sample_field(
            clearance_grid, points, resolution=resolution, origin=origin
        )
        report = SmoothingReport(
            status="spline",
            reason="ok",
            inserted_points=0,
            removed_points=0,
            depth=0,
            rounds=1,
            control_points=1,
            max_abs_curvature=0.0,
            min_clearance=float(clearance[0]),
            energy_j=0.0,
            length_m=0.0,
            rdp_vertices=1,
        )
        return SmoothPath(
            success=True,
            status="spline",
            xy=points.copy(),
            yaw=np.zeros(1),
            s=np.zeros(1),
            curvature=np.zeros(1),
            length_m=0.0,
            energy_j=0.0,
            clearance=clearance,
            min_clearance=float(clearance[0]),
            max_abs_curvature=0.0,
            kappa_max=params.kappa_max,
            curve=None,
            polyline=points.copy(),
            report=report,
        )

    kept = rdp_indices(points, params.rdp_epsilon_m)
    reduce_to = points[kept]
    if reduce_to.shape[0] < 2:
        reduce_to = points[[0, points.shape[0] - 1]]

    control = control_points_for(reduce_to, params.control_spacing_m)
    inserted = 0
    removed = 0
    depth = 0
    rounds = 0
    last_reason = "ok"
    last_control = int(control.shape[0])

    while True:
        rounds += 1
        last_control = int(control.shape[0])
        curve, reason, max_kappa, min_clearance, energy_j, fail_mask, clearance = (
            _assemble(
                control,
                params=params,
                clearance_field=clearance_grid,
                resolution=resolution,
                origin=origin,
                speed=speed,
                energy_params=energy_params,
                c_rr=c_rr,
                slope_rad=slope_rad,
            )
        )
        last_reason = reason

        if reason == "ok":
            report = SmoothingReport(
                status="spline",
                reason="ok",
                inserted_points=inserted,
                removed_points=removed,
                depth=depth,
                rounds=rounds,
                control_points=int(control.shape[0]),
                max_abs_curvature=float(max_kappa),
                min_clearance=float(min_clearance),
                energy_j=float(energy_j),
                length_m=float(curve.length_m),
                rdp_vertices=int(reduce_to.shape[0]),
            )
            return SmoothPath(
                success=True,
                status="spline",
                xy=curve.xy,
                yaw=curve.yaw,
                s=curve.s,
                curvature=curve.curvature,
                length_m=float(curve.length_m),
                energy_j=float(energy_j),
                clearance=clearance,
                min_clearance=float(min_clearance),
                max_abs_curvature=float(max_kappa),
                kappa_max=params.kappa_max,
                curve=curve,
                polyline=reduce_to,
                report=report,
            )

        if depth >= params.max_insert_depth:
            break

        before = int(control.shape[0])
        if reason == "clearance":
            fail_xy = curve.xy[fail_mask] if np.any(fail_mask) else None
            control = densify_control_points(control, fail_xy=fail_xy)
            inserted += max(0, int(control.shape[0]) - before)
        else:  # curvature / energy：放稀控制点
            thinned = thin_control_points(control)
            if thinned.shape[0] < before:
                control = thinned
                removed += before - int(control.shape[0])
            else:
                # 已经是最少控制点（4 个），无法再放稀。此时曲率过大的原因
                # 恰恰是"控制点太少、圆化半径太大把两个拐角连成一个大弯"，
                # 方向反过来：**加密**控制点让曲线贴合折线、把大弯拆回小弯。
                fail_xy = curve.xy[fail_mask] if np.any(fail_mask) else None
                control = densify_control_points(control, fail_xy=fail_xy)
                inserted += max(0, int(control.shape[0]) - before)
        depth += 1

        if control.shape[0] == before:
            # 既不能放稀也不能加密：继续只会得到同样的曲线，直接回退。
            last_reason = "insert_exhausted"
            break

    if last_reason == "ok":
        last_reason = "insert_exhausted"
    return _fallback(
        reduce_to,
        params=params,
        clearance_field=clearance_grid,
        resolution=resolution,
        origin=origin,
        speed=speed,
        energy_params=energy_params,
        c_rr=c_rr,
        slope_rad=slope_rad,
        reason=last_reason,
        inserted=inserted,
        removed=removed,
        depth=depth,
        rounds=rounds,
        control_points=last_control,
    )


def feasibility_stats(reports: list[SmoothingReport]) -> dict[str, float]:
    """曲率可满足性统计（7.7 第 6 条要求的报告口径，2026-10-29 重新定性）。

    **``success`` 与 ``curvature_satisfied`` 是两个独立的概念，这里只统计后者。**
    回退折线的曲率无界，必须计入 ``curvature_infeasible``，绝不能用
    "几何链路产出了结果"掩盖"曲率硬约束被违反"。

    * ``tasks``：任务总数；
    * ``spline``：产出 B 样条曲线的任务数（**不含**回退）；
    * ``curvature_satisfied``：曲率硬约束真正成立的任务数；
    * ``curvature_infeasible``：曲率约束被违反的任务数（**包含全部回退**）；
    * ``curvature_infeasible_rate`` = ``curvature_infeasible / tasks``，**主指标**；
    * ``inserted``：触发过控制点插入的任务数；
    * ``insertion_rate`` = ``inserted / tasks``；
    * ``pure_spline_rate``：一次通过、既没插入也没回退的比例。
    """
    total = len(reports)
    if total == 0:
        return {
            "tasks": 0,
            "spline": 0,
            "curvature_satisfied": 0,
            "curvature_infeasible": 0,
            "curvature_infeasible_rate": 0.0,
            "inserted": 0,
            "insertion_rate": 0.0,
            "pure_spline_rate": 0.0,
        }
    spline = sum(1 for report in reports if report.status == "spline")
    satisfied = sum(1 for report in reports if report.curvature_satisfied)
    infeasible = sum(1 for report in reports if report.curvature_infeasible)
    inserted = sum(1 for report in reports if report.used_insertion)
    pure = sum(
        1
        for report in reports
        if report.curvature_satisfied and not report.used_insertion
    )
    return {
        "tasks": total,
        "spline": spline,
        "curvature_satisfied": satisfied,
        "curvature_infeasible": infeasible,
        "curvature_infeasible_rate": infeasible / total,
        "inserted": inserted,
        "insertion_rate": inserted / total,
        "pure_spline_rate": pure / total,
    }


def fallback_rate(reports: list[SmoothingReport]) -> dict[str, float]:
    """已废弃：保留为 ``feasibility_stats`` 的别名，语义以新函数为准。

    "回退率"这个名字容易被误读成"降级成功"，而回退实际上是**曲率约束违反**。
    新代码请直接调用 ``feasibility_stats``，它把 ``curvature_infeasible``
    作为独立指标报告。
    """
    return feasibility_stats(reports)
