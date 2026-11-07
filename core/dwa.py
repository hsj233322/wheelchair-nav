"""DWA 局部规划（纯函数，无全局状态，不 import config）。

链路里的位置
------------
全局层（``core.astar``）给折线，平滑层（``core/bspline``）给可跟踪的参考轨迹，
本模块消费 **``trackable = True`` 的 SmoothPath**，在动态障碍存在时逐周期输出
速度指令并推进执行轨迹。

前置红线（交接文档 7.9）
-----------------------
只接受 ``trackable = True`` 的参考轨迹。回退折线在拐点处曲率无界，任何有限转向
能力的底盘都跟不动；拿它做局部规划，所有指标（到达时间、碰撞帧数、a_y 峰值）
都失去意义。``DwaPlanner`` 在构造时强制校验，不满足直接抛 ``NotTrackableError``。

舒适性为什么一个硬、一个软（7.9 第 3 条）
-----------------------------------------
* **横向加速度 ``a_y = v * ω`` 作硬约束**。理由三条：
  1. 它**只依赖本周期候选的 (v, ω)**，可以解析算出，不需要历史状态；
  2. 它是**乘人体舒适 / 侧翻风险的物理量**（ISO 2631-1 的横向分量），
     越界不是"略差"而是不可接受，属于可行性边界；
  3. 越界可以被**立即剔除**，让搜索空间只包含物理可行的候选。
* **jerk 作代价项**。理由：jerk 是**三阶量**，必须用至少三个历史周期差分；
  它依赖 planner 上一周期的选择，若做成硬约束，同一候选的可行性会随历史变化
  （周期性抖动 / 死锁），且离散差分对噪声极敏感。把它放进代价项，
  让"较平滑"的候选优先，但不允许因为一时算不出 jerk 就判定不可行。

这个区分写进 ``DwaParams`` 的字段注释与 ``evaluate_candidate`` 的实现顺序里：
**硬约束先剔，代价项后排序**。

绕行判据的落点：可行性，不是加分（7.9 第 4 条）
------------------------------------------------
沿用相对运动方向 δ 的几何构造，但**输出是布尔 + 剩余间距**，不是权重加分：
给定候选与一个动态障碍的预测位置序列，算

    δ            = 障碍相对速度方向 与 "障碍→本车" 连线的夹角
    d_center     = 该候选与障碍的**最小圆心距**（车体中心 <-> 障碍中心）
    R            = 车体物理外接圆半径 ``body_radius``
    r_o          = 障碍半径
    g_surface(δ) = ``required_gap(δ)`` —— **表面间隙要求**（不是圆心距！）

判定（可行性判据，不是权重）：

    d_center >= R + r_o + g_surface(δ)                                   (★)

不满足即**该候选不可行**。这与第 3 章"约束而非权重"的主线一致：
绕行不是"多给点分"，而是"必须满足"。

**(★) 的三项各有物理来源，缺一不可**（2026-10-30 修正）：

1. ``R + r_o``：两圆**不接触**的充要条件是 ``d_center >= R + r_o``。
   这是与 δ 无关的物理下限（本平台 0.58 + 0.30 = 0.88 m）；
2. ``g_surface(δ)``：两圆**表面之间**还应留出的策略间隙，量纲是"表面距"。

早期实现把它写成 ``required_gap(δ) + r_o``，**漏掉了车体半径 R**，于是同向档阈值只有
``0.25 + 0.30 = 0.55 m``，比 ``R + r_o = 0.88 m`` 小 0.33 m ——
即**允许圆心距小到两圆重叠 0.33 m**。实测最小圆心距恰好饱和在 0.550 m
（easy / hard 都是），说明规划器正贴着这个过宽的阈值行驶。详见
``required_gap`` / ``required_center_gap`` 的 docstring。

``g_surface(δ)`` 的物理含义：迎面（δ≈0）时相对速度大、留给反应的时间短，
要求更大的**表面**间隙；同向（δ≈π）时相对速度接近 0，可以更近。用余弦插值是最小假设，
比"一个常数阈值"更贴近相对运动学，且在 δ 两端连续可导。

单位：m, s, rad, m/s, rad/s, m/s^2, m/s^3。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9

# 边界相等容差：几何量常落在限值上（例如测算得到的 clearance 恰好等于 d_min，
# 或 a_y 恰好等于 a_y_max）。只用 EPS=1e-9 会把"恰好合法"判成违反，
# 产生**缺额为 0.0000 的假违反**（本轮验收补充 2 / 3a 都踩到这一点）。
# 用相对容差 1e-6：对 0.7 m 的 clearance 约 7e-7 m，远小于任何有物理意义的差异。
_BOUNDARY_TOL = 1e-6


def _boundary_tol(scale: float) -> float:
    """边界相等容差（相对量，随被比较的限值缩放）。"""
    return _BOUNDARY_TOL * max(1.0, abs(float(scale)))


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DwaParams:
    """DWA 参数。全部由调用方显式传入（core 不读配置）。

    所有阈值与权重都必须来自 ``algo`` 层配置（7.9 第 7 条），core 只接收数值。
    """

    # --- 动态窗口（7.9 第 1 条：四个窗口）---
    v_max: float
    w_max: float
    a_max: float
    alpha_max: float

    # --- 预测时域与步长（frozen_A.yaml: horizon 0.6 / dt 0.1）---
    horizon_s: float
    dt_s: float

    # --- 采样分辨率（候选空间）---
    v_samples: int = 9
    w_samples: int = 15

    # --- 安全距离 ---
    d_min: float = 0.7
    """**策略**安全半径 [m]：可降级但必须记录的约束（见 ``body_radius``）。"""

    body_radius: float = 0.58
    """**物理**外接圆半径 [m]（``platform.body_radius``）：绝不可违反的约束。

    两级安全约束（本轮验收阻断 1）：

    * ``clearance >= body_radius``：车身不撞墙的物理下限。**违反即碰撞、任务失败**，
      不能计入"到达"。它与地图膨胀、碰撞判定用的是同一个量。
    * ``clearance >= d_min``：策略裕度（默认 0.70 = 0.58 + 0.12）。**可以降级**，
      但必须记录违反幅度 ``max(d_min - clearance)``。

    历史上把它们合成一个 ``d_min`` 判据，导致 hard 档出现
    ``min_clearance = 0.371 < 0.58``（车身穿墙）却仍被记为"到达 8/8"。
    """

    v_nominal: float = 1.0
    """**标称巡航速度** [m/s]（来自 glob 层的 ``astar.v_cruise``）。

    速度保持项必须对**它**取偏差，不能对 ``v_max`` 取偏差：``v_max`` 是平台物理
    上限，把"接近上限"当作目标会让规划器在起步阶段永远偏好低速（因为加速度窗口
    每周期只放开 ``a_max * dt``，低速候选天然离窗口下界更近）。
    """

    stop_margin_m: float = 0.02
    """到达判据 [m]：距终点小于该值即视为到达。"""

    approach_slowdown_m: float = 0.60
    """接近终点开始限速的距离 [m]。

    为什么必须有：终端判据是点判据（``<= stop_margin_m``），而每周期位移是
    ``v * dt``。以 v=1.0 m/s、dt=0.1 s 计，单周期跨越 0.1 m，若全程不减速，
    距离 0.18 m 时会**直接冲过终点**（实测 houseexpo_d398cc42 冲到 2.2 m 外后
    被障碍逼停）。减速带把"可停住"变成候选可行性的一部分，而不是靠判据碰运气。
    """

    stop_speed_ratio: float = 0.35
    """终点减速带的末端速度比 ``v_nominal``：越靠近越慢，最小按该比例。"""

    # --- 动态绕行（可行性判据，不是权重）---
    #
    # 语义修正（2026-10-30）：下面两个量是**表面间隙**要求 g_surface(δ)，
    # 不是"圆心距阈值"。判据的完整形式是
    #     d_center >= body_radius + r_o + g_surface(δ)
    # 见 ``required_center_gap`` / 模块 docstring 的 (★) 式。
    gap_head_on_m: float = 0.60
    """迎面（δ=0）时的最小**表面间隙** g_surface [m]。

    圆心距阈值 = ``body_radius + r_o + gap_head_on_m``（默认 0.58 + 0.30 + 0.60 = 1.48 m）。
    """

    gap_overtake_m: float = 0.25
    """同向（δ=π）时的最小**表面间隙** g_surface [m]。

    圆心距阈值 = ``body_radius + r_o + gap_overtake_m``（默认 0.58 + 0.30 + 0.25 = 1.13 m）。
    早期实现漏掉 ``body_radius``，同向阈值只有 0.55 m < ``body_radius + r_o`` = 0.88 m，
    允许两圆重叠 0.33 m —— 本轮修正的核心。
    """

    # --- 舒适性：a_y 是硬约束 ---
    a_y_max: float = 0.8
    """横向加速度硬上限 [m/s^2]（``a_y = v * ω``）。"""

    spin_speed_floor: float = 0.25
    """**原地自转抑制**下的速度下限 [m/s]（相对 ``v_nominal`` 的比例）。

    ``a_y = v·ω`` 有个结构性漏洞：当 ``v → 0`` 时 ``a_y`` 也趋于 0，于是
    "原地以 ω_max 自转"可以同时满足 ``a_y`` 上限。实测
    `houseexpo_d0aeed69` 同向慢速场景：车在横向偏差仅 0.15 m 处把 ω 打满
    2.09 rad/s 并同时刹车，v 从 0.43 降到 0.01，**在原地自转 270 个周期**直到超时。

    物理上轮椅确实能原地转向，但"用转向代替行驶"不是路径跟踪该有的行为，
    而且会让横向偏差不收敛。因此规定：**ω 的绝对值超过一个门限时，
    必须保持不低于 ``spin_speed_floor * v_nominal`` 的前进速度**。
    门限取 ``alpha_max * dt_s``（一个周期内能放开的角速度增量）的倍数，
    与动态窗口一致。
    """

    spin_rate_gate_ratio: float = 3.0
    """触发前进速度下限的 ω 门限 = ``spin_rate_gate_ratio * alpha_max * dt_s``。"""

    lateral_deviation_max_m: float = 0.0
    """允许的最大横向偏离 [m]；``0`` 表示不限制。

    参考路径是全局层唯一验证过"整条满足 d_min 与曲率"的轨迹。绕行自由度若不
    受限，DWA 会走进全局层从未验证的窄处（实测 d0aeed69 因此锁死）。
    取 0.6 m 时仍留有充足绕行空间（一般走廊余量 0.3~0.5 m）。
    """

    lateral_margin_loss_weight: float = 0.5
    """偏离参考线造成的离障余量损失的**代价权重**（无单位）。

    参考路径是全局层唯一验证过的轨迹，偏离它等于走进未验证区域，因此要
    惩罚"偏离导致的 clearance 损失"。这个量**只作代价、不作硬约束**：
    回到参考线的过程必然横穿低 clearance 区，做成硬约束会把返回动作堵死
    （详见 evaluate_candidate 里横向偏离一节的说明）。
    """

    regression_tolerance_m: float = 0.01
    """"已在不合规位置"时允许的 clearance 恶化容差 [m]。

    避障时车可能被挤到 ``clearance < d_min`` 的位置（既成事实）。若要求未来点
    必须立刻合规，全部候选都会不可行并永久锁死；若完全不限，车会越陷越深。
    折中：只禁止"比当前位置更差超过该容差"，给数值抖动留余量。
    """

    # --- 舒适性：jerk 是代价项 ---
    jerk_weight: float = 0.15
    """jerk 代价权重 [s^3/m]，把 jerk RMS 折算进总代价。"""

    # --- 评价项权重（7.9 第 2 条）---
    w_heading: float = 0.1
    w_dist: float = 0.5
    w_vel: float = 0.3
    w_clearance: float = 0.2

    def __post_init__(self) -> None:
        for name in ("v_max", "w_max", "a_max", "alpha_max", "horizon_s", "dt_s"):
            value = float(getattr(self, name))
            if not value > 0:
                raise ValueError(f"{name} 必须 > 0，实际为 {value}")
        if not math.isfinite(self.horizon_s):
            raise ValueError("horizon_s 必须有限")
        if self.dt_s > self.horizon_s + EPS:
            raise ValueError(
                f"dt_s={self.dt_s} 不能大于 horizon_s={self.horizon_s}"
            )
        for name in ("v_samples", "w_samples"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} 必须 >= 1")
        if self.d_min < 0:
            raise ValueError(f"d_min 必须 >= 0，实际为 {self.d_min}")
        if not self.body_radius > 0:
            raise ValueError(f"body_radius 必须 > 0，实际为 {self.body_radius}")
        if self.d_min + EPS < self.body_radius:
            raise ValueError(
                f"d_min={self.d_min} 小于 body_radius={self.body_radius}："
                "策略裕度不能比车身还小，否则物理约束与策略约束的层级颠倒"
            )
        if not self.v_nominal > 0:
            raise ValueError(f"v_nominal 必须 > 0，实际为 {self.v_nominal}")
        if self.v_nominal > self.v_max + EPS:
            raise ValueError(
                f"v_nominal={self.v_nominal} 不能超过平台 v_max={self.v_max}"
            )
        if not self.a_y_max > 0:
            raise ValueError(f"a_y_max 必须 > 0，实际为 {self.a_y_max}")
        if not 0 < self.spin_speed_floor <= 1.0:
            raise ValueError(
                f"spin_speed_floor 必须在 (0, 1] 内，实际为 {self.spin_speed_floor}"
            )
        if not self.spin_rate_gate_ratio > 0:
            raise ValueError(
                f"spin_rate_gate_ratio 必须 > 0，实际为 {self.spin_rate_gate_ratio}"
            )
        if self.jerk_weight < 0:
            raise ValueError(f"jerk_weight 必须 >= 0，实际为 {self.jerk_weight}")
        if self.gap_overtake_m < 0 or self.gap_head_on_m < 0:
            raise ValueError("gap 阈值必须非负")

    @property
    def steps(self) -> int:
        """预测时域内的步数。"""
        return max(1, int(round(self.horizon_s / self.dt_s)))


# ---------------------------------------------------------------------------
# 参考轨迹
# ---------------------------------------------------------------------------


class NotTrackableError(ValueError):
    """参考轨迹不可跟踪（回退折线），不得进入局部规划。"""


@dataclass(frozen=True)
class Reference:
    """可跟踪的参考轨迹（由 ``SmoothPath`` 转换而来）。

    只保留局部规划需要的东西：采样点、朝向前馈、弧长参数、曲率。
    """

    xy: np.ndarray
    yaw: np.ndarray
    s: np.ndarray
    curvature: np.ndarray
    length_m: float
    kappa_max: float

    def __post_init__(self) -> None:
        pts = np.asarray(self.xy, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"参考轨迹 xy 必须是 (N,2)，实际 {pts.shape}")
        if pts.shape[0] < 2:
            raise ValueError("参考轨迹至少需要 2 个点")
        heading = np.asarray(self.yaw, dtype=float)
        arc = np.asarray(self.s, dtype=float)
        kappa = np.asarray(self.curvature, dtype=float)
        if heading.shape != arc.shape or arc.shape != kappa.shape:
            raise ValueError("yaw / s / curvature 长度必须一致")
        if not np.all(np.isfinite(kappa)):
            raise NotTrackableError(
                "参考轨迹含无界曲率（回退折线），不可跟踪，不得进入局部规划"
            )

    def project_windowed(
        self,
        xy: np.ndarray,
        arc_center: float,
        half_width_m: float = 1.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """只在 ``arc_center ± half_width_m`` 的弧长窗口内做投影。

        性能优化：全量投影要对全部参考线段建 (P, S, 2) 矩阵，是 DWA 的
        第一热点（56841 次 / 单场景 14.3 s）。而 DWA 每周期只在当前位置附近
        活动，远端线段根本不可能成为最近投影点。用上一周期的弧长做窗口，
        把 S 从 ~600 降到 ~40，单次投影成本降一个量级。

        窗口边界仍然取整段折线做一次**兜底**：若窗口内最近距离超过
        ``half_width_m``（说明车已跑出窗口，例如被障碍推远），退回全量投影，
        保证语义与 ``project`` 完全一致。
        """
        pts = np.asarray(xy, dtype=float)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        low = max(0.0, float(arc_center) - half_width_m)
        high = float(arc_center) + half_width_m
        start_index = int(np.searchsorted(self.s, low, side="left"))
        end_index = int(np.searchsorted(self.s, high, side="right"))
        start_index = max(0, min(start_index, self.s.shape[0] - 2))
        end_index = max(start_index + 2, min(end_index, self.s.shape[0]))
        return self._project_range(pts, start_index, end_index, single)

    def _project_range(
        self,
        pts: np.ndarray,
        start_index: int,
        end_index: int,
        single: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在指定线段区间 ``[start_index, end_index)`` 内做投影。"""
        ref = self.xy[start_index : end_index + 1]
        starts = ref[:-1]
        segments = ref[1:] - starts
        seg_len_sq = np.einsum("ij,ij->i", segments, segments)
        usable = seg_len_sq > EPS
        rel = pts[:, None, :] - starts[None, :, :]
        t = np.einsum("psi,si->ps", rel, segments) / np.where(
            usable, seg_len_sq, 1.0
        )[None, :]
        t = np.clip(t, 0.0, 1.0)
        candidate = starts[None, :, :] + t[:, :, None] * segments[None, :, :]
        dx = pts[:, None, 0] - candidate[:, :, 0]
        dy = pts[:, None, 1] - candidate[:, :, 1]
        distance = np.hypot(dx, dy)
        distance = np.where(usable[None, :], distance, np.inf)
        index = np.argmin(distance, axis=1)
        rows = np.arange(pts.shape[0])
        best_point = candidate[rows, index]
        best_distance = distance[rows, index]
        best_arc = self.s[start_index + index] + t[rows, index] * np.sqrt(
            np.maximum(seg_len_sq[index], EPS)
        )
        if single:
            return best_point[0], best_arc[:1], best_distance[:1]
        return best_point, best_arc, best_distance

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """把任意点投影到参考折线：返回 (投影点, 弧长, 距离)。

        实现是**向量化**的：一次性算全部 (点, 线段) 距离矩阵再取最小。
        逐段 for 循环在 135 候选 / 周期的规划里会把单周期拖到 2 s，不可接受。
        参考轨迹点数 ~10^3、查询点 ~10^1，矩阵规模 ~10^4，内存与耗时都可忽略。
        """
        pts = np.asarray(xy, dtype=float)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        ref = self.xy
        starts = ref[:-1]                      # (S, 2)
        segments = ref[1:] - starts            # (S, 2)
        seg_len_sq = np.einsum("ij,ij->i", segments, segments)   # (S,)
        usable = seg_len_sq > EPS

        # rel: (P, S, 2)
        rel = pts[:, None, :] - starts[None, :, :]
        t = np.einsum("psi,si->ps", rel, segments) / np.where(
            usable, seg_len_sq, 1.0
        )[None, :]
        t = np.clip(t, 0.0, 1.0)
        candidate = starts[None, :, :] + t[:, :, None] * segments[None, :, :]
        distance = np.linalg.norm(pts[:, None, :] - candidate, axis=2)  # (P, S)
        distance = np.where(usable[None, :], distance, np.inf)

        index = np.argmin(distance, axis=1)
        rows = np.arange(pts.shape[0])
        best_point = candidate[rows, index]
        best_distance = distance[rows, index]
        best_arc = self.s[index] + t[rows, index] * np.sqrt(
            np.maximum(seg_len_sq[index], EPS)
        )
        if single:
            return best_point[0], best_arc[:1], best_distance[:1]
        return best_point, best_arc, best_distance


def reference_from_smooth(smooth: object) -> Reference:
    """从 ``core.bspline.SmoothPath`` 构造参考轨迹，**强制校验 trackable**。

    ``smooth`` 用 ``object`` 类型避免 core 内部模块耦合；只读取约定属性。
    """
    trackable = getattr(smooth, "trackable", None)
    if trackable is None:
        raise ValueError("传入对象缺少 trackable 属性，无法判定是否可跟踪")
    if not trackable:
        raise NotTrackableError(
            "参考轨迹 trackable=False（曲率约束未满足），不得进入局部规划实验；"
            "请先确认全局+平滑层该任务是否可跟踪（见交接文档 7.9 前置红线）"
        )
    return Reference(
        xy=np.asarray(getattr(smooth, "xy"), dtype=float),
        yaw=np.asarray(getattr(smooth, "yaw"), dtype=float),
        s=np.asarray(getattr(smooth, "s"), dtype=float),
        curvature=np.asarray(getattr(smooth, "curvature"), dtype=float),
        length_m=float(getattr(smooth, "length_m")),
        kappa_max=float(getattr(smooth, "kappa_max")),
    )


# ---------------------------------------------------------------------------
# 动态障碍：确定性手工轨迹 + 预留显式轨迹接口
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObstacleTrack:
    """一个动态障碍的**显式**位置-时间轨迹（纯数据，无模型假设）。

    这是给 ETH/UCY 预留的统一接口：下一轮把行人数据读成两列 ``(t, xy)`` ，
    构造本对象即可接入，DWA 侧不需要任何改动。

    * ``times`` 严格递增 [s]，**绝对时间**，与 planner 的累计时间对齐；
    * ``xy`` 与 ``times`` 等长，(N, 2) [m]；
    * 轨迹范围外按**最近端点**外推（匀速假设下等于常数延拓）。
    """

    times: np.ndarray
    xy: np.ndarray
    radius_m: float
    label: str = "obstacle"

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        p = np.asarray(self.xy, dtype=float)
        if t.ndim != 1 or p.ndim != 2 or p.shape[1] != 2:
            raise ValueError("times 必须是 (N,)，xy 必须是 (N,2)")
        if t.shape[0] != p.shape[0]:
            raise ValueError("times 与 xy 长度必须一致")
        if t.shape[0] < 1:
            raise ValueError("轨迹至少需要 1 个采样")
        if t.shape[0] > 1 and np.any(np.diff(t) <= 0):
            raise ValueError("times 必须严格递增")
        if not np.isfinite(p).all():
            raise ValueError("xy 必须是有限数值")
        if not float(self.radius_m) > 0:
            raise ValueError(f"radius_m 必须 > 0，实际为 {self.radius_m}")

    def position_at(self, t_query: np.ndarray) -> np.ndarray:
        """在给定时刻查位置；轨迹范围外按最近端点外推。"""
        query = np.asarray(t_query, dtype=float)
        times = np.asarray(self.times, dtype=float)
        xy = np.asarray(self.xy, dtype=float)
        out = np.zeros((query.shape[0], 2), dtype=float)
        out[:, 0] = np.interp(query, times, xy[:, 0])
        out[:, 1] = np.interp(query, times, xy[:, 1])
        return out

    def velocity_at(self, t0: float) -> np.ndarray:
        """在 ``t0`` 处的瞬时速度 [m/s]（用相邻两段中心差分，端点按单侧差）。"""
        times = np.asarray(self.times, dtype=float)
        xy = np.asarray(self.xy, dtype=float)
        if times.shape[0] < 2:
            return np.zeros(2, dtype=float)
        index = int(np.clip(np.searchsorted(times, t0) - 1, 0, times.shape[0] - 2))
        dt = float(times[index + 1] - times[index])
        if dt <= EPS:
            return np.zeros(2, dtype=float)
        return (xy[index + 1] - xy[index]) / dt

    def predict(self, t0: float, times_ahead: np.ndarray) -> np.ndarray:
        return self.position_at(np.asarray(times_ahead, dtype=float))


def build_obstacle_track(
    *,
    start_xy: tuple[float, float],
    velocity: tuple[float, float],
    duration_s: float,
    radius_m: float,
    label: str = "obstacle",
    dt_s: float = 0.1,
    t_offset_s: float = 0.0,
) -> ObstacleTrack:
    """按**匀速直线**构造确定性手工轨迹（单元测试用）。

    7.9 第 5 条要求本轮只用确定性手工轨迹（迎面 / 横穿 / 同向慢速），
    这样测试可以解析核对，不依赖任何数据集。
    """
    steps = max(2, int(round(float(duration_s) / float(dt_s))) + 1)
    times = float(t_offset_s) + np.arange(steps, dtype=float) * float(dt_s)
    vx, vy = float(velocity[0]), float(velocity[1])
    dx = (times - float(t_offset_s)) * vx
    dy = (times - float(t_offset_s)) * vy
    xy = np.column_stack(
        [float(start_xy[0]) + dx, float(start_xy[1]) + dy]
    )
    return ObstacleTrack(times=times, xy=xy, radius_m=float(radius_m), label=label)


def make_scenario_obstacle(
    kind: str,
    *,
    start_xy: tuple[float, float],
    direction: tuple[float, float],
    speed: float,
    radius_m: float,
    duration_s: float = 20.0,
    t_offset_s: float = 0.0,
) -> ObstacleTrack:
    """三种确定性场景（迎面 / 横穿 / 同向慢速），供单元测试与验收图使用。

    ``direction`` 是**单位方向**，内部再乘 ``speed``。
    """
    norm = math.hypot(float(direction[0]), float(direction[1]))
    if norm <= EPS:
        raise ValueError("direction 不能为零向量")
    velocity = (float(direction[0]) / norm * float(speed),
                float(direction[1]) / norm * float(speed))
    names = {
        "head_on": "迎面",
        "crossing": "横穿",
        "same_direction": "同向慢速",
    }
    if kind not in names:
        raise ValueError(f"未知的动态障碍类型 {kind!r}，可选 {sorted(names)}")
    return build_obstacle_track(
        start_xy=start_xy,
        velocity=velocity,
        duration_s=duration_s,
        radius_m=radius_m,
        label=names[kind],
        t_offset_s=t_offset_s,
    )


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------


def required_gap(delta_rad: float, params: DwaParams) -> float:
    """**表面间隙**要求 ``g_surface(δ)`` [m]（绕行可行性判据的阈值的一部分）。

    公式（余弦插值，端点连续可导）：

        g_surface(δ) = gap_overtake + 0.5 * (1 + cos δ) * (gap_head_on - gap_overtake)

    ``δ`` 定义：障碍相对速度方向，与"障碍 → 本车"连线的夹角 [rad]，取 [0, π]。

    * ``δ = 0``（迎面）：相对速度最大、反应时间最短 → ``gap_head_on_m``；
    * ``δ = π``（同向）：相对速度接近 0 → ``gap_overtake_m``。

    **量纲声明（2026-10-30 修正，务必按此理解）**：返回值是**两个圆表面之间**
    要留出的距离，不是圆心距。判定必须写成

        d_center >= body_radius + r_o + required_gap(δ)

    绝不能只用 ``required_gap(δ) + r_o``：那样同向档只剩 0.55 m，
    比两圆不接触所需的 ``body_radius + r_o`` = 0.88 m 还小，等于允许穿透。
    完整的圆心距阈值请用 :func:`required_center_gap`。

    这是**可行性阈值**，不是代价权重：不满足就是"该候选不可行"。
    """
    delta = float(np.clip(abs(delta_rad), 0.0, math.pi))
    share = 0.5 * (1.0 + math.cos(delta))  # δ=0 -> 1，δ=π -> 0
    return float(
        params.gap_overtake_m
        + share * (params.gap_head_on_m - params.gap_overtake_m)
    )


def required_center_gap(
    delta_rad: float, params: DwaParams, obstacle_radius: float
) -> float:
    """动态间距判据的**圆心距阈值** [m]（交接文档 7.10 的修法）。

        required_center_gap = body_radius + r_o + required_gap(δ)
                            = R + r_o + g_surface(δ)

    三项的含义：

    * ``body_radius``：车体物理外接圆半径（``platform.body_radius``）；
    * ``r_o``：障碍半径；
    * ``required_gap(δ)``：表面间隙要求（迎面大、同向小）。

    结构性保证：因为 ``required_gap >= 0``，本函数**恒有**
    ``required_center_gap >= body_radius + r_o``，即"两圆不接触"这一物理下限
    永远被包含在内，不会被任何 δ 或参数组合挖掉。这条不变量由
    ``tests/test_dwa.py::test_center_gap_threshold_never_below_contact`` 锁定。
    """
    return (
        float(params.body_radius)
        + float(obstacle_radius)
        + required_gap(delta_rad, params)
    )


def relative_direction(
    obstacle_xy: np.ndarray, obstacle_velocity: np.ndarray, self_xy: np.ndarray
) -> float:
    """相对运动方向 ``δ`` [rad]，取 [0, π]。"""
    relative = np.asarray(obstacle_velocity, dtype=float)
    offset = np.asarray(self_xy, dtype=float) - np.asarray(obstacle_xy, dtype=float)
    offset_norm = float(np.linalg.norm(offset))
    velocity_norm = float(np.linalg.norm(relative))
    if offset_norm < EPS or velocity_norm < EPS:
        return math.pi  # 无方向信息 -> 按"最宽容"处理，交给 gap_overtake 兜底
    cosine = float(np.clip(relative @ offset / (velocity_norm * offset_norm), -1.0, 1.0))
    return float(math.acos(cosine))


# ---------------------------------------------------------------------------
# 候选与评价
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """一个 (v, ω) 候选及其预测结果。"""

    v: float
    w: float
    xy: np.ndarray
    """预测轨迹 (steps+1, 2)：含当前位置作为第 0 行。"""

    admissable: bool
    reject_reason: str
    cost: float
    min_clearance_static: float
    min_gap_dynamic: float
    """与动态障碍的最小**圆心距** [m]（``inf`` 表示本周期无动态障碍）。"""

    heading_error: float
    lateral_error: float
    a_y_max: float
    jerk_rms: float
    delta_rad: float
    gap_required: float
    """要求的最小**圆心距** [m] = ``body_radius + r_o + required_gap(δ)``。

    对应"最小圆心距的那个障碍"（即 ``min_gap_dynamic`` 的来源障碍），
    因此 ``min_gap_dynamic >= gap_required`` 就是该判据的布尔形式。
    """

    contact_gap_dynamic: float = math.inf
    """**接触阈值**（圆心距）[m] = ``body_radius + r_o``（该候选最近障碍的 r_o）。

    用于判"是否已经物理穿透"：``min_gap_dynamic < contact_gap_dynamic`` 即两圆重叠。
    它比 ``gap_required`` 弱（少了表面间隙项 ``required_gap(δ)``），
    两者不能混用：前者是碰撞，后者是策略间距不足。
    """

    physical_violation: bool = False
    """是否违反**物理约束** ``clearance >= body_radius``（穿越车体，绝不允许）。

    兜底分支**不得**选择该候选，见 ``simulate`` 里的兜底循环。
    """

    clearance_deficit: float = 0.0
    """策略裕度不足量 ``max(0, d_min - clearance)`` [m]（超出幅度，非布尔）。"""

    physical_deficit: float = 0.0
    """物理约束不足量 ``max(0, body_radius - clearance)`` [m]。非 0 即碰撞。"""

    a_y_deficit: float = 0.0
    """横向加速度超出量 ``max(0, a_y - a_y_max)`` [m/s^2]（超出幅度，非布尔）。"""


@dataclass(frozen=True)
class Command:
    """一个周期输出的速度指令与状态。"""

    t: float
    xy: np.ndarray
    yaw: float
    v: float
    w: float
    a_y: float
    a_linear: float
    a_angular: float
    jerk: float

    @property
    def speed(self) -> float:  # 兼容 metrics 的命名
        return self.v


@dataclass(frozen=True)
class DwaResult:
    """一次局部规划执行的结果。

    **两级安全约束的统计必须分开**（本轮验收阻断 1）：

    * ``physical_collision_frames`` / ``physical_violation_periods``：违反
      ``clearance >= body_radius``，即**车身穿墙**。这是碰撞，任务失败。
    * ``violation_periods`` / ``policy_violation_periods``：策略裕度
      （``d_min``）或动态间距不足。**可降级**，记录幅度。

    布尔率必须与**超出幅度**一起报（``max_clearance_deficit_m`` 等），
    否则"差 1 mm"与"差 30 cm"看起来一样。

    **成功判据（2026-10-31 口径修正）**：``success = 到达终点 ∧ 无物理违反``。
    ``timeout``（周期预算用尽）**不算成功** —— 成功率必须等于到达率。
    三种结束状态因此互斥且穷尽：``reached_goal`` / ``timeout`` / 物理违反。
    """

    success: bool
    """是否成功：**到达终点且无物理违反**（``timeout`` 不算成功）。"""

    reason: str
    """结束原因：``reached_goal`` / ``timeout`` / ``physical_violation``
    / ``reached_goal_with_collision`` / ``no_admissable_candidate``。"""
    commands: tuple[Command, ...]
    exit_xy: np.ndarray
    reached_goal: bool
    collision_frames: int
    """**物理碰撞帧数**：``clearance < body_radius`` 的周期数（车身穿墙）。

    注意与 ``violation_periods``（策略违反）**不是同一个量**，不能合并。
    """
    min_clearance_m: float
    min_gap_dynamic_m: float
    """最小**圆心距** [m]（车体中心到障碍中心，非表面距）。

    口径说明（2026-10-30 修正）：这是圆心距。判据是

        min 圆心距 >= body_radius + r_o + required_gap(δ)

    其中 ``required_gap(δ)`` 是**表面间隙**要求（0.25~0.60 m），
    ``body_radius + r_o`` 是**两圆不接触**的物理下限（默认 0.58 + 0.30 = 0.88 m）。
    因此 **``min_gap_dynamic_m >= body_radius + r_o`` 才等价于"无接触"**；
    早先写的 ``>= r_o`` 漏掉了车体半径，会把"两圆重叠 0.33 m"判成无接触。
    小于 ``body_radius + r_o`` 即物理穿透，计入碰撞。
    """
    periods: int
    violation_periods: int
    violation_period_flags: tuple[bool, ...]
    fallback_periods: int = 0
    """触发**停车兜底**（v=0）的周期数。"""

    physical_violation_periods: int = 0
    """违反**物理约束**的周期数（``clearance < body_radius``）。计入碰撞。"""

    policy_violation_periods: int = 0
    """仅违反**策略裕度**的周期数（``body_radius <= clearance < d_min``）。"""

    dynamic_violation_periods: int = 0
    """违反动态间距判据的周期数。"""

    max_clearance_deficit_m: float = 0.0
    """``max(0, d_min - clearance)`` 的最大值 [m]（策略裕度不足幅度）。"""

    max_physical_deficit_m: float = 0.0
    """``max(0, body_radius - clearance)`` 的最大值 [m]。非 0 即发生穿墙。"""

    max_a_y_deficit: float = 0.0
    """``max(0, a_y - a_y_max)`` 的最大值 [m/s^2]（横向加速度超出幅度）。"""

    max_gap_deficit_m: float = 0.0
    """``max(0, required_gap - min_gap)`` 的最大值 [m]（动态间距不足幅度）。"""

    max_accel_deficit: float = 0.0
    """``max(0, |a| - a_max)`` 的最大值 [m/s^2]（线加速度超出幅度）。

    加速度是**可降级但必须记录**的一类约束（与策略裕度同级），
    不是物理碰撞。这里把它显式化，避免"窗口边界隐含保证"被其他项覆盖后
    出现"违反率 0.000 但实际超限"的矛盾（本轮验收补充 3b）。
    """

    max_angular_deficit: float = 0.0
    """``max(0, |alpha| - alpha_max)`` 的最大值 [rad/s^2]。"""

    @property
    def constraint_violation_rate_per_period(self) -> float:
        if self.periods == 0:
            return 0.0
        return self.violation_periods / self.periods

    @property
    def constraint_violation_rate_per_task(self) -> float:
        """按任务计：只要任一周期违反即为 1。**口径必须在论文里声明。**"""
        return 0.0 if not self.violation_period_flags else float(
            any(self.violation_period_flags)
        )

    @property
    def physical_collision_rate_per_period(self) -> float:
        if self.periods == 0:
            return 0.0
        return self.physical_violation_periods / self.periods

    @property
    def task_failed_by_collision(self) -> bool:
        """是否因物理碰撞而任务失败。"""
        return self.physical_violation_periods > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "success": int(self.success),
            "reason": self.reason,
            "periods": self.periods,
            "violation_periods": self.violation_periods,
            "violation_rate_per_period": self.constraint_violation_rate_per_period,
            "violation_rate_per_task": self.constraint_violation_rate_per_task,
            "physical_collision_frames": self.collision_frames,
            "physical_violation_periods": self.physical_violation_periods,
            "policy_violation_periods": self.policy_violation_periods,
            "dynamic_violation_periods": self.dynamic_violation_periods,
            "reached_goal": int(self.reached_goal),
            "min_clearance_m": self.min_clearance_m,
            "min_gap_dynamic_m": self.min_gap_dynamic_m,
            "max_clearance_deficit_m": self.max_clearance_deficit_m,
            "max_physical_deficit_m": self.max_physical_deficit_m,
            "max_a_y_deficit": self.max_a_y_deficit,
            "max_gap_deficit_m": self.max_gap_deficit_m,
        }


def _sample_clearance(
    grid: np.ndarray,
    points: np.ndarray,
    *,
    resolution: float,
    origin: tuple[float, float],
    ox: float,
    oy: float,
) -> np.ndarray:
    """``core.bspline.sample_field`` 的**热路径内联版**（语义完全相同）。

    为什么单独实现一份：DWA 每个周期要评估 ~135 个候选、每个候选 7 个轨迹点，
    ``sample_field`` 是单项最大热点（单场景 37805 次调用 / 4.2 s）。
    这里省掉参数校验、``np.asarray`` 与重复的属性访问，并把 4 个角点的
    权重运算合并。**插值公式与边界约定必须与 ``sample_field`` 一致**
    （双线性、网格外按 0），有测试 ``test_inline_sampler_matches_bspline`` 锁定。
    """
    res = resolution
    gx = (points[:, 0] - ox) / res - 0.5
    gy = (points[:, 1] - oy) / res - 0.5
    rows, cols = grid.shape
    inside = (gx >= 0) & (gx <= cols - 1) & (gy >= 0) & (gy <= rows - 1)
    x0 = np.floor(gx).astype(np.int32)
    y0 = np.floor(gy).astype(np.int32)
    np.clip(x0, 0, cols - 2, out=x0)
    np.clip(y0, 0, rows - 2, out=y0)
    fx = gx - x0
    fy = gy - y0
    np.clip(fx, 0.0, 1.0, out=fx)
    np.clip(fy, 0.0, 1.0, out=fy)
    one_minus_fx = 1.0 - fx
    one_minus_fy = 1.0 - fy
    values = (
        grid[y0, x0] * one_minus_fx * one_minus_fy
        + grid[y0, x0 + 1] * fx * one_minus_fy
        + grid[y0 + 1, x0] * one_minus_fx * fy
        + grid[y0 + 1, x0 + 1] * fx * fy
    )
    values[~inside] = 0.0
    return values


def _window_bounds(
    previous: tuple[float, float], params: DwaParams
) -> tuple[float, float, float, float]:
    """四个动态窗口（7.9 第 1 条）：v / ω / a / α。

    以**上一周期指令**为中心，按 ``a_max * dt`` 与 ``alpha_max * dt`` 展开，
    并夹到平台物理上限。首周期 ``previous = (0, 0)``。
    """
    v_prev, w_prev = previous
    dt = params.dt_s
    v_lo = max(-params.v_max, v_prev - params.a_max * dt)
    v_hi = min(params.v_max, v_prev + params.a_max * dt)
    w_lo = max(-params.w_max, w_prev - params.alpha_max * dt)
    w_hi = min(params.w_max, w_prev + params.alpha_max * dt)
    return v_lo, v_hi, w_lo, w_hi


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _rollout(
    xy0: np.ndarray, yaw0: float, v: float, w: float, params: DwaParams
) -> tuple[np.ndarray, np.ndarray]:
    """按恒定 (v, ω) 展开预测轨迹（差速驱动模型，中点积分更稳）。

    返回 ``(轨迹 (steps+1,2), 逐步 yaw (steps+1,))``。
    """
    steps = params.steps
    dt = params.dt_s
    points = np.zeros((steps + 1, 2), dtype=float)
    headings = np.zeros(steps + 1, dtype=float)
    points[0] = np.asarray(xy0, dtype=float)
    headings[0] = float(yaw0)
    for k in range(steps):
        mid = headings[k] + 0.5 * w * dt
        points[k + 1, 0] = points[k, 0] + v * math.cos(mid) * dt
        points[k + 1, 1] = points[k, 1] + v * math.sin(mid) * dt
        headings[k + 1] = headings[k] + w * dt
    return points, headings


def a_y_violated(candidate: Candidate, params: DwaParams) -> bool:
    """候选是否违反横向加速度硬约束（``a_y = v·ω`` 超过 ``a_y_max``）。

    单独成函数，是为了让"兜底候选池"复用同一判定：
    ``a_y`` 可解析、与历史无关，**任何情况下都不允许被兜底绕过**——
    为绕开障碍而牺牲横向加速度上限，等于拿舒适性换安全，两条硬约束不该互相交换。
    """
    return candidate.a_y_max > params.a_y_max + EPS


def _approach_cap(distance_to_goal: float, params: DwaParams) -> float:
    """终点减速带的速度上限 [m/s]。

    在 ``approach_slowdown_m`` 内按线性插值从 ``v_nominal`` 降到
    ``stop_speed_ratio * v_nominal``，保证"能停住"这件事由速度上限保证，
    而不是指望离散判据刚好踩中。

    这是**可行性层面的约束**（超速冲向终点 = 该候选不可行），按 7.9 第 4 条的
    取向放在硬约束一侧，而不是代价项。
    """
    band = float(params.approach_slowdown_m)
    if band <= EPS:
        return params.v_nominal
    ratio = float(np.clip(distance_to_goal / band, 0.0, 1.0))
    low = params.stop_speed_ratio * params.v_nominal
    return float(low + ratio * (params.v_nominal - low))


def evaluate_candidate(
    *,
    v: float,
    w: float,
    xy0: np.ndarray,
    yaw0: float,
    reference: Reference,
    params: DwaParams,
    clearance_field: np.ndarray | None,
    resolution: float,
    origin: tuple[float, float],
    obstacle_positions: tuple[np.ndarray, ...],
    obstacle_velocities: tuple[np.ndarray, ...],
    obstacle_radii: tuple[float, ...],
    previous_commands: tuple[Command, ...],
    t_now: float,
    goal_xy: tuple[float, float] | np.ndarray | None = None,
    local_arc: float | None = None,
    spin_floor_active: bool = True,
) -> Candidate:
    """评价一个候选：**先硬约束剔除，后代价排序**。

    硬约束（任一不满足即 ``admissable=False``）：

    1. 静态离障：预测轨迹每点 ``clearance >= d_min``；
    2. 动态间距：与每个动态障碍的最小**圆心距**
       ``>= body_radius + r_o + required_gap(δ)``（可行性判据，含物理不接触下限）；
    3. **横向加速度 ``a_y = v * ω <= a_y_max``**（可解析、属物理可行性）。

    代价项（仅在可行候选中比较）：

    * 跟踪：横向偏差 + 航向偏差；
    * 离障：静态 clearance 留裕度；
    * 速度保持：偏离期望速度；
    * **jerk**：与上一周期指令的差分（三阶量的代理）。
    """
    points, headings = _rollout(np.asarray(xy0, dtype=float), yaw0, v, w, params)

    # ---- 硬约束 1：静态离障 ----
    static_clearance = np.inf
    static_fail_point = -1
    if clearance_field is not None:
        # 热路径：用内联版双线性采样（语义与 core.bspline.sample_field 一致，
        # 由测试锁定），单场景可省 2~3 s。
        ox, oy = float(origin[0]), float(origin[1])
        clearance_grid = np.asarray(clearance_field, dtype=float)
        sample = _sample_clearance(
            clearance_grid, points, resolution=resolution, origin=origin, ox=ox, oy=oy
        )
        # ---- 两级安全约束（本轮验收阻断 1）----
        #
        #   物理约束: clearance >= body_radius  -> 绝不违反，违反即碰撞/任务失败
        #   策略约束: clearance >= d_min        -> 可降级，但必须记录幅度
        #
        # 两个层级的**处理方式不同**，这是本函数最关键的区分：
        #   * 物理违反 -> 该候选**直接不可行**，兜底也不允许选它（会穿墙）；
        #   * 策略违反 -> 候选不可行，但兜底可以选（记录违反幅度）。
        #
        # 历史问题：两者合成一个 d_min 判据，于是"最小风险动作"可能选到
        # clearance < body_radius 的候选，hard 档出现 min_clearance = 0.371 m
        #（车身半宽 0.58 m）却仍被记为"到达"。
        here = float(sample[0]) if sample.size else 0.0
        future_sample = sample[1:] if sample.shape[0] > 1 else sample
        future_min = float(future_sample.min()) if future_sample.size else here
        static_clearance = min(here, future_min)

        # 物理违反：候选轨迹任一点（含当前位置）低于车身半径
        physical_fail = (
            static_clearance + _boundary_tol(params.body_radius) < params.body_radius
        )

        if here + EPS >= params.d_min:
            # A：当前位置满足策略裕度 -> 未来点也必须满足（含当前位置，取更严者）
            policy_fail = (
                static_clearance + _boundary_tol(params.d_min) < params.d_min
            )
        else:
            # B：当前位置已在策略边界内（避障被挤进窄处）
            #    -> 只禁止"继续恶化"：未来最小 clearance 不得低于当前位置太多，
            #       留 regression_tolerance_m 容差避免抖动锁死。
            policy_fail = (
                future_min + EPS < here - params.regression_tolerance_m
            )

        if physical_fail:
            static_fail_point = int(np.argmin(sample))
        elif policy_fail:
            static_fail_point = int(np.argmin(future_sample)) + 1
        else:
            static_fail_point = -1

    # ---- 硬约束 2：动态间距（绕行可行性判据）----
    #
    # 判据（2026-10-30 修正）：
    #     min 圆心距 >= body_radius + r_o + required_gap(δ)
    #                  \_________/   \__/   \________________/
    #                   车体半径    障碍半径   表面间隙要求 g_surface(δ)
    #
    # 三项缺一不可：
    #   * `body_radius + r_o` 是**两圆不接触**的充要条件，与 δ 无关（物理下限）；
    #   * `required_gap(δ)` 是两圆**表面之间**还要留出的策略间隙。
    #
    # 历史错误：阈值写成 `required_gap(δ) + r_o`，**漏掉车体半径**。
    # 同向档因此只有 0.25 + 0.30 = 0.55 m < body_radius + r_o = 0.88 m，
    # 允许两圆重叠 0.33 m；实测最小圆心距恰好饱和在 0.550 m（easy/hard 都是）。
    # 修正后阈值 0.58 + 0.30 + 0.25 = 1.13 m（同向）/ 1.48 m（迎面）。
    #
    # δ 必须**逐障碍**计算：用上一个障碍的 δ 去判当前障碍会把阈值算错
    # （多障碍时阈值与障碍不匹配）。
    dynamic_gap = np.inf
    delta_used = math.pi
    gap_required = params.gap_overtake_m
    contact_gap = float(params.body_radius)
    dynamic_fail = False
    for obstacle, velocity, radius in zip(
        obstacle_positions, obstacle_velocities, obstacle_radii
    ):
        gaps = np.linalg.norm(points[1:] - obstacle[1:], axis=1)
        gap = float(gaps.min()) if gaps.size else float("inf")
        delta_here = relative_direction(obstacle[0], velocity, points[0])
        # 该障碍自己的接触阈值与圆心距阈值
        contact_here = float(params.body_radius) + float(radius)
        required_here = contact_here + required_gap(delta_here, params)
        if gap < dynamic_gap:
            dynamic_gap = gap
            delta_used = delta_here
            gap_required = required_here
            contact_gap = contact_here
        if gap + EPS < required_here:
            dynamic_fail = True

    # ---- 硬约束 3：横向加速度 a_y = v * ω ----
    a_y = abs(float(v) * float(w))
    # 边界容差：`a_y_max` 恰好被取到时（例如 v=1.0, w=0.8 -> a_y=0.800）必须判**合规**。
    # 只用 EPS=1e-9 时，浮点乘法的末位误差会让等式边界被判成违反，
    # 于是"按任务违反率 50%"里混进了一批"恰好等于上限"的假违反（本轮验收补充 2）。
    # 用相对容差 1e-6（对应 0.8 的 8e-7 绝对量），远小于任何有意义的物理差异。
    a_y_tolerance = 1e-6 * max(1.0, params.a_y_max)
    a_y_fail = a_y > params.a_y_max + a_y_tolerance

    # ---- 硬约束 4：原地自转抑制 ----
    # a_y = v·ω 在 v→0 时失效，高 ω + 低速 = 原地打转。这里要求：
    # ω 超过门限时，v 必须不低于 spin_speed_floor * v_nominal（即"转向不能代替行驶"）。
    # ---- 硬约束 4：原地自转抑制 ----
    # a_y = v·ω 在 v→0 时失效，高 ω + 低速 = 原地打转。要求：ω 超门限时，
    # v 必须不低于 spin_speed_floor * v_nominal（"转向不能代替行驶"）。
    #
    # **门控由调用方传入的 spin_floor_active 决定**（而不是在这里算 v_upper）：
    # 是否"可达"是**速度窗口**的属性，只有 simulate 知道窗口上界。
    # 若无条件要求 v >= 0.25，上一周期 ω 一旦打满（ω 窗口被锁在 [1.881,2.09]），
    # 接下来几个周期里全部候选会同时违反 spin，车彻底锁死
    # （实测 houseexpo_30a18345 head_on：135/135 候选全判 spin，原地卡 191 周期）。
    spin_gate = params.spin_rate_gate_ratio * params.alpha_max * params.dt_s
    spin_floor = params.spin_speed_floor * params.v_nominal
    spin_fail = (
        spin_floor_active
        and abs(float(w)) > spin_gate + EPS
        and float(v) < spin_floor - EPS
    )

    # ---- 硬约束 5：终点减速带（超速逼近终点 = 不可行）----
    goal_xy = (
        np.asarray(goal_xy, dtype=float)
        if goal_xy is not None
        else np.asarray(reference.xy[-1], dtype=float)
    )
    distance_to_goal = float(np.linalg.norm(goal_xy - np.asarray(points[0], dtype=float)))
    approach_cap = _approach_cap(distance_to_goal, params)
    approach_fail = float(v) > approach_cap + EPS

    # ---- 代价项（只对可行候选有意义，失败候选给 inf）----
    # 用**窗口化投影**（以当前位置的弧长为窗口中心）：全量投影是 DWA 的第一
    # 性能热点（单场景 56841 次调用、14 s）。窗口 1.5 m 覆盖一个预测时域的
    # 推进量与常见绕行幅度；车若被推得超出窗口，project_windowed 内部会
    # 自动退回全量投影，语义不受影响。
    reference_arc_now = float(reference.project(points[0])[1][0]) if local_arc is None else float(local_arc)
    projection, arcs, lateral = reference.project_windowed(
        points[1:], reference_arc_now, 1.5
    )
    # 横向偏差按**预测轨迹末端**取，而不是全程平均：平均会被"起步阶段贴着
    # 参考线"的低速候选占便宜，导致规划器倾向于原地磨蹭。
    lateral_error = float(lateral[-1])

    # 横向偏离上限是**硬约束**：DWA 可以绕行，但不能无限制地离开参考路径。
    # 参考路径是全局层验证过"整条都满足 d_min 与曲率约束"的唯一轨迹；
    # 一旦偏离过远，就会走进全局层从未验证的窄处（实测
    # `houseexpo_d0aeed69` 同向慢速场景：参考路径在 x≈4.0 处宽度 1.62 m，
    # 而 DWA 偏到 x≈4.8 的窄走廊 —— 那里 clearance 只有 0.703，恰好卡在
    # d_min=0.70 边界上，随后必然掉到 0.587 并被锁死）。
    # 绕行的**自由度**必须被限制在"参考路径附近"，否则全局层的可行性保证失效。
    lateral_limit = float(getattr(params, "lateral_deviation_max_m", 0.0))
    # **同样的"已越界不能越锁越死"原则**：硬约束只在**当前位置尚在限内**时生效。
    # 若当前位置已经超出 lateral_limit（例如被障碍推离参考线），继续用硬约束
    # 会把"回到参考线"的所有候选一并拒掉 —— 实测 houseexpo_07432f2d 同向慢速：
    # 车偏离到 0.692 m（限 0.6），135/135 候选全判 lateral，原地卡 211 周期到超时。
    # 此时改为软代价（lateral_error 直接进 cost），只禁止"进一步走远"。
    here_projection_arc, _, here_lateral_arr = reference.project_windowed(
        points[0], reference_arc_now, 1.5
    )
    current_lateral = float(np.atleast_1d(here_lateral_arr)[0])
    over_limit_now = lateral_limit > 0 and current_lateral > lateral_limit + EPS
    if lateral_limit > 0 and not over_limit_now:
        lateral_fail = lateral_error > lateral_limit + EPS
    else:
        # 已越界：只禁止比现在更远（留 regression_tolerance_m 容差），
        # 否则回参考线的过渡动作会被判死。
        lateral_fail = (
            lateral_limit > 0
            and lateral_error > current_lateral + params.regression_tolerance_m + EPS
        )

    # 横向偏离的**硬约束只有固定米数上限**（lateral_limit）。
    #
    # 曾经试过把"偏离参考线损失的离障余量"也做成硬约束，结论是**不可行**：
    # 一旦车偏离参考线，回到参考线的过程本身就要横穿低 clearance 区，
    # 硬约束会把"回去"这个正确动作一并堵死，导致 371/402 个周期走兜底、
    # 原地锁死到超时（实测 `houseexpo_d0aeed69`）。而且"当前离线偏离是否超限"
    # 的判定还会随预测点/当前点取法而翻转，极易自相矛盾。
    #
    # 因此余量损失只作为**软代价**（lateral_margin_loss_penalty，见下方 cost），
    # 鼓励尽快回到参考线；硬约束只保留"离参考线不超过 lateral_limit 米"。
    here_projection, _, here_lateral = reference.project_windowed(
        points[0], reference_arc_now, 1.5
    )
    current_lateral = float(here_lateral[0]) if np.ndim(here_lateral) else float(here_lateral)
    reference_clearance_at_end = (
        float(
            _sample_clearance(
                clearance_grid,
                np.asarray(projection[-1:], dtype=float),
                resolution=resolution,
                origin=origin,
                ox=ox,
                oy=oy,
            )[0]
        )
        if clearance_field is not None and projection.shape[0] > 0
        else np.inf
    )
    candidate_clearance_at_end = (
        float(sample[-1]) if clearance_field is not None and sample.size else np.inf
    )
    margin_loss = max(0.0, reference_clearance_at_end - candidate_clearance_at_end)
    arc_now = reference_arc_now
    arc_end = float(arcs[-1])

    # 剩余弧长：**必须显式惩罚"过冲"**，但只能用**相对量**。
    #
    # 用绝对剩余距离 `w_dist * (length - arc_end)` 会把整条代价压垮：
    # 距终点 9.8 m 时该项就有 4.9，占代价的 87%，而**同一位置上所有候选的
    # 剩余距离几乎相同**——它不区分候选，纯粹抬高了公共底噪，
    # 反而让真正起区分作用的横向/航向/离障项被淹没（实测 d0aeed69 出现
    # v=0 & ω 打满的"原地自转"被判成最优，代价 5.65 里 4.9 都来自这一项）。
    #
    # 正确用法：只看**本周期推进了多少**，过冲（arc_end < arc_now）才罚。
    remaining_overrun = max(0.0, arc_now - arc_end)
    progress = arc_end - arc_now

    # 目标朝向：参考轨迹在投影弧长处的切向
    ref_index = int(np.clip(np.searchsorted(reference.s, arc_end), 0, reference.s.shape[0] - 1))
    heading_error = abs(_wrap(float(headings[-1]) - float(reference.yaw[ref_index])))

    # 速度保持对**标称巡航速度**取偏差（不是平台上限 v_max）。
    velocity_error = abs(float(v) - params.v_nominal)
    # 停滞惩罚：只在**本可前进**（v>0）却净推进为负时生效。
    # 不能无条件惩罚"停住"——前进被障碍挡住时，停车是**正确**的安全动作；
    # 若把停车也罚重，规划器会为了"动起来"而倒车（实测倒行 14 m 冲出可行域）。
    stall_penalty = 0.0
    # 停滞惩罚必须覆盖"**停车**"本身，而不只是"前进但净推进为负"。
    #
    # 之前只在 `progress <= 0 and v > 0` 时罚，导致 v=0 完全免罚。而速度动态窗口
    # 每周期只放开 0.08 m/s，`velocity_error = |v-1|*w_vel` 在 0 与 0.08 之间的
    # 差只有 0.024 —— 一旦横向误差有 0.001 量级的差异，"停车"就可能比"前进"便宜，
    # 车便停在终点前 0.379 m 处不再走（实测 houseexpo_07432f2d 原地耗尽 247 周期）。
    #
    # 语义上：**在还有余量前进时，停车不该是最优解**。因此只要
    # (a) 本周期净推进为负 或 (b) 速度明显低于"本可达到的速度"，
    # 就施加一项与后退同量级的惩罚。这样"停车/后退"都被明确排斥，
    # 而"被物理挡住真的走不了"仍由兜底分支处理（兜底不受代价项影响）。
    best_reachable_v = min(params.v_nominal, float(v) + params.a_max * params.dt_s)
    slow_while_able = float(v) < 0.5 * best_reachable_v - EPS
    if (progress <= EPS and float(v) > EPS) or slow_while_able:
        stall_penalty = params.w_vel * params.v_nominal * 2.0
    # 倒车代价必须**按标称速度计**，不能按 |v| 计。
    # 用 `w_vel * |v| * 2` 时，低速倒车只罚 0.048（|v|=0.08），而"前进但净推进为负"
    # 罚 0.6 —— 于是"倒一小步"总是比"试着前进"便宜，规划器会宁可在终点前 0.37 m
    # 处反复微倒车也不肯走最后一步（实测 houseexpo_07432f2d 因此超时）。
    # 倒车是应当被明确排斥的行为，其代价应与速度无关地固定在标称量级。
    reverse_penalty = 0.0
    if float(v) < -EPS:
        reverse_penalty = params.w_vel * params.v_nominal * 2.0

    # jerk 代理：**必须有至少两个历史周期**才是三阶量。
    #
    # 两个陷阱（都实测过）：
    # 1. 只有一个历史周期时只能算加速度（一阶差分 / dt），拿它当 jerk 惩罚会
    #    把"正常起步加速"误罚——代价项压过速度项后，规划器永远停在加速度窗口
    #    下界（v 恒为 a_max*dt = 0.08 m/s），任务跑不完。所以历史不足 2 个周期
    #    时 jerk 记 0。
    # 2. 即使有 2 个历史周期，"从静止重新起步"也会被算成大 jerk
    #    （历史 v=0,0 → 候选 v=0.08 得 jerk≈8，惩罚 1.2 远超全部收益），
    #    导致车让行一次后**再也起步不了**（实测 head_on 场景 t=10 停住后
    #    直到超时都不动）。因此这里按**加速度上限**归一化：
    #    只有超出 a_max 的那部分才算 jerk，物理上对应"绞痛感"的突变，
    #    而 a_max 以内的平滑加减速是允许的、不该被罚。
    jerk_proxy = 0.0
    if len(previous_commands) >= 2:
        last = previous_commands[-1]
        before_last = previous_commands[-2]
        raw_jerk = abs(
            (float(v) - 2.0 * last.v + before_last.v) / (params.dt_s ** 2)
        )
        # a_max 对应的"允许的二阶差"：在 dt 内，加速度从 -a_max 变到 +a_max
        # 是物理允许的最大变化率，其离散二阶差上界约为 2*a_max*dt/dt^2 = 2a_max/dt。
        allowed = 2.0 * params.a_max / params.dt_s
        jerk_proxy = max(0.0, raw_jerk - allowed)

    admissable = (
        static_fail_point < 0
        and not dynamic_fail
        and not a_y_fail
        and not spin_fail
        and not lateral_fail
        and not approach_fail
    )
    if a_y_fail:
        reason = "a_y"
    elif static_fail_point >= 0:
        reason = "clearance"
    elif dynamic_fail:
        reason = "dynamic_gap"
    elif spin_fail:
        reason = "spin"
    elif lateral_fail:
        reason = "lateral"
    elif approach_fail:
        reason = "approach_speed"
    else:
        reason = "ok"

    if not admissable:
        cost = float("inf")
    else:
        cost = (
            params.w_dist * lateral_error
            + params.w_heading * heading_error
            + params.w_vel * velocity_error
            + params.w_clearance * max(0.0, params.d_min * 2.0 - static_clearance)
            + params.jerk_weight * jerk_proxy
            + params.w_dist * remaining_overrun * 10.0
            + stall_penalty
            + reverse_penalty
            + params.lateral_margin_loss_weight * margin_loss
        )

    return Candidate(
        v=float(v),
        w=float(w),
        xy=points,
        admissable=admissable,
        reject_reason=reason,
        cost=float(cost),
        min_clearance_static=float(static_clearance),
        min_gap_dynamic=float(dynamic_gap),
        heading_error=float(heading_error),
        lateral_error=float(lateral_error),
        a_y_max=float(a_y),
        jerk_rms=float(jerk_proxy),
        delta_rad=float(delta_used),
        gap_required=float(gap_required),
        contact_gap_dynamic=float(contact_gap),
        physical_violation=bool(
            static_clearance + EPS < params.body_radius
            if clearance_field is not None
            else False
        ),
        clearance_deficit=(
            max(0.0, params.d_min - static_clearance)
            if clearance_field is not None
            else 0.0
        ),
        physical_deficit=(
            max(0.0, params.body_radius - static_clearance)
            if clearance_field is not None
            else 0.0
        ),
        a_y_deficit=max(0.0, float(a_y) - params.a_y_max - a_y_tolerance),
    )


# ---------------------------------------------------------------------------
# 规划器
# ---------------------------------------------------------------------------


@dataclass
class DwaPlanner:
    """逐周期 DWA 规划器：唯一持有状态（上一周期指令），其余全为纯函数。

    ``core/`` 的纯函数约束：本类是**显式状态容器**，不含模块级可变状态，
    也不读配置文件；所有阈值来自 ```params```。
    """

    reference: Reference
    params: DwaParams
    clearance_field: np.ndarray | None = None
    resolution: float = 0.1
    origin: tuple[float, float] = (0.0, 0.0)
    start_xy: np.ndarray | None = None
    start_yaw: float | None = None

    commands: list[Command] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 前置红线：不可跟踪的参考轨迹直接拒绝（防呆，Reference 已校验一次）
        if not np.all(np.isfinite(np.asarray(self.reference.curvature, dtype=float))):
            raise NotTrackableError("参考轨迹曲率非有限，不可跟踪")

    # -- 主循环 --------------------------------------------------------

    def simulate(
        self,
        obstacles: tuple[ObstacleTrack, ...] = (),
        *,
        max_periods: int | None = None,
        goal_xy: tuple[float, float] | None = None,
        period_budget_factor: float = 3.0,
    ) -> DwaResult:
        """执行逐周期规划，直到到达终点 / 超时 / 无可行候选。

        ``max_periods`` 默认按**任务长度自适应**：``参考长度 / v_max / dt`` 的
        ``period_budget_factor`` 倍。固定给一个大上限会让"走不动"的任务空转几百
        周期（单周期 ~0.06 s，600 周期就是 40 s），既慢又掩盖了"停滞"这种失败。

        返回的 ``DwaResult`` 里同时给出**按周期**与**按任务**两种约束违反率，
        论文必须声明用哪一个（7.9 第 6 条）。
        """
        ref = self.reference
        if max_periods is None:
            ideal = ref.length_m / max(self.params.v_max, EPS) / self.params.dt_s
            max_periods = int(math.ceil(ideal * period_budget_factor)) + 20
        start = (
            np.asarray(self.start_xy, dtype=float)
            if self.start_xy is not None
            else ref.xy[0].copy()
        )
        yaw = float(self.start_yaw) if self.start_yaw is not None else float(ref.yaw[0])
        goal = (
            np.asarray(goal_xy, dtype=float) if goal_xy is not None
            else ref.xy[-1].copy()
        )
        # 若起点不在参考路径上，先用参考路径起点对齐（本框架保证起终点重合）
        self.commands = []
        current = start.copy()
        t = 0.0
        flags: list[bool] = []
        flags_physical: list[bool] = []
        flags_policy: list[bool] = []
        flags_dynamic: list[bool] = []
        physical_collision_frames = 0
        min_clearance = np.inf
        min_gap = np.inf
        max_clearance_deficit = 0.0
        max_physical_deficit = 0.0
        max_a_y_deficit = 0.0
        max_gap_deficit = 0.0
        max_accel_deficit = 0.0
        max_angular_deficit = 0.0
        reason = "timeout"
        reached = False
        fallback_used = 0
        """触发安全兜底（停车）的周期数。"""

        # 到达容差必须**至少覆盖一个周期的位移**：每周期位移是 v*dt（可达 0.1 m），
        # 而 stop_margin_m 默认只有 0.02 m —— 点判据几乎不可能被"踩中"，
        # 车会在终点前 0.1~0.4 m 处反复微调直到超时
        # （实测 houseexpo_07432f2d 卡在弧长差 0.374 m 处）。
        # 容差取 max(stop_margin_m, v_nominal * dt_s)，语义是"再来一步就能到"。
        reach_tolerance = max(
            self.params.stop_margin_m,
            self.params.v_nominal * self.params.dt_s,
        )

        for period in range(max_periods):
            # 到达判据：**点判据 + 越线判据**。
            # 只看欧氏距离会漏掉"已经沿参考方向越过终点"的情形——此时车到终点的
            # 欧氏距离可能很大（实测 same_direction 场景越过 2 m 后卡死），但任务
            # 在路径跟踪语义上已经完成。用参考弧长投影判断"是否走完全程"。
            _, arcs_now, _ = ref.project(current)
            arc_now = float(arcs_now[0])
            if (
                float(np.linalg.norm(current - goal)) <= reach_tolerance + EPS
                or arc_now >= ref.length_m - reach_tolerance - EPS
            ):
                reached = True
                reason = "reached_goal"
                break

            obstacle_now = tuple(
                track.predict(t, np.array([t + self.params.dt_s * k
                                           for k in range(self.params.steps + 1)]))
                for track in obstacles
            )
            obstacle_velocity = tuple(track.velocity_at(t) for track in obstacles)
            obstacle_radii = tuple(float(track.radius_m) for track in obstacles)
            min_obstacle_radius = min(obstacle_radii) if obstacle_radii else None
            # 弧长每周期只算一次，传给全部候选（否则每个候选都要投影一次当前位置，
            # 135 候选 x 数百周期是主要开销之一）。
            local_arc_now = float(ref.project(current)[1][0])

            # 当前位置的 clearance：兜底"停车"需要它来判断是否已物理越界。
            if self.clearance_field is not None:
                current_clearance = float(
                    _sample_clearance(
                        np.asarray(self.clearance_field, dtype=float),
                        np.asarray(current, dtype=float).reshape(1, 2),
                        resolution=self.resolution,
                        origin=self.origin,
                        ox=float(self.origin[0]),
                        oy=float(self.origin[1]),
                    )[0]
                )
            else:
                current_clearance = float("inf")

            best: Candidate | None = None
            fallback: Candidate | None = None
            fallback_risk: tuple[float, float, float] | None = None
            v_prev, w_prev = (
                (self.commands[-1].v, self.commands[-1].w) if self.commands else (0.0, 0.0)
            )
            v_lo, v_hi, w_lo, w_hi = _window_bounds((v_prev, w_prev), self.params)
            # 终点减速带：收紧速度上界。
            #
            # **但绝不能把上界压到低于物理可达下限**（本轮验收补充 3b）：
            # 从 v_prev 减速时，一个周期内最多降 a_max*dt，所以
            # `v_lo` 就是"物理上还能减到多低"。若减速带给的 cap 低于 v_lo，
            # 直接把 v_hi 压过去会让 `np.linspace(v_lo, v_hi)` **反向采样**，
            # 采出低于 v_lo 的速度 —— 实测末端出现 1.0 -> 0.47 m/s（1.43 m/s²，
            # 远超 a_max=0.8），而违反率仍显示 0.000，因为加速度约束根本没被检查。
            #
            # 正确做法：cap 只在上界一侧生效；当 cap < v_lo 时取 v_lo（物理可达），
            # 并在**更早的距离**就开始减速（减速带本身已按距离线性插值，
            # 只要 band 足够长就不会出现这种冲突）。这里额外做一个显式保护，
            # 避免参数组合导致反向窗口。
            distance_to_goal = float(np.linalg.norm(goal - current))
            cap = _approach_cap(distance_to_goal, self.params)
            v_hi = min(v_hi, max(cap, v_lo))
            v_hi = max(v_hi, v_lo)
            # 原地自转抑制是否生效：取决于**本周期速度窗口**能否达到门槛。
            spin_floor = self.params.spin_speed_floor * self.params.v_nominal
            spin_floor_active = v_hi >= spin_floor - EPS
            v_options = np.linspace(v_lo, v_hi, self.params.v_samples)
            w_options = np.linspace(w_lo, w_hi, self.params.w_samples)

            for v_candidate in v_options:
                for w_candidate in w_options:
                    candidate = evaluate_candidate(
                        v=float(v_candidate),
                        w=float(w_candidate),
                        xy0=current,
                        yaw0=yaw,
                        reference=ref,
                        params=self.params,
                        clearance_field=self.clearance_field,
                        resolution=self.resolution,
                        origin=self.origin,
                        obstacle_positions=obstacle_now,
                        obstacle_velocities=obstacle_velocity,
                        obstacle_radii=obstacle_radii,
                        previous_commands=tuple(self.commands),
                        t_now=t,
                        goal_xy=goal,
                        local_arc=local_arc_now,
                        spin_floor_active=spin_floor_active,
                    )
                    if candidate.admissable and (
                        best is None or candidate.cost < best.cost
                    ):
                        best = candidate
                    # 兜底候选：按"最小风险"排序——先看动态间距余量，再看静态离障，
                    # 最后偏好更慢的速度。只在全部候选都不可行时启用。
                    #
                    # **a_y 越界的候选绝不进入兜底池**：a_y = v·ω 是可以解析算出的
                    # 物理量，不依赖任何"探测不到"的信息；为了绕开障碍而违反
                    # 横向加速度上限，等于用舒适性换安全，两条硬约束不该互相交换。
                    if a_y_violated(candidate, self.params):
                        continue
                    # 兜底选择的优先级（字典序比较 risk 元组，越大越好）：
                    #
                    #   1. **是否发生了物理穿透**（圆心距 < body_radius + r_o）
                    #      ——最根本。碰撞与否必须排在最前：为了"停车优先"而接受
                    #      穿透是本末倒置（实测 head_on 场景曾出现 0.258 m 圆心距）。
                    #      **阈值必须含车体半径**（2026-10-30 修正）：旧写法只比 r_o
                    #      （0.30 m），把"两圆重叠 0.33 m"漏判成没有穿透。
                    #   2. 本周期位移尽量小（-|v*dt| 越大越好），即"停车优先"。
                    #      若把它排到间距前面都无所谓，但**排到穿透前面**会奖励
                    #      倒车跑最远以求最大间距，实测倒行 28 m 冲出地图，所以
                    #      位移项必须在"没有穿透"的组内比较。
                    #   3. 动态间距余量；
                    #   4. 静态离障余量。
                    #
                    # 语义：所有候选都不满足全部硬约束时，先避免碰撞，再就地减速/停车，
                    # 把控制权交回下一次观测，而不是用激进动作去"挣脱"。
                    penetrated = (
                        candidate.min_gap_dynamic
                        < candidate.contact_gap_dynamic - EPS
                        if min_obstacle_radius is not None
                        else False
                    )
                    risk = (
                        -1.0 if penetrated else 0.0,
                        -abs(candidate.v) * self.params.dt_s,
                        candidate.min_gap_dynamic - candidate.gap_required,
                        candidate.min_clearance_static - self.params.d_min,
                    )
                    if fallback is None or risk > fallback_risk:
                        fallback = candidate
                        fallback_risk = risk

            if best is None:
                # **兜底动作 = 停车**（本轮验收阻断 2）。
                #
                # 旧实现是"选最小风险候选"，但任何非零速度都可能在下一周期把车
                # 推进墙体（hard 档实测 min_clearance = 0.371 m < body_radius 0.58 m，
                # 即车身穿墙却仍记为"到达"）。
                #
                # 新语义：所有候选都不满足约束时，**只执行 v = 0**。
                # 静止是唯一"不新增碰撞风险"的动作——位置不变，因此不可能比
                # 当前状态更差；它永远可行（除非当前位置本身已违反物理约束）。
                #
                # 停车的含义是"本周期不前进"，把决策权交回下一次观测；
                # 若障碍离开，下一周期自然恢复行驶（不需要任何解锁逻辑）。
                #
                # 让步保留 ω = 0（不原地转向）：原地转向虽不移动，但会消耗
                # 后续周期的角速度窗口，且对脱离困境没有帮助。
                if current_clearance + EPS < self.params.body_radius:
                    # 当前位置已经物理越界（既成事实，无法通过停车修正），
                    # 这是一个**硬失败**：任务失败，不再继续执行。
                    reason = "physical_violation"
                    break
                # **停车也必须受线/角加速度约束**（本轮验收补充 3b 的同源坑）：
                #
                # * 直接把 v 置 0 -> 实测 v 从 0.961 一步降到 0，即 9.6 m/s²
                #   （a_max = 0.8），"违反率 0"与曲线自相矛盾；
                # * 直接把 ω 置 0 -> 实测角加速度 2.986 rad/s²（alpha_max = 2.09），
                #   同样超限。
                #
                # 物理上本周期最多只能减到 `prev - rate_max*dt`。兜底就取该值：
                # 位移尽可能小（安全），同时不违反加速度约束（口径一致）。
                stop_v = max(0.0, v_prev - self.params.a_max * self.params.dt_s)
                w_step = self.params.alpha_max * self.params.dt_s
                # ω 的兜底值 = 朝 0 方向走**一步**，而不是直接取 w_step。
                # 之前写成 `copysign(w_step, w_prev)`，当 |w_prev| 较大时
                # （如 -0.448）会一步跳到 -0.209，角加速度 2.389 > 2.09 超限。
                if abs(w_prev) <= w_step:
                    stop_w = 0.0
                else:
                    stop_w = w_prev - math.copysign(w_step, w_prev)
                best = Candidate(
                    v=float(stop_v),
                    w=float(stop_w),
                    xy=np.repeat(current[None, :], self.params.steps + 1, axis=0),
                    admissable=False,
                    reject_reason="fallback_stop",
                    cost=float("inf"),
                    min_clearance_static=current_clearance,
                    min_gap_dynamic=float("inf"),
                    heading_error=0.0,
                    lateral_error=0.0,
                    a_y_max=0.0,
                    jerk_rms=0.0,
                    delta_rad=math.pi,
                    gap_required=0.0,
                    physical_violation=False,
                    clearance_deficit=max(0.0, self.params.d_min - current_clearance),
                    physical_deficit=max(0.0, self.params.body_radius - current_clearance),
                    a_y_deficit=0.0,
                )
                fallback_used += 1

            # ---- 分级记录（本轮验收阻断 1 + 补充 2）----
            #
            # 物理约束（clearance >= body_radius）：违反 = 碰撞 = **任务失败**。
            # 策略约束（clearance >= d_min）/ 动态间距：违反 = 降级，记录幅度。
            #
            # 三者都进 `flags`（作为"任一约束违反"的统一口径），但物理违反
            # 会额外触发 `physical_collision_frames` 递增并把 success 置 False。
            physical_violation = (
                best.min_clearance_static + _boundary_tol(self.params.body_radius)
                < self.params.body_radius
            )
            # 策略违反**不含**物理违反（互斥统计）：
            # clearance < body_radius 时必然也 < d_min，若都计入 policy 会重复计算，
            # 让"策略降级"的数量虚高、掩盖真正的碰撞数量。
            policy_violation = (
                not physical_violation
                and best.min_clearance_static + _boundary_tol(self.params.d_min)
                < self.params.d_min
            )
            gap_violation = best.min_gap_dynamic + EPS < best.gap_required
            periodic_violation = physical_violation or policy_violation or gap_violation
            flags.append(bool(periodic_violation))
            flags_physical.append(bool(physical_violation))
            flags_policy.append(bool(policy_violation))
            flags_dynamic.append(bool(gap_violation))

            if physical_violation:
                physical_collision_frames += 1
            # 碰撞帧数的口径：**物理约束违反**才算碰撞（车身穿墙）。
            # 策略裕度不足不是碰撞，只是降级——两者不能合并计数。
            min_clearance = min(min_clearance, best.min_clearance_static)
            min_gap = min(min_gap, best.min_gap_dynamic)
            # 超出幅度（最大值，非布尔）——布尔率会掩盖"差 1 mm"和"差 30 cm"
            max_clearance_deficit = max(
                max_clearance_deficit, float(best.clearance_deficit)
            )
            max_physical_deficit = max(
                max_physical_deficit, float(best.physical_deficit)
            )
            max_a_y_deficit = max(max_a_y_deficit, float(best.a_y_deficit))
            gap_excess = max(0.0, float(best.gap_required) - float(best.min_gap_dynamic))
            max_gap_deficit = max(max_gap_deficit, gap_excess)

            a_linear = (best.v - v_prev) / self.params.dt_s
            a_angular = (best.w - w_prev) / self.params.dt_s
            # **加速度约束必须显式检查并记录**（本轮验收补充 3b）。
            # 之前只靠动态窗口边界隐含保证；一旦边界被别的项（终点减速带）
            # 反向覆盖，就会出现 1.0 -> 0.47 m/s（1.43 m/s² > a_max = 0.8）
            # 却报"违反率 0.000"的自相矛盾。现在把它变成显式判据。
            accel_tolerance = 1e-6 * max(1.0, self.params.a_max)
            accel_deficit = max(
                0.0, abs(a_linear) - self.params.a_max - accel_tolerance
            )
            angular_tolerance = 1e-6 * max(1.0, self.params.alpha_max)
            angular_deficit = max(
                0.0, abs(a_angular) - self.params.alpha_max - angular_tolerance
            )
            max_accel_deficit = max(max_accel_deficit, accel_deficit)
            max_angular_deficit = max(max_angular_deficit, angular_deficit)
            if accel_deficit > EPS or angular_deficit > EPS:
                # 加速度超限与策略违反同属"可降级但必须记录"的一类，
                # 计入 flags_policy（不影响 bool 主口径的语义一致性）。
                flags_policy[-1] = True
                flags[-1] = True
            jerk = 0.0
            if len(self.commands) >= 2:
                jerk = abs(
                    (best.v - 2.0 * self.commands[-1].v + self.commands[-2].v)
                    / (self.params.dt_s ** 2)
                )
            command = Command(
                t=t,
                xy=current.copy(),
                yaw=yaw,
                v=best.v,
                w=best.w,
                a_y=best.a_y_max,
                a_linear=a_linear,
                a_angular=a_angular,
                jerk=jerk,
            )
            self.commands.append(command)

            # 推进一个周期（中点积分，与候选展开一致）
            mid = yaw + 0.5 * best.w * self.params.dt_s
            current = np.array(
                [
                    current[0] + best.v * math.cos(mid) * self.params.dt_s,
                    current[1] + best.v * math.sin(mid) * self.params.dt_s,
                ]
            )
            yaw = _wrap(yaw + best.w * self.params.dt_s)
            t += self.params.dt_s
        else:
            reason = "timeout"

        if not self.commands:
            reason = "no_admissable_candidate" if reason == "timeout" else reason

        # **碰撞即任务失败**（阻断 1）：物理约束违反不可被记为"到达"。
        #
        # **超时也不算成功**（2026-10-31 口径修正）：
        #     success = 到达终点 ∧ 无物理违反
        # 旧口径把 ``timeout`` 也算成功（语义是"有指令序列且没穿墙"），于是
        # hard 档打印出 "task_ok = 8/8"，掩盖了"1 个任务超时未完成"这一事实。
        # 成功率必须等于**到达率**，不能用"没撞墙"冒充"完成任务"。
        collision_failed = physical_collision_frames > 0
        success = bool(self.commands) and bool(reached) and not collision_failed
        if collision_failed and reason == "reached_goal":
            reason = "reached_goal_with_collision"

        return DwaResult(
            success=bool(success),
            reason=str(reason),
            commands=tuple(self.commands),
            exit_xy=current,
            reached_goal=bool(reached) and not collision_failed,
            collision_frames=int(physical_collision_frames),
            min_clearance_m=0.0 if not self.commands else float(min_clearance),
            min_gap_dynamic_m=0.0 if not self.commands else float(min_gap),
            periods=len(self.commands),
            violation_periods=int(sum(flags)),
            violation_period_flags=tuple(flags),
            fallback_periods=int(fallback_used),
            physical_violation_periods=int(sum(flags_physical)),
            policy_violation_periods=int(sum(flags_policy)),
            dynamic_violation_periods=int(sum(flags_dynamic)),
            max_clearance_deficit_m=float(max_clearance_deficit),
            max_physical_deficit_m=float(max_physical_deficit),
            max_a_y_deficit=float(max_a_y_deficit),
            max_gap_deficit_m=float(max_gap_deficit),
            max_accel_deficit=float(max_accel_deficit),
            max_angular_deficit=float(max_angular_deficit),
        )
