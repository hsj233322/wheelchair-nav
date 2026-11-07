"""局部规划器的统一接口（协议层，不含任何具体实现）。

为什么单独一个文件
------------------
本轮实现的是**约束化 DWA**（``core/dwa.py``）。下一轮要接**传统 DWA** 与
**MPPI** 做基线对比（交接文档 7.9 第 9 条）。三者必须共用同一套输入/输出契约，
否则"同样的任务、同样的参考轨迹、同样的动态障碍"这个前提就不成立，对比没有意义。

本模块只定义：

* ``LocalPlanner`` —— 协议（``typing.Protocol``），任何基线实现都遵守它；
* ``PlannerInput`` —— 一次局部规划的全部输入，**只读**；
* ``PlannerOutput`` —— 统一输出，含指标所需的全部量。

设计约束
--------
1. **只接受 trackable=True 的参考轨迹**。协议不重复校验，但每个实现必须在入口
   调用 ``core.dwa.reference_from_smooth``（或等价校验），这是 7.9 的前置红线。
   本模块的 ``assert_trackable`` 提供一处共享的断言入口，避免各实现各写一遍。
2. **约束违反率必须同时给出按周期与按任务两种口径**（``PlannerOutput``）。
3. 不得引入第三方依赖：基线实现只允许 numpy / scipy。

下一轮接入方法（示例）
----------------------
```python
class TraditionalDwa:            # 传统 DWA：全部代价项，无硬约束剔除
    name = "traditional_dwa"
    def plan(self, request: PlannerInput) -> PlannerOutput: ...

class MppiPlanner:               # MPPI：采样 + 指数加权平均
    name = "mppi"
    def plan(self, request: PlannerInput) -> PlannerOutput: ...
```
两者只需实现 ``plan(request) -> PlannerOutput``，即可直接进入本框架的评测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from core.dwa import (
    DwaParams,
    DwaResult,
    NotTrackableError,
    ObstacleTrack,
    Reference,
)


# ---------------------------------------------------------------------------
# 输入 / 输出契约
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerInput:
    """一次局部规划任务的**只读**输入。

    所有字段都与具体算法无关，基线实现可以原样消费。
    """

    reference: Reference
    """可跟踪的参考轨迹（**已校验** trackable=True）。"""

    clearance_field: np.ndarray
    """静态距离场 [m]（``core.astar.clearance_field`` 的输出）。"""

    resolution: float
    origin: tuple[float, float]
    start_xy: np.ndarray | None = None
    start_yaw: float | None = None
    goal_xy: tuple[float, float] | None = None

    obstacles: tuple[ObstacleTrack, ...] = ()
    """逐帧动态障碍。确定性手工轨迹或 ETH/UCY 读出的显式轨迹，接口一致。"""

    params: DwaParams | None = None
    """算法参数集合。基线实现可定义自己的参数类型并忽略本字段。"""


@dataclass(frozen=True)
class PlannerOutput:
    """统一输出：**指标计算所需的全部量**，与具体算法无关。"""

    planner: str
    """实现名，用于对比表分组（如 ``constrained_dwa`` / ``traditional_dwa``）。"""

    success: bool
    reason: str

    t: np.ndarray
    """逐周期时间 [s]。"""

    xy: np.ndarray
    """(N, 2) 执行轨迹（逐周期）。"""

    yaw: np.ndarray
    v: np.ndarray
    w: np.ndarray

    reached_goal: bool

    collision_frames: int
    min_clearance_m: float
    min_gap_dynamic_m: float

    violation_periods: int
    periods: int
    violation_period_flags: tuple[bool, ...]

    energy_j: float = 0.0
    """能耗 [J]，由调用方按实际几何与平台参数复算后写回。"""

    @property
    def constraint_violation_rate_per_period(self) -> float:
        return 0.0 if self.periods == 0 else self.violation_periods / self.periods

    @property
    def constraint_violation_rate_per_task(self) -> float:
        return 0.0 if not self.violation_period_flags else float(
            any(self.violation_period_flags)
        )


@runtime_checkable
class LocalPlanner(Protocol):
    """局部规划器协议。任何基线实现满足本协议即可进入评测。"""

    name: str

    def plan(self, request: PlannerInput) -> PlannerOutput:
        """执行一次任务级局部规划。"""
        ...


# ---------------------------------------------------------------------------
# 共享的前置校验
# ---------------------------------------------------------------------------


def assert_trackable(reference: Reference) -> None:
    """7.9 前置红线：不可跟踪的参考轨迹不得进入局部规划。

    各实现应在 ``plan`` 入口调用本函数，保证同一条红线只有一处定义。
    """
    curvature = np.asarray(reference.curvature, dtype=float)
    if not np.all(np.isfinite(curvature)):
        raise NotTrackableError(
            "参考轨迹曲率含无界值（回退折线），不可跟踪，不得进入局部规划实验"
        )


def output_from_dwa(
    result: DwaResult, *, planner: str = "constrained_dwa"
) -> PlannerOutput:
    """把本轮 ``DwaResult`` 转成统一输出，作为其他实现的格式参照。"""
    commands = result.commands
    count = len(commands)
    return PlannerOutput(
        planner=planner,
        success=bool(result.success),
        reason=str(result.reason),
        t=np.array([c.t for c in commands], dtype=float) if count else np.zeros(0),
        xy=(
            np.array([c.xy for c in commands], dtype=float)
            if count
            else np.zeros((0, 2))
        ),
        yaw=np.array([c.yaw for c in commands], dtype=float) if count else np.zeros(0),
        v=np.array([c.v for c in commands], dtype=float) if count else np.zeros(0),
        w=np.array([c.w for c in commands], dtype=float) if count else np.zeros(0),
        reached_goal=bool(result.reached_goal),
        collision_frames=int(result.collision_frames),
        min_clearance_m=float(result.min_clearance_m),
        min_gap_dynamic_m=float(result.min_gap_dynamic_m),
        violation_periods=int(result.violation_periods),
        periods=int(result.periods),
        violation_period_flags=tuple(result.violation_period_flags),
    )


# ---------------------------------------------------------------------------
# 注册表：下一轮基线在此登记即可
# ---------------------------------------------------------------------------

PLANNER_REGISTRY: dict[str, type] = {}


def register_planner(name: str, factory: type) -> None:
    """把基线实现登记进评测入口，供 ``analysis`` 层按名字选算法。"""
    if name in PLANNER_REGISTRY:
        raise ValueError(f"局部规划器 {name!r} 已注册，不允许重名覆盖")
    PLANNER_REGISTRY[name] = factory


def available_planners() -> tuple[str, ...]:
    return tuple(sorted(PLANNER_REGISTRY))
