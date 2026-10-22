"""轨迹指标计算（纯函数）。

指标口径（交接文档 §4.6）：

* 路径长度 / 能耗 / 完成任务时间：描述量；
* **平顺性用 jerk RMS + 平均绝对角速度**；
* ``总转弯角`` 只作辅助指标 —— 它越大不代表越差（旧论文正是栽在这个反向上），
  不要用它单独论证“更平顺”；
* 最小离障距离、碰撞帧数、约束违反率、成功率：安全与约束证据。

所有函数接受 numpy 数组，不读配置、不写文件。空轨迹（规划失败）由
``summarize`` 统一返回全 0，成功与否由 runner 的 success 列单独记录。
"""

from __future__ import annotations

import math

import numpy as np

METRIC_NAMES: tuple[str, ...] = (
    "path_length_m",
    "energy_j",
    "completion_time_s",
    "total_turn_deg",
    "mean_abs_yaw_rate_dps",
    "jerk_rms_mps3",
    "min_clearance_m",
    "collision_frames",
    "constraint_violation_rate",
)


def _as_array(values: object, name: str, *, ndim: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} 必须是 {ndim} 维数组，实际形状为 {array.shape}")
    return array


def _as_time(t: object) -> np.ndarray:
    time = _as_array(t, "t")
    if time.size > 1 and np.any(np.diff(time) <= 0):
        raise ValueError("t 必须严格递增")
    return time


def path_length(xy: object) -> float:
    """折线路径长度 [m]。"""
    points = _as_array(xy, "xy", ndim=2)
    if points.shape[1] != 2:
        raise ValueError(f"xy 必须是 (N, 2) 数组，实际形状为 {points.shape}")
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def completion_time(t: object) -> float:
    """完成任务时间 [s] = t[-1] - t[0]。"""
    time = _as_time(t)
    if time.size < 2:
        return 0.0
    return float(time[-1] - time[0])


def total_turn_angle(yaw: object) -> float:
    """总转弯角 [rad]（辅助指标，见模块 docstring）。"""
    heading = _as_array(yaw, "yaw")
    if heading.size < 2:
        return 0.0
    turns = np.mod(np.diff(heading) + math.pi, 2.0 * math.pi) - math.pi
    return float(np.sum(np.abs(turns)))


def mean_abs_yaw_rate(t: object, yaw: object) -> float:
    """平均绝对角速度 [rad/s]（平顺性主指标之一）。"""
    time = _as_time(t)
    heading = _as_array(yaw, "yaw")
    if time.size < 2:
        return 0.0
    if time.size != heading.size:
        raise ValueError("t 与 yaw 长度不一致")
    turns = np.mod(np.diff(heading) + math.pi, 2.0 * math.pi) - math.pi
    return float(np.mean(np.abs(turns / np.diff(time))))


def jerk_rms(t: object, speed: object) -> float:
    """jerk 的均方根 [m/s^3]（平顺性主指标之一）。

    对速度做两次数值微分（``numpy.gradient`` 支持非均匀采样）。
    少于 4 个采样点时返回 0.0：此时 jerk 没有定义，不应产生 NaN 污染 CSV。
    """
    time = _as_time(t)
    velocity = _as_array(speed, "speed")
    if time.size != velocity.size:
        raise ValueError("t 与 speed 长度不一致")
    if time.size < 4:
        return 0.0
    # edge_order=2：端点也用二阶差分，避免把截断误差算成 jerk。
    accel = np.gradient(velocity, time, edge_order=2)
    jerk = np.gradient(accel, time, edge_order=2)
    return float(np.sqrt(np.mean(jerk * jerk)))


def min_clearance(clearance: object) -> float:
    """最小离障距离 [m]；空轨迹返回 0.0。"""
    values = _as_array(clearance, "clearance")
    return 0.0 if values.size == 0 else float(np.min(values))


def collision_frames(collision: object) -> int:
    """碰撞帧数（车身中心到障碍物表面距离 < body_radius 的采样点数）。"""
    flags = np.asarray(collision, dtype=bool)
    if flags.ndim != 1:
        raise ValueError(f"collision 必须是一维数组，实际形状为 {flags.shape}")
    return int(np.count_nonzero(flags))


def constraint_violation_rate(violations: object) -> float:
    """约束违反率：违反任一硬约束（离障、能耗预算、碰撞）的采样点占比。"""
    flags = np.asarray(violations, dtype=bool)
    if flags.ndim != 1:
        raise ValueError(f"violations 必须是一维数组，实际形状为 {flags.shape}")
    if flags.size == 0:
        return 0.0
    return float(np.mean(flags))


def summarize(
    *,
    t: object,
    xy: object,
    yaw: object,
    speed: object,
    energy_j: object,
    clearance: object,
    collision: object,
    violations: object,
) -> dict[str, float | int]:
    """把一条轨迹压成一行指标；空轨迹返回全 0。"""
    time = _as_time(t)
    points = _as_array(xy, "xy", ndim=2)
    heading = _as_array(yaw, "yaw")
    velocity = _as_array(speed, "speed")
    energy_values = _as_array(energy_j, "energy_j")
    clearance_values = _as_array(clearance, "clearance")
    collision_flags = np.asarray(collision, dtype=bool)
    violation_flags = np.asarray(violations, dtype=bool)

    if time.size == 0:
        return {name: 0.0 for name in METRIC_NAMES}

    lengths = {
        "t": time.size,
        "xy": points.shape[0],
        "yaw": heading.size,
        "speed": velocity.size,
        "energy_j": energy_values.size,
        "clearance": clearance_values.size,
        "collision": collision_flags.size,
        "violations": violation_flags.size,
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"轨迹各数组长度不一致: {lengths}")
    if points.shape[1] != 2:
        raise ValueError(f"xy 必须是 (N, 2) 数组，实际形状为 {points.shape}")

    return {
        "path_length_m": path_length(points),
        "energy_j": float(energy_values[-1]),
        "completion_time_s": completion_time(time),
        "total_turn_deg": math.degrees(total_turn_angle(heading)),
        "mean_abs_yaw_rate_dps": math.degrees(mean_abs_yaw_rate(time, heading)),
        "jerk_rms_mps3": jerk_rms(time, velocity),
        "min_clearance_m": min_clearance(clearance_values),
        "collision_frames": collision_frames(collision_flags),
        "constraint_violation_rate": constraint_violation_rate(violation_flags),
    }
