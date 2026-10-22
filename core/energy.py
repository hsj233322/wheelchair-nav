"""智能轮椅能耗与电池模型（纯函数，无全局状态，不 import config）。

模型（第 2 章）：

    F_roll = m g C_rr
    F_grade = m g sin(slope)
    F_inertia = m a
    P_mech = (F_roll + F_grade + F_inertia) * v
    P_elec = P_mech / eta + P_aux + I^2 R_batt,   I = P_elec^bus / V_batt

符号与坐标约定：

* ``speed`` 是车速大小，非负 [m/s]；
* ``slope_rad`` 是地面沿 +x 方向的有符号俯仰角 [rad]，正值表示向 +x 上坡；
* ``direction_x`` 是单位运动方向的 x 分量（cos(yaw)），因此坡度功率为
  ``m*g*sin(slope)*speed*direction_x``；
* 能量一律指从电池取出的电能 [J]；下坡/制动**不回收**（保守假设，暂不建模再生制动）。

单位：m, s, rad, kg, W, J, V, Ohm。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

J_PER_WH = 3600.0


@dataclass(frozen=True)
class EnergyParams:
    """由 platform.yaml 构造的显式参数集合；core 不读配置文件。"""

    mass: float
    g: float = 9.81
    eta: float = 0.85
    p_aux: float = 0.0
    battery_voltage: float = 24.0
    battery_resistance: float = 0.08

    def __post_init__(self) -> None:
        if not self.mass > 0:
            raise ValueError(f"mass 必须为正，实际为 {self.mass}")
        if not self.g > 0:
            raise ValueError(f"g 必须为正，实际为 {self.g}")
        if not 0 < self.eta <= 1:
            raise ValueError(f"eta 必须在 (0, 1] 内，实际为 {self.eta}")
        if self.p_aux < 0:
            raise ValueError(f"p_aux 必须 >= 0，实际为 {self.p_aux}")
        if not self.battery_voltage > 0:
            raise ValueError(f"battery_voltage 必须为正，实际为 {self.battery_voltage}")
        if self.battery_resistance < 0:
            raise ValueError(
                f"battery_resistance 必须 >= 0，实际为 {self.battery_resistance}"
            )


def _check_speed(speed: float) -> float:
    speed = float(speed)
    if not math.isfinite(speed) or speed < 0:
        raise ValueError(f"speed 必须是非负有限数，实际为 {speed}")
    return speed


def rolling_power(mass: float, g: float, c_rr: float, speed: float) -> float:
    """滚动阻力功率 [W]。"""
    if mass <= 0 or g <= 0:
        raise ValueError("mass 与 g 必须为正")
    if c_rr < 0:
        raise ValueError(f"c_rr 必须 >= 0，实际为 {c_rr}")
    return mass * g * c_rr * _check_speed(speed)


def grade_power(
    mass: float, g: float, slope_rad: float, speed: float, direction_x: float = 1.0
) -> float:
    """坡度阻力功率 [W]；下坡时可为负。"""
    if mass <= 0 or g <= 0:
        raise ValueError("mass 与 g 必须为正")
    return mass * g * math.sin(float(slope_rad)) * _check_speed(speed) * float(direction_x)


def acceleration_power(mass: float, accel: float, speed: float) -> float:
    """惯性功率 [W]。"""
    if mass <= 0:
        raise ValueError("mass 必须为正")
    return mass * float(accel) * _check_speed(speed)


def mechanical_power(
    mass: float,
    g: float,
    c_rr: float,
    slope_rad: float,
    speed: float,
    *,
    accel: float = 0.0,
    direction_x: float = 1.0,
) -> float:
    """轮端机械功率 [W]：滚阻 + 坡度 + 惯性。"""
    return (
        rolling_power(mass, g, c_rr, speed)
        + grade_power(mass, g, slope_rad, speed, direction_x)
        + acceleration_power(mass, accel, speed)
    )


def motor_power(mech_power: float, eta: float) -> float:
    """电机从电池取用的功率（不含辅助负载与电池内阻损耗）[W]。

    驱动时除以传动效率；制动/下坡时按耗散处理（不回收）。
    """
    if not 0 < eta <= 1:
        raise ValueError(f"eta 必须在 (0, 1] 内，实际为 {eta}")
    if mech_power >= 0:
        return mech_power / eta
    return 0.0


def electrical_power(mech_power: float, params: EnergyParams) -> float:
    """电池端总功率 [W]，含辅助负载与内阻损耗。"""
    bus_power = motor_power(mech_power, params.eta) + params.p_aux
    if bus_power <= 0:
        return bus_power
    current = bus_power / params.battery_voltage
    return bus_power + current * current * params.battery_resistance


def battery_current(power_w: float, params: EnergyParams) -> float:
    """电池电流 [A]；power_w 为电池端功率。"""
    return float(power_w) / params.battery_voltage


def step_energy(
    distance: float,
    speed: float,
    *,
    accel: float = 0.0,
    c_rr: float,
    slope_rad: float = 0.0,
    direction_x: float = 1.0,
    params: EnergyParams,
) -> float:
    """以恒定 speed 走完 distance 的电能 [J]。

    ``distance / speed`` 为这一步的时长；speed 必须 > 0。全局规划里用
    accel=0 的稳态估算作为能量预算的记账口径。
    """
    distance = float(distance)
    speed = _check_speed(speed)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError(f"distance 必须是非负有限数，实际为 {distance}")
    if distance == 0:
        return 0.0
    if speed <= 0:
        raise ValueError("distance > 0 时 speed 必须 > 0")
    mech = mechanical_power(
        params.mass,
        params.g,
        c_rr,
        slope_rad,
        speed,
        accel=accel,
        direction_x=direction_x,
    )
    return electrical_power(mech, params) * (distance / speed)


def auxiliary_energy(duration: float, params: EnergyParams) -> float:
    """原地不动（如原地转向）duration 秒的耗电 [J]。"""
    duration = float(duration)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(f"duration 必须是非负有限数，实际为 {duration}")
    return electrical_power(0.0, params) * duration


def energy_wh(joules: float) -> float:
    """J -> Wh。"""
    return float(joules) / J_PER_WH


def energy_from_wh(wh: float) -> float:
    """Wh -> J。"""
    return float(wh) * J_PER_WH
