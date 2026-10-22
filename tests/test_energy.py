import math

import pytest

from core import energy
from core.energy import EnergyParams

BASE = EnergyParams(
    mass=100.0,
    g=9.81,
    eta=0.85,
    p_aux=15.0,
    battery_voltage=24.0,
    battery_resistance=0.08,
)
NO_AUX = EnergyParams(
    mass=100.0,
    g=9.81,
    eta=1.0,
    p_aux=0.0,
    battery_voltage=24.0,
    battery_resistance=0.0,
)


def step(
    params: EnergyParams = BASE,
    *,
    distance: float = 1.0,
    speed: float = 1.0,
    c_rr: float = 0.015,
    slope_deg: float = 0.0,
    direction_x: float = 1.0,
) -> float:
    return energy.step_energy(
        distance,
        speed,
        c_rr=c_rr,
        slope_rad=math.radians(slope_deg),
        direction_x=direction_x,
        params=params,
    )


def test_step_energy_linear_in_distance() -> None:
    one = step(distance=1.0)
    two = step(distance=2.0)
    assert two == pytest.approx(2.0 * one, rel=1e-12)


def test_step_energy_monotonic_in_mass() -> None:
    light = step(EnergyParams(mass=80.0, eta=1.0, p_aux=0.0, battery_resistance=0.0))
    heavy = step(EnergyParams(mass=130.0, eta=1.0, p_aux=0.0, battery_resistance=0.0))
    assert light < heavy


def test_step_energy_monotonic_in_rolling_resistance() -> None:
    smooth = step(c_rr=0.010)
    rough = step(c_rr=0.030)
    assert smooth < rough


def test_step_energy_monotonic_in_auxiliary_power() -> None:
    without = step(EnergyParams(mass=100.0, p_aux=0.0))
    with_aux = step(EnergyParams(mass=100.0, p_aux=15.0))
    assert without < with_aux


def test_lower_efficiency_costs_more() -> None:
    efficient = step(EnergyParams(mass=100.0, eta=0.9, p_aux=0.0, battery_resistance=0.0))
    lossy = step(EnergyParams(mass=100.0, eta=0.7, p_aux=0.0, battery_resistance=0.0))
    assert lossy > efficient


def test_battery_resistance_adds_loss() -> None:
    ideal = step(EnergyParams(mass=100.0, battery_resistance=0.0))
    resistive = step(EnergyParams(mass=100.0, battery_resistance=0.2))
    assert resistive > ideal


def test_uphill_flat_downhill_ordering() -> None:
    uphill = step(slope_deg=4.0, direction_x=1.0)
    flat = step(slope_deg=0.0)
    downhill = step(slope_deg=4.0, direction_x=-1.0)
    assert uphill > flat > downhill


def test_downhill_without_aux_or_regeneration_is_free() -> None:
    assert step(NO_AUX, slope_deg=-5.0, direction_x=1.0) == pytest.approx(0.0)


def test_zero_distance_is_free_even_at_zero_speed() -> None:
    assert energy.step_energy(0.0, 0.0, c_rr=0.02, params=BASE) == 0.0


def test_positive_distance_at_zero_speed_raises() -> None:
    with pytest.raises(ValueError):
        energy.step_energy(1.0, 0.0, c_rr=0.02, params=BASE)


def test_negative_speed_raises() -> None:
    with pytest.raises(ValueError):
        energy.step_energy(1.0, -1.0, c_rr=0.02, params=BASE)


def test_auxiliary_energy_scales_with_duration() -> None:
    one = energy.auxiliary_energy(1.0, BASE)
    ten = energy.auxiliary_energy(10.0, BASE)
    assert one > 0.0
    assert ten == pytest.approx(10.0 * one, rel=1e-12)


def test_wh_conversion_round_trip() -> None:
    assert energy.energy_wh(3600.0) == pytest.approx(1.0)
    assert energy.energy_from_wh(1.0) == pytest.approx(3600.0)
    assert energy.energy_wh(energy.energy_from_wh(2.5)) == pytest.approx(2.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass": 0.0},
        {"mass": 100.0, "eta": 0.0},
        {"mass": 100.0, "eta": 1.2},
        {"mass": 100.0, "p_aux": -1.0},
        {"mass": 100.0, "battery_voltage": 0.0},
        {"mass": 100.0, "battery_resistance": -0.1},
    ],
)
def test_invalid_energy_params_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EnergyParams(**kwargs)
