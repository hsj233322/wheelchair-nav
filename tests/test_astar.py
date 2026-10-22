import math

import numpy as np
import pytest

from core.astar import AStarParams, clearance_field, find_path
from core.energy import EnergyParams

ENERGY = EnergyParams(
    mass=100.0,
    g=9.81,
    eta=1.0,
    p_aux=0.0,
    battery_voltage=24.0,
    battery_resistance=0.0,
)


def params(**overrides: object) -> AStarParams:
    base: dict[str, object] = {
        "e_budget": 1e9,
        "d_min": 0.0,
        "kappa_max": 3.0,
        "kappa_window_m": 0.6,
    }
    base.update(overrides)
    return AStarParams(**base)  # type: ignore[arg-type]


def grid_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[char == "#" for char in row] for row in rows], dtype=bool)


def test_empty_map_takes_straight_line() -> None:
    occupancy = np.zeros((20, 20), dtype=bool)
    result = find_path(
        occupancy,
        0.5,
        (0.25, 0.25),
        (9.75, 9.75),
        params(),
        energy_params=ENERGY,
        speed=1.0,
    )
    assert result.success
    assert result.reason == "ok"
    assert result.length_m == pytest.approx(math.hypot(9.5, 9.5), rel=1e-12)
    assert result.total_energy_j > 0.0
    assert len(result.xy) == 20


def test_path_goes_through_gap_not_through_wall() -> None:
    rows = []
    for row in range(21):
        cells = ["."] * 21
        if not (9 <= row <= 11):
            cells[10] = "#"
        rows.append("".join(cells))
    occupancy = grid_from_rows(rows)
    result = find_path(
        occupancy,
        0.5,
        (0.25, 0.25),
        (10.25, 0.25),
        params(),
        energy_params=ENERGY,
        speed=1.0,
    )
    assert result.success
    for row, col in result.cells:
        assert not occupancy[row, col]
    assert any(row in (9, 10, 11) and col == 10 for row, col in result.cells)


def test_wall_across_map_is_unsolvable() -> None:
    occupancy = np.zeros((21, 21), dtype=bool)
    occupancy[:, 10] = True
    result = find_path(
        occupancy,
        0.5,
        (0.25, 0.25),
        (10.25, 0.25),
        params(),
        energy_params=ENERGY,
        speed=1.0,
    )
    assert not result.success
    assert result.reason == "no_path"
    assert result.expanded > 0
    assert result.xy.shape == (0, 2)
    assert result.length_m == 0.0


def test_energy_budget_is_hard_constraint() -> None:
    occupancy = np.zeros((10, 10), dtype=bool)
    args = dict(
        occupancy=occupancy,
        resolution=0.5,
        start_xy=(0.25, 0.25),
        goal_xy=(4.75, 4.75),
        energy_params=ENERGY,
        speed=1.0,
    )
    starved = find_path(**args, params=params(e_budget=0.0))  # type: ignore[arg-type]
    assert not starved.success
    assert starved.reason == "no_path"

    funded = find_path(**args, params=params(e_budget=1000.0))  # type: ignore[arg-type]
    assert funded.success
    assert funded.total_energy_j <= 1000.0 + 1e-9
    assert np.all(np.diff(funded.energy_j) >= -1e-9)


def test_clearance_constraint_blocks_narrow_gap() -> None:
    occupancy = np.zeros((15, 21), dtype=bool)
    occupancy[7, :] = True
    occupancy[7, 9:12] = False
    args = dict(
        occupancy=occupancy,
        resolution=0.5,
        start_xy=(1.25, 1.25),
        goal_xy=(9.25, 6.25),
        energy_params=ENERGY,
        speed=1.0,
    )
    too_wide = find_path(**args, params=params(d_min=1.0))  # type: ignore[arg-type]
    assert not too_wide.success
    assert too_wide.reason == "no_path"

    fitting = find_path(**args, params=params(d_min=0.6))  # type: ignore[arg-type]
    assert fitting.success
    assert fitting.clearance.min() >= 0.6 - 1e-9


def test_start_and_goal_failure_reasons() -> None:
    occupancy = np.zeros((10, 10), dtype=bool)
    occupancy[0, 0] = True
    blocked_start = find_path(
        occupancy,
        0.5,
        (0.25, 0.25),
        (4.75, 4.75),
        params(),
        energy_params=ENERGY,
        speed=1.0,
    )
    assert not blocked_start.success
    assert blocked_start.reason == "start_blocked"

    occupancy = np.zeros((10, 10), dtype=bool)
    occupancy[9, 9] = True
    blocked_goal = find_path(
        occupancy,
        0.5,
        (0.25, 0.25),
        (4.75, 4.75),
        params(),
        energy_params=ENERGY,
        speed=1.0,
    )
    assert not blocked_goal.success
    assert blocked_goal.reason == "goal_blocked"


def test_start_violating_clearance_is_reported() -> None:
    occupancy = np.zeros((10, 10), dtype=bool)
    result = find_path(
        occupancy,
        0.5,
        (0.25, 0.25),
        (4.75, 4.75),
        params(d_min=3.0),
        energy_params=ENERGY,
        speed=1.0,
    )
    assert not result.success
    assert result.reason == "start_violates_clearance"


def test_start_equals_goal_returns_single_point() -> None:
    occupancy = np.zeros((5, 5), dtype=bool)
    result = find_path(
        occupancy,
        0.5,
        (1.25, 1.25),
        (1.25, 1.25),
        params(),
        energy_params=ENERGY,
        speed=1.0,
    )
    assert result.success
    assert result.length_m == 0.0
    assert result.total_energy_j == 0.0
    assert result.xy.shape == (1, 2)


def test_same_inputs_give_identical_results() -> None:
    occupancy = np.zeros((12, 12), dtype=bool)
    occupancy[3:9, 6] = True
    kwargs = dict(
        occupancy=occupancy,
        resolution=0.5,
        start_xy=(0.25, 0.25),
        goal_xy=(5.75, 5.75),
        params=params(),
        energy_params=ENERGY,
        speed=1.0,
    )
    first = find_path(**kwargs)
    second = find_path(**kwargs)
    assert np.array_equal(first.cells, second.cells)
    assert np.array_equal(first.energy_j, second.energy_j)
    assert first.expanded == second.expanded


def test_curvature_limit_blocks_impossible_turn() -> None:
    occupancy = np.zeros((20, 20), dtype=bool)
    tight = params(kappa_max=0.2)
    straight = find_path(
        occupancy,
        0.5,
        (0.25, 0.25),
        (5.25, 0.25),
        tight,
        energy_params=ENERGY,
        speed=1.0,
    )
    assert straight.success

    needs_turn = find_path(
        occupancy,
        0.5,
        (0.25, 0.25),
        (2.75, 5.25),
        tight,
        energy_params=ENERGY,
        speed=1.0,
    )
    assert not needs_turn.success
    assert needs_turn.reason == "no_path"


def test_pareto_front_trades_length_for_energy() -> None:
    occupancy = np.zeros((20, 20), dtype=bool)
    c_rr = np.full((20, 20), 0.01)
    c_rr[6:14, 6:14] = 0.25
    result = find_path(
        occupancy,
        0.5,
        (0.25, 0.25),
        (9.75, 9.75),
        params(collect_pareto=True),
        energy_params=ENERGY,
        speed=1.0,
        c_rr=c_rr,
    )
    assert result.success
    assert result.pareto_length_m.size >= 2
    assert np.all(np.diff(result.pareto_length_m) >= 0)
    assert np.all(np.diff(result.pareto_energy_j) <= 0)
    shortest = result.pareto_length_m[0]
    assert result.length_m == pytest.approx(shortest)
    assert result.total_energy_j == pytest.approx(result.pareto_energy_j[0])


def test_clearance_field_matches_wall_distance() -> None:
    occupancy = np.zeros((5, 5), dtype=bool)
    occupancy[2, 2] = True
    clear = clearance_field(occupancy, 1.0)
    assert clear[2, 2] == 0.0
    assert clear[2, 1] == pytest.approx(0.5)
    assert clear[1, 1] == pytest.approx(math.hypot(1.0, 1.0) - 0.5)
    # 地图边界视为障碍
    assert clear[0, 0] == pytest.approx(0.5)


def test_invalid_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        find_path(
            np.zeros((0, 0), dtype=bool),
            0.5,
            (0.0, 0.0),
            (1.0, 1.0),
            params(),
            energy_params=ENERGY,
            speed=1.0,
        )
    with pytest.raises(ValueError):
        find_path(
            np.zeros((5, 5), dtype=bool),
            0.5,
            (0.0, 0.0),
            (1.0, 1.0),
            params(),
            energy_params=ENERGY,
            speed=0.0,
        )
