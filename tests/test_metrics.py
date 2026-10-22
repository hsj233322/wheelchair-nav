import math

import numpy as np
import pytest

from core import metrics


def test_path_length_of_straight_line() -> None:
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    assert metrics.path_length(xy) == pytest.approx(7.0)


def test_path_length_of_single_point_is_zero() -> None:
    assert metrics.path_length(np.array([[1.0, 2.0]])) == 0.0


def test_completion_time() -> None:
    assert metrics.completion_time(np.array([0.0, 1.5, 4.0])) == pytest.approx(4.0)
    assert metrics.completion_time(np.array([2.0])) == 0.0


def test_total_turn_angle_of_right_angle() -> None:
    yaw = np.array([0.0, 0.0, math.pi / 2, math.pi / 2])
    assert metrics.total_turn_angle(yaw) == pytest.approx(math.pi / 2)


def test_total_turn_angle_wraps_correctly() -> None:
    yaw = np.array([math.radians(170.0), math.radians(-170.0)])
    assert metrics.total_turn_angle(yaw) == pytest.approx(math.radians(20.0))


def test_mean_abs_yaw_rate() -> None:
    t = np.array([0.0, 1.0, 2.0])
    yaw = np.array([0.0, 0.5, 0.5])
    # |0.5 / 1| 与 |0 / 1| 的平均
    assert metrics.mean_abs_yaw_rate(t, yaw) == pytest.approx(0.25)


def test_jerk_rms_of_constant_speed_is_zero() -> None:
    t = np.linspace(0.0, 2.0, 41)
    speed = np.full_like(t, 0.8)
    assert metrics.jerk_rms(t, speed) == pytest.approx(0.0, abs=1e-12)


def test_jerk_rms_of_quadratic_speed_equals_constant_jerk() -> None:
    t = np.linspace(0.0, 2.0, 101)
    speed = 0.5 * t**2  # a = t, jerk = 1
    assert metrics.jerk_rms(t, speed) == pytest.approx(1.0, rel=1e-6)


def test_jerk_rms_needs_four_samples() -> None:
    t = np.array([0.0, 0.1, 0.2])
    speed = np.array([0.0, 1.0, 2.0])
    assert metrics.jerk_rms(t, speed) == 0.0


def test_clearance_and_collision_counts() -> None:
    clearance = np.array([0.9, 0.5, 0.2])
    collision = clearance < 0.58
    assert metrics.min_clearance(clearance) == pytest.approx(0.2)
    assert metrics.collision_frames(collision) == 2
    assert metrics.constraint_violation_rate(np.array([False, True, True])) == pytest.approx(
        2.0 / 3.0
    )


def test_summarize_reports_all_metric_names() -> None:
    t = np.linspace(0.0, 2.0, 21)
    xy = np.column_stack([np.linspace(0.0, 1.6, 21), np.zeros_like(t)])
    yaw = np.zeros_like(t)
    speed = np.full_like(t, 0.8)
    energy_j = np.linspace(0.0, 100.0, 21)
    clearance = np.full_like(t, 0.9)
    collision = np.zeros_like(t, dtype=bool)
    violations = np.zeros_like(t, dtype=bool)

    summary = metrics.summarize(
        t=t,
        xy=xy,
        yaw=yaw,
        speed=speed,
        energy_j=energy_j,
        clearance=clearance,
        collision=collision,
        violations=violations,
    )
    assert tuple(summary) == metrics.METRIC_NAMES
    assert summary["path_length_m"] == pytest.approx(1.6)
    assert summary["energy_j"] == pytest.approx(100.0)
    assert summary["completion_time_s"] == pytest.approx(2.0)
    assert summary["total_turn_deg"] == pytest.approx(0.0)
    assert summary["mean_abs_yaw_rate_dps"] == pytest.approx(0.0)
    assert summary["jerk_rms_mps3"] == pytest.approx(0.0, abs=1e-12)
    assert summary["min_clearance_m"] == pytest.approx(0.9)
    assert summary["collision_frames"] == 0
    assert summary["constraint_violation_rate"] == pytest.approx(0.0)


def test_summarize_empty_trajectory_returns_zeros() -> None:
    summary = metrics.summarize(
        t=np.zeros(0),
        xy=np.zeros((0, 2)),
        yaw=np.zeros(0),
        speed=np.zeros(0),
        energy_j=np.zeros(0),
        clearance=np.zeros(0),
        collision=np.zeros(0, dtype=bool),
        violations=np.zeros(0, dtype=bool),
    )
    assert tuple(summary) == metrics.METRIC_NAMES
    assert all(value == 0 for value in summary.values())


def test_summarize_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        metrics.summarize(
            t=np.zeros(3),
            xy=np.zeros((2, 2)),
            yaw=np.zeros(2),
            speed=np.zeros(2),
            energy_j=np.zeros(2),
            clearance=np.zeros(2),
            collision=np.zeros(2, dtype=bool),
            violations=np.zeros(2, dtype=bool),
        )


def test_non_monotonic_time_rejected() -> None:
    with pytest.raises(ValueError):
        metrics.mean_abs_yaw_rate(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
