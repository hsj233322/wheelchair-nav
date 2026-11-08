"""``env/pedestrians.py`` 的测试：坐标往返、重采样帧数、穿墙剔除、不出界、同 seed 哈希一致。

测试分两层：

1. **合成数据**（不依赖数据集）：几何/时间/筛选规则可以用解析值核对；
2. **真实数据**（``data/eth_ucy/`` 存在时）：核对格式、帧步长、时长与可用性口径。

数据不存在时真实数据层自动跳过（``data/`` 不进 git，见 ``data/README.md``）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core import astar
from env import pedestrians as ped

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "eth_ucy"
REAL_SEQUENCES = ("eth", "hotel", "zara01", "students001")


# ---------------------------------------------------------------------------
# 锚点映射
# ---------------------------------------------------------------------------


def test_anchor_mapping_round_trip() -> None:
    """世界 <-> 局部坐标往返一致（含旋转与缩放）。"""
    mapping = ped.AnchorMapping(origin_xy=(3.5, -1.25), rotation_deg=37.0, scale=1.0)
    local = np.array([[0.0, 0.0], [1.0, 2.0], [-3.5, 4.25], [10.0, -7.0]])
    world = mapping.apply(local)
    back = mapping.inverse(world)
    assert np.allclose(back, local, atol=1e-12)
    # 旋转保距（scale = 1）
    assert np.isclose(
        np.linalg.norm(world[1] - world[0]), np.linalg.norm(local[1] - local[0])
    )


def test_anchor_mapping_scales_distances() -> None:
    mapping = ped.AnchorMapping(origin_xy=(0.0, 0.0), rotation_deg=0.0, scale=2.0)
    world = mapping.apply(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(world, [[2.0, 0.0], [0.0, 2.0]])


def test_anchor_mapping_rejects_bad_parameters() -> None:
    with pytest.raises(ped.PedestrianDataError):
        ped.AnchorMapping(origin_xy=(0.0, 0.0), scale=0.0)
    with pytest.raises(ped.PedestrianDataError):
        ped.AnchorMapping(origin_xy=(0.0, float("nan")), scale=1.0)


# ---------------------------------------------------------------------------
# 时间对齐
# ---------------------------------------------------------------------------


def test_resample_frame_count_and_spacing() -> None:
    """帧数 = floor((t_end - t_start)/dt) + 1，步长严格等于 dt。"""
    times = np.arange(5, dtype=float) * 0.4          # 0, 0.4, ..., 1.6
    xy = np.column_stack([times, times * 0.0])
    out_t, out_xy = ped.resample_track(times, xy, dt_s=0.1)
    assert out_t.shape[0] == int(np.floor(1.6 / 0.1)) + 1 == 17
    assert np.allclose(np.diff(out_t), 0.1)
    # 线性轨迹：插值后仍是等间隔直线
    assert np.allclose(out_xy[:, 0], out_t)
    assert np.allclose(out_xy[:, 1], 0.0)


def test_resample_is_densification_not_decimation() -> None:
    """0.4 s -> 0.1 s 是**加密**：样本数变多，信息不丢（端点保留）。"""
    times = np.arange(10, dtype=float) * 0.4
    xy = np.column_stack([np.sin(times), np.cos(times)])
    out_t, out_xy = ped.resample_track(times, xy, dt_s=0.1)
    assert out_t.shape[0] > times.shape[0]
    assert np.isclose(out_t[0], times[0]) and np.isclose(out_t[-1], times[-1])
    assert np.isclose(out_xy[0, 0], xy[0, 0]) and np.isclose(out_xy[-1, 1], xy[-1, 1])


def test_resample_rejects_non_increasing_times() -> None:
    with pytest.raises(ped.PedestrianDataError):
        ped.resample_track(
            np.array([0.0, 0.4, 0.4]), np.zeros((3, 2)), dt_s=0.1
        )


def test_observation_runs_split_at_holes() -> None:
    """观测空洞（间隔 > max_gap_s）必须切成段，绝不跨空洞插值。"""
    times = np.array([0.0, 0.4, 0.8, 5.0, 5.4, 5.8])
    runs = ped.observation_runs(times, max_gap_s=0.6)
    assert runs == [(0, 3), (3, 6)]
    # 阈值大于空洞时只剩一段（用来验证阈值语义）
    assert ped.observation_runs(times, max_gap_s=10.0) == [(0, 6)]


def test_smooth_velocity_matches_constant_speed() -> None:
    """匀速直线运动：平滑速度等于真实速度（窗口不引入偏差）。"""
    dt = 0.1
    times = np.arange(40, dtype=float) * dt
    xy = np.column_stack([0.7 * times, -0.3 * times])
    velocity = ped.smooth_velocity(xy, dt, window=5)
    assert np.allclose(velocity, [0.7, -0.3], atol=1e-9)


def test_smooth_velocity_suppresses_noise() -> None:
    """速度平滑必须显著降低逐周期抖动（否则 DWA 的相对运动方向会反复翻转）。"""
    rng = np.random.default_rng(0)
    dt = 0.1
    times = np.arange(200, dtype=float) * dt
    clean = np.column_stack([0.8 * times, np.zeros_like(times)])
    noisy = clean + rng.normal(scale=0.01, size=clean.shape)
    raw = ped.smooth_velocity(noisy, dt, window=1)
    smoothed = ped.smooth_velocity(noisy, dt, window=5)
    assert np.std(smoothed[:, 0]) < 0.5 * np.std(raw[:, 0])
    assert np.std(smoothed[:, 1]) < 0.5 * np.std(raw[:, 1])


# ---------------------------------------------------------------------------
# 可用性筛选（合成场景）
# ---------------------------------------------------------------------------


def _synthetic_scene(size: int = 20) -> tuple[np.ndarray, np.ndarray, float, tuple[float, float]]:
    """``size x size`` 栅格（0.1 m）：四周一圈墙，中间自由。

    默认 20x20 时自由区只有 1.8x1.8 m，``clearance`` 最大 0.9~1.0 m，
    与 ``body_radius = 0.58`` 接近，判据容易整体触发；需要"中间自由、只有局部
    贴墙"的测试请传更大的 ``size``。
    """
    occupancy = np.zeros((size, size), dtype=bool)
    occupancy[0, :] = occupancy[-1, :] = True
    occupancy[:, 0] = occupancy[:, -1] = True
    clearance = astar.clearance_field(occupancy, 0.1)
    return occupancy, clearance, 0.1, (0.0, 0.0)


def _synthetic_raw(
    xy: np.ndarray, *, frames: np.ndarray | None = None, ids: np.ndarray | None = None
) -> ped.RawSequence:
    count = xy.shape[0]
    return ped.RawSequence(
        name="synthetic",
        source=Path("<synthetic>"),
        frame=np.asarray(frames if frames is not None else np.arange(count) * 10, dtype=float),
        ids=np.asarray(ids if ids is not None else np.ones(count), dtype=float),
        xy_local=np.asarray(xy, dtype=float),
    )


def test_filter_removes_wall_crossing_and_out_of_bounds() -> None:
    """穿墙段被剔除、出界段被剔除；两者都**不**进入 ObstacleTrack。"""
    occupancy, clearance, resolution, origin = _synthetic_scene(80)
    # 一条从自由区中心一路走到栅格外（出界）的轨迹
    xs = np.linspace(4.0, -0.5, 60)
    xy = np.column_stack([xs, np.full(60, 4.0)])
    raw = _synthetic_raw(xy)
    tracks = ped.build_sequence_tracks(
        raw,
        scene_occupancy=occupancy,
        clearance_field=clearance,
        resolution=resolution,
        origin=origin,
        anchor=ped.AnchorMapping(origin_xy=(0.0, 0.0)),
        dt_s=0.1,
        body_radius=0.58,
        min_samples=5,
    )
    report = tracks.report
    assert report.samples_out_of_bounds > 0, "越界样本必须被标记"
    assert report.out_of_bounds_ratio > 0.0
    assert report.sample_usable_ratio <= 1.0
    # 所有保留下来的样本都必须在栅格内且不穿墙
    for track in tracks.tracks:
        values = clearance[
            np.floor(track.xy[:, 1] / resolution).astype(int),
            np.floor(track.xy[:, 0] / resolution).astype(int),
        ]
        assert np.all(values >= 0.58 - 1e-9)


def test_filter_drops_pedestrian_walking_through_walls() -> None:
    """长时间整体穿墙的行人整条丢弃（连续穿墙 > 阈值）。"""
    occupancy, clearance, resolution, origin = _synthetic_scene(80)
    xy = np.column_stack([np.linspace(1.0, 3.0, 80), np.full(80, 3.0)])
    # 人为把 clearance 场改成"处处贴墙"：模拟行人整段落在墙里
    wall_clearance = np.zeros_like(clearance) + 0.10
    raw = _synthetic_raw(xy)
    tracks = ped.build_sequence_tracks(
        raw,
        scene_occupancy=occupancy,
        clearance_field=wall_clearance,
        resolution=resolution,
        origin=origin,
        anchor=ped.AnchorMapping(origin_xy=(0.0, 0.0)),
        dt_s=0.1,
        body_radius=0.58,
        max_consecutive_wall_crossings=3,
        min_samples=5,
    )
    assert tracks.report.dropped_consecutive_wall_crossing == 1
    assert tracks.report.pedestrians_used == 0
    assert tracks.tracks == ()
    assert tracks.report.wall_crossing_ratio > 0.0
    # 丢弃的样本不得计入 samples_used（否则可用率会 > 1）
    assert tracks.report.samples_used == 0
    assert tracks.report.sample_usable_ratio <= 1.0


def test_segments_do_not_bridge_filtered_gaps() -> None:
    """剔除中段后必须断开成两段，不得让行人"瞬移"穿过被剔除的墙。

    关键参数：墙块要**短于** ``max_consecutive_wall_crossings``，否则整段会被
    "连续穿墙"规则丢弃 —— 那是另一条规则，由
    ``test_filter_drops_pedestrian_walking_through_walls`` 覆盖。
    注意重采样是**加密**：原始 0.4 s 采样、dt = 0.1 s，所以 1 个栅格单元的墙块
    （轨迹每 0.4 s 只走 0.025 m，对应 ~1.16 s）会命中约 12 个重采样周期；
    阈值取 30 以保证"只切段、不丢段"。
    """
    occupancy, clearance, resolution, origin = _synthetic_scene(80)
    xy = np.column_stack([np.linspace(1.0, 6.0, 200), np.full(200, 4.0)])
    broken = clearance.copy()
    # 中段人为压成贴墙（两侧仍合规）
    broken[:, 30:31] = 0.10
    raw = _synthetic_raw(xy)
    tracks = ped.build_sequence_tracks(
        raw,
        scene_occupancy=occupancy,
        clearance_field=broken,
        resolution=resolution,
        origin=origin,
        anchor=ped.AnchorMapping(origin_xy=(0.0, 0.0)),
        dt_s=0.1,
        body_radius=0.58,
        max_consecutive_wall_crossings=30,
        min_samples=3,
    )
    assert tracks.report.dropped_consecutive_wall_crossing == 0, "墙块短于阈值，不该整段丢弃"
    assert len(tracks.tracks) >= 2, "被剔除的中段必须把轨迹切成多段"
    for track in tracks.tracks:
        assert track.times_s.shape[0] == track.xy.shape[0]
        assert np.all(np.diff(track.times_s) > 0)
        # 段内不得再出现贴墙样本
        values = broken[
            np.floor(track.xy[:, 1] / resolution).astype(int),
            np.floor(track.xy[:, 0] / resolution).astype(int),
        ]
        assert np.all(values >= 0.58 - 1e-9)
    # 墙块里的样本必须真的被剔除（分母仍守恒）
    assert tracks.report.samples_wall_crossing > 0
    assert (
        tracks.report.samples_used + tracks.report.samples_wall_crossing
        == tracks.report.samples_resampled
    )


def test_to_obstacle_tracks_interface() -> None:
    """与 DWA 对接：时间严格递增、半径正确、可被 core.dwa 直接消费。"""
    track = ped.PedestrianTrack(
        pedestrian_id=7,
        times_s=np.arange(5, dtype=float) * 0.1,
        xy=np.column_stack([np.arange(5, dtype=float) * 0.1, np.zeros(5)]),
        velocity=np.tile([1.0, 0.0], (5, 1)),
        radius_m=0.30,
    )
    obstacles = ped.to_obstacle_tracks((track,), t_offset_s=2.5)
    assert len(obstacles) == 1
    obstacle = obstacles[0]
    assert np.isclose(obstacle.times[0], 2.5)
    assert obstacle.radius_m == pytest.approx(0.30)
    assert obstacle.label == "pedestrian_7"
    assert obstacle.position_at(np.array([2.6])).shape == (1, 2)


def test_stable_seed_is_deterministic() -> None:
    assert ped.stable_seed("a|b", "ped42") == ped.stable_seed("a|b", "ped42")
    assert ped.stable_seed("a|b", "ped42") != ped.stable_seed("a|b", "ped43")


def test_largest_free_rectangle_fits_inside_free_space() -> None:
    free = np.zeros((40, 60), dtype=bool)
    free[5:35, 10:50] = True
    x0, y0, x1, y1 = ped.largest_free_rectangle(free, 0.1, (0.0, 0.0), stride=2)
    assert x0 >= 1.0 - 1e-9 and y0 >= 0.5 - 1e-9
    assert x1 <= 5.0 + 1e-9 and y1 <= 3.5 + 1e-9
    assert (x1 - x0) > 3.0 and (y1 - y0) > 2.0


def test_loader_rejects_bad_rows(tmp_path: Path) -> None:
    good = tmp_path / "ok.txt"
    good.write_text("0 1 0.0 0.0\n10 1 0.4 0.4\n", encoding="utf-8")
    raw = ped.load_sequence(good)
    assert raw.xy_local.shape == (2, 2)
    assert raw.pedestrian_ids == (1,)
    assert np.isclose(raw.times_s[-1], 0.4)
    assert np.isclose(raw.extent_m[0], 0.4)

    bad = tmp_path / "bad.txt"
    bad.write_text("0 1 0.0\n", encoding="utf-8")
    with pytest.raises(ped.PedestrianDataError):
        ped.load_sequence(bad)
    missing = tmp_path / "nope.txt"
    with pytest.raises(ped.PedestrianDataError):
        ped.load_sequence(missing)


def test_usability_report_dict_has_consistent_denominator() -> None:
    """三条比例共用同一分母：used + oob + wall == resampled，且可用率 <= 1。"""
    occupancy, clearance, resolution, origin = _synthetic_scene(80)
    xy = np.column_stack([np.linspace(2.0, 5.0, 120), np.full(120, 4.0)])
    raw = _synthetic_raw(xy)
    tracks = ped.build_sequence_tracks(
        raw,
        scene_occupancy=occupancy,
        clearance_field=clearance,
        resolution=resolution,
        origin=origin,
        anchor=ped.AnchorMapping(origin_xy=(0.0, 0.0)),
        dt_s=0.1,
        body_radius=0.58,
        min_samples=5,
    )
    report = tracks.report
    assert (
        report.samples_used + report.samples_out_of_bounds + report.samples_wall_crossing
        == report.samples_resampled
    )
    assert report.sample_usable_ratio <= 1.0
    assert report.samples_resampled > report.samples_raw  # 0.4 s -> 0.1 s 加密
    assert tracks.tracks, "这段轨迹完全合规，必须保留"


# ---------------------------------------------------------------------------
# 真实数据（数据缺失时跳过）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", REAL_SEQUENCES)
def test_real_sequence_format(name: str) -> None:
    path = DATA / f"{name}.txt"
    if not path.is_file():
        pytest.skip(f"缺少 {path}（见 data/README.md）")
    raw = ped.load_sequence(path, name=name)
    assert raw.xy_local.shape[1] == 2
    assert raw.frame.size == raw.ids.size == raw.xy_local.shape[0]
    assert np.all(np.diff(np.unique(raw.frame)) > 0)
    # 数据集帧步长恒为 10（2.5 fps）：时间步长必须是 0.4 s 的整数倍
    steps = np.diff(np.unique(raw.frame))
    assert np.all(steps % ped.DATASET_FRAME_STEP == 0)
    assert raw.duration_s > 60.0
    assert raw.extent_m[0] > 5.0 and raw.extent_m[1] > 5.0
    assert len(raw.pedestrian_ids) > 20


@pytest.mark.parametrize("name", REAL_SEQUENCES)
def test_real_sequence_is_usable_in_a_frozen_scene(name: str) -> None:
    """真实序列 -> 锚点 -> 筛选：必须产出非空轨迹，且全部不穿墙、不出界。"""
    path = DATA / f"{name}.txt"
    if not path.is_file():
        pytest.skip(f"缺少 {path}（见 data/README.md）")
    from runner import config as config_module
    from runner import experiment

    resolved = config_module.load_resolved(
        "configs/frozen_A.yaml",
        ROOT / "config" / "scenarios" / "houseexpo_d398cc42.yaml",
    )
    scene = experiment.build_scene(resolved.scenario)
    raw = ped.load_sequence(path, name=name)
    anchor, conflict = ped.fit_anchor(
        raw,
        clearance_field=scene.clearance,
        occupancy=scene.occupancy,
        resolution=scene.resolution,
        origin=scene.origin,
        free_clearance_m=0.9,
        body_radius=float(resolved.platform["body_radius"]),
        rotation_step_deg=15.0,
        offset_half_range_m=1.0,
        offset_step_m=1.0,
    )
    assert conflict < 0.25, f"{name} 在该户型里放不下（冲突率 {conflict:.3f}）"
    tracks = ped.build_sequence_tracks(
        raw,
        scene_occupancy=scene.occupancy,
        clearance_field=scene.clearance,
        resolution=scene.resolution,
        origin=scene.origin,
        anchor=anchor,
        dt_s=0.1,
        body_radius=float(resolved.platform["body_radius"]),
        min_samples=10,
    )
    assert tracks.tracks, "筛选后应至少保留一条轨迹"
    assert tracks.report.sample_usable_ratio > 0.5
    for track in tracks.tracks:
        cols = np.floor(
            (track.xy[:, 0] - scene.origin[0]) / scene.resolution
        ).astype(int)
        rows = np.floor(
            (track.xy[:, 1] - scene.origin[1]) / scene.resolution
        ).astype(int)
        in_bounds = (
            (rows >= 0)
            & (rows < scene.occupancy.shape[0])
            & (cols >= 0)
            & (cols < scene.occupancy.shape[1])
        )
        assert in_bounds.all(), "筛选后不得再有出界样本"
        assert np.all(
            scene.clearance[rows, cols] >= float(resolved.platform["body_radius"]) - 1e-9
        ), "筛选后不得再有穿墙样本"


def test_same_seed_gives_same_window_hash_inputs() -> None:
    """同 seed：时间窗起点与任务哈希输入逐字段一致。"""
    path = DATA / "students003.txt"
    if not path.is_file():
        pytest.skip(f"缺少 {path}（见 data/README.md）")
    from analysis import pedestrians as analysis_ped

    config = analysis_ped.load_pedestrian_config(
        ROOT / "config" / "pedestrian_sequences.yaml"
    )
    first = analysis_ped.run_episode(
        ROOT / "config" / "scenarios" / "houseexpo_d398cc42.yaml",
        "students003",
        config=config,
        seed=7,
    )
    second = analysis_ped.run_episode(
        ROOT / "config" / "scenarios" / "houseexpo_d398cc42.yaml",
        "students003",
        config=config,
        seed=7,
    )
    assert first.window_start_s == second.window_start_s
    assert first.pedestrian_count == second.pedestrian_count
    assert first.episode.config_hash == second.episode.config_hash
    assert first.episode.periods == second.episode.periods
