"""HouseExpo 户型 -> 冻结场景 YAML（``config/scenarios/*.yaml``），带场景质量档案。

生成流程（每个户型）：

1. ``env/floorplan.py`` 读外轮廓多边形与房间包围盒；
2. 轮廓 -> 占据栅格（``--resolution``），约定与 README 一致；
3. 房间类型 -> 地形材质（``config/terrain_rules.yaml``，**假设，非实测**），
   按确定顺序写成 ``terrain.patches``；
4. 自由区 = ``clearance >= max(physical_radius, algo.astar.d_min)``：场景层只声明
   **物理**半径 ``physical_radius = platform.body_radius``（写进 ``map.inflation_m``），
   策略安全半径由 ``algo.astar.d_min`` 决定（采样与连通性分析按它侵蚀，保证冻结场景
   在基准档下必有解）；
5. **按难度档采样起终点**（``config/scenario_selection.yaml`` 指定户型）：
   * easy   ：同一房间内采样，路径 >= ``easy_min_path_len_m``；
   * medium ：跨 1~2 个门洞，路径 >= ``min_path_len_m``；
   * hard   ：跨 >= 3 个门洞，路径 >= ``min_path_len_m``；
   采样用 ``core.astar`` 验证有解，失败重试；不能达标的户型跳过（不输出）。
6. 计算 ``env/scene_quality.py`` 的质量档案（跨门次数、房间数、连通性、地形占比、
   通道宽度），写入场景的 ``quality`` 块，并汇总到 ``docs/scene_metadata.csv``；
7. 计算 ``content_hash`` 并写 YAML（固定 seed，同输入两次运行结果一致）。

另外生成一组**窄门场景**（``narrowdoor_w080/w100/w120``，门洞 0.8/1.0/1.2 m），
供后续通过性实验使用：它们按门洞宽度对应的较小安全半径验证有解，
并分开标注 ``quality.feasible_at_physical_radius`` 与 ``feasible_at_base_d_min``。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy import ndimage

from core import astar
from env import floorplan, scene_quality, terrain_partition
from runner import config as config_module
from runner import experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_PREFIX = "houseexpo"
NARROW_PREFIX = "narrowdoor"
_STRUCTURE = np.ones((3, 3), dtype=int)

_RULES_KEYS = {
    "version",
    "default_class",
    "classes",
    "material_partition",
    "note",
}
_PARTITION_KEYS = {
    "version",
    "zone_size_m",
    "default_tier",
    "min_free_cells",
    "tiers",
}
_CLASS_KEYS = {"c_rr", "slope_deg", "label_zh", "description"}
_SELECTION_KEYS = {"version", "note", "min_path_len_m", "easy_min_path_len_m", "tiers"}
_MAX_ABS_SLOPE_DEG = 30.0


class SkipScene(Exception):
    """该户型在当前配置/档位下无法产出达标场景（跳过，不写文件）。"""


@dataclass(frozen=True)
class TerrainRules:
    """``config/terrain_rules.yaml`` 的内容（**受控材质因子**）。

    场景层的地形是"真实户型几何 + 按受控面积占比叠加的材质分区"，不是实测材质
    分布；材质库与三档占比目标全部来自配置，代码不写死任何 ``c_rr``。
    """

    default_class: str
    classes: dict[str, dict[str, Any]]
    partition: terrain_partition.PartitionConfig
    note: str


@dataclass(frozen=True)
class Selection:
    """``config/scenario_selection.yaml``：按难度档指定要固化的户型。"""

    min_path_len_m: float
    easy_min_path_len_m: float
    tiers: dict[str, tuple[str, ...]]


def load_terrain_rules(path: str | Path) -> TerrainRules:
    """读取并校验地形规则（材质库 + 受控材质分布三档）。

    规则属于**受控实验因子**，不参与三层模型参数，也不进 config hash；
    它决定的是场景文件里的 ``terrain.patches``（因此进场景 content_hash）。
    """
    doc = config_module.read_config(path)
    unknown = sorted(set(doc) - _RULES_KEYS)
    if unknown:
        raise config_module.ConfigError(f"terrain_rules 含未知字段: {', '.join(unknown)}")
    for key in ("default_class", "classes", "material_partition"):
        if key not in doc:
            raise config_module.ConfigError(f"terrain_rules 缺少必填字段: {key}")

    default_class = str(doc["default_class"])
    raw_classes = doc["classes"]
    if not isinstance(raw_classes, dict) or not raw_classes:
        raise config_module.ConfigError("terrain_rules.classes 必须是非空映射")
    classes: dict[str, dict[str, Any]] = {}
    for name, props in raw_classes.items():
        if not isinstance(props, dict):
            raise config_module.ConfigError(f"terrain_rules.classes.{name} 必须是映射")
        unknown_props = sorted(set(props) - _CLASS_KEYS)
        if unknown_props:
            raise config_module.ConfigError(
                f"terrain_rules.classes.{name} 含未知字段: {', '.join(unknown_props)}"
            )
        if "c_rr" not in props or "slope_deg" not in props:
            raise config_module.ConfigError(
                f"terrain_rules.classes.{name} 必须有 c_rr 与 slope_deg"
            )
        c_rr = float(props["c_rr"])
        slope = float(props["slope_deg"])
        if c_rr < 0:
            raise config_module.ConfigError(f"terrain_rules.classes.{name}.c_rr 必须 >= 0")
        if abs(slope) > _MAX_ABS_SLOPE_DEG:
            raise config_module.ConfigError(
                f"terrain_rules.classes.{name}.slope_deg 超出 ±{_MAX_ABS_SLOPE_DEG}"
            )
        classes[str(name)] = {
            "c_rr": c_rr,
            "slope_deg": slope,
            "label_zh": str(props.get("label_zh", name)),
        }
    if default_class not in classes:
        raise config_module.ConfigError(
            f"terrain_rules.default_class = {default_class!r} 不在 classes 里"
        )

    partition = load_partition_config(doc["material_partition"], classes, default_class)
    return TerrainRules(
        default_class=default_class,
        classes=classes,
        partition=partition,
        note=str(doc.get("note", "")),
    )


def load_partition_config(
    doc: Mapping[str, Any],
    classes: Mapping[str, Mapping[str, Any]],
    default_class: str,
) -> terrain_partition.PartitionConfig:
    """``terrain_rules.material_partition`` -> :class:`PartitionConfig`。"""
    if not isinstance(doc, Mapping):
        raise config_module.ConfigError("terrain_rules.material_partition 必须是映射")
    unknown = sorted(set(doc) - _PARTITION_KEYS)
    if unknown:
        raise config_module.ConfigError(
            f"terrain_rules.material_partition 含未知字段: {', '.join(unknown)}"
        )
    zone_size_m = float(doc.get("zone_size_m", 2.2))
    if not zone_size_m > 0:
        raise config_module.ConfigError(
            f"material_partition.zone_size_m 必须 > 0，实际为 {zone_size_m}"
        )
    min_free_cells = int(doc.get("min_free_cells", 4))
    if min_free_cells < 1:
        raise config_module.ConfigError("material_partition.min_free_cells 必须 >= 1")
    default_tier = str(doc.get("default_tier", terrain_partition.MIXED))
    raw_tiers = doc.get("tiers")
    if not isinstance(raw_tiers, Mapping) or not raw_tiers:
        raise config_module.ConfigError("material_partition.tiers 必须是非空映射")

    material_classes = {
        name: terrain_partition.MaterialClass(
            name=name,
            c_rr=float(props["c_rr"]),
            slope_deg=float(props["slope_deg"]),
            label_zh=str(props.get("label_zh", name)),
        )
        for name, props in classes.items()
    }

    tiers: dict[str, terrain_partition.TerrainTier] = {}
    for tier_name, raw_shares in raw_tiers.items():
        tier_name = str(tier_name)
        if not isinstance(raw_shares, Mapping) or not raw_shares:
            raise config_module.ConfigError(
                f"material_partition.tiers.{tier_name} 必须是非空映射"
            )
        shares: dict[str, float] = {}
        for class_name, share in raw_shares.items():
            class_name = str(class_name)
            if class_name not in material_classes:
                raise config_module.ConfigError(
                    f"material_partition.tiers.{tier_name} 引用了未定义材质 {class_name!r}"
                )
            value = float(share)
            if not value >= 0:
                raise config_module.ConfigError(
                    f"material_partition.tiers.{tier_name}.{class_name} 必须 >= 0"
                )
            shares[class_name] = value
        total = sum(shares.values())
        if abs(total - 1.0) > 1e-6:
            raise config_module.ConfigError(
                f"material_partition.tiers.{tier_name} 的占比之和必须为 1，实际为 {total}"
            )
        tiers[tier_name] = terrain_partition.TerrainTier(
            name=tier_name, shares=shares
        )
    if default_tier not in tiers:
        raise config_module.ConfigError(
            f"material_partition.default_tier = {default_tier!r} 不在 tiers 里"
        )
    return terrain_partition.PartitionConfig(
        zone_size_m=zone_size_m,
        default_tier=default_tier,
        classes=material_classes,
        tiers=tiers,
        default_class=default_class,
        min_free_cells=min_free_cells,
        note=str(doc.get("note", "")),
    )


def load_selection(path: str | Path) -> Selection:
    doc = config_module.read_config(path)
    unknown = sorted(set(doc) - _SELECTION_KEYS)
    if unknown:
        raise config_module.ConfigError(f"scenario_selection 含未知字段: {', '.join(unknown)}")
    tiers_doc = doc.get("tiers")
    if not isinstance(tiers_doc, dict) or not tiers_doc:
        raise config_module.ConfigError("scenario_selection.tiers 必须是非空映射")
    tiers: dict[str, tuple[str, ...]] = {}
    for tier, ids in tiers_doc.items():
        tier = str(tier)
        if tier not in {scene_quality.EASY, scene_quality.MEDIUM, scene_quality.HARD}:
            raise config_module.ConfigError(f"scenario_selection.tiers.{tier} 不是合法档位")
        if not isinstance(ids, (list, tuple)) or not ids:
            raise config_module.ConfigError(f"scenario_selection.tiers.{tier} 必须是非空列表")
        tiers[tier] = tuple(str(item) for item in ids)
    return Selection(
        min_path_len_m=float(doc.get("min_path_len_m", 8.0)),
        easy_min_path_len_m=float(doc.get("easy_min_path_len_m", 4.0)),
        tiers=tiers,
    )


def physical_radius(platform: dict[str, Any]) -> float:
    """场景层只保留**物理**外接圆半径：platform.body_radius（用于地图膨胀与碰撞判定）。

    策略裕度（原 platform.inflation）已移到 algo.astar.d_min，不再写进场景安全半径，
    这样 d_min 才是自由的实验变量（见 configs/dmin_sweep/）。
    """
    return float(platform["body_radius"])


def _clip_rect(
    rect: tuple[float, float, float, float], bounds: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    bx0, by0, bx1, by1 = bounds
    return (max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1))


def _scenario_patches(
    zones: Sequence[terrain_partition.MaterialZone],
    rules: TerrainRules,
    bounds: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    """材质分区 -> ``terrain.patches``（按确定顺序，后画覆盖先画）。"""
    return terrain_partition.zones_to_patches(
        zones, rules.partition.classes, bounds=bounds
    )


def terrain_partition_for(
    occupancy: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
    rules: TerrainRules,
    *,
    tier: str,
    zone_size_m: float | None = None,
) -> tuple[list[terrain_partition.MaterialZone], np.ndarray]:
    """占据栅格 -> （材质分区, 逐单元材质名网格）。

    **自由区 = 轮廓内的可通行区域**（``~occupancy``，即 ``clearance > 0``），
    与安全半径无关：材质是地面属性，不该随 ``d_min`` 改变。
    """
    free = ~np.asarray(occupancy, dtype=bool)
    return terrain_partition.partition_free_area(
        free,
        resolution,
        origin,
        rules.partition,
        tier=tier,
        zone_size_m=zone_size_m,
        min_free_cells=rules.partition.min_free_cells,
    )


def terrain_partition_grid_from_scenario(
    scenario: Mapping[str, Any], scene: experiment.Scene
) -> np.ndarray:
    """从**已固化的场景**重建逐单元材质名网格（与 runner 的 patches 采样一致）。

    采样规则与 :func:`runner.experiment.build_scene` 相同：单元中心落在矩形内即命中，
    后画覆盖先画。分析模块用它做"路线上的 ``c_rr`` 分布"，不需要重新生成分区。
    """
    rows, cols = (int(scene.occupancy.shape[0]), int(scene.occupancy.shape[1]))
    terrain = scenario["terrain"]
    grid = np.full(
        (rows, cols), "__free__", dtype="<U16"
    )  # 先填一个不会与材质名冲突的占位
    xs = float(scene.origin[0]) + (np.arange(cols) + 0.5) * scene.resolution
    ys = float(scene.origin[1]) + (np.arange(rows) + 0.5) * scene.resolution
    for patch in terrain.get("patches", []):
        x0, y0, x1, y1 = (float(value) for value in patch["rect"])
        mask = ((xs >= x0) & (xs <= x1))[None, :] & ((ys >= y0) & (ys <= y1))[:, None]
        grid[mask] = str(patch["id"])
    grid[grid == "__free__"] = str(
        (terrain.get("default") or {}).get("class", "default")
    )
    return grid


def _partition_metadata(
    rules: TerrainRules,
    *,
    tier: str,
    zones: Sequence[terrain_partition.MaterialZone],
    zone_size_m: float,
) -> dict[str, Any]:
    """写进场景 ``terrain.partition`` 的受控因子声明（论文可追溯用）。"""
    return {
        "kind": "controlled_factor",
        "tier": str(tier),
        "tier_label_zh": terrain_partition.TIER_LABELS_ZH.get(str(tier), str(tier)),
        "zone_size_m": round(float(zone_size_m), 3),
        "expected_c_rr": round(
            terrain_partition.expected_c_rr(rules.partition.tier(tier), rules.partition.classes),
            5,
        ),
        "classes": {
            name: {
                "c_rr": float(props["c_rr"]),
                "slope_deg": float(props["slope_deg"]),
                "label_zh": str(props["label_zh"]),
            }
            for name, props in rules.classes.items()
        },
        "target_shares": {
            name: float(share) for name, share in rules.partition.tier(tier).shares.items()
        },
        "zones": int(len(zones)),
        "note": (
            "受控实验因子：真实户型几何 + 按受控面积占比叠加的材质分区；"
            "c_rr 取自公开文献范围，非实测标定，需配合 ±30% 敏感性分析。"
        ),
    }


def _scenario_dict(
    *,
    plan: floorplan.FloorPlan,
    source_path: Path,
    resolution: float,
    shape: tuple[int, int],
    origin: tuple[float, float],
    patches: list[dict[str, Any]],
    inflation_m: float,
    default_class: str,
    classes: dict[str, dict[str, Any]],
    note: str,
    start: tuple[float, float],
    goal: tuple[float, float],
    quality: dict[str, Any],
    seed: int,
    prefix: str = DEFAULT_PREFIX,
    partition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scenario: dict[str, Any] = {
        "id": f"{prefix}_{plan.id[:8]}",
        "content_hash": "",  # 占位；最后回填
        "source": {
            "dataset": "HouseExpo",
            "file": source_path.name,
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "url": "https://github.com/TeaganLi/HouseExpo",
            "version": "45e2b2505f6ea1fe49c0203f14efb7ce20b94e7c",
            "license": "MIT",
            "note": f"env/build_scenarios.py 生成；seed={seed}",
        },
        "description": (
            f"HouseExpo {plan.id[:8]}：{plan.room_num} 个房间，"
            f"由外轮廓多边形与房间包围盒转换；地形材质为**受控实验因子**"
            f"（真实几何 + 按受控占比叠加的材质分区），非实测材质分布。"
        ),
        "map": {
            "resolution": resolution,
            "shape": [shape[0], shape[1]],
            "origin": [round(origin[0], 6), round(origin[1], 6)],
            "polygons": [
                [[round(x, 6), round(y, 6)] for x, y in polygon]
                for polygon in plan.polygons
            ],
            "inflation_m": round(inflation_m, 6),
        },
        "start": [round(start[0], 6), round(start[1], 6)],
        "goal": [round(goal[0], 6), round(goal[1], 6)],
        "terrain": {
            "default": {
                **dict(classes[default_class]),
                "class": str(default_class),
            },
            "patches": patches,
            "note": note,
        },
        "quality": quality,
    }
    if partition is not None:
        scenario["terrain"]["partition"] = partition
    scenario["content_hash"] = config_module.compute_scenario_hash(scenario)
    return scenario


def build_one(
    source_path: str | Path,
    *,
    tier: str,
    resolution: float,
    platform: dict[str, Any],
    algo: dict[str, Any],
    rules: TerrainRules,
    selection: Selection,
    seed: int,
    core_clearance_m: float,
    max_attempts: int,
    min_start_goal_m: float = 4.0,
    prefix: str = DEFAULT_PREFIX,
    terrain_tier: str | None = None,
    zone_size_m: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """把一个 HouseExpo JSON 变成指定难度档的场景；不达标时抛 :class:`SkipScene`。

    ``tier`` 是**难度档**（easy/medium/hard，只看瓶颈宽度）；
    ``terrain_tier`` 是**材质分布档**（low_resistance/mixed/high_resistance，
    受控实验因子），默认按 ``plan_id`` 哈希确定性地分配到其中一档。
    """
    source_path = Path(source_path)
    plan = floorplan.load_floorplan(source_path)
    bounds = plan.bbox
    rows, cols, origin = floorplan.grid_shape_for_bounds(bounds, resolution)
    inflation_m = physical_radius(platform)
    setup = experiment.planner_setup(platform, algo)
    base_d_min = float(setup.astar_params.d_min)
    if base_d_min + 1e-9 < inflation_m:
        raise config_module.ConfigError(
            f"algo.astar.d_min = {base_d_min} 小于物理半径 body_radius = {inflation_m}；"
            "规划器会给出撞到障碍物的路径"
        )

    occupancy = floorplan.rasterize_polygons(
        plan.polygons, resolution, (rows, cols), origin
    )
    clearance = astar.clearance_field(occupancy, resolution)
    # 采样与连通性分析都按 base d_min 侵蚀：保证冻结出来的场景在该档下必有解。
    # 物理半径只写进场景 / 用于窄门组与预览，不参与这里的安全半径。
    required = max(inflation_m, base_d_min)
    free = clearance + 1e-9 >= required
    if not free.any():
        raise SkipScene(f"侵蚀 {required:.2f} m 后没有自由单元（地图 {rows}x{cols}）")

    segmentation = scene_quality.segment_rooms(
        clearance, core_clearance_m=core_clearance_m
    )
    components = scene_quality.component_stats(free)
    pairs = scene_quality.room_pair_analysis(clearance, segmentation.labels, free, resolution)

    # 受控材质因子：材质档由 plan_id 派生（不随采样随机数漂移），
    # 几何完全来自真实户型；分区覆盖**全部自由区**（clearance > 0）。
    resolved_terrain_tier = str(
        terrain_tier or terrain_partition.tier_for_plan(plan.id, tuple(rules.partition.tiers))
    )
    active_zone_size = float(
        zone_size_m if zone_size_m is not None else rules.partition.zone_size_m
    )
    zones, class_grid = terrain_partition_for(
        occupancy,
        resolution,
        origin,
        rules,
        tier=resolved_terrain_tier,
        zone_size_m=active_zone_size,
    )
    patches = _scenario_patches(zones, rules, bounds)
    partition_meta = _partition_metadata(
        rules, tier=resolved_terrain_tier, zones=zones, zone_size_m=active_zone_size
    )

    def world(cell: np.ndarray) -> tuple[float, float]:
        return (
            origin[0] + (int(cell[1]) + 0.5) * resolution,
            origin[1] + (int(cell[0]) + 0.5) * resolution,
        )

    component_labels, _ = ndimage.label(free, structure=_STRUCTURE)

    # 采样：难度只看"必经瓶颈宽度"，所以统一走 scene_quality.sample_task_cells
    # （跨房间对 + 同房间极值对），最终以起终点算出的 bottleneck_width_m 定档。
    min_path_len = (
        selection.easy_min_path_len_m
        if tier == scene_quality.EASY
        else selection.min_path_len_m
    )
    rng = np.random.default_rng(seed)
    samples = scene_quality.sample_task_cells(
        clearance,
        segmentation.labels,
        free,
        resolution,
        tier=tier,
        rng=rng,
        min_start_goal_m=min_start_goal_m,
        max_samples=max_attempts,
    )
    if not samples:
        raise SkipScene(
            f"该档位没有可用采样（房间数 {segmentation.room_count}，"
            f"跨房间对 {len(pairs)}）"
        )

    probe_start = world(np.asarray(samples[0][0]))
    probe = _scenario_dict(
        plan=plan,
        source_path=source_path,
        resolution=resolution,
        shape=(rows, cols),
        origin=origin,
        patches=patches,
        inflation_m=inflation_m,
        default_class=rules.default_class,
        classes=rules.classes,
        note=rules.note,
        start=probe_start,
        goal=probe_start,
        quality={},
        seed=seed,
        prefix=prefix,
        partition=partition_meta,
    )
    scene = experiment.build_scene(probe)

    attempts = 0
    best: tuple[float, str, float] | None = None
    for cell_a_tuple, cell_b_tuple in samples:
        if attempts >= max_attempts:
            break
        cell_a = np.asarray(cell_a_tuple)
        cell_b = np.asarray(cell_b_tuple)
        start = world(cell_a)
        goal = world(cell_b)
        attempts += 1
        trial = replace(
            scene,
            start_xy=start,
            goal_xy=goal,
            start_cell=(int(cell_a[0]), int(cell_a[1])),
            goal_cell=(int(cell_b[0]), int(cell_b[1])),
        )
        result = experiment.plan_path(trial, setup)
        if not result.success:
            continue
        quality = scene_quality.build_quality(
            path_len_m=result.length_m,
            cells=result.cells,
            labels=segmentation.labels,
            clearance=clearance,
            free_mask=free,
            step_length=result.step_length,
            class_grid=class_grid,
            default_class=rules.default_class,
        )
        if best is None or result.length_m > best[0]:
            best = (result.length_m, quality.grade, quality.clearance_min_m)
        if result.length_m + 1e-9 < min_path_len or quality.grade != tier:
            continue
        quality_dict = quality.to_dict()
        quality_dict.update(
            {
                "group": "houseexpo",
                "tier_declared": tier,
                "physical_radius_m": round(inflation_m, 3),
                "base_d_min_m": round(base_d_min, 3),
                "required_clearance_m": round(required, 3),
                "core_clearance_m": core_clearance_m,
                "seed": seed,
            }
        )
        scenario = _scenario_dict(
            plan=plan,
            source_path=source_path,
            resolution=resolution,
            shape=(rows, cols),
            origin=origin,
            patches=patches,
            inflation_m=inflation_m,
            default_class=rules.default_class,
            classes=rules.classes,
            note=rules.note,
            start=start,
            goal=goal,
            quality=quality_dict,
            seed=seed,
            prefix=prefix,
            partition=partition_meta,
        )
        config_module.validate_scenario(scenario)
        experiment.build_scene(scenario)
        info = {
            "id": scenario["id"],
            "group": "houseexpo",
            "tier": tier,
            "terrain_tier": resolved_terrain_tier,
            "grade": quality.grade,
            "shape": [rows, cols],
            "n_components": quality.n_components,
            "unreachable_ratio": quality.unreachable_ratio,
            "default_terrain_ratio": quality.default_terrain_ratio,
            "path_len_m": quality.path_len_m,
            "room_label_changes": quality.room_label_changes,
            "rooms_visited": quality.rooms_visited,
            "clearance_min_m": quality.clearance_min_m,
            "clearance_median_m": quality.clearance_median_m,
            "clearance_below_090_ratio": quality.clearance_below_090_ratio,
            "bottleneck_width_m": quality.bottleneck_width_m,
            "bottleneck_crossings": quality.bottleneck_crossings,
            "min_interface_width_m": quality.min_interface_width_m,
            "attempts": attempts,
            "source": source_path.name,
            "start": scenario["start"],
            "goal": scenario["goal"],
            "content_hash": scenario["content_hash"],
        }
        return scenario, info

    detail = (
        f"最好结果 path={best[0]:.2f} m grade={best[1]}（要求 >= {min_path_len} m 且 {tier}）"
        if best is not None
        else "没有任何可行采样"
    )
    raise SkipScene(f"{attempts} 次采样未达标：{detail}")


# ---------------------------------------------------------------------------
# 窄门场景（通过性实验用的独立一组）
# ---------------------------------------------------------------------------


def _narrow_door_walls(width_m: float) -> list[list[float]]:
    half = width_m / 2.0
    y0, y1 = 2.2 - half, 2.2 + half
    return [
        [0.0, 0.0, 8.4, 0.2],      # 下外墙
        [0.0, 4.2, 8.4, 4.4],      # 上外墙
        [0.0, 0.0, 0.2, 4.4],      # 左外墙
        [8.2, 0.0, 8.4, 4.4],      # 右外墙
        [4.0, 0.2, 4.2, y0],       # 中间隔墙（门洞以下）
        [4.0, y1, 4.2, 4.2],       # 中间隔墙（门洞以上）
    ]


def build_narrow_door(
    width_m: float,
    *,
    resolution: float,
    platform: dict[str, Any],
    algo: dict[str, Any],
    rules: TerrainRules,
    seed: int,
    prefix: str = NARROW_PREFIX,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """生成一个门洞宽度固定的两房间场景，用于通过性实验。

    验证用的安全半径取 ``width_m / 2 - resolution``（车身刚好能过、留一个单元余量），
    并分别记录两个口径的可行性：
    ``feasible_at_physical_radius``（门宽 >= 2 x body_radius）与
    ``feasible_at_base_d_min``（门宽 >= 2 x 基准 d_min，本组按设计为 False）。
    """
    inflation_m = physical_radius(platform)
    verified_clearance = max(resolution, width_m / 2.0 - resolution)
    setup = experiment.planner_setup(platform, algo)
    base_d_min = float(setup.astar_params.d_min)
    narrow_setup = replace(
        setup,
        astar_params=replace(setup.astar_params, d_min=verified_clearance),
    )
    walls = _narrow_door_walls(width_m)
    scenario: dict[str, Any] = {
        "id": f"{prefix}_w{int(round(width_m * 100)):03d}",
        "content_hash": "",
        "source": {
            "dataset": "synthetic",
            "file": "env/build_scenarios.py::_narrow_door_walls",
            "sha256": hashlib.sha256(
                repr(_narrow_door_walls(width_m)).encode("utf-8")
            ).hexdigest(),
            "url": "https://github.com/TeaganLi/HouseExpo",
            "version": "n/a",
            "license": "n/a",
            "note": "受控门洞宽度的合成场景，用于通过性实验",
        },
        "description": (
            f"两房间 + 中间隔墙门洞 {width_m:.1f} m 的受控场景（合成，非 HouseExpo 户型）；"
            f"地形材质取 config/terrain_rules.yaml 的 default_class（受控因子，非实测）。"
        ),
        "map": {
            "resolution": resolution,
            "shape": [
                int(math.ceil(4.4 / resolution)),
                int(math.ceil(8.4 / resolution)),
            ],
            "origin": [0.0, 0.0],
            "walls": walls,
            "inflation_m": round(inflation_m, 6),
        },
        "start": [1.2, 2.2],
        "goal": [7.2, 2.2],
        "terrain": {
            "default": dict(rules.classes[rules.default_class]),
            "patches": [],
            "note": rules.note,
        },
        "quality": {},
    }

    scene = experiment.build_scene(scenario)
    result = experiment.plan_path(scene, narrow_setup)
    if not result.success:
        raise SkipScene(f"窄门 {width_m} m 在 clearance={verified_clearance:.2f} m 下无解")
    # 用几何方法量一下门洞实际宽度（区域级二分），作为自检
    half_mask = np.zeros(scene.occupancy.shape, dtype=bool)
    half_mask[:, : scene.occupancy.shape[1] // 2] = True
    other_mask = ~half_mask
    measured = scene_quality.widest_connection_m(
        scene.clearance,
        half_mask & ~scene.occupancy,
        other_mask & ~scene.occupancy,
    )
    if abs(measured - width_m) > 3.0 * resolution:
        raise SkipScene(
            f"窄门 {width_m} m 实测宽度 {measured:.3f} m 偏差过大（分辨率 {resolution}）"
        )

    segmentation = scene_quality.segment_rooms(scene.clearance, core_clearance_m=0.9)
    quality = scene_quality.build_quality(
        path_len_m=result.length_m,
        cells=result.cells,
        labels=segmentation.labels,
        clearance=scene.clearance,
        free_mask=scene.clearance + 1e-9 >= verified_clearance,
        step_length=result.step_length,
        class_grid=None,
        default_class=rules.default_class,
    )
    quality_dict = quality.to_dict()
    quality_dict.update(
        {
            "group": "narrow_door",
            "tier_declared": "passability_fixture",
            "narrow_door_width_m": round(width_m, 3),
            "measured_door_width_m": round(measured, 3),
            "verified_clearance_m": round(verified_clearance, 3),
            "physical_radius_m": round(inflation_m, 3),
            "base_d_min_m": round(base_d_min, 3),
            "feasible_at_physical_radius": bool(width_m >= 2.0 * inflation_m - 1e-9),
            "feasible_at_base_d_min": bool(width_m >= 2.0 * base_d_min - 1e-9),
            "seed": seed,
            "note": "受控门洞宽度，供后续通过性/局部规划实验使用",
        }
    )
    scenario["quality"] = quality_dict
    scenario["content_hash"] = config_module.compute_scenario_hash(scenario)
    config_module.validate_scenario(scenario)
    info = {
        "id": scenario["id"],
        "group": "narrow_door",
        "tier": "passability_fixture",
        "grade": quality.grade,
        "shape": scenario["map"]["shape"],
        "n_components": quality.n_components,
        "unreachable_ratio": quality.unreachable_ratio,
        "default_terrain_ratio": 0.0,
        "path_len_m": quality.path_len_m,
        "room_label_changes": quality.room_label_changes,
        "rooms_visited": quality.rooms_visited,
        "clearance_min_m": quality.clearance_min_m,
        "clearance_median_m": quality.clearance_median_m,
        "clearance_below_090_ratio": quality.clearance_below_090_ratio,
        "bottleneck_width_m": quality.bottleneck_width_m,
        "bottleneck_crossings": quality.bottleneck_crossings,
        "min_interface_width_m": quality.min_interface_width_m,
        "attempts": 1,
        "source": scenario["source"]["file"],
        "start": scenario["start"],
        "goal": scenario["goal"],
        "content_hash": scenario["content_hash"],
    }
    return scenario, info


# ---------------------------------------------------------------------------
# 写盘
# ---------------------------------------------------------------------------


def write_scenario(scenario: dict[str, Any], path: Path) -> None:
    """以确定性格式写 YAML（叶子列表紧凑，长多边形不折行）。"""
    text = yaml.safe_dump(
        scenario,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=None,
        width=10**6,
    )
    path.write_text(text, encoding="utf-8")


METADATA_COLUMNS = (
    "id",
    "group",
    "tier",
    "terrain_tier",
    "grade",
    "path_len_m",
    "room_label_changes",
    "rooms_visited",
    "n_components",
    "unreachable_ratio",
    "clearance_min_m",
    "clearance_median_m",
    "clearance_below_090_ratio",
    "bottleneck_width_m",
    "bottleneck_crossings",
    "min_interface_width_m",
    "default_terrain_ratio",
    "shape_rows",
    "shape_cols",
    "start",
    "goal",
    "attempts",
    "source",
    "content_hash",
)


def write_metadata(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["shape_rows"] = row["shape"][0]
            record["shape_cols"] = row["shape"][1]
            record["start"] = f"{row['start'][0]},{row['start'][1]}"
            record["goal"] = f"{row['goal'][0]},{row['goal'][1]}"
            record.setdefault("terrain_tier", "")
            writer.writerow({key: record[key] for key in METADATA_COLUMNS})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m env.build_scenarios",
        description="HouseExpo 户型 -> config/scenarios/*.yaml（分档采样 + 质量档案 + 内容哈希）",
    )
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "houseexpo"))
    parser.add_argument("--out-dir", default=str(CONFIG_ROOT / "scenarios"))
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--platform", default=None, help="默认 config/platform.yaml")
    parser.add_argument("--algo", default="configs/frozen_A.yaml")
    parser.add_argument("--terrain-rules", default=None, help="默认 config/terrain_rules.yaml")
    parser.add_argument(
        "--selection", default=None, help="默认 config/scenario_selection.yaml"
    )
    parser.add_argument(
        "--metadata-out", default=str(PROJECT_ROOT / "docs" / "scene_metadata.csv")
    )
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--narrow-resolution", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--core-clearance-m", type=float, default=0.9)
    parser.add_argument("--max-attempts", type=int, default=24)
    parser.add_argument(
        "--min-start-goal-m", type=float, default=4.0,
        help="起终点最小直线距离（不超过房间对测地距离的 0.6 倍）",
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--terrain-tier", default=None,
        choices=list(terrain_partition.TIER_IDS),
        help=(
            "材质分布档（受控因子）。默认按 plan_id 哈希确定性分配到低阻/混合/高阻三档之一；"
            "指定后本批场景全部使用该档"
        ),
    )
    parser.add_argument(
        "--zone-size-m", type=float, default=None,
        help="材质分区目标边长 [m]（默认取 terrain_rules.material_partition.zone_size_m）",
    )
    parser.add_argument(
        "--only", default=None,
        help="只生成指定档位（easy/medium/hard/narrow），逗号分隔",
    )
    parser.add_argument(
        "--keep-existing", action="store_true", help="不删除本 prefix 的旧场景文件"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_root = Path(args.config_root) if args.config_root else CONFIG_ROOT
    platform_path = args.platform or config_root / "platform.yaml"
    rules_path = args.terrain_rules or config_root / "terrain_rules.yaml"
    selection_path = args.selection or config_root / "scenario_selection.yaml"

    try:
        platform = config_module.validate_platform(config_module.read_config(platform_path))
        algo = config_module.validate_algo(
            config_module.read_config(config_module.resolve_input(args.algo, config_root))
        )
        rules = load_terrain_rules(rules_path)
        selection = load_selection(selection_path)
    except config_module.ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    only = {item.strip() for item in args.only.split(",")} if args.only else None
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        for stale in list(out_dir.glob(f"{args.prefix}_*.yaml")) + list(
            out_dir.glob(f"{NARROW_PREFIX}_*.yaml")
        ):
            stale.unlink()

    rows: list[dict[str, Any]] = []
    written = skipped = 0

    if only is None or only & {scene_quality.EASY, scene_quality.MEDIUM, scene_quality.HARD}:
        for tier in (scene_quality.EASY, scene_quality.MEDIUM, scene_quality.HARD):
            if only is not None and tier not in only:
                continue
            for ids_index, plan_id in enumerate(selection.tiers.get(tier, ())):
                source = data_dir / f"{plan_id}.json"
                if not source.is_file():
                    skipped += 1
                    print(f"SKIP {plan_id[:8]} ({tier}): 数据文件不存在 {source.name}")
                    continue
                scene_seed = args.seed + ids_index + {"easy": 0, "medium": 100, "hard": 200}[tier]
                try:
                    scenario, info = build_one(
                        source,
                        tier=tier,
                        resolution=args.resolution,
                        platform=platform,
                        algo=algo,
                        rules=rules,
                        selection=selection,
                        seed=scene_seed,
                        core_clearance_m=args.core_clearance_m,
                        max_attempts=args.max_attempts,
                        min_start_goal_m=args.min_start_goal_m,
                        prefix=args.prefix,
                        terrain_tier=args.terrain_tier,
                        zone_size_m=args.zone_size_m,
                    )
                except (SkipScene, config_module.ConfigError) as exc:
                    skipped += 1
                    print(f"SKIP {plan_id[:8]} ({tier}): {exc}")
                    continue
                target = out_dir / f"{args.prefix}_{plan_id[:8]}.yaml"
                write_scenario(scenario, target)
                rows.append(info)
                written += 1
                print(
                    f"OK   {info['tier']:<6} {info['terrain_tier']:<16} {info['id']:<20} "
                    f"{info['shape'][0]:>3}x{info['shape'][1]:<3} "
                    f"len={info['path_len_m']:>6.2f} m label_chg={info['room_label_changes']} "
                    f"rooms={info['rooms_visited']} c_min={info['clearance_min_m']:.2f} "
                    f"below0.9={info['clearance_below_090_ratio']:.2f} "
                    f"bneck={info['bottleneck_width_m']:.2f} comps={info['n_components']} "
                    f"tries={info['attempts']:>2} hash={info['content_hash'][:8]} -> {target.name}"
                )

    if only is None or "narrow" in only:
        for index, width in enumerate((0.8, 1.0, 1.2)):
            try:
                scenario, info = build_narrow_door(
                    width,
                    resolution=args.narrow_resolution,
                    platform=platform,
                    algo=algo,
                    rules=rules,
                    seed=args.seed + 300 + index,
                    prefix=NARROW_PREFIX,
                )
            except (SkipScene, config_module.ConfigError) as exc:
                skipped += 1
                print(f"SKIP narrowdoor {width} m: {exc}")
                continue
            target = out_dir / f"{NARROW_PREFIX}_w{int(round(width * 100)):03d}.yaml"
            write_scenario(scenario, target)
            info["group"] = "narrow_door"
            rows.append(info)
            written += 1
            print(
                f"OK   narrow {info['id']:<20} door={width:.1f} m measured="
                f"{scenario['quality']['measured_door_width_m']:.2f} m "
                f"verified_clearance={scenario['quality']['verified_clearance_m']:.2f} m "
                f"physical_ok={scenario['quality']['feasible_at_physical_radius']} "
                f"base_d_min_ok={scenario['quality']['feasible_at_base_d_min']} "
                f"hash={info['content_hash'][:8]} -> {target.name}"
            )

    if rows:
        write_metadata(rows, Path(args.metadata_out))
    print(
        f"\ngenerated {written} scenes, skipped {skipped}; "
        f"resolution={args.resolution} m (narrow {args.narrow_resolution} m), "
        f"seed={args.seed}, physical_radius={physical_radius(platform):.2f} m "
        f"(platform.body_radius), base_d_min={algo['astar']['d_min']} m (algo), "
        f"terrain_tier={args.terrain_tier or 'per-plan(auto)'} "
        f"zone_size_m={args.zone_size_m or rules.partition.zone_size_m} m, "
        f"metadata -> {args.metadata_out}"
    )
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
