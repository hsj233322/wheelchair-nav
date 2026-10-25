"""HouseExpo 户型 -> 固化场景 YAML（``config/scenarios/houseexpo_*.yaml``）。

每个户型的处理流程：

1. ``env/floorplan.py`` 读外轮廓多边形与房间包围盒；
2. 轮廓 -> 占据栅格（``--resolution`` 可配），约定与 README 一致；
3. 房间类型 -> 地形材质（``config/terrain_rules.yaml``，**假设，非实测**），
   按确定顺序写成 ``terrain.patches``（后画的覆盖先画的）；
4. 自由区判定：``clearance >= max(inflation_m, algo.astar.d_min)``，其中
   ``inflation_m = platform.body_radius + platform.inflation``（不许写死）；
5. 取自由区最大的 8 连通分量，在其中随机采样起终点，用 ``core.astar`` 验证有解；
   采样/规划失败就重试，重试上限用尽则**跳过**该户型，绝不输出不可行场景；
6. 计算 ``content_hash`` 并写 YAML（固定 seed，同输入两次运行结果一致）。

生成过程全程复用 runner 的校验与规划代码（``config.validate_scenario``、
``experiment.build_scene``、``experiment.plan_path``），因此生成出来的场景
一定能被实验框架加载并规划。
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from scipy import ndimage

from core import astar
from env import floorplan
from runner import config as config_module
from runner import experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"
DEFAULT_PREFIX = "houseexpo"

_RULES_KEYS = {
    "version",
    "default_class",
    "classes",
    "generic_room_types",
    "room_type_to_class",
    "note",
}
_MAX_ABS_SLOPE_DEG = 30.0


class SkipScene(Exception):
    """该户型在当前配置下无法产出可行场景（跳过，不写文件）。"""


@dataclass(frozen=True)
class TerrainRules:
    """``config/terrain_rules.yaml`` 的内容；仅用于把房间类型映射成材质。"""

    default_class: str
    classes: dict[str, dict[str, float]]
    room_type_to_class: dict[str, str]
    generic_types: tuple[str, ...]
    note: str


def load_terrain_rules(path: str | Path) -> TerrainRules:
    """读取并校验地形规则；规则属于假设，不参与三层模型参数。"""
    doc = config_module.read_config(path)
    unknown = sorted(set(doc) - _RULES_KEYS)
    if unknown:
        raise config_module.ConfigError(
            f"terrain_rules 含未知字段: {', '.join(unknown)}"
        )
    for key in ("default_class", "classes", "room_type_to_class"):
        if key not in doc:
            raise config_module.ConfigError(f"terrain_rules 缺少必填字段: {key}")

    default_class = str(doc["default_class"])
    raw_classes = doc["classes"]
    if not isinstance(raw_classes, dict) or not raw_classes:
        raise config_module.ConfigError("terrain_rules.classes 必须是非空映射")
    classes: dict[str, dict[str, float]] = {}
    for name, props in raw_classes.items():
        if not isinstance(props, dict):
            raise config_module.ConfigError(f"terrain_rules.classes.{name} 必须是映射")
        unknown_props = sorted(set(props) - {"c_rr", "slope_deg"})
        if unknown_props:
            raise config_module.ConfigError(
                f"terrain_rules.classes.{name} 含未知字段: {', '.join(unknown_props)}"
            )
        c_rr = float(props["c_rr"])
        slope = float(props["slope_deg"])
        if c_rr < 0:
            raise config_module.ConfigError(f"terrain_rules.classes.{name}.c_rr 必须 >= 0")
        if abs(slope) > _MAX_ABS_SLOPE_DEG:
            raise config_module.ConfigError(
                f"terrain_rules.classes.{name}.slope_deg 超出 ±{_MAX_ABS_SLOPE_DEG}"
            )
        classes[str(name)] = {"c_rr": c_rr, "slope_deg": slope}
    if default_class not in classes:
        raise config_module.ConfigError(
            f"terrain_rules.default_class = {default_class!r} 不在 classes 里"
        )

    raw_mapping = doc["room_type_to_class"]
    if not isinstance(raw_mapping, dict):
        raise config_module.ConfigError("terrain_rules.room_type_to_class 必须是映射")
    mapping: dict[str, str] = {}
    for room_type, class_name in raw_mapping.items():
        normalized = floorplan.normalize_category(str(room_type))
        class_name = str(class_name)
        if class_name not in classes:
            raise config_module.ConfigError(
                f"terrain_rules.room_type_to_class.{room_type} = {class_name!r} "
                "不在 classes 里"
            )
        mapping[normalized] = class_name

    generic = doc.get("generic_room_types", [])
    if not isinstance(generic, (list, tuple)):
        raise config_module.ConfigError("terrain_rules.generic_room_types 必须是列表")
    generic_types = tuple(floorplan.normalize_category(str(item)) for item in generic)
    note = str(doc.get("note", ""))
    return TerrainRules(
        default_class=default_class,
        classes=classes,
        room_type_to_class=mapping,
        generic_types=generic_types,
        note=note,
    )


def inflation_radius(platform: dict[str, Any]) -> float:
    """车身安全半径 = platform.body_radius + platform.inflation（不写死）。"""
    return float(platform["body_radius"]) + float(platform["inflation"])


def _clip_rect(
    rect: tuple[float, float, float, float], bounds: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    bx0, by0, bx1, by1 = bounds
    return (max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1))


def _scenario_patches(
    plan: floorplan.FloorPlan,
    rules: TerrainRules,
    bounds: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    """房间包围盒 -> terrain.patches（按确定顺序，后画覆盖先画）。"""
    ordered = floorplan.ordered_room_boxes(
        plan.rooms,
        rules.room_type_to_class,
        default_class=rules.default_class,
        generic_types=rules.generic_types,
    )
    patches: list[dict[str, Any]] = []
    for class_name, rect in ordered:
        x0, y0, x1, y1 = _clip_rect(rect, bounds)
        if x1 <= x0 or y1 <= y0:
            continue
        props = rules.classes[class_name]
        patches.append(
            {
                "id": class_name,
                "rect": [round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)],
                "c_rr": props["c_rr"],
                "slope_deg": props["slope_deg"],
            }
        )
    return patches


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
    classes: dict[str, dict[str, float]],
    note: str,
    start: tuple[float, float],
    goal: tuple[float, float],
    seed: int,
    prefix: str = DEFAULT_PREFIX,
) -> dict[str, Any]:
    source_bytes = source_path.read_bytes()
    scenario: dict[str, Any] = {
        "id": f"{prefix}_{plan.id[:8]}",
        "content_hash": "",  # 占位；最后回填
        "source": {
            "dataset": "HouseExpo",
            "file": source_path.name,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "url": "https://github.com/TeaganLi/HouseExpo",
            "version": "45e2b2505f6ea1fe49c0203f14efb7ce20b94e7c",
            "license": "MIT",
            "note": f"env/build_scenarios.py 生成；seed={seed}",
        },
        "description": (
            f"HouseExpo {plan.id[:8]}：{plan.room_num} 个房间，"
            f"由外轮廓多边形与房间包围盒转换；地形标签为基于房间类型的假设，非实测数据。"
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
            "default": dict(classes[default_class]),
            "patches": patches,
            "note": note,
        },
    }
    scenario["content_hash"] = config_module.compute_scenario_hash(scenario)
    return scenario


def _largest_component(free_mask: np.ndarray) -> np.ndarray | None:
    """自由区的最大 8 连通分量（没有自由单元时返回 None）。"""
    labeled, count = ndimage.label(free_mask, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return None
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    if sizes.max() == 0:
        return None
    return labeled == int(sizes.argmax())


def build_one(
    source_path: str | Path,
    *,
    resolution: float,
    platform: dict[str, Any],
    algo: dict[str, Any],
    rules: TerrainRules,
    seed: int,
    min_start_goal_m: float,
    max_attempts: int,
    prefix: str = DEFAULT_PREFIX,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """把一个 HouseExpo JSON 变成场景 dict；不可行时抛 :class:`SkipScene`。"""
    source_path = Path(source_path)
    plan = floorplan.load_floorplan(source_path)
    bounds = plan.bbox
    rows, cols, origin = floorplan.grid_shape_for_bounds(bounds, resolution)
    inflation_m = inflation_radius(platform)
    setup = experiment.planner_setup(platform, algo)
    if inflation_m > setup.astar_params.d_min + 1e-9:
        raise config_module.ConfigError(
            f"inflation = body_radius + inflation = {inflation_m} "
            f"大于 algo.astar.d_min = {setup.astar_params.d_min}"
        )

    patches = _scenario_patches(plan, rules, bounds)

    # 先用纯函数算出占据栅格与距离场，找出可行自由区（不依赖 build_scene 的起终点校验）。
    occupancy = floorplan.rasterize_polygons(
        plan.polygons, resolution, (rows, cols), origin
    )
    clearance = astar.clearance_field(occupancy, resolution)
    required_clearance = max(inflation_m, setup.astar_params.d_min)
    free_mask = clearance + 1e-9 >= required_clearance
    component = _largest_component(free_mask)
    if component is None:
        raise SkipScene(
            f"侵蚀 {required_clearance:.2f} m 后没有自由单元（地图 {rows}x{cols}）"
        )
    cells = np.argwhere(component)
    if cells.shape[0] < 2:
        raise SkipScene("自由区最大连通分量只有 1 个单元")

    def world(cell: np.ndarray) -> tuple[float, float]:
        return (
            origin[0] + (int(cell[1]) + 0.5) * resolution,
            origin[1] + (int(cell[0]) + 0.5) * resolution,
        )

    probe_cell = cells[0]
    probe_start = world(probe_cell)
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
        seed=seed,
        prefix=prefix,
    )
    scene = experiment.build_scene(probe)

    # 场景内采样距离下限：小户型按分量尺寸自适应，避免把可用场景全判掉。
    span = cells.max(axis=0) - cells.min(axis=0)
    diameter = float(np.hypot(span[0], span[1])) * resolution
    min_distance = min(min_start_goal_m, 0.5 * diameter)

    rng = np.random.default_rng(seed)
    failures: dict[str, int] = {}
    for _ in range(max_attempts):
        first, second = rng.integers(0, cells.shape[0], size=2)
        if first == second:
            continue
        cell_a = cells[int(first)]
        cell_b = cells[int(second)]
        start = world(cell_a)
        goal = world(cell_b)
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) < min_distance:
            failures["too_close"] = failures.get("too_close", 0) + 1
            continue
        trial = replace(
            scene,
            start_xy=start,
            goal_xy=goal,
            start_cell=(int(cell_a[0]), int(cell_a[1])),
            goal_cell=(int(cell_b[0]), int(cell_b[1])),
        )
        result = experiment.plan_path(trial, setup)
        if result.success:
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
                seed=seed,
                prefix=prefix,
            )
            config_module.validate_scenario(scenario)
            experiment.build_scene(scenario)
            info = {
                "id": scenario["id"],
                "source": source_path.name,
                "shape": [rows, cols],
                "inflation_m": inflation_m,
                "required_clearance_m": required_clearance,
                "free_cells": int(component.sum()),
                "start": scenario["start"],
                "goal": scenario["goal"],
                "path_length_m": round(result.length_m, 3),
                "energy_j": round(result.total_energy_j, 1),
                "attempts": sum(failures.values()) + 1,
                "attempt_failures": dict(failures),
                "content_hash": scenario["content_hash"],
            }
            return scenario, info
        failures[result.reason] = failures.get(result.reason, 0) + 1

    raise SkipScene(
        f"{max_attempts} 次采样都没有可行解（失败原因: {failures or '全部距离过近'}）"
    )


def write_scenario(scenario: dict[str, Any], path: Path) -> None:
    """以确定性格式写 YAML。

    ``default_flow_style=None``：叶子列表（坐标对、rect）写成紧凑的 ``[x, y]``，
    外层结构仍然是块状；``width`` 调大以免长多边形被折行。
    """
    text = yaml.safe_dump(
        scenario,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=None,
        width=10**6,
    )
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m env.build_scenarios",
        description="HouseExpo 户型 -> config/scenarios/*.yaml（固定 seed + 内容哈希）",
    )
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "houseexpo"))
    parser.add_argument("--out-dir", default=str(CONFIG_ROOT / "scenarios"))
    parser.add_argument("--config-root", default=None)
    parser.add_argument("--platform", default=None, help="默认 config/platform.yaml")
    parser.add_argument("--algo", default="configs/frozen_A.yaml")
    parser.add_argument("--terrain-rules", default=None, help="默认 config/terrain_rules.yaml")
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--min-start-goal-m", type=float, default=3.0)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--max-scenes", type=int, default=20)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_root = Path(args.config_root) if args.config_root else CONFIG_ROOT
    platform_path = args.platform or config_root / "platform.yaml"
    rules_path = args.terrain_rules or config_root / "terrain_rules.yaml"

    try:
        platform = config_module.validate_platform(
            config_module.read_config(platform_path)
        )
        algo = config_module.validate_algo(
            config_module.read_config(config_module.resolve_input(args.algo, config_root))
        )
        rules = load_terrain_rules(rules_path)
    except config_module.ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.json"))[: args.max_scenes]
    if not files:
        print(f"没有在 {data_dir} 找到户型 JSON；见 data/README.md 的下载方式", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for index, source in enumerate(files):
        scene_seed = args.seed + index
        try:
            scenario, info = build_one(
                source,
                resolution=args.resolution,
                platform=platform,
                algo=algo,
                rules=rules,
                seed=scene_seed,
                min_start_goal_m=args.min_start_goal_m,
                max_attempts=args.max_attempts,
                prefix=args.prefix,
            )
        except (SkipScene, config_module.ConfigError) as exc:
            skipped += 1
            print(f"SKIP {source.stem[:8]} (seed={scene_seed}): {exc}")
            continue
        target = out_dir / f"{args.prefix}_{source.stem[:8]}.yaml"
        write_scenario(scenario, target)
        written += 1
        print(
            f"OK   {info['id']:<20} {info['shape'][0]:>3}x{info['shape'][1]:<3} "
            f"free={info['free_cells']:>6} "
            f"path={info['path_length_m']:>6.2f} m E={info['energy_j']:>7.1f} J "
            f"tries={info['attempts']:>2} hash={info['content_hash'][:8]} "
            f"-> {target.name}"
        )

    print(
        f"\ngenerated {written} scenes, skipped {skipped}; "
        f"resolution={args.resolution} m, seed={args.seed} (scene i uses seed+i), "
        f"inflation={inflation_radius(platform):.2f} m "
        f"(body_radius={platform['body_radius']} + inflation={platform['inflation']})"
    )
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
