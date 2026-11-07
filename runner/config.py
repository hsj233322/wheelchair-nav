"""三层配置的加载、校验与配置哈希。

层与所有权（交接文档 §4.3）：

* ``platform``：平台物理参数，实验不可变；
* ``scenario``：地图、路面、坡度，实验不可变；
* ``algo``：算法超参，**唯一允许在实验中变化**的层；
* ``run``：执行层（seed / repeats / out_dir / dt），不是模型参数，
  不参与配置哈希，只被记录。

本模块属于 runner，不属于 core：core 只接收显式参数，不得 import 本模块。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

SCHEMA_VERSION = 1
DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"


class ConfigError(ValueError):
    """配置文件缺失、字段非法或跨层冲突。"""


# ---------------------------------------------------------------------------
# YAML 读取：拒绝重复键（冻结配置里出现重复键一定是事故）
# ---------------------------------------------------------------------------


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):  # type: ignore[no-untyped-def]
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = key_node.start_mark.line + 1
            raise ConfigError(f"{loader.name}:{line}: 重复的配置键 {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {path}: {exc}") from exc
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: YAML 解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: 顶层必须是映射（dict），实际为 {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# 基础校验工具
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _check_number(
    value: Any,
    where: str,
    *,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
    maximum: float | None = None,
) -> float:
    if not _is_number(value):
        raise ConfigError(f"{where} 必须是有限数值，实际为 {value!r}")
    number = float(value)
    if minimum is not None:
        if number < minimum or (not minimum_inclusive and number == minimum):
            operator = ">=" if minimum_inclusive else ">"
            raise ConfigError(f"{where} 必须 {operator} {minimum}，实际为 {number}")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{where} 必须 <= {maximum}，实际为 {number}")
    return number


def _check_int(value: Any, where: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} 必须是整数，实际为 {value!r}")
    if value < minimum:
        raise ConfigError(f"{where} 必须 >= {minimum}，实际为 {value}")
    return value


def _check_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} 必须是布尔值（true/false），实际为 {value!r}")
    return value


def _check_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} 必须是非空字符串，实际为 {value!r}")
    return value


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} 必须是映射，实际为 {type(value).__name__}")
    return value


def _require_keys(mapping: Mapping[str, Any], required: set[str], where: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ConfigError(f"{where} 缺少必填字段: {', '.join(missing)}")


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"{where} 含未知字段: {', '.join(unknown)}")


def _check_xy(value: Any, where: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{where} 必须是 [x, y]，实际为 {value!r}")
    return (
        _check_number(value[0], f"{where}[0]"),
        _check_number(value[1], f"{where}[1]"),
    )


def _check_rect(value: Any, where: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ConfigError(f"{where} 必须是 [x0, y0, x1, y1]，实际为 {value!r}")
    x0, y0, x1, y1 = (_check_number(item, f"{where}[{i}]") for i, item in enumerate(value))
    if x1 <= x0 or y1 <= y0:
        raise ConfigError(f"{where} 必须满足 x1 > x0 且 y1 > y0，实际为 {value!r}")
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# 分层校验
# ---------------------------------------------------------------------------

_PLATFORM_FIELDS: dict[str, tuple[float, bool]] = {
    "mass": (0.0, False),
    "g": (0.0, False),
    "body_radius": (0.0, False),
    "v_max": (0.0, False),
    "w_max": (0.0, False),
    "a_max": (0.0, False),
    "alpha_max": (0.0, False),
    "eta": (0.0, False),
    "p_aux": (0.0, True),
    "battery_voltage": (0.0, False),
    "battery_resistance": (0.0, True),
}


def validate_platform(doc: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(doc, "platform")
    _reject_unknown(doc, set(_PLATFORM_FIELDS), "platform")
    _require_keys(doc, set(_PLATFORM_FIELDS), "platform")
    for key, (minimum, inclusive) in _PLATFORM_FIELDS.items():
        _check_number(doc[key], f"platform.{key}", minimum=minimum, minimum_inclusive=inclusive)
    eta = float(doc["eta"])
    if eta > 1.0:
        raise ConfigError(f"platform.eta 是传动效率，必须 <= 1.0，实际为 {eta}")
    return dict(doc)


_SCENARIO_KEYS = {
    "id",
    "description",
    "source",
    "content_hash",
    "map",
    "start",
    "goal",
    "terrain",
    "quality",
}
_QUALITY_KEYS = {
    "grade",
    "group",
    "tier_declared",
    "path_len_m",
    "room_label_changes",
    "rooms_visited",
    "n_components",
    "unreachable_ratio",
    "largest_component_ratio",
    "clearance_min_m",
    "clearance_p10_m",
    "clearance_median_m",
    "clearance_below_090_ratio",
    "clearance_threshold_m",
    "bottleneck_width_m",
    "bottleneck_crossings",
    "bottleneck_min_width_m",
    "interface_widths_m",
    "min_interface_width_m",
    "terrain_ratios",
    "default_terrain_ratio",
    "narrow_door_width_m",
    "measured_door_width_m",
    "verified_clearance_m",
    "feasible_at_physical_radius",
    "feasible_at_base_d_min",
    "physical_radius_m",
    "base_d_min_m",
    "required_clearance_m",
    "core_clearance_m",
    "seed",
    "note",
}
_MAP_KEYS = {"resolution", "shape", "origin", "walls", "polygons", "inflation_m"}
_TERRAIN_KEYS = {"default", "patches", "note", "partition"}
_TERRAIN_PATCH_KEYS = {"id", "rect", "c_rr", "slope_deg"}
_TERRAIN_PARTITION_KEYS = {
    "kind",
    "tier",
    "tier_label_zh",
    "zone_size_m",
    "expected_c_rr",
    "classes",
    "target_shares",
    "zones",
    "note",
}
_SOURCE_KEYS = {"dataset", "file", "sha256", "url", "version", "license", "note"}
_MAX_ABS_SLOPE_DEG = 30.0


def _check_polygon(value: Any, where: str) -> list[tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ConfigError(f"{where} 必须是至少 3 个 [x, y] 顶点的列表")
    return [
        _check_xy(point, f"{where}[{index}]") for index, point in enumerate(value)
    ]



def _check_terrain_values(doc: Mapping[str, Any], where: str) -> None:
    """校验 c_rr / slope_deg 的取值；键的存在性与合法性由调用方负责。"""
    _check_number(doc["c_rr"], f"{where}.c_rr", minimum=0.0)
    _check_number(
        doc["slope_deg"],
        f"{where}.slope_deg",
        minimum=-_MAX_ABS_SLOPE_DEG,
        maximum=_MAX_ABS_SLOPE_DEG,
    )


def validate_scenario(doc: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(doc, "scenario")
    _reject_unknown(doc, _SCENARIO_KEYS, "scenario")
    _require_keys(doc, {"id", "map", "start", "goal", "terrain"}, "scenario")
    _check_str(doc["id"], "scenario.id")
    if "description" in doc and not isinstance(doc["description"], str):
        raise ConfigError("scenario.description 必须是字符串")
    if "source" in doc:
        source = _require_mapping(doc["source"], "scenario.source")
        _reject_unknown(source, _SOURCE_KEYS, "scenario.source")
        for key, value in source.items():
            if not isinstance(value, str):
                raise ConfigError(f"scenario.source.{key} 必须是字符串，实际为 {value!r}")
    if "content_hash" in doc:
        _check_str(doc["content_hash"], "scenario.content_hash")
        expected = compute_scenario_hash(doc)
        if doc["content_hash"] != expected:
            raise ConfigError(
                "scenario.content_hash 与场景内容不一致："
                f"文件为 {doc['content_hash'][:16]}…，实算为 {expected[:16]}…；"
                "手工改过场景内容就必须重算哈希（env/build_scenarios.py 会重算）"
            )

    map_doc = _require_mapping(doc["map"], "scenario.map")
    _reject_unknown(map_doc, _MAP_KEYS, "scenario.map")
    _require_keys(map_doc, {"resolution", "shape"}, "scenario.map")

    resolution = _check_number(
        map_doc["resolution"], "scenario.map.resolution", minimum=0.0, minimum_inclusive=False
    )
    shape = map_doc["shape"]
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ConfigError(f"scenario.map.shape 必须是 [rows, cols]，实际为 {shape!r}")
    rows = _check_int(shape[0], "scenario.map.shape[0]", minimum=1)
    cols = _check_int(shape[1], "scenario.map.shape[1]", minimum=1)

    origin = map_doc.get("origin", [0.0, 0.0])
    ox, oy = _check_xy(origin, "scenario.map.origin")
    x_min, y_min = ox, oy
    x_max, y_max = ox + cols * resolution, oy + rows * resolution

    has_walls = "walls" in map_doc
    has_polygons = "polygons" in map_doc
    if has_walls == has_polygons:
        raise ConfigError(
            "scenario.map 必须且只能给出 walls（轴对齐矩形墙）或 "
            "polygons（户型外轮廓多边形）之一"
        )

    if has_walls:
        walls = map_doc["walls"]
        if not isinstance(walls, (list, tuple)):
            raise ConfigError(
                f"scenario.map.walls 必须是矩形列表，实际为 {type(walls).__name__}"
            )
        rects = [_check_rect(rect, f"scenario.map.walls[{i}]") for i, rect in enumerate(walls)]
        for index, (x0, y0, x1, y1) in enumerate(rects):
            if (
                x0 < x_min - 1e-9
                or y0 < y_min - 1e-9
                or x1 > x_max + 1e-9
                or y1 > y_max + 1e-9
            ):
                raise ConfigError(
                    f"scenario.map.walls[{index}] 超出地图范围 "
                    f"[{x_min}, {x_max}] x [{y_min}, {y_max}]"
                )
    else:
        polygons = map_doc["polygons"]
        if not isinstance(polygons, (list, tuple)) or not polygons:
            raise ConfigError("scenario.map.polygons 必须是非空的多边形列表")
        for index, polygon in enumerate(polygons):
            points = _check_polygon(polygon, f"scenario.map.polygons[{index}]")
            for point_index, (x, y) in enumerate(points):
                if not (
                    x_min - 1e-6 <= x <= x_max + 1e-6
                    and y_min - 1e-6 <= y <= y_max + 1e-6
                ):
                    raise ConfigError(
                        f"scenario.map.polygons[{index}][{point_index}] = {[x, y]} "
                        f"超出地图范围 [{x_min}, {x_max}] x [{y_min}, {y_max}]"
                    )

    if "inflation_m" in map_doc:
        _check_number(map_doc["inflation_m"], "scenario.map.inflation_m", minimum=0.0)

    if "quality" in doc:
        quality = _require_mapping(doc["quality"], "scenario.quality")
        _reject_unknown(quality, _QUALITY_KEYS, "scenario.quality")
        for key, value in quality.items():
            if isinstance(value, (dict, list)):
                try:
                    json.dumps(value, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(
                        f"scenario.quality.{key} 必须是 JSON 可序列化内容: {exc}"
                    ) from exc
            elif not isinstance(value, (str, int, float, bool)):
                raise ConfigError(f"scenario.quality.{key} 类型不支持: {value!r}")

    start = _check_xy(doc["start"], "scenario.start")
    goal = _check_xy(doc["goal"], "scenario.goal")
    for name, (x, y) in (("start", start), ("goal", goal)):
        if not (x_min <= x < x_max and y_min <= y < y_max):
            raise ConfigError(
                f"scenario.{name} = {[x, y]} 不在半开区间 "
                f"[{x_min}, {x_max}) x [{y_min}, {y_max}) 内"
            )

    terrain = _require_mapping(doc["terrain"], "scenario.terrain")
    _reject_unknown(terrain, _TERRAIN_KEYS, "scenario.terrain")
    _require_keys(terrain, {"default"}, "scenario.terrain")
    if "note" in terrain and not isinstance(terrain["note"], str):
        raise ConfigError("scenario.terrain.note 必须是字符串")
    default_terrain = _require_mapping(terrain["default"], "scenario.terrain.default")
    _reject_unknown(
        default_terrain,
        {"c_rr", "slope_deg", "label_zh", "class"},
        "scenario.terrain.default",
    )
    _require_keys(default_terrain, {"c_rr", "slope_deg"}, "scenario.terrain.default")
    _check_terrain_values(default_terrain, "scenario.terrain.default")
    patches = terrain.get("patches", [])
    if not isinstance(patches, (list, tuple)):
        raise ConfigError("scenario.terrain.patches 必须是列表")
    for index, patch in enumerate(patches):
        where = f"scenario.terrain.patches[{index}]"
        _require_mapping(patch, where)
        _reject_unknown(patch, _TERRAIN_PATCH_KEYS, where)
        _require_keys(patch, _TERRAIN_PATCH_KEYS, where)
        _check_str(patch["id"], f"{where}.id")
        x0, y0, x1, y1 = _check_rect(patch["rect"], f"{where}.rect")
        if x0 < x_min - 1e-9 or y0 < y_min - 1e-9 or x1 > x_max + 1e-9 or y1 > y_max + 1e-9:
            raise ConfigError(f"{where}.rect 超出地图范围")
        _check_terrain_values(patch, where)
    if "partition" in terrain:
        _validate_terrain_partition(
            terrain["partition"], "scenario.terrain.partition"
        )
    return dict(doc)


def _validate_terrain_partition(value: Any, where: str) -> None:
    """``terrain.partition`` 是**受控材质因子**的声明（论文可追溯用）。

    它只描述"材质怎么分"，不参与任何算法参数；这里做结构校验并强制
    ``kind == "controlled_factor"``，避免有人把它当成实测材质分布。
    """
    doc = _require_mapping(value, where)
    _reject_unknown(doc, _TERRAIN_PARTITION_KEYS, where)
    _require_keys(doc, {"kind", "tier", "classes", "target_shares"}, where)
    if str(doc["kind"]) != "controlled_factor":
        raise ConfigError(
            f"{where}.kind 必须是 'controlled_factor'（材质分布是受控实验因子，"
            f"不是实测数据），实际为 {doc['kind']!r}"
        )
    _check_str(doc["tier"], f"{where}.tier")
    if "tier_label_zh" in doc:
        _check_str(doc["tier_label_zh"], f"{where}.tier_label_zh")
    if "zone_size_m" in doc:
        _check_number(doc["zone_size_m"], f"{where}.zone_size_m", minimum=0.0)
    if "expected_c_rr" in doc:
        _check_number(doc["expected_c_rr"], f"{where}.expected_c_rr", minimum=0.0)
    if "zones" in doc:
        _check_int(doc["zones"], f"{where}.zones", minimum=0)
    if "note" in doc:
        _check_str(doc["note"], f"{where}.note")

    classes = _require_mapping(doc["classes"], f"{where}.classes")
    if not classes:
        raise ConfigError(f"{where}.classes 必须是非空映射")
    for name, props in classes.items():
        class_where = f"{where}.classes.{name}"
        props = _require_mapping(props, class_where)
        _reject_unknown(props, {"c_rr", "slope_deg", "label_zh"}, class_where)
        _require_keys(props, {"c_rr", "slope_deg"}, class_where)
        _check_terrain_values(props, class_where)
        if "label_zh" in props:
            _check_str(props["label_zh"], f"{class_where}.label_zh")

    shares = _require_mapping(doc["target_shares"], f"{where}.target_shares")
    if not shares:
        raise ConfigError(f"{where}.target_shares 必须是非空映射")
    total = 0.0
    for name, share in shares.items():
        if str(name) not in classes:
            raise ConfigError(f"{where}.target_shares.{name} 引用了未声明材质")
        total += _check_number(share, f"{where}.target_shares.{name}", minimum=0.0)
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"{where}.target_shares 的占比之和必须为 1，实际为 {total}")


_ALGO_KEYS = {"id", "astar", "bspline", "dwa", "comfort"}
_ASTAR_REQUIRED = {"e_budget", "d_min", "kappa_max"}
_ASTAR_OPTIONAL = {
    "kappa_window_m",
    "v_cruise",
    "max_labels_per_state",
    "max_expansions",
    # 曲率裕度前馈（2026-10-29）：全局层在 d_min 之外**额外**预留的安全裕度 [m]，
    # 使得 A* 规划出的折线离障 = d_min + clearance_margin_m，从而给平滑切角留出空间。
    # 它只抬高 A* 的离障门槛，**不改变 A* 的搜索逻辑**（仍是同一套 8 邻域多目标标签）。
    "clearance_margin_m",
    # 第 3 章"硬约束 vs 软惩罚"对照开关（2026-11-06）：
    #   enforce_dmin=True （默认）-> d_min 是硬约束，违反率恒为 0；
    #   enforce_dmin=False        -> d_min 转为代价项 w*max(0, d_min-clearance)。
    # 只影响优化方式，不新增任何算法能力。
    "enforce_dmin",
    "soft_penalty_weight",
    # 第 3 章 Pareto 分析：把搜索跑完并返回终点的 (长度, 能耗) 非支配集。
    "collect_pareto",
}
# 平滑专属参数。注意这里**故意不包含** kappa_max / d_min / e_budget：
# 它们与 A* 是同一个约束，必须共用 astar 段的值，不允许在 bspline 段重复声明
# （重复声明就等于允许两套口径分裂，正是 7.7 第 2 条禁止的）。
_BSPLINE_KEYS = {
    "control_spacing_m",
    "rdp_epsilon_m",
    "sample_step_m",
    "max_insert_depth",
}
_DWA_KEYS = {
    "w_heading",
    "w_dist",
    "w_vel",
    "horizon",
    "dt",
    # 6 个采样与安全参数（2026-10-29 接入 DWA 里程碑，均在 algo 层、冻结使用）
    "v_samples",
    "w_samples",
    "w_clearance",
    "jerk_weight",
    "gap_head_on_m",
    "gap_overtake_m",
    "lateral_deviation_max_m",
}
_COMFORT_KEYS = {"a_y_max", "jerk_max"}


def validate_algo(doc: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(doc, "algo")
    _reject_unknown(doc, _ALGO_KEYS, "algo")
    _require_keys(doc, {"id", "astar"}, "algo")
    _check_str(doc["id"], "algo.id")

    astar = _require_mapping(doc["astar"], "algo.astar")
    _reject_unknown(astar, _ASTAR_REQUIRED | _ASTAR_OPTIONAL, "algo.astar")
    _require_keys(astar, _ASTAR_REQUIRED, "algo.astar")
    _check_number(astar["e_budget"], "algo.astar.e_budget", minimum=0.0,
                  minimum_inclusive=False)
    _check_number(astar["d_min"], "algo.astar.d_min", minimum=0.0)
    _check_number(astar["kappa_max"], "algo.astar.kappa_max", minimum=0.0,
                  minimum_inclusive=False)
    if "kappa_window_m" in astar:
        _check_number(astar["kappa_window_m"], "algo.astar.kappa_window_m", minimum=0.0,
                      minimum_inclusive=False)
    if "v_cruise" in astar:
        _check_number(astar["v_cruise"], "algo.astar.v_cruise", minimum=0.0,
                      minimum_inclusive=False)
    if "max_labels_per_state" in astar:
        _check_int(astar["max_labels_per_state"], "algo.astar.max_labels_per_state", minimum=1)
    if "max_expansions" in astar:
        _check_int(astar["max_expansions"], "algo.astar.max_expansions", minimum=1)
    if "clearance_margin_m" in astar:
        _check_number(astar["clearance_margin_m"], "algo.astar.clearance_margin_m",
                      minimum=0.0)
    if "enforce_dmin" in astar:
        _check_bool(astar["enforce_dmin"], "algo.astar.enforce_dmin")
    if "collect_pareto" in astar:
        _check_bool(astar["collect_pareto"], "algo.astar.collect_pareto")
    if "soft_penalty_weight" in astar:
        _check_number(astar["soft_penalty_weight"], "algo.astar.soft_penalty_weight",
                      minimum=0.0)

    if "bspline" in doc:
        bspline = _require_mapping(doc["bspline"], "algo.bspline")
        # 先查"共用约束被重复声明"：这类错误必须给出**为什么**不允许，
        # 而不是笼统地报成"未知字段"。
        shared = sorted(set(bspline) & {"kappa_max", "d_min", "e_budget"})
        if shared:
            raise ConfigError(
                "algo.bspline 不得重复声明 " + "、".join(shared) + "；"
                "曲率上限 / 离障距离 / 能耗预算与 A* 是同一组硬约束，"
                "必须共用 algo.astar 下的同名参数（见交接文档 7.7 第 2 条）"
            )
        _reject_unknown(bspline, _BSPLINE_KEYS, "algo.bspline")
        for key in ("control_spacing_m", "rdp_epsilon_m", "sample_step_m"):
            if key in bspline:
                _check_number(bspline[key], f"algo.bspline.{key}", minimum=0.0,
                              minimum_inclusive=False)
        if "max_insert_depth" in bspline:
            _check_int(bspline["max_insert_depth"], "algo.bspline.max_insert_depth",
                       minimum=0)
            if int(bspline["max_insert_depth"]) > 8:
                raise ConfigError(
                    f"algo.bspline.max_insert_depth 必须 <= 8，实际为 "
                    f"{bspline['max_insert_depth']}"
                )
    if "dwa" in doc:
        dwa = _require_mapping(doc["dwa"], "algo.dwa")
        _reject_unknown(dwa, _DWA_KEYS, "algo.dwa")
        for key in _DWA_KEYS & set(dwa):
            _check_number(dwa[key], f"algo.dwa.{key}", minimum=0.0)
        for key in ("v_samples", "w_samples"):
            if key in dwa and float(dwa[key]) < 1.0:
                raise ConfigError(f"algo.dwa.{key} 必须 >= 1，实际为 {dwa[key]}")
        if "dt" in dwa and "horizon" in dwa:
            if float(dwa["dt"]) > float(dwa["horizon"]) + 1e-9:
                raise ConfigError(
                    f"algo.dwa.dt = {dwa['dt']} 不能大于 algo.dwa.horizon = "
                    f"{dwa['horizon']}"
                )
    if "comfort" in doc:
        comfort = _require_mapping(doc["comfort"], "algo.comfort")
        _reject_unknown(comfort, _COMFORT_KEYS, "algo.comfort")
        for key in _COMFORT_KEYS & set(comfort):
            _check_number(comfort[key], f"algo.comfort.{key}", minimum=0.0,
                          minimum_inclusive=False)
    return dict(doc)


_RUN_KEYS = {"seed", "repeats", "out_dir", "dt"}


def validate_run(doc: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(doc, "run")
    _reject_unknown(doc, _RUN_KEYS, "run")
    _require_keys(doc, {"seed", "repeats"}, "run")
    _check_int(doc["seed"], "run.seed", minimum=0)
    _check_int(doc["repeats"], "run.repeats", minimum=1)
    if "out_dir" in doc:
        _check_str(doc["out_dir"], "run.out_dir")
    if "dt" in doc:
        _check_number(doc["dt"], "run.dt", minimum=0.0, minimum_inclusive=False,
                      maximum=0.5)
    return dict(doc)


def validate_cross(
    platform: Mapping[str, Any], scenario: Mapping[str, Any], algo: Mapping[str, Any]
) -> None:
    """跨层一致性。

    参数归属（2026-10-25 解耦）：

    * platform.body_radius 是**物理**外接圆半径，用于地图膨胀与碰撞判定；
    * algo.astar.d_min 是**策略**安全半径（含物理半径 + 裕度），是实验变量；
    * scenario.map.inflation_m 记录该场景要求的最小物理膨胀半径（= body_radius）。

    因此这里只要求 ``d_min >= body_radius``（不能比车身还小），
    以及 ``d_min >= map.inflation_m``（不能违反场景声明的物理半径）。
    """
    astar = algo["astar"]
    d_min = float(astar["d_min"])
    body_radius = float(platform["body_radius"])
    if d_min < body_radius - 1e-9:
        raise ConfigError(
            f"algo.astar.d_min = {d_min} 小于 platform.body_radius = {body_radius}；"
            "这样允许的路径在物理上已经撞到障碍物"
        )
    inflation_m = scenario.get("map", {}).get("inflation_m")
    if inflation_m is not None and float(inflation_m) > d_min + 1e-9:
        raise ConfigError(
            f"场景声明的物理膨胀半径 scenario.map.inflation_m = {inflation_m} "
            f"大于 algo.astar.d_min = {d_min}；规划器会给出撞到障碍物的路径"
        )
    v_cruise = float(astar.get("v_cruise", platform["v_max"]))
    if v_cruise > float(platform["v_max"]) + 1e-9:
        raise ConfigError(
            f"algo.astar.v_cruise = {v_cruise} 超过 platform.v_max = {platform['v_max']}"
        )


# ---------------------------------------------------------------------------
# 解析后的配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedConfig:
    platform: dict[str, Any]
    scenario: dict[str, Any]
    algo: dict[str, Any]
    run: dict[str, Any]
    config_hash: str
    sources: dict[str, str]

    def snapshot(
        self, *, git_commit: str | None = None, git_dirty: bool | None = None
    ) -> dict[str, Any]:
        """写入 results/<run_id>/config.json 的完整快照。

        三个模型层原样落盘（与 YAML 逐字段一致）；run 层是生效值（可能被 CLI 覆盖）。
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "git": {"commit": git_commit, "dirty": git_dirty},
            "sources": dict(self.sources),
            "platform": copy.deepcopy(self.platform),
            "scenario": copy.deepcopy(self.scenario),
            "algo": copy.deepcopy(self.algo),
            "run": copy.deepcopy(self.run),
        }


def resolve_input(path: str | Path, config_root: Path) -> Path:
    """先按当前工作目录找，再按 config/ 根目录找。

    这样交接文档里的命令 ``python -m runner.experiment configs/frozen_A.yaml
    scenarios/house_01.yaml`` 在仓库根目录可以直接跑。
    """
    raw = Path(path).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, config_root / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = "；".join(str(candidate) for candidate in candidates)
    raise ConfigError(f"找不到配置文件 {str(path)!r}（已尝试：{tried}）")


def _canonical_payload(
    platform: Mapping[str, Any], scenario: Mapping[str, Any], algo: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "platform": platform,
        "scenario": scenario,
        "algo": algo,
    }


def compute_config_hash(
    platform: Mapping[str, Any], scenario: Mapping[str, Any], algo: Mapping[str, Any]
) -> str:
    """三层模型参数的规范化 JSON 的 sha256。

    run 层（seed/repeats/out_dir/dt）不参与：同一份模型配置换种子重跑，
    配置哈希不变，种子单独记录在 config.json 与 metrics.csv 中。
    """
    blob = json.dumps(
        _canonical_payload(platform, scenario, algo),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_scenario_hash(scenario: Mapping[str, Any]) -> str:
    """场景内容的规范化 JSON 的 sha256（``content_hash`` 字段本身不参与）。

    与 config hash 的区别：config hash 覆盖 platform + scenario + algo，
    场景哈希只覆盖一份场景文件，用来确认「这个场景没有被手改过」。
    改场景必须重算哈希（``env/build_scenarios.py`` 会自动重算）。
    """
    payload = {key: value for key, value in scenario.items() if key != "content_hash"}
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def read_config(path: str | Path) -> dict[str, Any]:
    """读取一个 YAML 配置（拒绝重复键），不含任何 schema 校验。"""
    return _read_yaml(Path(path))


def astar_effective_params(
    algo: Mapping[str, Any], platform: Mapping[str, Any]
) -> dict[str, Any]:
    """把 algo.astar 补齐默认值，供 runner 直接构造 core.astar.AStarParams。

    **曲率裕度前馈**：``d_min`` 是**策略安全半径**（平滑后必须守住的硬约束）；
    ``clearance_margin_m`` 是在它之上、**只给全局层**额外预留的裕度。因此传给 A* 的
    是 ``d_min + clearance_margin_m``，而平滑侧仍校验 ``d_min`` 本身。

    这样做的依据是解析判据（Δθ = 45°、κ_max = 1.5 1/m 时切角深度）：

        Δ_切角 = r (1/cos(Δθ/2) − 1)，  r = 1/κ_max = 0.667 m  ⇒  Δ ≈ 55 mm

    若全局折线的离障只有 ``d_min + 11.6 mm``，任何满足 κ ≤ κ_max 的平滑曲线都必然
    越界（几何上无解）。前馈就是把这段**已知会被切掉的深度**提前在全局层留出来。
    它只抬高离障门槛，**不改 A* 的搜索逻辑**（同一套 8 邻域多目标标签）。
    """
    astar = algo["astar"]
    margin = float(astar.get("clearance_margin_m", 0.0))
    d_min = float(astar["d_min"])
    return {
        "e_budget": float(astar["e_budget"]),
        "d_min": d_min,
        "d_min_astar": d_min + margin,
        "clearance_margin_m": margin,
        "kappa_max": float(astar["kappa_max"]),
        "kappa_window_m": float(astar.get("kappa_window_m", 0.6)),
        "v_cruise": float(astar.get("v_cruise", platform["v_max"])),
        "max_labels_per_state": int(astar.get("max_labels_per_state", 64)),
        "max_expansions": int(astar.get("max_expansions", 2_000_000)),
        # 第 3 章软/硬对照开关（默认 True / 0.0 = 既有行为）
        "enforce_dmin": bool(astar.get("enforce_dmin", True)),
        "soft_penalty_weight": float(astar.get("soft_penalty_weight", 0.0)),
        "collect_pareto": bool(astar.get("collect_pareto", False)),
    }


def bspline_effective_params(
    algo: Mapping[str, Any], platform: Mapping[str, Any]
) -> dict[str, Any]:
    """平滑参数：**约束三件套从 astar 段取**，平滑专属参数取 bspline 段。

    这是 7.7 第 2 条"共用同一个 kappa_max"的落地位置——平滑侧与搜索侧唯一
    的约束来源都是 ``algo.astar``，不存在第二份可被改动的拷贝。``d_min`` 与
    ``e_budget`` 同理：平滑会把几何长度改掉，但约束口径不能跟着变。

    **注意 d_min 取 ``astar["d_min"]`` 而不是 ``d_min_astar``**：前馈裕度
    ``clearance_margin_m`` 只作用于全局层（让 A* 走得更保守），平滑侧要守住的仍然是
    策略安全半径 ``d_min`` 本身。若这里也取 ``d_min + margin``，前馈就从
    "留出切角空间"变成了"偷偷抬高安全约束"，那才是真正的口径污染。
    """
    astar = astar_effective_params(algo, platform)
    bspline = algo.get("bspline") or {}
    return {
        "kappa_max": float(astar["kappa_max"]),
        "d_min": float(astar["d_min"]),
        "e_budget": float(astar["e_budget"]),
        "clearance_margin_m": float(astar["clearance_margin_m"]),
        "control_spacing_m": float(bspline.get("control_spacing_m", 0.8)),
        "rdp_epsilon_m": float(bspline.get("rdp_epsilon_m", 0.15)),
        "sample_step_m": float(bspline.get("sample_step_m", 0.02)),
        "max_insert_depth": int(bspline.get("max_insert_depth", 3)),
    }


def dwa_effective_params(
    algo: Mapping[str, Any], platform: Mapping[str, Any]
) -> dict[str, Any]:
    """DWA 参数：窗口来自 platform，权重/阈值/时域来自 algo，约束共用 astar/comfort。

    归属（交接文档 7.9 第 7 条，全部来自配置，core 只接收数值）：

    * ``v_max`` / ``w_max`` / ``a_max`` / ``alpha_max`` -> **platform**（物理上限）
    * ``horizon`` / ``dt`` / 权重 / 采样数 -> **algo.dwa**
    * ``a_y_max`` -> **algo.comfort.a_y_max**（与 A*、B 样条同属舒适性硬约束）
    * ``d_min`` -> **algo.astar.d_min**（策略半径，与全局层同一个口径）
    * ``v_nominal`` -> **algo.astar.v_cruise**（全局层标称巡航速度）
    """
    astar = astar_effective_params(algo, platform)
    dw = algo.get("dwa") or {}
    comfort = algo.get("comfort") or {}
    return {
        "v_max": float(platform["v_max"]),
        "w_max": float(platform["w_max"]),
        "a_max": float(platform["a_max"]),
        "alpha_max": float(platform["alpha_max"]),
        "horizon_s": float(dw.get("horizon", 0.6)),
        "dt_s": float(dw.get("dt", 0.1)),
        "v_samples": int(dw.get("v_samples", 9)),
        "w_samples": int(dw.get("w_samples", 15)),
        "d_min": float(astar["d_min"]),
        # 物理外接圆：来自 platform，与地图膨胀/碰撞判定同一个量（阻断 1）
        "body_radius": float(platform["body_radius"]),
        "v_nominal": float(astar["v_cruise"]),
        "a_y_max": float(comfort.get("a_y_max", 0.8)),
        "jerk_weight": float(dw.get("jerk_weight", 0.15)),
        "w_heading": float(dw.get("w_heading", 0.1)),
        "w_dist": float(dw.get("w_dist", 0.5)),
        "w_vel": float(dw.get("w_vel", 0.3)),
        "w_clearance": float(dw.get("w_clearance", 0.2)),
        "gap_head_on_m": float(dw.get("gap_head_on_m", 0.6)),
        "gap_overtake_m": float(dw.get("gap_overtake_m", 0.25)),
        "lateral_deviation_max_m": float(dw.get("lateral_deviation_max_m", 0.6)),
    }


def load_resolved(
    algo_path: str | Path,
    scenario_path: str | Path,
    *,
    platform_path: str | Path | None = None,
    run_path: str | Path | None = None,
    config_root: str | Path | None = None,
    seed: int | None = None,
    repeats: int | None = None,
) -> ResolvedConfig:
    root = Path(config_root) if config_root is not None else DEFAULT_CONFIG_ROOT
    platform_file = resolve_input(platform_path or "platform.yaml", root)
    run_file = resolve_input(run_path or "run.yaml", root)
    algo_file = resolve_input(algo_path, root)
    scenario_file = resolve_input(scenario_path, root)

    platform = validate_platform(_read_yaml(platform_file))
    scenario = validate_scenario(_read_yaml(scenario_file))
    algo = validate_algo(_read_yaml(algo_file))
    run = validate_run(_read_yaml(run_file))

    if seed is not None:
        run["seed"] = _check_int(seed, "run.seed", minimum=0)
    if repeats is not None:
        run["repeats"] = _check_int(repeats, "run.repeats", minimum=1)

    validate_cross(platform, scenario, algo)
    config_hash = compute_config_hash(platform, scenario, algo)
    return ResolvedConfig(
        platform=platform,
        scenario=scenario,
        algo=algo,
        run=run,
        config_hash=config_hash,
        sources={
            "platform": str(platform_file),
            "scenario": str(scenario_file),
            "algo": str(algo_file),
            "run": str(run_file),
        },
    )
