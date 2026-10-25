"""批量筛查 HouseExpo 户型：在给定安全半径下能否跨房间，以及门洞有多宽。

用途：挑选"能做出中/难场景"的户型。HouseExpo 的自由区轮廓把每道内墙描成宽约
0.12 m 的狭缝，门洞是狭缝中断处；把自由区按安全半径（默认
``platform.body_radius + platform.inflation`` = 0.70 m）侵蚀后，窄门会被切断，
"能否跨房间"因此取决于门洞宽度。本工具把这件事量化，输出一张 CSV。

```bash
uv run python -m env.screen_floorplans --out docs/floorplan_screening.csv
```

每个户型一行（房间、门洞、跨门次数都是**从几何推断**的，规则见
``env/scene_quality.py``）：

* ``free_cells`` / ``eroded_cells``：轮廓内自由单元数 / 侵蚀后仍可行单元数；
* ``n_components`` / ``unreachable_ratio``：侵蚀后可行区的连通分量数与不可达占比；
* ``seg_rooms``：clearance 分割出的房间数；
* ``cross_room_m`` / ``crossings``：**最远**的跨房间测地距离及其路径上的跨门次数；
* ``max_crossings`` / ``crossings_len_m``：**跨门次数最多**的房间对及其测地距离；
* ``widest_door_m``：该房间对之间最宽连通通道的宽度 [m]；小于 2 x 安全半径
  意味着在当前安全半径下不可达。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core import astar
from env import floorplan, scene_quality
from runner import config as config_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"

CSV_COLUMNS = (
    "id",
    "file",
    "room_num",
    "extent_x_m",
    "extent_y_m",
    "free_cells",
    "eroded_cells",
    "n_components",
    "unreachable_ratio",
    "seg_rooms",
    "cross_room_m",
    "crossings",
    "max_crossings",
    "crossings_len_m",
    "widest_door_m",
)


def screen_one(
    path: Path,
    *,
    resolution: float,
    required_clearance: float,
    core_clearance_m: float,
) -> dict[str, Any]:
    plan = floorplan.load_floorplan(path)
    rows, cols, origin = floorplan.grid_shape_for_bounds(plan.bbox, resolution)
    occupancy = floorplan.rasterize_polygons(
        plan.polygons, resolution, (rows, cols), origin
    )
    clearance = astar.clearance_field(occupancy, resolution)
    free = clearance > 0.0
    eroded = clearance + 1e-9 >= required_clearance
    components = scene_quality.component_stats(eroded)
    segmentation = scene_quality.segment_rooms(
        clearance, core_clearance_m=core_clearance_m
    )

    row: dict[str, Any] = {
        "id": plan.id[:8],
        "file": path.name,
        "room_num": plan.room_num,
        "extent_x_m": round(plan.bbox[2] - plan.bbox[0], 2),
        "extent_y_m": round(plan.bbox[3] - plan.bbox[1], 2),
        "free_cells": int(free.sum()),
        "eroded_cells": int(eroded.sum()),
        "n_components": int(components["n_components"]),
        "unreachable_ratio": round(float(components["unreachable_ratio"]), 4),
        "seg_rooms": int(segmentation.room_count),
        "cross_room_m": 0.0,
        "crossings": 0,
        "max_crossings": 0,
        "crossings_len_m": 0.0,
        "widest_door_m": 0.0,
    }
    if not eroded.any() or segmentation.room_count < 2:
        return row

    pairs = scene_quality.room_pair_analysis(
        clearance, segmentation.labels, eroded, resolution
    )
    if not pairs:
        return row

    farthest = max(pairs, key=lambda pair: pair.geodesic_m)
    most_crossings = max(pairs, key=lambda pair: (pair.crossings, pair.geodesic_m))
    mask_a = segmentation.labels == most_crossings.room_a
    mask_b = segmentation.labels == most_crossings.room_b
    row["cross_room_m"] = round(farthest.geodesic_m, 2)
    row["crossings"] = int(farthest.crossings)
    row["max_crossings"] = int(most_crossings.crossings)
    row["crossings_len_m"] = round(most_crossings.geodesic_m, 2)
    row["widest_door_m"] = round(
        scene_quality.widest_connection_m(clearance, mask_a, mask_b), 2
    )
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m env.screen_floorplans",
        description="筛查 HouseExpo 户型的跨房间可行性与门洞宽度，输出 CSV",
    )
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "houseexpo"))
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "docs" / "floorplan_screening.csv")
    )
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--core-clearance-m", type=float, default=0.9)
    parser.add_argument(
        "--required-clearance-m",
        type=float,
        default=None,
        help="默认取 platform.body_radius + platform.inflation",
    )
    parser.add_argument("--max-files", type=int, default=0, help="0 表示全部")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    platform = config_module.validate_platform(
        config_module.read_config(CONFIG_ROOT / "platform.yaml")
    )
    required = (
        float(args.required_clearance_m)
        if args.required_clearance_m is not None
        else float(platform["body_radius"]) + float(platform["inflation"])
    )
    files = sorted(Path(args.data_dir).glob("*.json"))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        print(f"没有在 {args.data_dir} 找到户型 JSON", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for index, path in enumerate(files):
            row = screen_one(
                path,
                resolution=args.resolution,
                required_clearance=required,
                core_clearance_m=args.core_clearance_m,
            )
            rows.append(row)
            writer.writerow(row)
            if (index + 1) % 25 == 0:
                print(f"  screened {index + 1}/{len(files)}", file=sys.stderr)

    cross_room = [row for row in rows if row["cross_room_m"] >= 8.0 and row["crossings"] >= 1]
    hard = [row for row in rows if row["max_crossings"] >= 3 and row["crossings_len_m"] >= 8.0]
    print(
        f"screened {len(rows)} floorplans at required clearance {required:.2f} m "
        f"-> {out_path}\n"
        f"cross-room >= 8 m with >= 1 crossing: {len(cross_room)}; "
        f">= 3 crossings with >= 8 m: {len(hard)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
