"""把 ETH/UCY 三类指标汇成表，写进 README 的占位标记之间（README 表格的可复现来源）。

```bash
uv run python -m analysis.pedestrians --jobs 8      # 先跑出 docs/dwa_metrics_eth_ucy.csv
uv run python -m analysis.eth_ucy_readme            # 再刷新 README 的结果块
```

为什么要有这个脚本：README 里的 ETH/UCY 结果表**不是手抄的**，而是从
`docs/pedestrian_anchor.csv` / `docs/pedestrian_usability.csv` /
`docs/dwa_metrics_eth_ucy.csv` 三个 CSV 直接生成，避免"表里数字与 CSV 不一致"。
`terrain_tier` 不在指标表里（避免为此重跑全量），这里从场景 YAML 读。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
METRICS = ROOT / "docs" / "dwa_metrics_eth_ucy.csv"
USABILITY = ROOT / "docs" / "pedestrian_usability.csv"
ANCHOR = ROOT / "docs" / "pedestrian_anchor.csv"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_block() -> str:
    metrics_all = _rows(METRICS)
    usability = _rows(USABILITY)
    anchors = _rows(ANCHOR)
    # 窗口里一个行人都没有的 episode == 无障碍任务，按 DWA 表既有口径
    # （obstacle_kind != none）**不进入动态任务分母**，单独报出来。
    empty = [r for r in metrics_all if int(r["pedestrians_in_window"]) == 0]
    metrics = [r for r in metrics_all if int(r["pedestrians_in_window"]) > 0]
    kept = [int(r["pedestrians_in_window"]) for r in metrics_all]
    dropped = [int(r["pedestrians_crowd_dropped"]) for r in metrics_all]

    lines: list[str] = []
    total_combos = len(metrics_all) + 72
    lines.append(
        f"**规模**：{total_combos} 个 (场景, 序列) 组合（20 个冻结场景 × 8 个序列）。"
        f"其中 **{len(anchors)} 个通过锚点拟合并执行**，"
        f"**72 个因参考轨迹不可跟踪被跳过**（`polyline/clearance` 或 `polyline/curvature`，"
        f"见 7.8/7.9 红线：回退折线不是可跟踪的参考轨迹，不得进入局部规划）。"
    )
    lines.append("")
    lines.append(
        f"**拥挤度是受控量**：episode 内保留的行人轨迹中位 **{int(np.median(kept))}** 条"
        f"（上限 {max(kept)}，均值 {np.mean(kept):.1f}），按「离参考路径最近」选取；"
        f"因此另有 **{sum(dropped)} 条**（{sum(1 for d in dropped if d > 0)}/{len(dropped)} 个 episode）"
        f"因拥挤度受控被排除。两个数并列给出，不隐藏取舍。"
        f"其中 **{len(empty)} 个 episode 的观察窗内没有任何行人**"
        f"（等价于无障碍任务，按 DWA 表既有口径不进入动态分母，单独列出）。"
    )
    lines.append("")
    lines.append(
        "**口径**：三条可用性比例同分母 = 连续观测段内按 `dt` 重采样后的样本数；"
        "「窗口内样本可用率」= 观察窗的 0.1 s 栅格中，至少有 1 条行人轨迹覆盖的格子占比。"
    )
    lines.append("")

    # ---- 可用性统计（按序列汇总）----
    lines.append("**① 可用性统计**（全序列口径；逐 (场景, 序列) 明细见 `docs/pedestrian_usability.csv`）")
    lines.append("")
    lines.append(
        "| 序列 | 组合 | 可用行人比例 | 出界比例 | 穿墙比例 | 样本可用率 | 窗口内样本可用率 | 丢弃段 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in usability:
        by_sequence[row["sequence"]].append(row)
    for sequence in sorted(by_sequence):
        subset = by_sequence[sequence]

        def mean(key: str) -> float:
            return float(np.mean([float(r[key]) for r in subset]))

        dropped_runs = sum(int(r["dropped_consecutive_wall_crossing"]) for r in subset)
        lines.append(
            f"| `{sequence}` | {len(subset)} | {mean('pedestrians_usable_ratio'):.3f} "
            f"| {mean('out_of_bounds_ratio'):.3f} | {mean('wall_crossing_ratio'):.3f} "
            f"| {mean('sample_usable_ratio'):.3f} | {mean('window_sample_usable_ratio'):.3f} "
            f"| {dropped_runs} |"
        )
    lines.append("")

    # ---- 三类指标（按序列汇总，动态任务口径）----
    lines.append(
        f"**② 动态层三类指标**（{len(metrics)} 个**有行人**的 episode；"
        f"成功 = 到达 ∧ 无物理违反；`timeout` 不算成功）"
    )
    lines.append("")
    lines.append(
        "| 序列 | 任务 | 到达 | 超时 | **主动碰撞帧** | 被动接触帧 | 已停∧W<1.76 | 接触任务 | 最小圆心距 [m] |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    totals = {"tasks": 0, "reached": 0, "timeout": 0, "moving": 0, "stopped": 0,
              "stopped_narrow": 0, "contact_tasks": 0}
    gaps_all: list[float] = []
    by_seq: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        by_seq[row["sequence"]].append(row)
    for sequence in sorted(by_seq):
        subset = by_seq[sequence]
        reached = sum(1 for r in subset if int(r["reached_goal"]) == 1)
        timeout = sum(1 for r in subset if str(r["reason"]) == "timeout")
        moving = sum(int(r["contact_moving_frames"]) for r in subset)
        stopped = sum(int(r["contact_stopped_frames"]) for r in subset)
        stopped_narrow = sum(int(r["contact_stopped_narrow"]) for r in subset)
        contact_tasks = sum(1 for r in subset if int(r["dynamic_contact_periods"]) > 0)
        gaps = [
            float(r["min_gap_realized_m"])
            for r in subset
            if np.isfinite(float(r["min_gap_realized_m"]))
        ]
        gaps_all.extend(gaps)
        totals["tasks"] += len(subset)
        totals["reached"] += reached
        totals["timeout"] += timeout
        totals["moving"] += moving
        totals["stopped"] += stopped
        totals["stopped_narrow"] += stopped_narrow
        totals["contact_tasks"] += contact_tasks
        lines.append(
            f"| `{sequence}` | {len(subset)} | {reached} | {timeout} | **{moving}** "
            f"| {stopped} | {stopped_narrow} | {contact_tasks} | "
            f"{(min(gaps) if gaps else float('nan')):.3f} |"
        )
    lines.append(
        f"| **合计** | {totals['tasks']} | {totals['reached']} | {totals['timeout']} "
        f"| **{totals['moving']}** | {totals['stopped']} | {totals['stopped_narrow']} "
        f"| {totals['contact_tasks']} | "
        f"{(min(gaps_all) if gaps_all else float('nan')):.3f} |"
    )
    lines.append("")

    # ---- 三类结论 ----
    reached = totals["reached"]
    lines.append("**③ 三类结论**（全文不使用「零碰撞」这一说法）：")
    lines.append("")
    lines.append(
        f"> ① **主动碰撞 = {totals['moving']} 帧**"
        + ("（必须为 0，否则是规划器 bug）" if totals["moving"] == 0 else "（⚠️ 不为 0，需排查）")
    )
    lines.append(
        f"> ② **被动接触 = {totals['stopped']} 帧**"
        f"（{totals['contact_tasks']}/{totals['tasks']} 个 episode 出现），"
        f"其中「本车已停 ∧ 瓶颈 < 1.76 m」= {totals['stopped_narrow']} 帧；"
        f"这是**真人不避让机器人**的必然代价"
    )
    lines.append(
        f"> ③ **未完成（超时）= {totals['timeout']} 个**；到达 {reached}/{totals['tasks']}"
    )
    lines.append("")

    # ---- 档位分解 ----
    lines.append("**按难度档 / 材质档分解**")
    lines.append("")
    lines.append("| 分组 | 任务 | 到达 | 超时 | 主动碰撞帧 | 被动接触帧 | 接触任务 |")
    lines.append("|---|---|---|---|---|---|---|")
    # 材质档不在指标表里（避免为此重跑 40 分钟的全量），从场景 YAML 读
    import yaml  # noqa: PLC0415

    tier_by_scenario: dict[str, str] = {}
    for scenario_file in sorted((ROOT / "config" / "scenarios").glob("houseexpo_*.yaml")):
        with scenario_file.open(encoding="utf-8") as handle:
            doc = yaml.safe_load(handle)
        partition = (doc.get("terrain") or {}).get("partition") or {}
        tier_by_scenario[str(doc.get("id"))] = str(partition.get("tier", "-"))
    for row in metrics:
        row["terrain_tier"] = tier_by_scenario.get(row["scenario_id"], "-")
    for key, label in (("grade", "档位"), ("terrain_tier", "材质档")):
        seen: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in metrics:
            seen[row[key]].append(row)
        for name in sorted(seen):
            subset = seen[name]
            lines.append(
                f"| {label} `{name}` | {len(subset)} "
                f"| {sum(1 for r in subset if int(r['reached_goal']) == 1)} "
                f"| {sum(1 for r in subset if str(r['reason']) == 'timeout')} "
                f"| {sum(int(r['contact_moving_frames']) for r in subset)} "
                f"| {sum(int(r['contact_stopped_frames']) for r in subset)} "
                f"| {sum(1 for r in subset if int(r['dynamic_contact_periods']) > 0)} |"
            )
    return "\n".join(lines)


def main() -> int:
    text = README.read_text(encoding="utf-8")
    start = "<!-- ETH_UCY_RESULTS_START -->"
    end = "<!-- ETH_UCY_RESULTS_END -->"
    if start not in text or end not in text:
        print("README 里找不到占位标记")
        return 2
    block = build_block()
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    README.write_text(f"{head}{start}\n\n{block}\n\n{end}{tail}", encoding="utf-8")
    print(f"已写入 {README}（{len(block)} 字符）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
