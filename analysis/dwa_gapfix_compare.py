"""动态间距判据修正的**前后对比**（交接文档 7.10，2026-10-30）。

修正内容（``core/dwa.py``）：

    旧：圆心距阈值 = required_gap(δ) + r_o            （同向 0.55 m < 0.88 m，允许穿透）
    新：圆心距阈值 = body_radius + r_o + required_gap(δ)（同向 1.13 m，迎面 1.48 m）

本脚本把两次运行的 ``docs/dwa_metrics*.csv`` 按 ``(场景, 障碍类型)`` **配对**
比较，并画出一张"每个任务的圆心距最小值：修正前 vs 修正后"的哑铃图。

```bash
uv run python -m analysis.dwa_gapfix_compare \
    --before docs/dwa_metrics_before.csv \
    --after docs/dwa_metrics.csv \
    --csv docs/dwa_gapfix_compare.csv \
    --figure docs/dwa/gapfix_before_after.png
```

配对原则（交接文档 7.8 第 1 条）：子集成分变化的对比必须报**配对差**，
不能只报各自的中位数。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
)
TIER_ORDER = ("easy", "medium", "hard")
COLOR_BEFORE = "#d62728"
COLOR_AFTER = "#2ca02c"
COLOR_CONTACT = "#d62728"


def _setup_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["font.monospace"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 读入
# ---------------------------------------------------------------------------


def read_metrics(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["scenario_id"], row["obstacle_kind"]): row for row in rows}


def _number(row: dict[str, str] | None, key: str) -> float:
    """缺失列 / "inf" / "nan" 一律记 ``inf``（表示"无约束/未记录"）。"""
    if row is None or key not in row or row[key] in ("", "inf", "nan"):
        return float("inf")
    try:
        return float(row[key])
    except ValueError:
        return float("inf")


def _integer(row: dict[str, str] | None, key: str) -> int:
    if row is None or key not in row or row[key] in ("", "nan"):
        return 0
    try:
        return int(float(row[key]))
    except ValueError:
        return 0


@dataclass(frozen=True)
class Pair:
    """一个 (场景, 障碍类型) 的前后配对。"""

    scenario_id: str
    grade: str
    obstacle_kind: str
    trackable_before: bool
    trackable_after: bool
    task_ok_before: bool
    task_ok_after: bool
    reached_before: bool
    reached_after: bool
    reason_before: str
    reason_after: str
    gap_before: float
    gap_after: float
    contact_periods_before: int
    contact_periods_after: int
    collision_frames_before: int
    collision_frames_after: int
    fallback_before: int
    fallback_after: int
    periods_before: int
    periods_after: int
    time_before: float
    time_after: float

    @property
    def key(self) -> str:
        return f"{self.scenario_id}/{self.obstacle_kind}"

    @property
    def executable_before(self) -> bool:
        """修正前是否真的执行过（``skipped_untrackable`` 的周期数为 0）。"""
        return self.periods_before > 0

    @property
    def executable_after(self) -> bool:
        return self.periods_after > 0

    @property
    def gap_delta(self) -> float:
        if not (math.isfinite(self.gap_before) and math.isfinite(self.gap_after)):
            return float("nan")
        return self.gap_after - self.gap_before


def build_pairs(
    before: dict[tuple[str, str], dict[str, str]],
    after: dict[tuple[str, str], dict[str, str]],
    *,
    gap_column: str,
) -> list[Pair]:
    pairs: list[Pair] = []
    for key in sorted(set(before) | set(after)):
        old = before.get(key)
        new = after.get(key)
        template = new or old
        assert template is not None
        pairs.append(
            Pair(
                scenario_id=key[0],
                grade=str(template.get("grade", "-")),
                obstacle_kind=key[1],
                trackable_before=bool(_integer(old, "trackable")),
                trackable_after=bool(_integer(new, "trackable")),
                task_ok_before=bool(_integer(old, "task_ok")),
                task_ok_after=bool(_integer(new, "task_ok")),
                reached_before=bool(_integer(old, "reached_goal")),
                reached_after=bool(_integer(new, "reached_goal")),
                reason_before=str((old or {}).get("reason", "-")),
                reason_after=str((new or {}).get("reason", "-")),
                gap_before=_number(old, gap_column),
                gap_after=_number(new, gap_column),
                contact_periods_before=_integer(old, "dynamic_contact_periods"),
                contact_periods_after=_integer(new, "dynamic_contact_periods"),
                collision_frames_before=_integer(old, "physical_collision_frames"),
                collision_frames_after=_integer(new, "physical_collision_frames"),
                fallback_before=_integer(old, "fell_back_periods"),
                fallback_after=_integer(new, "fell_back_periods"),
                periods_before=_integer(old, "periods"),
                periods_after=_integer(new, "periods"),
                time_before=_number(old, "completion_time_s"),
                time_after=_number(new, "completion_time_s"),
            )
        )
    return pairs


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def _finite(values: Sequence[float]) -> np.ndarray:
    return np.array([v for v in values if math.isfinite(v)], dtype=float)


def _median(values: Sequence[float]) -> str:
    data = _finite(values)
    return f"{float(np.median(data)):.3f}" if data.size else "nan"


@dataclass(frozen=True)
class GroupSummary:
    group: str
    obstacle_kind: str
    tasks: int
    trackable: int
    ok_before: int
    ok_after: int
    reached_before: int
    reached_after: int
    contact_tasks_before: int
    contact_tasks_after: int
    contact_periods_before: int
    contact_periods_after: int
    collision_frames_before: int
    collision_frames_after: int
    fallback_before: int
    fallback_after: int
    gap_min_before: str
    gap_min_after: str
    gap_p10_before: str
    gap_p10_after: str
    gap_median_before: str
    gap_median_after: str
    time_median_before: str
    time_median_after: str


def summarise(pairs: Sequence[Pair], *, dynamic_only: bool = True) -> list[GroupSummary]:
    rows: list[GroupSummary] = []
    pool = [p for p in pairs if p.obstacle_kind != "none"] if dynamic_only else list(pairs)
    groups: list[tuple[str, list[Pair]]] = []
    for grade in TIER_ORDER:
        subset = [p for p in pool if p.grade == grade]
        if subset:
            groups.append((grade, subset))
    groups.append(("all", list(pool)))

    for group_name, subset in groups:
        for kind in ("all", "head_on", "crossing", "same_direction"):
            bucket = (
                subset if kind == "all" else [p for p in subset if p.obstacle_kind == kind]
            )
            if not bucket:
                continue
            # 只有**真正执行过**的任务才进间距分布（skipped_untrackable 无轨迹）
            ran = [p for p in bucket if p.executable_before or p.executable_after]
            before_gaps = _finite([p.gap_before for p in ran])
            after_gaps = _finite([p.gap_after for p in ran])
            rows.append(
                GroupSummary(
                    group=group_name,
                    obstacle_kind=kind,
                    tasks=len(bucket),
                    trackable=sum(
                        1 for p in bucket if p.trackable_before or p.trackable_after
                    ),
                    ok_before=sum(1 for p in ran if p.task_ok_before),
                    ok_after=sum(1 for p in ran if p.task_ok_after),
                    reached_before=sum(1 for p in ran if p.reached_before),
                    reached_after=sum(1 for p in ran if p.reached_after),
                    contact_tasks_before=sum(
                        1 for p in ran if p.contact_periods_before > 0
                    ),
                    contact_tasks_after=sum(
                        1 for p in ran if p.contact_periods_after > 0
                    ),
                    contact_periods_before=sum(p.contact_periods_before for p in ran),
                    contact_periods_after=sum(p.contact_periods_after for p in ran),
                    collision_frames_before=sum(
                        p.collision_frames_before for p in ran
                    ),
                    collision_frames_after=sum(
                        p.collision_frames_after for p in ran
                    ),
                    fallback_before=sum(p.fallback_before for p in ran),
                    fallback_after=sum(p.fallback_after for p in ran),
                    gap_min_before=(
                        f"{before_gaps.min():.3f}" if before_gaps.size else "nan"
                    ),
                    gap_min_after=(
                        f"{after_gaps.min():.3f}" if after_gaps.size else "nan"
                    ),
                    gap_p10_before=(
                        f"{float(np.percentile(before_gaps, 10)):.3f}"
                        if before_gaps.size
                        else "nan"
                    ),
                    gap_p10_after=(
                        f"{float(np.percentile(after_gaps, 10)):.3f}"
                        if after_gaps.size
                        else "nan"
                    ),
                    gap_median_before=_median([p.gap_before for p in ran]),
                    gap_median_after=_median([p.gap_after for p in ran]),
                    time_median_before=_median([p.time_before for p in ran]),
                    time_median_after=_median([p.time_after for p in ran]),
                )
            )
    return rows


SUMMARY_COLUMNS: tuple[str, ...] = (
    "group",
    "obstacle_kind",
    "tasks",
    "trackable_tasks",
    "task_ok_before",
    "task_ok_after",
    "reached_goal_before",
    "reached_goal_after",
    "contact_tasks_before",
    "contact_tasks_after",
    "dynamic_contact_periods_before",
    "dynamic_contact_periods_after",
    "physical_collision_frames_before",
    "physical_collision_frames_after",
    "fallback_periods_before",
    "fallback_periods_after",
    "gap_min_before_m",
    "gap_min_after_m",
    "gap_p10_before_m",
    "gap_p10_after_m",
    "gap_median_before_m",
    "gap_median_after_m",
    "completion_time_median_before_s",
    "completion_time_median_after_s",
)


def write_summary(rows: Sequence[GroupSummary], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "group": row.group,
                    "obstacle_kind": row.obstacle_kind,
                    "tasks": row.tasks,
                    "trackable_tasks": row.trackable,
                    "task_ok_before": row.ok_before,
                    "task_ok_after": row.ok_after,
                    "reached_goal_before": row.reached_before,
                    "reached_goal_after": row.reached_after,
                    "contact_tasks_before": row.contact_tasks_before,
                    "contact_tasks_after": row.contact_tasks_after,
                    "dynamic_contact_periods_before": row.contact_periods_before,
                    "dynamic_contact_periods_after": row.contact_periods_after,
                    "physical_collision_frames_before": row.collision_frames_before,
                    "physical_collision_frames_after": row.collision_frames_after,
                    "fallback_periods_before": row.fallback_before,
                    "fallback_periods_after": row.fallback_after,
                    "gap_min_before_m": row.gap_min_before,
                    "gap_min_after_m": row.gap_min_after,
                    "gap_p10_before_m": row.gap_p10_before,
                    "gap_p10_after_m": row.gap_p10_after,
                    "gap_median_before_m": row.gap_median_before,
                    "gap_median_after_m": row.gap_median_after,
                    "completion_time_median_before_s": row.time_median_before,
                    "completion_time_median_after_s": row.time_median_after,
                }
            )
    return path


def paired_transitions(pairs: Sequence[Pair]) -> dict[str, int]:
    """配对转移（7.8 第 1 条：必须报配对差，不能只报各自中位数）。

    只统计**真正执行过**的任务：``skipped_untrackable``（周期数为 0）单列，
    它前后都不可能"成功"，混进转移会伪造出 24 个 "kept_fail"。
    """
    pool = [p for p in pairs if p.obstacle_kind != "none"]
    ran = [p for p in pool if p.executable_before or p.executable_after]
    counts = {
        "skipped_untrackable": sum(1 for p in pool if not p.executable_before),
        "kept_ok": sum(1 for p in ran if p.task_ok_before and p.task_ok_after),
        "ok_to_fail": sum(1 for p in ran if p.task_ok_before and not p.task_ok_after),
        "fail_to_ok": sum(1 for p in ran if not p.task_ok_before and p.task_ok_after),
        "kept_fail": sum(1 for p in ran if not p.task_ok_before and not p.task_ok_after),
        "reached_then_timeout": sum(
            1
            for p in ran
            if p.reached_before and not p.reached_after and p.reason_after == "timeout"
        ),
        # 完成口径（reached_goal）与 task_ok 在 2026-10-31 之后**等价**：
        # success = 到达 ∧ 无物理违反，``timeout`` 不再算成功（成功率 = 到达率）。
        "reached_then_not_reached": sum(
            1 for p in ran if p.reached_before and not p.reached_after
        ),
        "not_reached_then_reached": sum(
            1 for p in ran if not p.reached_before and p.reached_after
        ),
        "new_dynamic_contact": sum(
            1
            for p in ran
            if p.contact_periods_before == 0 and p.contact_periods_after > 0
        ),
        "contact_removed": sum(
            1
            for p in ran
            if p.contact_periods_before > 0 and p.contact_periods_after == 0
        ),
    }
    deltas = _finite([p.gap_delta for p in ran])
    counts["gap_delta_median_mm"] = (
        int(round(float(np.median(deltas)) * 1000)) if deltas.size else 0
    )
    counts["gap_increased"] = int(np.sum(deltas > 1e-9)) if deltas.size else 0
    counts["gap_unchanged"] = int(np.sum(np.abs(deltas) <= 1e-9)) if deltas.size else 0
    counts["gap_decreased"] = int(np.sum(deltas < -1e-9)) if deltas.size else 0
    return counts


def format_summary(rows: Sequence[GroupSummary]) -> str:
    header = (
        f"{'档位':<8}{'障碍':<12}{'任务(可跟踪)':>14}{'到达前/后':>11}"
        f"{'接触任务前/后':>13}{'接触帧前/后':>13}"
        f"{'兜底周期前/后':>15}{'圆心距min前/后 [m]':>21}{'圆心距p50前/后':>17}"
    )
    lines = [header, "-" * len(header)]
    label = {
        "all": "全部",
        "head_on": "迎面",
        "crossing": "横穿",
        "same_direction": "同向慢速",
    }
    for row in rows:
        lines.append(
            f"{row.group:<8}{label.get(row.obstacle_kind, row.obstacle_kind):<12}"
            f"{f'{row.tasks}({row.trackable})':>14}"
            f"{row.reached_before:>5}/{row.reached_after:<5}"
            f"{row.contact_tasks_before:>6}/{row.contact_tasks_after:<6}"
            f"{row.contact_periods_before:>6}/{row.contact_periods_after:<6}"
            f"{row.fallback_before:>7}/{row.fallback_after:<7}"
            f"{row.gap_min_before:>10}/{row.gap_min_after:<10}"
            f"{row.gap_median_before:>8}/{row.gap_median_after:<8}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------


def plot_comparison(
    pairs: Sequence[Pair],
    out_path: Path,
    *,
    gap_label: str,
    contact_gap: float = 0.88,
) -> Path:
    """哑铃图 + 直方图：每个任务的圆心距最小值（执行口径）修正前后对比。"""
    _setup_fonts()
    pool = [
        p
        for p in pairs
        if p.obstacle_kind != "none"
        and math.isfinite(p.gap_before)
        and math.isfinite(p.gap_after)
    ]
    pool.sort(key=lambda p: (p.grade, p.obstacle_kind, p.gap_after))
    figure, axes = plt.subplots(
        1, 2, figsize=(14.5, 7.6), constrained_layout=True,
        gridspec_kw={"width_ratios": [2.1, 1.0]},
    )
    axis = axes[0]
    if pool:
        positions = np.arange(len(pool))
        axis.vlines(
            positions,
            [p.gap_before for p in pool],
            [p.gap_after for p in pool],
            color="0.65",
            linewidth=1.4,
            zorder=1,
        )
        axis.scatter(
            positions, [p.gap_before for p in pool], s=42, color=COLOR_BEFORE,
            label="修正前（阈值 = required_gap(δ) + r_o）", zorder=3,
        )
        axis.scatter(
            positions, [p.gap_after for p in pool], s=42, color=COLOR_AFTER,
            marker="D", label="修正后（阈值 = body_radius + r_o + required_gap(δ)）",
            zorder=3,
        )
        axis.axhline(
            contact_gap, color=COLOR_CONTACT, linestyle="--", linewidth=1.8,
            label=f"两圆不接触门槛 = {contact_gap:.2f} m",
        )
        axis.axhspan(0.0, contact_gap, color=COLOR_CONTACT, alpha=0.08)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [f"{p.scenario_id.replace('houseexpo_', '')}\n{p.grade[:3]}/"
             f"{ {'head_on': '迎面', 'crossing': '横穿', 'same_direction': '同向'}.get(p.obstacle_kind, p.obstacle_kind) }"
             for p in pool],
            fontsize=6.5, rotation=90,
        )
    axis.set_ylabel(f"每个任务的圆心距最小值 [{gap_label}]")
    axis.set_title("配对对比：圆心距最小值（虚线以下 = 两圆重叠）", fontsize=11)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left", fontsize=8.5)

    axis = axes[1]
    if pool:
        bins = np.linspace(0.0, max(2.6, max(p.gap_after for p in pool) + 0.1), 28)
        axis.hist(
            [p.gap_before for p in pool], bins=bins, color=COLOR_BEFORE, alpha=0.55,
            label="修正前",
        )
        axis.hist(
            [p.gap_after for p in pool], bins=bins, color=COLOR_AFTER, alpha=0.55,
            label="修正后",
        )
        axis.axvline(contact_gap, color=COLOR_CONTACT, linestyle="--", linewidth=1.8)
        axis.axvspan(0.0, contact_gap, color=COLOR_CONTACT, alpha=0.08)
        axis.annotate(
            f"两圆重叠区\n< {contact_gap:.2f} m", xy=(contact_gap / 2, 0.5),
            xytext=(contact_gap / 2, 0.5), xycoords=("data", "axes fraction"),
            ha="center", fontsize=9, color=COLOR_CONTACT,
        )
    axis.set_xlabel("圆心距最小值 [m]")
    axis.set_ylabel("任务数")
    axis.set_title("分布对比", fontsize=11)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.dwa_gapfix_compare",
        description="动态间距判据修正前后对比（配对）",
    )
    parser.add_argument("--before", default=str(PROJECT_ROOT / "docs" / "dwa_metrics_before.csv"))
    parser.add_argument("--after", default=str(PROJECT_ROOT / "docs" / "dwa_metrics.csv"))
    parser.add_argument(
        "--csv", default=str(PROJECT_ROOT / "docs" / "dwa_gapfix_compare.csv")
    )
    parser.add_argument(
        "--figure", default=str(PROJECT_ROOT / "docs" / "dwa" / "gapfix_before_after.png")
    )
    parser.add_argument(
        "--gap-column",
        default="min_gap_realized_m",
        help="对比用的圆心距列；旧 CSV 无该列时退回 min_gap_dynamic_m",
    )
    parser.add_argument(
        "--contact-gap",
        type=float,
        default=0.88,
        help="两圆不接触门槛 body_radius + r_o [m]（CSV 有 contact_gap_m 列时以列为准）",
    )
    parser.add_argument("--no-figure", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    before = read_metrics(Path(args.before))
    after = read_metrics(Path(args.after))
    gap_column = args.gap_column
    if not any(gap_column in row for row in before.values()):
        print(
            f"[note] 修正前 CSV 没有 {gap_column} 列，退回 min_gap_dynamic_m（预测口径）"
        )
        gap_column = "min_gap_dynamic_m"
    pairs = build_pairs(before, after, gap_column=gap_column)
    rows = summarise(pairs)
    transitions = paired_transitions(pairs)

    print(f"配对任务数 = {len(pairs)}（含无障碍任务）")
    print(
        f"圆心距口径 = {gap_column}"
        f"{'（执行口径：逐周期实测）' if gap_column.startswith('min_gap_realized') else '（预测口径）'}"
    )
    print("\n按档位 × 障碍类型（前 / 后）")
    print(format_summary(rows))
    print("\n配对转移（只统计带动态障碍的任务）")
    for name, value in transitions.items():
        print(f"  {name:<22}{value}")
    path = write_summary(rows, Path(args.csv))
    print(f"\n对比表: {path}")
    if not args.no_figure:
        # 门槛必须取**带动态障碍**的那一行：无障碍任务的 contact_gap_m = body_radius
        # （0.58），不是 body_radius + r_o（0.88），拿错会把门槛画低 0.30 m。
        contact = next(
            (
                _number(row, "contact_gap_m")
                for row in after.values()
                if row.get("obstacle_kind") != "none"
                and _number(row, "contact_gap_m") < float("inf")
            ),
            args.contact_gap,
        )
        figure = plot_comparison(
            pairs,
            Path(args.figure),
            gap_label=gap_column,
            contact_gap=float(contact),
        )
        print(f"对比图: {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
