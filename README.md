# wheelchair-nav

《能耗与舒适性约束下的智能轮椅动态路径规划方法研究》的可复现仿真平台骨架。

纯 Python + CPU，无 ROS、无实车。目标不是"再写一个 A*"，而是让论文里的每个数字都能追回
唯一一份配置、唯一一次运行。

## 快速开始

```bash
uv sync                 # 国内网络慢可用：uv sync --default-index https://pypi.tuna.tsinghua.edu.cn/simple
uv run pytest
uv run python -m runner.experiment configs/frozen_A.yaml scenarios/house_01.yaml
```

HouseExpo 真实户型场景（数据子集需先按 `data/README.md` 下载）：

```bash
uv run python -m env.build_scenarios                      # 户型 -> config/scenarios/houseexpo_*.yaml
uv run python -m analysis.preview config/scenarios/houseexpo_*.yaml   # 出图 -> docs/preview/
```

最后一条会在 `results/<run_id>/` 下产出：

| 文件 | 内容 |
|---|---|
| `config.json` | platform / scenario / algo 三层快照 + config hash + git commit + seed |
| `metrics.csv` | 首列为 config hash，每次重复一行 |
| `trajectory.npz` | t / xy / yaw / speed / yaw_rate / energy_j / clearance / collision |

`run_id` 形如 `frozen_A-03fba3b5-s42-20261022T144516Z`，时间戳只用于区分目录；
同一 seed 重跑，`metrics.csv` 除 `run_id` 外逐字段一致。

## 目录

```
wheelchair-nav/
  pyproject.toml
  config/
    platform.yaml          # 第 1 层：平台
    run.yaml               # 执行层：seed / repeats / out_dir / dt（不参与 hash）
    terrain_rules.yaml     # 地形标签规则（房间类型 -> 材质的假设）
    scenarios/             # 第 2 层：house_01（手写）+ houseexpo_*（真实户型，已固化）
    configs/frozen_A.yaml  # 第 3 层：算法（唯一允许变化的层）
  core/                    # 纯函数，不 import config / runner
    astar.py               # 约束化 A*（离障 + 能耗预算 + 曲率窗口）
    energy.py              # 滚阻 / 坡度 / 惯性 / 传动效率 / 电池内阻
    metrics.py             # 指标口径
  env/                     # 公开数据集 -> core 能吃的数组
    floorplan.py           # HouseExpo JSON -> 占据栅格 + 地形标签（纯函数）
    build_scenarios.py     # 户型 -> 冻结场景 YAML（采样起终点 + A* 验证 + 内容哈希）
  runner/
    config.py              # 三层加载、严格校验、config hash、场景哈希
    experiment.py          # 一个 spec -> 一批 run -> 落盘
  analysis/
    preview.py             # 一条命令出图：栅格 + 膨胀层 + 起终点 + A* 路径 + 地形
  data/                    # HouseExpo 取样子集（进退数据，不进 git；来源见 data/README.md）
  docs/preview/            # 里程碑验收图
  tests/
  results/                 # 每次运行一个目录（git 忽略）
```

## 参数三层 + 执行层

| 层 | 内容 | 能否作为实验变量 |
|---|---|---|
| platform | 质量、尺寸、最大速度/加速度、传动效率、辅助功率、电池参数 | 否 |
| scenario | 地图、墙体、路面滚阻、坡度 | 否 |
| algo | 能耗预算、离障距离、曲率上限、权重、时域 | **是，唯一可变层** |
| run | seed、repeats、out_dir、dt | 不是模型参数，只被记录 |

`runner/config.py` 会拒绝：

- 三层文件里的未知字段、重复 YAML 键、缺失必填字段；
- algo 文件里出现 platform/scenario 字段（例如 `mass`）——防止"偷偷改平台"；
- `d_min < body_radius`、`v_cruise > v_max` 这类跨层矛盾。

config hash 只覆盖 platform + scenario + algo。换 seed 重跑，hash 不变；这样
"配置"与"重复"是两个维度，符合第 5 章参数治理的叙事。

## 关键约定

- **坐标**：世界单位 m；数组第 0 行对应 y 最小侧，第 0 列对应 x 最小侧；单元中心为
  `origin + (i + 0.5) * resolution`。
- **栅格化（矩形墙）**：`map.walls` 的矩形与单元有任何重叠即判为障碍（保守），
  薄墙不会漏；户型轮廓 `map.polygons` 的规则见下面「HouseExpo 轮廓」。
- **离障距离**：单元中心到最近障碍物表面的距离；地图边界外视为障碍。
- **能耗记账**：全局层用稳态巡航（`accel=0`）估算每步电能，作为硬预算；
  下坡/制动不回收（保守）。电池内阻损耗按 `I²R` 计入。
- **曲率**：8 邻域单步转角只能是 45° 的倍数，直接套 `Δθ ≤ κ·ds` 会禁止一切转弯。
  因此 A* 用**曲率窗口**：`max_turn = kappa_max * kappa_window_m`（默认 1.5 × 0.6 =
  0.9 rad，允许 45°、禁止 90°）。精确曲率由后续 `core/bspline.py` 在平滑阶段保证，
  两处共用同一个 `kappa_max`。
- **平顺性指标**：以 jerk RMS + 平均绝对角速度为主；`total_turn_deg` 只作辅助，
  不能单独用来论证"更平顺"（旧论文的坑）。
- **参考轨迹**：当前按恒定巡航速度采样、朝向受 `w_max` 限速，因此 jerk 恒为 0、
  角速度是几何量。它是链路验证用的参考轨迹，不是控制器输出；真实速度/朝向平滑
  在 DWA + B 样条里程碑补上。
- **HouseExpo 轮廓**：`map.polygons` 用「单元中心是否在多边形内」判定自由/障碍
  （外轮廓是零厚度边界，不是有厚度的墙，所以不套 `walls` 的保守重叠规则）。
  扫描线填充的栅格面积与轮廓 shoelace 面积误差 < 2%（`tests/test_scenarios_registry.py` 有校验）。
- **膨胀半径**：`inflation_m = platform.body_radius + platform.inflation`（当前
  0.58 + 0.12 = 0.70 m），由 `env/build_scenarios.py` 算好并写进场景的
  `map.inflation_m`，不写死在任何代码里。runner 校验 `algo.astar.d_min >= inflation_m`。
  A* 的 clearance 硬约束在数学上等价于按该半径膨胀栅格，所以场景里**不另存一份膨胀栅格**，
  避免把安全半径重复计算两次；预览图里的红色带就是这个膨胀层。
- **地形标签是假设**：HouseExpo 只有几何、没有地面材质。材质按房间类型映射
  （厨房/卫生间 -> 瓷砖，卧室/客厅 -> 木地板，走廊/门厅/未标注 -> 水泥），规则写在
  `config/terrain_rules.yaml`，场景里带 `terrain.note` 声明。**这不是实测数据**，
  论文中必须按假设表述，并配合 `c_rr` 敏感性分析使用。
- **场景内容哈希**：每个场景带 `content_hash`（sha256，不含该字段本身），
  `runner/config.py` 加载时校验；改场景内容必须重算哈希（`env/build_scenarios.py` 自动重算）。

## 相对旧代码修掉的系统性问题

| 旧代码问题 | 这里的处理 |
|---|---|
| g 累计能量 J、h 返回长度 m，启发式失效 | g 与 h 都以米计，能耗只作为资源约束 |
| 参数硬编码在类内 | 全部来自三层 YAML，core 不读配置 |
| 转角 >135° 惩罚返回 0（掉头不罚） | 转角用硬上限，不存在"越界反而不罚" |
| `self.goal_x` 隐藏状态 | 状态显式放进函数参数与 `AStarParams` |
| 同一算法多副本 | 一个 `core/astar.py`，配置分文件冻结 |
| 单函数 166 行 | 每个函数单一职责，core 全部纯函数 |
| 旧数据因手工调参不可沿用 | 新框架全部重跑，种子与 hash 落盘 |

## 当前边界（故意没做）

- `core/bspline.py`、`core/dwa.py`、`env/pedestrians.py`、`analysis/tables.py` +
  `figures.py` 尚未实现；`frozen_A.yaml` 里的 `dwa` / `comfort` 字段先冻结占位。
- **HouseExpo 只有户型外轮廓与房间包围盒**：没有户内隔墙、没有门洞几何、没有地面材质。
  因此 `polygons` 决定可通行区域，`room_category` 只用于地形标签；"房间"目前不是
  通行性约束。这一点必须在论文里说明，并作为场景来源的已知简化。
- 失败 run 的指标全 0，`success=0` 与 `reason` 是唯一区分；`constraint_violation_rate`
  只在成功 run 上有意义。
- 未实现 `jerk` / 横向加速度的**硬约束**校验；它们目前只是报告量，等局部层落地。
- 参考轨迹恒速，jerk 恒为 0：**此阶段不得产出任何平顺性结论**。

## 下一步

里程碑顺序（2026-10-25 调整，见交接文档 §7）：

```
1. env/floorplan.py    HouseExpo -> 占据栅格 + 地形分区   [已完成 2026-10-25]
2. core/bspline.py     RDP + 曲率约束 B 样条               <- 当前
3. core/dwa.py         绕行判据 + 舒适性代价 + 曲率限速
4. env/pedestrians.py  ETH/UCY -> 逐帧动态障碍
5. analysis/           tables / figures，30 次重复 + Wilcoxon / Fisher
```

把 ETH/UCY 从第 2 位挪到第 4 位，理由：**行人轨迹是 DWA 的输入**。DWA 还没有
落地时接入轨迹，它无法影响任何规划结果，只能当装饰；等 DWA 有了动态障碍接口，
行人轨迹才能验证"绕行可行性判据"这类主张。先做 B 样条是因为它接着全局层的
曲率约束，链路上紧挨着已完成的 A*。

每个里程碑都要求出图，验收方式是看图，不看代码。
