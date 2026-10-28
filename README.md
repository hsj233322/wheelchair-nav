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
uv run python -m env.screen_floorplans                    # 筛查 260 个候选 -> docs/floorplan_screening.csv
uv run python -m env.build_scenarios                      # 分档采样 -> config/scenarios/*.yaml + 窄门组
uv run python -m analysis.dmin_sweep                      # d_min 四档扫描 -> docs/dmin_sweep.csv
uv run python -m analysis.preview config/scenarios/houseexpo_*.yaml   # 出图 -> docs/preview/
uv run python -m analysis.preview config/scenarios/houseexpo_5427f9b0.yaml \
    --algo configs/dmin_sweep/dmin_110.yaml               # 指定档位出图（文件名带 __dmin_110 后缀）
```

场景质量档案汇总在 `docs/scene_metadata.csv`（每个场景一行：路径长度、房间标签变化、
房间数、连通分量数、不可达占比、clearance 分布、最宽瓶颈、地形占比、难度档）。

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
    scenario_selection.yaml# 场景选取（按难度档指定要固化的户型，依据筛查 CSV）
    scenarios/             # 第 2 层：house_01（手写）+ houseexpo_*（真实户型）+ narrowdoor_*（窄门）
    configs/frozen_A.yaml  # 第 3 层：算法（唯一允许变化的层）
    configs/dmin_sweep/    # d_min 四档（0.58/0.70/0.90/1.10），只改 astar.d_min
  core/                    # 纯函数，不 import config / runner
    astar.py               # 约束化 A*（离障 + 能耗预算 + 曲率窗口）
    energy.py              # 滚阻 / 坡度 / 惯性 / 传动效率 / 电池内阻
    metrics.py             # 指标口径
  env/                     # 公开数据集 -> core 能吃的数组
    floorplan.py           # HouseExpo JSON -> 占据栅格 + 地形标签（纯函数）
    scene_quality.py       # 房间分割 / 标签变化 / clearance 硬指标 / 难度分档（纯函数）
    screen_floorplans.py   # 批量筛查户型能否跨房间 -> docs/floorplan_screening.csv
    build_scenarios.py     # 户型 -> 冻结场景 YAML（分档采样 + A* 验证 + 质量档案 + 内容哈希）
  runner/
    config.py              # 三层加载、严格校验、config hash、场景哈希
    experiment.py          # 一个 spec -> 一批 run -> 落盘
  analysis/
    preview.py             # 一条命令出图：栅格 + 物理半径 + 策略裕度 + A* 路径 + 地形
    dmin_sweep.py          # d_min 档位扫描 -> docs/dmin_sweep.csv + 对比表
  data/                    # HouseExpo 候选池（不进 git；来源与取样规则见 data/README.md）
  docs/preview/            # 里程碑验收图
  docs/scene_metadata.csv  # 场景质量元数据表
  docs/dmin_sweep.csv      # d_min 扫描逐场景明细
  docs/floorplan_screening.csv  # 260 个候选户型的筛查结果
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
- **HouseExpo 轮廓与「墙 / 门洞」的来源（2026-10-25 更正）**：`verts` 是**自由区轮廓**，
  不是简单的房间外框。它把每道内墙的两侧各描一遍，于是内墙在填充结果里表现为宽约
  **0.12 m** 的狭缝（实测 7e80c5f4：2135 条狭缝，p25 = 中位 = p75 = 0.120 m），
  **门洞就是狭缝中断处**。所以预览图里的内墙与门洞**来自 HouseExpo 原始数据**，
  转换代码没有推断它们——`env/floorplan.py` 只做一件事：扫描线填充多边形，
  单元中心在内部即自由。栅格面积与轮廓 shoelace 面积误差 < 2%
  （`tests/test_scenarios_registry.py` 有校验）。
  数据里没有独立的「墙/门」对象列表，所以下面这些量是**我们的几何推断**：
  * **房间**：`clearance >= core_clearance_m`（默认 0.9 m）作房间核心，其余自由单元
    归属最近核心（等价于对距离场做 watershed）。门洞比房间窄，因此在核心之间断开；
  * **`room_label_changes`**：路径上**房间标签**变化次数（连续 <3 格的短游程会被合并）。
    **它不等于门洞个数**：走廊若比 0.9 m 窄，走廊内没有核心，会被 watershed 按最近核心
    切成若干段。实测 `houseexpo_5427f9b0`：7 次标签变化里有 5 次发生在同一条 1.50 m
    宽的走廊上（该走廊 clearance 恒为 0.750 m < 0.9 m）。论文里引用时必须按这个口径写；
  * **`interface_widths_m` / `min_interface_width_m`**：每次标签变化处分界面上的
    `2 x max(clearance)`，同样可能包含走廊切段，只作参考。
- **不依赖房间分割的硬指标**（推荐作为论文主指标）：
  * `clearance_min_m` / `clearance_p10_m` / `clearance_median_m`：路径 clearance 的最小值 /
    10% 分位 / 中位数；
  * `clearance_below_090_ratio`：路径上 `clearance < clearance_threshold_m`（默认 0.9 m）
    的**路段长度占比**（按单元间长度加权）；
  * `bottleneck_width_m`：起点到终点的**最宽瓶颈**宽度（对 clearance 做二分，
    最大连通阈值 x2）。这是"无论怎么绕都必须挤过的最窄通道"，与 A* 走哪条路、
    与房间分割都无关；
  * `bottleneck_crossings` / `bottleneck_min_width_m`：沿路径 clearance 局部极小的个数
    （`scipy.signal.find_peaks` + 显著度）与其中最窄宽度。
- **参数归属（2026-10-25 解耦）**：场景层只声明**物理**外接圆半径
  `map.inflation_m = platform.body_radius`（0.58 m，用于地图膨胀与碰撞判定）；
  **策略**安全半径 `algo.astar.d_min`（默认 0.70 m = 0.58 + 0.12 裕度）是算法层参数，
  也是实验变量，档位配置在 `config/configs/dmin_sweep/`。跨层校验只要求
  `d_min >= body_radius`（以及 `d_min >= map.inflation_m`）；调大 d_min 不再被场景拖住。
  A* 的 clearance 硬约束在数学上等价于按该半径膨胀栅格，所以场景里**不另存膨胀栅格**。
  预览图里深红带是物理半径、橙色带是物理半径到 d_min 之间的策略裕度。
- **地形标签是假设**：HouseExpo 只有几何、没有地面材质。材质按房间类型映射
  （厨房/卫生间 -> 瓷砖，卧室/客厅 -> 木地板，走廊/门厅/未标注 -> 水泥），规则写在
  `config/terrain_rules.yaml`，场景里带 `terrain.note` 声明。**这不是实测数据**，
  论文中必须按假设表述，并配合 `c_rr` 敏感性分析使用。
- **场景难度分档**（`env/scene_quality.py::grade_scene`）：**只看 `bottleneck_width_m`**
  —— 起终点之间"最宽瓶颈"的宽度（对 clearance 二分，取最大连通阈值 x2）。它与 A* 走哪条路
  无关、与 `algo.astar.d_min` 无关，所以同一场景在任何档位下档位标签都一样，场景表不会自相矛盾。
  阈值依据：
  * `hard`：`< 1.6 m`。1.40 m = 2 x 基准 d_min(0.70) 是最小可通行宽度，余量不足 0.2 m（两格）算难；
  * `medium`：`1.6 ~ 2.0 m`；
  * `easy`：`>= 2.0 m`，余量 >= 0.6 m（约 1.5 x 车身半径），通道基本不构成约束。
  参考量：1.16 m = 2 x body_radius（物理最小）、1.40 m = 2 x 基准 d_min（策略最小）。
  `room_label_changes` / `bottleneck_crossings` 降级为**描述性统计（路径相关）**，不参与分档。
  **分档对象**：档位标签描述的是**冻结任务**（场景自带的 start/goal，已被 content_hash 覆盖）
  的瓶颈宽度；同一栋房子不同端点对的瓶颈可以不同，所以重新采样出来的任务会**单独复算档位**
  （必须落在同一档，否则丢弃）。因此"场景档位"与"某档是否存在任务"是两个问题，
  后者要看 d_min 扫描的重采样结果（见下文）。
  采样规则：easy 在同一房间内采样且路径 ≥4 m，medium/hard 在不同房间之间采样且路径 ≥8 m，
  全部用 `core.astar` 在基准 d_min 下验证有解，并把结果按档位复核（`bottleneck_width_m` 定档），
  不达标就跳过（`config/scenario_selection.yaml`，每档多备候选）。
- **窄门组**：`narrowdoor_w080/w100/w120` 是受控门洞（0.8/1.0/1.2 m）的**合成**场景，
  供通过性实验使用。它们按 `width/2 - 一个单元` 的安全半径验证有解，并分开记录两个口径：
  `feasible_at_physical_radius`（门宽 ≥ 2 x 0.58 m）与 `feasible_at_base_d_min`
  （门宽 ≥ 2 x 0.70 m）。其中 1.2 m 门洞对物理半径够、对基准 d_min 不够——这正是
  通过性实验要测的边界，不是 bug。
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
- **HouseExpo 的内墙与门洞在数据里（见「关键约定」的更正），但都是轮廓几何**：
  没有独立的墙/门对象、没有门扇、没有房间多边形（只有房间包围盒）。
  房间分割、`room_label_changes`、分界面宽度都是我们按固定参数推断的，参数与规则写在
  `env/scene_quality.py`，换参数会让 `quality` 数值变化（场景内容哈希也会变，需重跑生成器）。
  `clearance_*` 与 `bottleneck_width_m` 不受分割影响，建议作为论文主指标。
- **基准 d_min = 0.70 m 把多数户型切成孤岛**：实测 260 个候选里，按 0.70 m 侵蚀后仍有
  ≥8 m 跨房间路径的只有 44 个。原因是 HouseExpo 门洞多在 1.4~1.6 m，卡在 2 × 0.70 m
  的门槛上；当前 20 个户型场景就是从这些候选里按瓶颈宽度分档选出的。
  解耦后这条限制只作用于**基准档**：`d_min = 0.58 m` 时门槛降到 1.16 m，可用范围更大。
- **"该档不可行"必须区分两种原因**（2026-10-25 修正）：沿用场景冻结的起终点时，
  高档位失败可能只是"冻结任务在新档不合法"；要判断"该档是否存在任务"，
  必须在 `clearance >= d_min` 的可行域内、同一连通分量内**重新采样起终点**
  （`env.scene_quality.sample_task_cells`，保持原难度档结构）。
  `docs/dmin_sweep.csv` 同时给出固定起终点（`frozen_*`）与重采样（`resampled_*`）两套结果。
- 失败 run 的指标全 0，`success=0` 与 `reason` 是唯一区分；`constraint_violation_rate`
  只在成功 run 上有意义。
- 未实现 `jerk` / 横向加速度的**硬约束**校验；它们目前只是报告量，等局部层落地。
- 参考轨迹恒速，jerk 恒为 0：**此阶段不得产出任何平顺性结论**。

## 场景分档统计（2026-10-25 修正后，20 个户型场景）

| 档位 | 个数 | 冻结任务瓶颈 W [m] | 0.90 档重采样任务 W [m] | 路径中位 [m] | 能耗中位 [J] |
|---|---|---|---|---|---|
| easy | 8 | 2.40 ~ 6.70 | 2.10 ~ 7.30（8 个都有） | 8.67 | 355.6 |
| medium | 5 | 1.62 ~ 1.71 | 1.90（4 个有任务） | 10.64 | 469.4 |
| hard | 7 | 1.46 ~ 1.50 | 无任务（1.5 < 2 x 0.90） | 11.69 | 459.3 |
| 窄门组 | 3 | 0.75 / 0.95 / 1.15 | 通过性实验用，不参与三档 | 6.05 | – |

**分档对象说明（回答"medium 瓶颈 1.62~1.71 与 0.90 档 medium 4/5 可行是否矛盾"）**：
不矛盾，两者是**不同的端点对**。

* 场景的档位标签是**冻结任务**（场景自带的 start/goal，已被 content_hash 覆盖）的属性：
  `quality.bottleneck_width_m` 就是这对起终点之间的最宽瓶颈；
* d_min 扫描里"重新采样"出来的是**另一个任务**，它有自己的瓶颈宽度
  （`resampled_bottleneck_width_m`）。同一栋房子里不同房间对之间的最宽连通宽度不同，
  所以 medium 场景冻结任务 1.62~1.71 m、而 0.90 档新采样的任务 1.90 m，可以同时成立；
  1.90 m 仍落在 medium 档（1.6~2.0），且 ≥ 2 x 0.90 = 1.80 m，因此在 0.90 档可行。
* 结论口径：**"场景档位"描述冻结任务；"某档是否存在任务"要看重采样**。
  核对过 `docs/dmin_sweep.csv` 用的是重新固化后的这 20 个场景（`tier` 与
  `frozen_bottleneck_width_m` 与 `docs/scene_metadata.csv` 逐字段一致）。

## d_min 参数扫描（第 3 章 Pareto 轴）

```bash
uv run python -m analysis.dmin_sweep --max-samples 24   # 4 档 x 20 场景 x 两个口径
```

档位配置：`config/configs/dmin_sweep/dmin_{058,070,090,110}.yaml`（只改 `astar.d_min`）。
每行两组列：**固定起终点**（沿用场景冻结的任务）与**按档位重新采样**
（在该档可行域内同分量重采样、保持原难度档结构）。中位数只在可行任务上统计：

| d_min [m] | 固定:可行 | 固定:路径中位 [m] | 固定:能耗中位 [J] | 固定:离障中位 [m] | 重采样:成功率 | 重采样:可行任务（95% Wilson CI） | 重采样:路径中位 [m] | 重采样:能耗中位 [J] | 重采样:离障中位 [m] |
|---|---|---|---|---|---|---|---|---|---|
| 0.58 | 20/20 | 10.97 | 442.4 | 0.691 | 20/20 | 20/20（100%，CI 84–100%） | 11.72 | 440.1 | 0.582 |
| 0.70 | 20/20 | 11.09 | 450.5 | 0.740 | 20/20 | 20/20（100%，CI 84–100%） | 12.97 | 519.5 | 0.712 |
| 0.90 | 8/20 | 8.67 | 355.6 | 1.764 | 12/20 | **12/20（60%，CI 39–78%）** | 8.22 | 352.1 | 1.250 |
| 1.10 | 8/20 | 8.67 | 355.6 | 1.764 | 8/20 | 8/20（40%，CI 22–61%） | 7.51 | 282.4 | 1.444 |

**确定性指标（与采样预算无关，建议作为论文主指标）**：

| d_min [m] | 可行面积占比中位（固定锚点） | 可行面积占比中位（重采样锚点） |
|---|---|---|
| 0.58 | 0.593 | 0.593 |
| 0.70 | 0.542 | 0.542 |
| 0.90 | 0.195 | 0.274 |
| 1.10 | 0.102 | 0.000 |

`feasible_area_ratio`（`env/scene_quality.py::feasible_area_ratio`）的定义：
**场景自由区（`clearance > 0`）中，满足 `clearance >= d_min` 且与锚点（起点）同属一个
连通分量的单元面积占比**（单元等面积，所以就是面积之比）。锚点自身不满足门槛时返回 0，
表示"这个任务在该档不可用"。它不依赖采样次数，因此比"可行任务数"更适合写进论文；
中位数里的 0（例如 1.10 档重采样锚点列）表示过半场景在该档没有任务。

**"可行任务数"的使用限制**：它是 20 个场景上的比例量，受**采样预算**（24 次/场景）与
随机种子影响。表里给出 95% Wilson 得分区间，采样失败会**低估**真实可行性，
所以它是下界性质的证据；写论文时与 `feasible_area_ratio` 搭配使用。

三点解读：

1. **0.58 与 0.70 之间：路径长度/能耗不变，只有离障变**（最小离障 0.740 -> 0.691 -> 0.582）。
   8 邻域栅格上贴着约束边界走不额外付出长度代价，所以放宽 d_min 的收益是**可行域变大**，
   不是路径变短。这一条本身就是第 3 章可写的结论。
2. **0.90 档：固定起终点 8/20，重新采样后 12/20（CI 39–78%）**。原来的"0.90 断崖"
   有一半是假象：4 个 medium 场景的冻结任务在 0.90 下不可用，但同一分量内重新采样后
   仍有中档任务（W = 1.90 m）。剩下 8 个 hard 场景（1.46~1.50 m）在该档**物理上**不存在任务。
   确定性指标同样显示收缩：可行面积占比中位从 0.542（0.70）降到 0.195（0.90）。
3. 1.10 档只剩 8 个 easy 场景（W ≥ 2.10 m > 2 x 1.10 = 2.20 m 附近），与物理预期一致；
   不同档位的路径长/能耗中位数统计的是不同任务集合，**不可直接横向比较**。

重新采样后的 hard 场景预览图：`docs/preview/houseexpo_5427f9b0__resampled_dmin070.png`
（起点/终点取自 `docs/dmin_sweep.csv` 的 `resampled_start/goal`，路径 18.54 m、瓶颈 1.50 m）；
0.90 档被重新采样救回来的中档场景见 `docs/preview/houseexpo_1241c62f__resampled_dmin090.png`。

## 下一步

里程碑顺序（2026-10-25 调整，见交接文档 §7）：

```
1. env/floorplan.py             HouseExpo -> 占据栅格 + 地形分区        [已完成 2026-10-25]
2. env/scene_quality.py 等      场景质量：元数据 + 难度分档 + 窄门组     [已完成 2026-10-25]
3. 平台/约束解耦 + d_min 扫描    物理半径与策略裕度分层 + 4 档扫描        [已完成 2026-10-25]
4. 场景口径二次修正            按档重采样 + 瓶颈宽度分档（本次）           [已完成 2026-10-25]
5. core/bspline.py              RDP + 曲率约束 B 样条                    <- 当前
6. core/dwa.py                  绕行判据 + 舒适性代价 + 曲率限速
7. env/pedestrians.py           ETH/UCY -> 逐帧动态障碍
8. analysis/                    tables / figures，30 次重复 + Wilcoxon / Fisher
```

把 ETH/UCY 从第 2 位挪到第 4 位，理由：**行人轨迹是 DWA 的输入**。DWA 还没有
落地时接入轨迹，它无法影响任何规划结果，只能当装饰；等 DWA 有了动态障碍接口，
行人轨迹才能验证"绕行可行性判据"这类主张。先做 B 样条是因为它接着全局层的
曲率约束，链路上紧挨着已完成的 A*。

每个里程碑都要求出图，验收方式是看图，不看代码。
