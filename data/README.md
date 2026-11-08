# data/ ：外部数据集（体积大，不进 git）

本目录只放**原始数据**与取得方式的记录；转换代码在 `env/` 下，
转换结果落在 `config/scenarios/` 与 `docs/`（这些进 git）。

---

# data/houseexpo：HouseExpo 取样子集

## 来源与版本

| 项 | 值 |
|---|---|
| 数据集 | HouseExpo: A Large-scale 2D Indoor Layout Dataset（IROS 2020） |
| 仓库 | https://github.com/TeaganLi/HouseExpo |
| 论文 | arXiv:1903.09845 |
| 许可证 | MIT（仓库 LICENSE） |
| 版本 | `master` @ commit `45e2b2505f6ea1fe49c0203f14efb7ce20b94e7c` |
| 取数日期 | 2026-10-25 |
| 原始文件 | `HouseExpo/json.tar.gz`（25.6 MB，35 126 个户型 JSON） |

## 本目录的候选池（260 个户型）与最终选取

候选池规则（可复现，不挑图）：

1. 官方演示列表 `HouseExpo/map_id_10.txt` 的全部 10 个 id；
2. 官方训练列表 `experiments/map_id_train.txt` 按文件顺序的前 250 个 id。

共 260 个，占原始数据集的 0.7%。文件为 `data/houseexpo/<id>.json`。

候选池不等于最终场景集。流程是：

```
env/screen_floorplans.py   260 个候选 -> docs/floorplan_screening.csv（跨房间距离/跨门次数/门洞宽度）
config/scenario_selection.yaml  按难度档挑户型（easy 4 / medium 6 / hard 4）
env/build_scenarios.py     分档采样起终点 -> config/scenarios/houseexpo_*.yaml（+ narrowdoor_* 合成窄门组）
```

实测结论（2026-10-25）：260 个候选里，按 0.70 m 安全半径侵蚀后仍有 ≥8 m 跨房间路径的
只有 44 个，其中跨 ≥3 个门洞的只有 7 个。原因是 HouseExpo 的门洞多在 1.4~1.6 m，
正好卡在 `2 × 0.70 m` 的可通行门槛上。

## 重新下载

```bash
curl -sSL -o /tmp/json.tar.gz \
  https://raw.githubusercontent.com/TeaganLi/HouseExpo/master/HouseExpo/json.tar.gz
IDS=$(cat map_id_10.txt; head -250 map_id_train.txt)
tar -xzf /tmp/json.tar.gz -C data/houseexpo --strip-components=1 \
  $(for id in $IDS; do printf 'json/%s.json ' "$id"; done)
```

`data/houseexpo/` 已加入 `.gitignore`，不进版本库；本 README 进版本库，
以便任何人按上面的规则复原同一份候选池。

## 数据格式（HouseExpo 单文件）

```json
{
  "id": "7e80c5f4...",
  "room_num": 6,
  "bbox": {"min": [x1, y1], "max": [x2, y2]},
  "verts": [[x, y], ...],                       // 自由区轮廓多边形，单位 m
  "room_category": {"Kitchen": [[x1,y1,x2,y2]], ...}   // 房间包围盒
}
```

### 内墙与门洞在哪里（重要）

`verts` 是**自由区轮廓**：它沿着每道内墙的两侧各走一遍，因此在填充结果里
内墙表现为宽约 **0.12 m** 的狭缝，**门洞是狭缝中断的地方**。实测（`7e80c5f4`：
101 个顶点、2135 条狭缝，狭缝宽 p25 = 中位 = p75 = 0.120 m）说明这是数据固有的
几何，不是转换代码推断出来的——`env/floorplan.py` 只做「单元中心在多边形内即自由」。

数据里**没有**：独立墙/门对象、门扇、房间多边形（只有房间包围盒）、地面材质。
因此房间分割、跨门次数、门洞宽度必须由几何推断，规则与参数见
`env/scene_quality.py` 与仓库 README 的「关键约定」。

## 相关代码

* 转换：`env/floorplan.py`（纯函数）
* 几何推断（房间/门洞/难度）：`env/scene_quality.py`（纯函数）
* 候选筛查：`env/screen_floorplans.py` -> `docs/floorplan_screening.csv`
* 场景固化：`env/build_scenarios.py` -> `config/scenarios/houseexpo_*.yaml`、`narrowdoor_*.yaml`
* 质量表：`docs/scene_metadata.csv`
* 可视化：`analysis/preview.py`

---

# data/eth_ucy：ETH/UCY 行人轨迹子集（2026-11-07 接入）

| 项 | 值 |
|---|---|
| 数据集 | ETH Walking Pedestrians (BIWI) + UCY Crowds |
| 上游仓库 | https://github.com/StanfordASL/Trajectron-plus-plus |
| 上游路径 | `experiments/pedestrians/raw/eth/{train,test}/`（这两个数据集的**原始 obsmat 文本**镜像） |
| 上游版本 | `master`（2026-11-07 取） |
| 许可证 | 数据版权归原作者（BIWI / ETH / UCY）；Trajectron++ 仓库为 MIT。**仅用于学术研究**，论文须引用原始数据集论文 |
| 取数日期 | 2026-11-07 |
| 取数方式 | 本机到 `github.com` / `raw.githubusercontent.com` 被网络策略拦截（curl 超时 / SSL 失败），改走 **jsDelivr CDN**：`https://cdn.jsdelivr.net/gh/StanfordASL/Trajectron-plus-plus@master/experiments/pedestrians/raw/eth/<path>` |

放进 `data/eth_ucy/` 的文件（**不进 git**）：

| 本地文件 | 上游文件 | 序列 | 行数 | 行人 | 局部包围盒 [m] |
|---|---|---|---|---|---|
| `eth.txt` | `test/biwi_eth.txt` | ETH | 5492 | 360 | 22.1 × 16.4 |
| `hotel.txt` | `train/biwi_hotel_train.txt` | Hotel | 4946 | 311 | 7.6 × 14.5 |
| `zara01.txt` | `train/crowds_zara01_train.txt` | Zara01 | 4307 | 125 | 15.6 × 12.7 |
| `zara02.txt` | `train/crowds_zara02_train.txt` | Zara02 | 7621 | 171 | 15.8 × 14.2 |
| `zara03.txt` | `train/crowds_zara03_train.txt` | Zara03 | 3708 | 106 | 15.9 × 12.8 |
| `students001.txt` | `train/students001_train.txt` | Students001 | 18353 | 375 | 15.9 × 14.1 |
| `students003.txt` | `train/students003_train.txt` | Students003 | 15641 | 364 | 15.6 × 14.0 |
| `uni_examples.txt` | `train/uni_examples_train.txt` | Univ | 2266 | 96 | 16.0 × 13.8 |

## 格式与两条必须声明的口径

1. **格式**：每行 `frame id x y`（空白分隔，无表头）；帧步长恒为 **10**，
   采样率 **2.5 fps（0.4 s）**。本框架重采样到 `dt = 0.1 s`（**加密**，不是抽稀），
   速度用重采样后的中心差分 + 0.5 s 滑动平均（直接对 0.4 s 原始帧差分会把标注
   噪声放大 4 倍）。
2. **坐标是局部场地坐标**：ETH/UCY 的坐标属于各自场地（校园 / 酒店大堂 / 火车站
   前街道），原点与朝向与 HouseExpo 户型**毫无关系**。因此把行人放进户型必须显式
   经过**锚点映射**（原点 / 旋转 / 缩放）：声明在
   `config/pedestrian_sequences.yaml::anchor`，由
   `env/pedestrians.py::fit_anchor` 依据**场景几何**（自由区最大内接矩形 +
   行人圆与墙的冲突率）确定到一个具体值，逐 (场景, 序列) 记录在
   `docs/pedestrian_anchor.csv`。`scale` 固定 **1.0**：缩放会连带改变步行速度，
   本框架不靠缩小数据去"凑"可用率。

## 复现下载

```bash
BASE=https://cdn.jsdelivr.net/gh/StanfordASL/Trajectron-plus-plus@master/experiments/pedestrians/raw/eth
mkdir -p data/eth_ucy
curl -sL "$BASE/test/biwi_eth.txt"             -o data/eth_ucy/eth.txt
curl -sL "$BASE/train/biwi_hotel_train.txt"    -o data/eth_ucy/hotel.txt
curl -sL "$BASE/train/crowds_zara01_train.txt" -o data/eth_ucy/zara01.txt
curl -sL "$BASE/train/crowds_zara02_train.txt" -o data/eth_ucy/zara02.txt
curl -sL "$BASE/train/crowds_zara03_train.txt" -o data/eth_ucy/zara03.txt
curl -sL "$BASE/train/students001_train.txt"   -o data/eth_ucy/students001.txt
curl -sL "$BASE/train/students003_train.txt"   -o data/eth_ucy/students003.txt
curl -sL "$BASE/train/uni_examples_train.txt"  -o data/eth_ucy/uni_examples.txt
```

## 引用（论文参考文献）

* A. Lerner, Y. Chrysanthou, D. Lischinski. *Crowds by Example.* Computer Graphics
  Forum (Eurographics), 2007. — **UCY 数据集的原始来源**。
* S. Pellegrini, A. Ess, K. Schindler, L. van Gool. *You'll Never Walk Alone: Modeling
  Social Behavior for Multi-target Tracking.* ICCV, 2009. — **ETH/BIWI 数据集的原始来源**。
* Trajectron++ / jsDelivr 只作为**文本镜像与下载通道**被引用，不是行人数据集的原始出处。

## 相关代码

* 读取 / 锚点 / 重采样 / 筛选 / 对接 DWA：`env/pedestrians.py`（纯函数）
* 锚点标定与可用性统计：`analysis/pedestrians.py` -> `docs/pedestrian_anchor.csv`、
  `docs/pedestrian_usability.csv`
* 动态层三类指标（主动碰撞 / 被动接触 / 未完成）：`docs/dwa_metrics_eth_ucy.csv`
* 可视化：`docs/pedestrians/*.png`
