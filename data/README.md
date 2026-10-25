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
