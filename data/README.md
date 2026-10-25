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

## 本目录的取样子集（16 个户型）

选取规则（可复现，不挑图）：

1. 官方演示列表 `HouseExpo/map_id_10.txt` 的全部 10 个 id；
2. 官方训练列表 `experiments/map_id_train.txt` 按文件顺序的前 6 个 id。

共 16 个，占原始数据集的 0.05%。文件为 `data/houseexpo/<id>.json`。

## 重新下载

```bash
curl -sSL -o /tmp/json.tar.gz \
  https://raw.githubusercontent.com/TeaganLi/HouseExpo/master/HouseExpo/json.tar.gz
IDS=$(cat map_id_10.txt; head -6 map_id_train.txt)
tar -xzf /tmp/json.tar.gz -C data/houseexpo --strip-components=1 \
  $(for id in $IDS; do printf 'json/%s.json ' "$id"; done)
```

`data/houseexpo/` 已加入 `.gitignore`，不进版本库；本 README 进版本库，
以便任何人按上面的规则复原同一份子集。

## 数据格式（HouseExpo 单文件）

```json
{
  "id": "7e80c5f4...",
  "room_num": 6,
  "bbox": {"min": [x1, y1], "max": [x2, y2]},
  "verts": [[x, y], ...],                       // 户型外轮廓多边形，单位 m
  "room_category": {"Kitchen": [[x1,y1,x2,y2]], ...}   // 房间包围盒
}
```

只有外轮廓 + 房间包围盒，**没有墙厚、门洞几何与地面材质**；因此：

* 占据栅格 = 轮廓多边形内为自由，之外（含地图边界）为障碍；
* 地形标签只能按房间类型映射（`config/terrain_rules.yaml`），并声明为假设。

## 相关代码

* 转换：`env/floorplan.py`（纯函数）
* 场景固化：`env/build_scenarios.py` -> `config/scenarios/houseexpo_*.yaml`
* 可视化：`analysis/preview.py`
