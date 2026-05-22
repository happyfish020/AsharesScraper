# BK / TS / SW 现役依赖清单

## 目标

这份清单回答三个问题：

1. 哪些脚本还在直接读取 `BK` / 东方财富板块体系
2. 哪些链路已经可以切换到 `SW` / `cn_local_industry_*`
3. 哪些旧链路可以安全退役，至少不再作为 `V8 primary path`

结论先行：

- `BK` 仍然残留在一部分 `leader / monthly basic / compatibility` 链路里
- `SW2021 + cn_local_industry_* + cn_ga_*` 已经是当前 `V8` 主方向
- `cn_board_*` 不能立刻整体删除，但已经不应该再当 `V8` 主源

## 分类定义

### `BK_ACTIVE`

脚本仍然直接依赖 `BK%`、东方财富概念/板块、或 `cn_board_*` 的 `BK` 语义。

### `SW_READY`

脚本已经主要依赖 `SW2021`、`cn_local_industry_*`、`cn_board_industry_member_hist` 或 `cn_ga_*`，可以视作当前主路径。

### `LEGACY_ONLY`

脚本或表仍可保留作兼容/历史支持，但不应继续作为 `V8 primary path` 扩展。

## 总表

| Path | Current read path | Classification | Can switch to SW? | Recommended action | Notes |
| --- | --- | --- | --- | --- | --- |
| `docs/DDL/cn_market.cn_stock_leader_score_v1.sql` | `cn_board_member_map_d` + `BK%` or `801%.SI` | `BK_ACTIVE` | Partial | Keep short-term, plan migration | 2026-05-21 注释已写明 `BK%` 产出停止后扩到 `801%.SI` |
| `app/tools/sync_cn_stock_daily_basic_from_tushare.py` | `calendar_source=board-map` 时过滤 `sector_id LIKE 'BK%%'` | `BK_ACTIVE` | Yes | Replace `BK%%` calendar with price or SW map | 这是显式 BK 过滤 |
| `app/tools/sync_cn_stock_fundamental_monthly.py` | `calendar_source=board-map` 时过滤 `sector_id LIKE 'BK%%'` | `BK_ACTIVE` | Yes | Replace `BK%%` calendar with price or SW map | 月度补数仍有 BK 兼容逻辑 |
| `app/tasks/board_membership_refresh_task.py` | 刷 `concept` + `industry`，写 `cn_board_*` 后重建 `cn_board_member_map_d` | `LEGACY_ONLY` | Partial | Keep for board-history maintenance only | 仍是 board 资产维护链，不是 V8 主 alpha 源 |
| `app/tools/backfill_concept_history_from_tushare.py` | 写 `cn_board_concept_master` / `cn_board_concept_member_hist` | `LEGACY_ONLY` | No | Keep only if concept-history still needed | 服务 concept legacy 资产 |
| `app/tasks/rotation_sector_snapshot_task.py` | 依赖 `cn_board_member_map_d` 覆盖 | `LEGACY_ONLY` | Partial | Migrate later to local-industry / GA sources | 现在仍压在 board map 上 |
| `scripts/build_cn_stock_mainline_strength_daily.py` | 读 `cn_stock_leader_score_daily` + `cn_mainline_lifecycle_daily` + `cn_board_member_map_d` + `cn_local_industry_map_hist` + optional `cn_ga_mainline_radar_daily` | `TRANSITIONAL` | Yes | Remove direct `cn_board_member_map_d` dependency | 这是过渡态，不是纯 BK，但还没完全脱板块 |
| `scripts/build_local_industry_map_hist.py` | 优先从 `cn_board_member_map_d` 提取，再 fallback Tushare | `TRANSITIONAL` | Yes | Prefer direct SW member history over board map | 目前仍把 board map 当 source 1 |
| `data_pipeline/builders/mainline_strength_daily.py` | `cn_local_industry_map_hist` + `cn_local_industry_proxy_daily` | `SW_READY` | Already | Keep as V8 primary path | 已经是 local industry 主链 |
| `data_pipeline/builders/industry_proxy_daily.py` | `cn_local_industry_map_hist` / `cn_local_industry_proxy_daily` | `SW_READY` | Already | Keep as V8 primary path | 不再走 BK |
| `data_pipeline/builders/sw_industry_member_hist.py` | `SW2021` local history builder | `SW_READY` | Already | Keep as V8 primary path | 这是替代链路核心 |
| `app/tools/build_cn_stock_leader_sw_l1_latest_snap.py` | `cn_board_industry_member_hist` + `TUSHARE_SW2021_L1` | `SW_READY` | Already | Keep as SW-compatible snapshot path | 明确锚定 SW L1 |
| `data_fetcher/build_local_industry_proxy.py` | `cn_board_industry_member_hist`，并要求 `SW L1` coverage | `SW_READY` | Already | Keep as V8 primary path | 已经不是 BK 概念体系 |
| `data_fetcher/fetch_akshare_boards.py` | 东方财富行业/概念板块抓取 | `LEGACY_ONLY` | No | Keep only for board daily / concept inventory | 这是 AK/东财原始资产采集器 |
| `app/tools/rebuild_rotation_three_tables_from_map.py` | 读 `cn_board_concept_master` | `LEGACY_ONLY` | Partial | Freeze unless rotation keeps concept boards | 属于旧板块 rotation 资产 |

## 一、仍在读 BK 的脚本

这些脚本还存在明确 `BK` 读取或过滤逻辑。

| Path | Evidence | Risk | Recommendation |
| --- | --- | --- | --- |
| `docs/DDL/cn_market.cn_stock_leader_score_v1.sql` | 直接写了 `LIKE 'BK%' OR LIKE '801%.SI'` | taxonomy 混合 | 继续保留兼容，但不要当 V8 最终形态 |
| `app/tools/sync_cn_stock_daily_basic_from_tushare.py` | `calendar_source == "board-map"` 时 `sector_id LIKE 'BK%%'` | calendar 污染 / source expansion 污染 | 改成 `price` 或 `SW` calendar |
| `app/tools/sync_cn_stock_fundamental_monthly.py` | `calendar_source == "board-map"` 时 `sector_id LIKE 'BK%%'` | 月频基本面窗口被 BK 绑住 | 改成 `price` 或 `SW` calendar |

### 判断

- 这几条是最直接、最明确的 `BK` 现役依赖
- 它们不是“隐式 legacy”，而是代码里明写了 `BK%%`
- 这些地方最应该优先去 BK

## 二、已经可以切到 SW 的脚本

这些链路已经有可用的 `SW` / `local_industry` 替代路线，或者本身已经在用。

| Path | Current source | Status | Recommendation |
| --- | --- | --- | --- |
| `data_pipeline/builders/mainline_strength_daily.py` | `cn_local_industry_map_hist` + `cn_local_industry_proxy_daily` | Already SW-first | 维持主路径 |
| `data_pipeline/builders/industry_proxy_daily.py` | `cn_local_industry_*` | Already SW-first | 维持主路径 |
| `data_pipeline/builders/sw_industry_member_hist.py` | `SW2021` | Already SW-first | 维持主路径 |
| `app/tools/build_cn_stock_leader_sw_l1_latest_snap.py` | `cn_board_industry_member_hist` + `TUSHARE_SW2021_L1` | Already SW-first | 维持主路径 |
| `data_fetcher/build_local_industry_proxy.py` | `cn_board_industry_member_hist` with SW L1 coverage | Already SW-first | 维持主路径 |
| `scripts/build_cn_stock_mainline_strength_daily.py` | 混合 `board_map + local_industry + GA` | Can migrate | 去掉 `cn_board_member_map_d` 直接依赖 |
| `scripts/build_local_industry_map_hist.py` | 优先 board_map，再 fallback Tushare | Can migrate | 反转优先级，改成 SW history first |

### 判断

- 真正的 `V8 primary path` 应该站在这一组上
- `TRANSITIONAL` 的两个脚本还能跑，但建议继续去板块化

## 三、可以安全退役的范围

这里的“安全退役”不是立刻删表删代码，而是：

- 不再作为 `V8 primary path`
- 不再新增上游依赖
- 只保留为兼容 / 历史支持 / inventory

| Path | Why not primary anymore | Retirement stance |
| --- | --- | --- |
| `data_fetcher/fetch_akshare_boards.py` | 只负责东方财富行业/概念板块抓取，不是 SW/local-industry 主链 | 可降级为 legacy board asset |
| `app/tools/backfill_concept_history_from_tushare.py` | 服务 `cn_board_concept_*` 历史，不是 V8 主 market-context | 可降级为 legacy concept-history tool |
| `app/tasks/board_membership_refresh_task.py` | 继续维护 board history / daily map，但不应主导 V8 mainline 定义 | 可保留但不再扩权 |
| `app/tools/rebuild_rotation_three_tables_from_map.py` | 仍围绕 `cn_board_concept_master` | 若 rotation 迁完可退役 |
| `app/tasks/rotation_sector_snapshot_task.py` | 当前仍吃 `cn_board_member_map_d`，不是最终态 | 待 rotation 完成去 board 化后退役旧实现 |

### 判断

- 这批东西现在更适合看成 `legacy compatibility surface`
- 不是“现在就删”，而是“停止继续把它们当主依赖”

## 四、建议优先级

### P0

先改掉显式 `BK%%` calendar 过滤：

- `app/tools/sync_cn_stock_daily_basic_from_tushare.py`
- `app/tools/sync_cn_stock_fundamental_monthly.py`

### P1

继续推进去板块化：

- `scripts/build_cn_stock_mainline_strength_daily.py`
- `scripts/build_local_industry_map_hist.py`

### P2

把下列链路正式标注为 legacy：

- `fetch_akshare_boards.py`
- `backfill_concept_history_from_tushare.py`
- `board_membership_refresh_task.py`
- 旧版 `rotation_*` board-map 实现

## 最终结论

当前仓库状态可以概括成一句话：

- `BK` 还活着，但主要活在兼容层
- `SW2021 + cn_local_industry_* + cn_ga_*` 才是当前应该继续强化的主路径
- 下一步不该再增加新的 `BK` 读路径，而应该逐步把显式 `BK%%` 过滤点清掉
