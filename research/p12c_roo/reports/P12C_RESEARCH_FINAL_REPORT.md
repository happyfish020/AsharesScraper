# P12c Collapse Confirmation Matrix — 最终研究报告

## 执行摘要

**结论: STOP_P12C_LINE / HOLD_RESEARCH_ONLY** — P12c Collapse Confirmation Matrix 在主线级和市场级两个层面都未能通过验证，现阶段只保留为归档研究材料，不进入实盘、政策或择时链路。

| 阶段 | 结果 | 通过 |
|------|------|:----:|
| Phase 1: Cross-Mainline | 11/12 主题正确分类 | ✅ |
| Phase 2: Danger-Zone | 50.1% 危险区比率 (320/639 日期) | ❌ |
| Phase 3: Persistence | 0 个多日窗口 | ❌ |
| Phase 4: False Positive | 0% 噪声率 | ✅ |
| Phase 5: Philosophy | 哲学一致性通过 | ✅ |
| **最终裁决** | **2/5 阶段通过** | **STOP** |

---

## 0. 归档补充结论（2026-05-23）

2026-05-23 新增的归档与市场元审计对 P12c 做了最后一次复核，结论不是“继续优化”，而是正式收口:

| 层级 | 裁决 | 关键依据 |
|------|------|----------|
| 主线级 | `STOP_P12C_LINE` | 5 个阶段仅通过 2 个；`TRUE_COLLAPSE` 仅占 0.04% |
| 市场级 | `HOLD_RESEARCH_ONLY` | `danger_ratio` 在 94.1% 的交易日高于 50%；`TRUE_COLLAPSE` 仅出现 2 个日期 |
| 最终状态 | `ARCHIVED` | 不可用于交易、政策判断或市场择时 |

### 边界锁定（最终版）

| 锁定项 | 状态 | 含义 |
|--------|------|------|
| `STOP_P12C_LINE` | Locked | 核心架构无法产出可执行信号 |
| `HOLD_RESEARCH_ONLY` | Locked | 市场级聚合噪声过高，不能进生产 |
| `DO_NOT_REPAIR_TRUE_MASK` | Locked | 不能通过把 AND 改 OR 来“修复”信号 |
| `DO_NOT_PROMOTE_TO_POLICY` | Locked | 不具备政策级解释力 |
| `DO_NOT_CONNECT_TO_TRADING` | Locked | 接入 P11d/P15 只会拖累现有链路 |

这意味着 P12c 的研究任务已经从“调参/修补”切换为“归档/留痕”，后续只保留历史诊断参考价值。

---

## 1. 修复内容 (v3)

### 1.1 主题映射修复
- **问题**: `_map_theme_bucket` 将中文关键词（如"半导体"）与 `mainline_id`（如 `801010.SI`）匹配，导致所有主题落入 OTHER
- **修复**: 添加 `_load_mainline_name_map()` 从 `cn_mainline_radar_daily` 加载 `mainline_id→mainline_name` 映射，使用中文行业名称进行主题分类
- **结果**: 从 1 个主题桶（OTHER）提升到 11 个正确分类的主题桶

### 1.2 时间段扩展
- **之前**: 2025-10-01 → 2026-01-31（4 个月）
- **之后**: 2025-01-01 → 2026-05-01（16 个月，80,000 行数据）

### 1.3 阈值扫描优化
- **之前**: 每个参数组合重建 research frame（36×10min = 360min）
- **之后**: 只重新应用分类逻辑（36×1s = 36s）
- **方法**: 新增 `_reclassify()` 函数，在已有基础列上重新计算 `group_b_leader_deterioration`、`confirmation_group_count`、`collapse_confidence_score` 和 `collapse_state`

---

## 2. 阈值扫描结果

扫描了 36 个参数组合（4 scores × 3 groups × 3 leader_min），**所有组合结果完全一致**:

| 参数 | 测试值 | 影响 |
|------|--------|:----:|
| `true_collapse_min_score` | 0.60, 0.65, 0.70, 0.74 | **无影响** |
| `true_collapse_min_groups` | 2, 3, 4 | **无影响** |
| `leader_island_min` | 0.50, 0.60, 0.68 | **无影响** |

### 原因分析

**`leader_island_min` 无影响**:
- `group_b_leader_deterioration` 有 5 个条件，其中 `leader_island_score >= 0.38`（cond1）已覆盖绝大多数行
- 只有极少数行仅靠 `leader_island_min` 触发，且这些行不会同时触发其他 3 个 group

**`true_collapse_min_groups` 和 `true_collapse_min_score` 无影响**:
- `TRUE_COLLAPSE_CONFIRMATION` 的 `true_mask` 要求**所有 4 个 group 同时触发**（AND 条件）
- 4-group 交集（`all4`）只有 **36 行**，且它们的 `collapse_confidence_score` 均为 **1.0**
- 因此无论 `min_groups` 或 `min_score` 如何设置，这 36 行始终被分类为 TRUE_COLLAPSE

---

## 3. 核心问题诊断

### 3.1 TRUE_COLLAPSE 信号极度稀疏

| 指标 | 值 |
|------|-----|
| 总行数 | 80,000 |
| TRUE_COLLAPSE 行数 | 36 (0.04%) |
| 出现日期 | 仅 2 天 (2026-01-09, 2026-01-26) |
| 涉及主线数 | 27 个 |
| 4-group 交集行数 | 36 (score 均为 1.0) |

### 3.2 DIFFUSION_WARNING 过度敏感

| 指标 | 值 |
|------|-----|
| DIFFUSION_WARNING 行数 | 47,949 (59.9%) |
| 危险区比率 | 50.1% (320/639 交易日) |
| 问题 | 超过一半的日期被标记为"危险"，失去信号区分度 |

### 3.3 持久性为零

| 指标 | 值 |
|------|-----|
| 多日窗口数 | 0 |
| 重复 collapse 事件 | 0 |
| 问题 | TRUE_COLLAPSE 信号不连续，无法形成可交易的持续性信号 |

---

## 4. 市场级元审计补充

### 4.1 日度聚合统计

| 指标 | 值 |
|------|-----|
| 交易日总数 | 320 |
| 平均每日跟踪主线数 | 131 |
| 平均 `danger_ratio` | 76.52% |
| `danger_ratio` 最大值 | 100.00% |
| `danger_ratio` 最小值 | 6.67% |
| 平均 `TC ratio` | 0.0643% |
| 有 `TC` 事件的日期 | 2 / 320 (0.6%) |
| `danger_ratio > 50%` 的日期 | 301 / 320 (94.1%) |
| `TC ratio < 1%` 的日期 | 318 / 320 (99.4%) |

### 4.2 市场状态分布

| Regime | Days | 占比 |
|--------|:----:|:----:|
| `RANGE_CHOP_HIGH` | 186 | 58.1% |
| `RANGE_CHOP_MID` | 93 | 29.1% |
| `DEFENSIVE_SHIFT` | 23 | 7.2% |
| `RANGE_CHOP_LOW` | 18 | 5.6% |

### 4.3 市场级失败原因

1. **`danger_ratio` 长期处于高位噪声区**: 94.1% 的交易日高于 50%，无法承担风险分层功能。
2. **`TRUE_COLLAPSE` 极度稀疏**: 16 个月仅 2 个日期触发，无法支持择时或预警。
3. **聚合层没有修复底层缺陷**: 主线级分类逻辑的问题被原样放大到市场级。
4. **Regime 区分度不足**: 市场大部分时间都落在 `RANGE_CHOP` 类状态，无法验证 P12c 的跨状态稳定性。

---

## 5. 最终建议（替代原修补方案）

### 5.1 明确不做的事

1. **不修 `true_mask`**: 不通过放宽 AND/OR 来人为增加信号数量。
2. **不把 P12c 接到交易链路**: 不作为 P11d、P15 或其他策略的输入条件。
3. **不提升为政策/市场总控指标**: 当前解释力和稳定性都不够。
4. **不继续投入维护算力**: 除非未来开启全新的研究线，否则不再为 P12c 做持续迭代。

### 5.2 保留价值

1. **作为历史失败样本归档**: 记录“严格多组 AND 确认”在真实市场数据上的失效方式。
2. **作为背景诊断材料**: `danger_ratio` 仅能用于回溯性说明，不可作为可执行信号。
3. **作为新研究线的反例约束**: 若未来开启 `P12d` 或替代方案，应避免复制当前架构瓶颈。

### 5.3 主线工作建议

继续将主工作流聚焦在 **P11d → P15**，把 P12c 视为已结束的研究分支，而不是待上线模块。

---

## 6. 文件清单

| 文件 | 说明 |
|------|------|
| `run_p12c_research.py` | 研究运行脚本 (v3) |
| `reports/full_pipeline_20260522_201509/` | 最后一次完整流水线输出 |
| `reports/threshold_scan_20260522_220541/` | 阈值扫描结果 |
| `reports/market_aggregate_20260523_070551/` | 市场级聚合输出 |
| `reports/p12c_archive_market_meta_20260523_113749/` | 归档与市场风险元审计 |
| `reports/P12C_RESEARCH_FINAL_REPORT.md` | 本报告 |

### 关键输出文件

- `full_pipeline_20260522_201509/P12C_FINAL_DECISION.md` — 最终决策
- `full_pipeline_20260522_201509/cross_mainline_repeatability.csv` — 主题分类结果
- `full_pipeline_20260522_201509/risk_reward_deterioration_review.md` — 危险区分析
- `full_pipeline_20260522_201509/signal_repeatability_review.md` — 信号重复性
- `full_pipeline_20260522_201509/background_noise_sector_audit.md` — 噪声审计
- `threshold_scan_20260522_220541/threshold_scan_results.csv` — 36 组合扫描结果
- `market_aggregate_20260523_070551/market_collapse_daily.csv` — 320 日市场聚合诊断
- `p12c_archive_market_meta_20260523_113749/P12C_ARCHIVE_AND_MARKET_RISK_META_AUDIT.md` — 归档元审计结论
- `p12c_archive_market_meta_20260523_113749/P12C_EXPERIMENT_LEDGER.md` — 实验台账与边界锁定

---

*报告最后整合时间: 2026-05-23*
*研究脚本: `research/p12c_roo/run_p12c_research.py`*
*依赖: `GrowthAlpha_V7.system_layers.p12c_collapse_confirmation_matrix`*
