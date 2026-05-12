# 财务数据表层次结构说明

> 适用系统：GrowthAlpha V8 / AsharesScraperV2
> 最后更新：2026-05-10

---

## 概述

以下 7 张表名称相近，但分属 **3 个不同层次**，外加 1 张独立旁路系统。
它们不是平行的重复设计，而是一条从原始季报到每日财务快照的流水线。

---

## 数据流向（三层 Pipeline）

```
┌─────────────────────────────────────────────────────────────────────┐
│  Tier 1：原始季报（Tushare API 直接入库，只读）                       │
│                                                                     │
│  cn_stock_fina_indicator     ~255K 行  2008~today  季度             │
│    30 列：eps / roe / grossprofit_margin / debt_to_assets /         │
│           or_yoy / netprofit_yoy / ocfps / current_ratio 等         │
│                                                                     │
│  cn_stock_income             ~287K 行  1996~today  季度             │
│    25 列：total_revenue / revenue / n_income_attr_p /               │
│           ebit / ebitda / undist_profit 等（完整利润表）              │
│                                                                     │
│  cn_stock_balancesheet       ~264K 行  2001~today  季度             │
│    30 列：total_assets / total_liab / inventories /                 │
│           fix_assets / goodwill 等（完整资产负债表）                  │
│                                                                     │
│  特点：有 raw_payload 字段（原始 JSON），列多，宽表                    │
│  用途：数据溯源、补数、跑批脚本入库用                                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  build_stock_fundamental_daily.py
                           │  Step 1：INSERT INTO cn_local_stock_*_q
                           │  抽取关键列，去掉 raw_payload
                           │  筛选 ann_date IS NOT NULL
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Tier 2：本地精简季报（staging，流水线中间表）                        │
│                                                                     │
│  cn_local_stock_fina_indicator_q                                    │
│    6 列：revenue_yoy / profit_yoy / roe /                           │
│          gross_margin / debt_to_assets / ocfps                      │
│                                                                     │
│  cn_local_stock_income_q                                            │
│    3 列：total_revenue / revenue / n_income_attr_p                  │
│                                                                     │
│  cn_local_stock_balancesheet_q                                      │
│    5 列：inventory / fixed_assets / total_assets /                  │
│          total_liab / contract_liability                            │
│                                                                     │
│  特点：窄表，无 raw_payload，ON DUPLICATE KEY UPDATE 幂等写入         │
│  用途：流水线中间层，为 Tier3 物化提供干净的季报输入                   │
│  注意：仅保留最近一次 build 覆盖的日期范围内的数据                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  build_stock_fundamental_daily.py
                           │  Step 2：MATERIALIZE_SQL
                           │  季报 × cn_stock_daily_price
                           │  → 按 ann_date 前向填充，每日一行
                           │  → 使用 LEAD(ann_date) 确定每份报告有效期
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Tier 3：每日财务快照（日频，供 alpha 引擎 JOIN）                     │
│                                                                     │
│  cn_stock_fundamental_daily                                         │
│    14 列：trade_date + report_end_date + ann_date +                 │
│           Tier2 三表字段合并                                         │
│                                                                     │
│  特点：日频，每天持有该股当前最新已公告季报的财务值                     │
│         无前视偏差（以 ann_date 而非 end_date 对齐）                  │
│  用途：build_stock_quality_score_daily.py 的主源                     │
│         → 生成 cn_stock_quality_score_daily（Factor 1 + 2）          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 独立旁路：旧版月度质量评分

```
┌─────────────────────────────────────────────────────────────────────┐
│  cn_stock_fundamental_quality_snap                                  │
│    ~694K 行  2008~today  月度粒度（basic_trade_date = 月末）          │
│                                                                     │
│  特点：                                                             │
│    - 自带独立的 pass/fail 规则体系：                                  │
│        pass_eps_positive / pass_revenue_growth_5/10                 │
│        pass_debt_to_eqt_lt_2 / pass_gross_margin_positive          │
│    - 有 quality_core_score / quality_total_score                    │
│    - 同时 JOIN 了 cn_stock_daily_basic（市值、PE、PB、PS）            │
│                                                                     │
│  与 V8 pipeline 的关系：                                             │
│    ⚠️ 独立系统，与 cn_stock_quality_score_daily 互不相通              │
│    - quality_snap：旧版（V7/legacy），月度，pass/fail 规则            │
│    - quality_score_daily：V8 新版，日频，5 维子分数连续值             │
│    两套评分逻辑不同，不要混用                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 冗余分析

| 冗余类型 | 具体情况 | 结论 |
|---|---|---|
| Tier1 → Tier2 | `cn_local_stock_*_q` 是 `cn_stock_*` 的列子集 | 合理。去掉 `raw_payload` 节省空间，只保留 pipeline 需要的 14 列 |
| Tier2 → Tier3 | `fundamental_daily` 字段 = Tier2 三表合并 + trade_date | 合理。季度 → 日频展开，便于与价格表 JOIN，避免重复计算 |
| 两套质量评分 | `quality_snap` vs `quality_score_daily` | 真正的设计重叠。旧版保留作历史参考，V8 不再依赖 |

---

## 当前数据新鲜度（2026-05-10 审核）

| 表 | 行数 | 覆盖范围 | 备注 |
|---|---|---|---|
| `cn_stock_fina_indicator` | 255K | 2008~2026-03 | 原始季报，正常 |
| `cn_stock_income` | 287K | 1996~2026-03 | 原始季报，正常 |
| `cn_stock_balancesheet` | 264K | 2001~2026-03 | 原始季报，正常 |
| `cn_local_stock_fina_indicator_q` | 42K | 2024~2025 | ⚠️ 仅最近一次 build 的窗口 |
| `cn_local_stock_income_q` | 42K | 2024~2025 | ⚠️ 同上 |
| `cn_local_stock_balancesheet_q` | 42K | 2024~2025 | ⚠️ 同上 |
| `cn_stock_fundamental_daily` | 285K | **2026 only** | ⚠️ 需回填 2010~2025 |
| `cn_stock_fundamental_quality_snap` | 694K | 2008~2026-04 | 旧版，覆盖完整 |

---

## 回填说明

`cn_local_stock_*_q` 的数据范围仅反映上次 build 的窗口，**不代表数据损坏**。
重新运行 `build_stock_fundamental_daily.py` 时，Step 1 会用
`end_date BETWEEN (start-2年) AND end` 重新从 Tier1 同步，历史数据不会丢失。

回填 2010~2025 的正确命令：

```powershell
cd d:\LHJ\PythonWS\MarketScraper\AsharesScraperV2

# Step 0：fundamental daily 回填
python scripts/build_stock_fundamental_daily.py `
    --start 2010-01-01 --end 2025-12-31 `
    --chunk-months 3 --replace

# Step 0.5：quality score 回填
python scripts/build_stock_quality_score_daily.py `
    --start 2010-01-01 --end 2025-12-31 `
    --db-user cn_opr_red --db-password sec_Bobo123 --replace

# Step 6：unified alpha 回填
python scripts/build_unified_alpha_score_daily.py `
    --start 2010-01-01 --end 2025-12-31 `
    --db-user cn_opr_red --db-password sec_Bobo123 --replace
```

---

## 相关文件

| 文件 | 说明 |
|---|---|
| `scripts/build_stock_fundamental_daily.py` | Tier1 → Tier2 → Tier3 构建器 |
| `data_pipeline/builders/stock_fundamental_daily.py` | 实际构建逻辑（SQL + 分块） |
| `scripts/build_stock_quality_score_daily.py` | Tier3 → quality score |
| `docs/DDL/ga_mainline_data_backfill_system.sql` | Tier2 staging 表 DDL |
| `docs/UNIFIED_ALPHA_ENGINE.md` | V8 alpha 引擎总文档 |
