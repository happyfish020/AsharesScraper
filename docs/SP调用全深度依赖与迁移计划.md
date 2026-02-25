# SP调用全深度依赖与迁移计划

## 1. 范围与目标

本文聚焦 `AshareScraperV2` 在仓库中已出现的板块轮动相关 SP 调用链，输出三部分：

1. 外部调用入口与参数签名
2. SP 内部全深度依赖（Tables / Views / SPs）
3. Oracle -> MySQL 迁移计划（含核验与切换）

数据来源：

1. `tests/backtest/runner.py`
2. `tests/grid_runner.py`
3. `tests/xp_sweep_runner.py`
4. `docs/板块轮动.md`
5. `docs/BackTest/1.md`

## 2. 外部入口 SP（应用直接调用）

### 2.1 回测主入口

`SECOPR.SP_RUN_SECTOR_ROT_BACKTEST`

调用来源：

1. `tests/backtest/runner.py`
2. `tests/grid_runner.py`
3. `tests/xp_sweep_runner.py`

参数（以 `tests/backtest/runner.py` 的命名调用为准）：

1. `p_run_id`
2. `p_start_dt`
3. `p_end_dt`
4. `p_sector_type`
5. `p_enter_p`
6. `p_exit_p`
7. `p_exit_consecutive`
8. `p_top_k`
9. `p_min_hold`
10. `p_rebalance_freq`
11. `p_weight_mode`
12. `p_full_rebuild`
13. `p_cost_rate`（该脚本已传入）

### 2.2 结果校验入口

1. `SECOPR.SP_VALIDATE_SECTOR_ROT_RUN(run_id)`
2. `SECOPR.SP_VALIDATE_AGAINST_BASELINE(p_run_id, p_baseline_id, p_min_alpha)`

调用来源：

1. `tests/backtest/runner.py`
2. `tests/grid_runner.py`
3. `tests/xp_sweep_runner.py`

### 2.3 日常刷新总入口

`SECOPR.SP_ROTATION_DAILY_REFRESH(run_id, trade_date, p_force, p_refresh_energy)`

来源：`docs/板块轮动.md`（作为日常 EOD 总控入口）。

## 3. SP 内部全深度依赖（按调用层级）

说明：以下为仓库文档和调用代码可确认的依赖图。最终生产真值仍需用数据库元数据递归查询（见第 5 节 SQL）。

### 3.1 L0：总控 / 直接入口层

1. `SP_ROTATION_DAILY_REFRESH`
2. `SP_RUN_SECTOR_ROT_BACKTEST`
3. `SP_VALIDATE_SECTOR_ROT_RUN`
4. `SP_VALIDATE_AGAINST_BASELINE`

### 3.2 L1：内部编排 SP 层（由总控 SP 调用）

1. `SP_BACKFILL_SECTOR_ENERGY_SNAP`
2. `SP_REFRESH_SECTOR_ENERGY_SNAP`
3. `SP_BUILD_SECTOR_ROTATION_RANKED_LATEST`
4. `SP_BUILD_SECTOR_ROTATION_SIGNAL_LATEST`
5. `SP_BACKFILL_ROT_BT_FROM_PRICE`
6. `SP_REFRESH_ROTATION_SNAP_ALL`
7. `SP_REFRESH_ROTATION_ENTRY_SNAP`
8. `SP_REFRESH_ROTATION_HOLDING_SNAP`
9. `SP_REFRESH_ROTATION_EXIT_SNAP`

### 3.3 L2：核心视图层（计算真源）

1. `CN_SECTOR_EOD_AGG_V`
2. `CN_SECTOR_EOD_FEATURE_V`
3. `CN_SECTOR_ENERGY_V`
4. `CN_SECTOR_ROTATION_TRANSITION_V`
5. `CN_SECTOR_ROTATION_NAMED_V`（部分报表/快照场景）
6. `V_ROTATION_ENTRY_EXEC_V1`（可选执行视图）
7. `V_ROTATION_EXIT_EXEC_V1`（可选执行视图）

### 3.4 L3：事实表 / 快照表层

1. `CN_STOCK_DAILY_PRICE`
2. `CN_SECTOR_ENERGY_SNAP_T` -- 
3. `CN_SECTOR_ROTATION_RANKED_T` --- 
4. `CN_SECTOR_ROTATION_SIGNAL_T` -- 
5. `CN_SECTOR_ROT_BT_DAILY_T` - 
6. `CN_SECTOR_ROT_POS_DAILY_T` - 
7. `CN_ROTATION_ENTRY_SNAP_T` - 
8. `CN_ROTATION_HOLDING_SNAP_T` - 
9. `CN_ROTATION_EXIT_SNAP_T` - 
10. `CN_SECTOR_ROT_BASELINE_T`  - 
11. `CN_BASELINE_REGISTRY_T`  - 
12. `CN_BASELINE_DECISION_T`  - 

### 3.5 L4：索引 / 调度对象（已出现）

1. `PK_SECTOR_ENERGY_SNAP`
2. `CN_BIC_PK`
3. `CN_BCC_PK`
4. `JOB_ROT_REFRESH_DAILY`（调度层）

## 4. 迁移计划（Oracle -> MySQL）

迁移原则（与你当前要求一致）：

1. 全部业务对象优先迁到 MySQL
2. 仅 `CN_STOCK_DAILY_PRICE` 暂时保留 Oracle + MySQL 双通道，待迁移稳定后再退 Oracle

### Phase 0：冻结与盘点

1. 冻结当前 SP 版本和对象清单（表、视图、SP、索引、作业）
2. 对每个入口 SP 固化输入/输出契约（参数、返回、副作用表）
3. 明确每张表的主键、唯一键、增量更新键

### Phase 1：表层迁移

1. 在 MySQL 建立 L3 全量表结构
2. 除 `CN_STOCK_DAILY_PRICE` 外，写入链路切至 MySQL
3. `CN_STOCK_DAILY_PRICE` 保持双写或 MySQL 主写 + Oracle 对账回写

### Phase 2：视图层迁移

1. 迁移 `CN_SECTOR_EOD_AGG_V`、`CN_SECTOR_EOD_FEATURE_V`、`CN_SECTOR_ENERGY_V`、`CN_SECTOR_ROTATION_TRANSITION_V`
2. 将 Oracle 特性改写为 MySQL 8 可用语法（窗口函数、日期函数、空值函数）
3. 对关键视图建立同口径对账（行数、主键覆盖、指标 diff）

### Phase 3：SP 层迁移

1. 优先迁移 `SP_ROTATION_DAILY_REFRESH` 及其 L1 内部 SP
2. 保留外部调用签名不变，先做兼容包装
3. 校验 SP 副作用表（snap/ranked/signal/bt/pos）逐日一致性

### Phase 4：验证层迁移

1. 迁移 `SP_VALIDATE_SECTOR_ROT_RUN` 与 `SP_VALIDATE_AGAINST_BASELINE`
2. 在 MySQL 侧落地 baseline 决策链
3. 保证 `tests/*runner.py` 可无感切换

### Phase 5：切换与下线

1. 切换读路径到 MySQL
2. 保留 `CN_STOCK_DAILY_PRICE` Oracle 兜底期
3. 达到稳定窗口后移除 `CN_STOCK_DAILY_PRICE` Oracle 路径

## 5. 全深度依赖核验 SQL（必须在数据库执行）

### 5.1 递归依赖树（对象级）

```sql
WITH RECURSIVE_DEPS AS (
  SELECT
    d.OWNER,
    d.NAME,
    d.TYPE,
    d.REFERENCED_OWNER,
    d.REFERENCED_NAME,
    d.REFERENCED_TYPE,
    1 AS LV
  FROM ALL_DEPENDENCIES d
  WHERE d.OWNER = 'SECOPR'
    AND d.NAME IN (
      'SP_ROTATION_DAILY_REFRESH',
      'SP_RUN_SECTOR_ROT_BACKTEST',
      'SP_VALIDATE_SECTOR_ROT_RUN',
      'SP_VALIDATE_AGAINST_BASELINE'
    )
  UNION ALL
  SELECT
    d.OWNER,
    d.NAME,
    d.TYPE,
    d.REFERENCED_OWNER,
    d.REFERENCED_NAME,
    d.REFERENCED_TYPE,
    r.LV + 1
  FROM ALL_DEPENDENCIES d
  JOIN RECURSIVE_DEPS r
    ON d.OWNER = r.REFERENCED_OWNER
   AND d.NAME  = r.REFERENCED_NAME
)
SELECT DISTINCT
  OWNER, NAME, TYPE,
  REFERENCED_OWNER, REFERENCED_NAME, REFERENCED_TYPE,
  LV
FROM RECURSIVE_DEPS
ORDER BY LV, OWNER, NAME, REFERENCED_TYPE, REFERENCED_NAME;
```

### 5.2 动态 SQL 补漏（源码扫描）

```sql
SELECT
  s.OWNER, s.NAME, s.TYPE, s.LINE, s.TEXT
FROM ALL_SOURCE s
WHERE s.OWNER = 'SECOPR'
  AND s.NAME IN (
    'SP_ROTATION_DAILY_REFRESH',
    'SP_RUN_SECTOR_ROT_BACKTEST',
    'SP_VALIDATE_SECTOR_ROT_RUN',
    'SP_VALIDATE_AGAINST_BASELINE'
  )
  AND (
    UPPER(s.TEXT) LIKE '%EXECUTE IMMEDIATE%'
    OR UPPER(s.TEXT) LIKE '%CN_%'
    OR UPPER(s.TEXT) LIKE '%SP_%'
  )
ORDER BY s.NAME, s.LINE;
```

## 6. 迁移验收门槛

1. 同一 `trade_date` 下，MySQL 与 Oracle 的 `ranked/signal/snap` 行数一致
2. 核心指标（`ENTRY_CNT`、`N_POS`、`energy_pct` 分布）偏差在阈值内
3. `tests/backtest/runner.py` 全流程可在 MySQL 端跑通
4. 仅 `CN_STOCK_DAILY_PRICE` 存在 Oracle 兜底，其余对象全部 MySQL 化

## 7. 当前缺口（需补）

1. 仓库内未包含上述 SP 的完整 DDL/源码，当前依赖树含“文档推断”成分
2. 需在数据库执行第 5 节 SQL，导出真实全量依赖并回填此文档
3. 建议新增 `sql/` 下对象 DDL 与依赖快照，避免迁移期漂移
## 8. 板块聚合视图性能优化（已落地）

目标：将 `CN_BOARD_CONCEPT_EOD_AGG_V` 与 `CN_BOARD_INDUSTRY_EOD_AGG_V` 从“全历史重算”改为“日常信号可用的近窗计算”。

### 8.1 视图 SQL 优化

1. 将成分表作为静态快照维度，仅取 `ASOF_DATE = MAX(ASOF_DATE)`。
2. 对 `CN_STOCK_DAILY_PRICE` 仅取最新交易日（daily-only）：
   - `trade_date = max_trade_date`
3. 去掉不必要的全历史窗口函数，使用 `PRE_CLOSE / CHANGE / CHG_PCT` 计算 `ret_eff`。

对应 DDL：

1. `docs/DDL/cn_market.cn_board_concept_eod_agg_v.sql`
2. `docs/DDL/cn_market.cn_board_industry_eod_agg_v.sql`

### 8.2 索引优化（已脚本化）

已在迁移脚本中加入性能索引确保步骤（幂等）：

1. `CN_BOARD_CONCEPT_CONS(ASOF_DATE, SYMBOL, CONCEPT_ID)`
2. `CN_BOARD_INDUSTRY_CONS(ASOF_DATE, SYMBOL, BOARD_ID)`

脚本位置：

1. `app/tools/migrate_sp_doc_tables_views.py`
2. 入口函数：`ensure_perf_indexes(...)`

执行方式：

```bash
python app/tools/migrate_sp_doc_tables_views.py --apply
```

### 8.2.1 SP 迁移（首个落地）

1. 已新增 MySQL 版本 `SP_ROTATION_DAILY_REFRESH` DDL：
   - `docs/DDL/cn_market.sp_rotation_daily_refresh.sql`
2. 该 SP 已在 `cn_market` 库创建并联调通过（`runner.py --tasks rotation`）。
3. 已同步落地 L1 过程（MySQL）：
   - `docs/DDL/cn_market.sp_build_sector_rotation_ranked_latest.sql`
   - `docs/DDL/cn_market.sp_build_sector_rotation_signal_latest.sql`
   - `docs/DDL/cn_market.sp_refresh_rotation_snap_all.sql`
4. 当前编排关系：
   - `SP_ROTATION_DAILY_REFRESH` -> `SP_BUILD_SECTOR_ROTATION_RANKED_LATEST`
   - `SP_ROTATION_DAILY_REFRESH` -> `SP_BUILD_SECTOR_ROTATION_SIGNAL_LATEST`
   - `SP_ROTATION_DAILY_REFRESH` -> `SP_BACKFILL_ROT_BT_FROM_PRICE`
   - `SP_ROTATION_DAILY_REFRESH` -> `SP_REPAIR_ROT_BT_NAV`
   - `SP_ROTATION_DAILY_REFRESH` -> `SP_REFRESH_ROTATION_SNAP_ALL`

### 8.2.2 验证 SP 迁移（已落地）

1. 已新增 MySQL 版本验证过程：
   - `docs/DDL/cn_market.sp_validate_sector_rot_run.sql`
   - `docs/DDL/cn_market.sp_validate_against_baseline.sql`
2. 已在 `cn_market` 库创建并可直接调用：
   - `CALL cn_market.SP_VALIDATE_SECTOR_ROT_RUN(:run_id)`
   - `CALL cn_market.SP_VALIDATE_AGAINST_BASELINE(:run_id, :baseline_id, :min_alpha)`
3. 当前实测结果：
   - `SP_VALIDATE_AGAINST_BASELINE` 可正常写入 `CN_BASELINE_DECISION_T`
   - `SP_VALIDATE_SECTOR_ROT_RUN` 已用于发现并验证修复历史 `CN_SECTOR_ROT_BT_DAILY_T` 的 `NAV` 空值问题

### 8.2.3 NAV 修复过程（已落地）

1. 已新增：
   - `docs/DDL/cn_market.sp_repair_rot_bt_nav.sql`
2. 用途：按 `run_id` 对 `CN_SECTOR_ROT_BT_DAILY_T` 的 `NAV` 空值做前值填充（forward-fill）。
3. 当前状态：历史 `NAV` 空值已修复为 0 行，`SP_VALIDATE_SECTOR_ROT_RUN` 返回 `PASS`。

### 8.2.4 Oracle vs MySQL 一键对账（已落地）

1. 脚本：
   - `app/tools/reconcile_rotation_oracle_mysql.py`
2. 用法：
   - `python app/tools/reconcile_rotation_oracle_mysql.py`
   - 可选参数：`--run-id`、`--trade-date`、`--output`
   - 门禁参数：`--max-mismatches N`、`--fail-on-mismatch`
3. 输出：
   - 生成 markdown 报告到 `audit_reports/reconcile_rotation_*.md`
4. 当前对账结论（最新一次）：
   - 存在差异属于预期（MySQL 迁移链路已跑到 `2026-02-25`，Oracle 侧仍停留旧状态）。

### 8.3 使用约束与分流

1. 当前两张板块聚合视图是“日常信号视图”，不再覆盖全历史。
2. 回测场景需使用独立全历史视图/表（建议新增 `_BT` 版本），避免与日常链路争抢性能。
3. 日常 SP/任务调用必须携带 `trade_date` 条件，避免无谓全表扫描。

### 8.4 依赖关系补充

`CN_SECTOR_EOD_AGG_V`
  --> `CN_BOARD_CONCEPT_EOD_AGG_V`
  --> `CN_BOARD_INDUSTRY_EOD_AGG_V`
