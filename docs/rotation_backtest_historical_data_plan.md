# Rotation 回测方案（历史数据重点）

## 1. 目标

建立可重跑、可审计的历史回测数据链路，覆盖：

1. 历史板块映射（symbol -> sector）
2. 历史板块日级特征与状态
3. 历史信号与回测主表

本方案用于 `cn_market` 库，优先保证历史一致性，再保证性能。

---

## 2. 历史数据硬约束

1. `cn_board_industry_cons` / `cn_board_concept_cons` 仅近年快照，不可作为历史映射源。
2. 历史映射必须使用 `cn_board_member_map_d`（按 `trade_date`）。
3. 如果 `cn_board_member_map_d` 缺日期覆盖，必须先从 `cn_board_*_member_hist` 重建映射，再跑回测链路。

---

## 3. 历史表分层与用途

### A. 映射层（历史基础）

1. `cn_board_industry_member_hist`
2. `cn_board_concept_member_hist`
3. `cn_board_member_map_d`（核心历史映射表）

用途：

1. 提供每个交易日的板块成分映射
2. 为后续板块聚合/rotation计算提供唯一历史口径

### B. 板块特征层（历史中间结果）

1. `cn_sector_eod_hist_t`

用途：

1. 存储按日板块特征（members/amount/up_ratio/score/sector_pass等）
2. 作为 ranked/signal 生成的底座

### C. 状态与信号层（历史输出）

1. `cn_sector_rotation_ranked_t`
2. `cn_sector_rotation_signal_t`

用途：

1. `ranked_t`：板块状态、tier、theme_rank、score（按日）
2. `signal_t`：ENTER/EXIT/WATCH 信号（按日）

### D. 回测结果层（策略输出）

1. `cn_sector_rot_bt_daily_t`
2. `cn_sector_rot_pos_daily_t`
3. `cn_rotation_entry_snap_t`
4. `cn_rotation_holding_snap_t`
5. `cn_rotation_exit_snap_t`

---

## 4. 历史回填标准流程

## Step 0: Pre-flight（必做）

```sql
-- 映射覆盖检查
SELECT MIN(trade_date) AS min_d, MAX(trade_date) AS max_d, COUNT(DISTINCT trade_date) AS d_cnt
FROM cn_board_member_map_d
WHERE trade_date BETWEEN :D1 AND :D2;

-- 交易日覆盖检查
SELECT COUNT(DISTINCT TRADE_DATE) AS price_days
FROM cn_stock_daily_price
WHERE TRADE_DATE BETWEEN :D1 AND :D2;
```

若 `map` 覆盖不足：先执行 `sp_build_board_member_map(:D1,:D2)` 或重建 `*_member_hist` 后再建 map。

## Step 1: 生成 `cn_sector_eod_hist_t`

按月/小块运行 `sp_refresh_sector_eod_hist`（或 monthly driver）。

原则：

1. 仅按目标日期范围增量/重算
2. 不使用 `*_cons` 参与历史映射

## Step 2: 生成 `ranked/signal`

推荐使用基表驱动（当前 `hybrid_base` 路径），避免深层视图全历史扫描。

原则：

1. 每块都带 `trade_date` 范围
2. 可用临时表承接窗口计算
3. 需要前值时只取“月初前1条”，不回扫全历史

## Step 3: 生成回测与快照

运行回测与快照 SP 链路，产出：

1. `cn_sector_rot_bt_daily_t`
2. `cn_sector_rot_pos_daily_t`
3. `cn_rotation_*_snap_t`

---

## 5. 性能策略（历史回填）

1. 单进程串行（优先稳定）
2. 月度分块（`months-per-chunk=1`）
3. 小事务 + 重试（deadlock/lock wait）
4. 视图去依赖：优先基表 + temp table
5. 每块落日志（成功/失败/月耗时）

---

## 6. 验收标准（必须全部通过）

```sql
-- 1) 三表日期覆盖一致
WITH p AS (
  SELECT DISTINCT TRADE_DATE d
  FROM cn_stock_daily_price
  WHERE TRADE_DATE BETWEEN :D1 AND :D2
),
e AS (SELECT DISTINCT trade_date d FROM cn_sector_eod_hist_t WHERE trade_date BETWEEN :D1 AND :D2),
r AS (SELECT DISTINCT trade_date d FROM cn_sector_rotation_ranked_t WHERE trade_date BETWEEN :D1 AND :D2),
s AS (SELECT DISTINCT signal_date d FROM cn_sector_rotation_signal_t WHERE signal_date BETWEEN :D1 AND :D2)
SELECT
  (SELECT COUNT(*) FROM p) AS price_days,
  (SELECT COUNT(*) FROM e) AS eod_days,
  (SELECT COUNT(*) FROM r) AS ranked_days,
  (SELECT COUNT(*) FROM s) AS signal_days;
```

```sql
-- 2) 每日行数一致性抽检
SELECT p.TRADE_DATE,
       COALESCE(e.c,0) AS eod_c,
       COALESCE(r.c,0) AS ranked_c,
       COALESCE(s.c,0) AS signal_c
FROM (SELECT DISTINCT TRADE_DATE FROM cn_stock_daily_price WHERE TRADE_DATE BETWEEN :D1 AND :D2) p
LEFT JOIN (SELECT trade_date d, COUNT(*) c FROM cn_sector_eod_hist_t WHERE trade_date BETWEEN :D1 AND :D2 GROUP BY trade_date) e ON e.d=p.TRADE_DATE
LEFT JOIN (SELECT trade_date d, COUNT(*) c FROM cn_sector_rotation_ranked_t WHERE trade_date BETWEEN :D1 AND :D2 GROUP BY trade_date) r ON r.d=p.TRADE_DATE
LEFT JOIN (SELECT signal_date d, COUNT(*) c FROM cn_sector_rotation_signal_t WHERE signal_date BETWEEN :D1 AND :D2 GROUP BY signal_date) s ON s.d=p.TRADE_DATE
ORDER BY p.TRADE_DATE;
```

验收口径：

1. `price_days == eod_days == ranked_days == signal_days`
2. 抽检日期 `eod_c == ranked_c == signal_c`

---

## 7. 运行建议（当前推荐命令）

```bash
python -m app.tools.run_rotation_monthly_backfill \
  --start-ym 2000-05 \
  --end-ym 2026-02 \
  --rank-signal-mode hybrid_base \
  --months-per-chunk 1 \
  --clear-first 0 \
  --retries 8 \
  --retry-sleep-sec 3 \
  --continue-on-error \
  --log-file logs/rotation_monthly_backfill_manual.log
```

---

## 8. 需要你审核确认的点

1. 历史映射是否统一强制 `cn_board_member_map_d`（禁用 `*_cons`）
2. `hybrid_base` 是否作为历史默认模式
3. 月度分块与失败继续策略（`--continue-on-error`）是否符合运营要求
4. 验收SQL是否满足你的审计口径

