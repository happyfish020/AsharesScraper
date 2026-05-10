-- ============================================================================
-- sql/create_cn_mainline_strength_daily.sql
-- GrowthAlpha V8 — P0 Upstream Mainline Strength Backfill
-- DDL for cn_mainline_strength_daily
--
-- Industry-level mainline strength score computed daily from:
--   1. cn_ga_mainline_radar_daily   — mainline_score, leader_density, breakout_ratio, etc.
--   2. cn_industry_capital_flow_daily — concentration_score (optional, may be empty)
--   3. cn_stock_leader_score_daily   — per-stock leader scores for leader_count
--   4. cn_ga_stock_role_map_daily    — stock-to-mainline role mapping
--   5. cn_local_industry_map_hist    — industry name mapping
--   6. cn_stock_daily_price          — price/volume data
--   7. cn_stock_daily_basic          — market cap / turnover data
--
-- Primary key: (trade_date, industry_id)
-- ============================================================================

CREATE TABLE IF NOT EXISTS `cn_mainline_strength_daily` (
    `trade_date`                    date            NOT NULL COMMENT '交易日期',
    `industry_id`                   varchar(32)     NOT NULL COMMENT '行业ID (SW L1)',
    `industry_name`                 varchar(128)    DEFAULT NULL COMMENT '行业名称',

    `mainline_strength`             decimal(18,8)   DEFAULT NULL COMMENT '主线强度综合评分 [0,1]',
    `trend_alignment_score`         decimal(18,8)   DEFAULT NULL COMMENT '趋势一致性评分 [0,1]',
    `breakout_ratio`                decimal(18,8)   DEFAULT NULL COMMENT '突破比例: 突破形态个股占比 [0,1]',
    `new_high_ratio`                decimal(18,8)   DEFAULT NULL COMMENT '新高比例: 创年内新高个股占比 [0,1]',

    `leader_density`                decimal(18,8)   DEFAULT NULL COMMENT '龙头密度: 行业内龙头股占比 [0,1]',
    `leader_count`                  int             DEFAULT NULL COMMENT '龙头股数量',
    `strong_stock_count`            int             DEFAULT NULL COMMENT '强势股数量',

    `capital_concentration_score`   decimal(18,8)   DEFAULT NULL COMMENT '资金集中度评分 [0,1]',
    `rotation_rank`                 int             DEFAULT NULL COMMENT '轮动排名 (1=最强)',

    `mainline_phase`                varchar(32)     DEFAULT NULL COMMENT '主线阶段: EMERGING/EXPANDING/DOMINANT/DIVERGING/DECAYING/UNKNOWN',

    `created_at`                    timestamp       NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`                    timestamp       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (`trade_date`, `industry_id`),
    KEY `idx_mainline_strength_daily_date` (`trade_date`),
    KEY `idx_mainline_strength_daily_strength` (`mainline_strength`),
    KEY `idx_mainline_strength_daily_phase` (`mainline_phase`, `trade_date`),
    KEY `idx_mainline_strength_daily_rank` (`rotation_rank`, `trade_date`),
    KEY `idx_mainline_strength_daily_industry` (`industry_id`, `trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='行业主线强度日表 — P0 Upstream Mainline Strength Backfill';
