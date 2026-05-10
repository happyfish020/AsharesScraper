-- ============================================================================
-- sql/create_cn_stock_mainline_strength_daily.sql
-- GrowthAlpha V8 — P3 Unified Alpha Engine
-- DDL for cn_stock_mainline_strength_daily
--
-- Industry-level mainline strength score computed daily from leader scores,
-- lifecycle states, price data, and industry membership.
-- ============================================================================

CREATE TABLE IF NOT EXISTS `cn_stock_mainline_strength_daily` (
    `trade_date`                date            NOT NULL COMMENT '交易日期',
    `industry_id`               varchar(32)     NOT NULL COMMENT '行业ID (SW L1)',
    `industry_name`             varchar(128)    DEFAULT NULL COMMENT '行业名称',
    `mainline_strength_score`   decimal(18,8)   DEFAULT NULL COMMENT '主线强度综合评分 [0,1]',
    `leader_density`            decimal(18,8)   DEFAULT NULL COMMENT '龙头密度: 行业内龙头股占比 [0,1]',
    `avg_leader_score`          decimal(18,8)   DEFAULT NULL COMMENT '平均龙头评分 [0,1]',
    `top_leader_score`          decimal(18,8)   DEFAULT NULL COMMENT '最高龙头评分 [0,1]',
    `breakout_ratio`            decimal(18,8)   DEFAULT NULL COMMENT '突破比例: 突破形态个股占比 [0,1]',
    `trend_alignment`           decimal(18,8)   DEFAULT NULL COMMENT '趋势一致性评分 [0,1]',
    `breadth_score`             decimal(18,8)   DEFAULT NULL COMMENT '宽度评分: 行业参与度 [0,1]',
    `acceleration_score`        decimal(18,8)   DEFAULT NULL COMMENT '加速评分: 动量变化 [0,1]',
    `lifecycle_bonus`           decimal(18,8)   DEFAULT NULL COMMENT '生命周期加成 [0,1]',
    `rank_in_market`            int             DEFAULT NULL COMMENT '全市场行业排名 (1=最强)',
    `is_active_mainline`        tinyint(1)      DEFAULT 0 COMMENT '是否为活跃主线 (score>=0.65)',
    `created_at`                timestamp       NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`                timestamp       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`trade_date`, `industry_id`),
    KEY `idx_mainline_strength_daily_date` (`trade_date`),
    KEY `idx_mainline_strength_daily_score` (`mainline_strength_score`),
    KEY `idx_mainline_strength_daily_active` (`is_active_mainline`, `trade_date`),
    KEY `idx_mainline_strength_daily_rank` (`rank_in_market`, `trade_date`),
    KEY `idx_mainline_strength_daily_industry` (`industry_id`, `trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='行业主线强度日表 — P3 Unified Alpha Engine';
