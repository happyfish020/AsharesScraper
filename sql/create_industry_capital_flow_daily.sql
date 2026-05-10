-- ============================================================================
-- sql/create_cn_industry_capital_flow_daily.sql
-- GrowthAlpha V8 — P3 Unified Alpha Engine
-- DDL for cn_industry_capital_flow_daily
--
-- Industry-level capital flow and concentration metrics computed daily from
-- stock price, volume, and industry membership data.
-- ============================================================================

CREATE TABLE IF NOT EXISTS `cn_industry_capital_flow_daily` (
    `trade_date`                date            NOT NULL COMMENT '交易日期',
    `industry_id`               varchar(32)     NOT NULL COMMENT '行业ID (SW L1)',
    `industry_name`             varchar(128)    DEFAULT NULL COMMENT '行业名称',
    `total_amount`              decimal(24,4)   DEFAULT NULL COMMENT '行业总成交额',
    `total_turnover`            decimal(24,4)   DEFAULT NULL COMMENT '行业总换手率',
    `avg_change_pct`            decimal(18,8)   DEFAULT NULL COMMENT '行业平均涨跌幅',
    `volume_ratio`              decimal(18,8)   DEFAULT NULL COMMENT '量比 (当日量/均量)',
    `market_share`              decimal(18,8)   DEFAULT NULL COMMENT '全市场成交额占比 [0,1]',
    `amount_rank`               int             DEFAULT NULL COMMENT '成交额排名 (1=最高)',
    `flow_strength_score`       decimal(18,8)   DEFAULT NULL COMMENT '资金流强度评分 [0,1]',
    `rotation_speed_score`      decimal(18,8)   DEFAULT NULL COMMENT '轮动速度评分 [0,1]',
    `concentration_score`       decimal(18,8)   DEFAULT NULL COMMENT '集中度评分 [0,1]',
    `capital_flow_score`        decimal(18,8)   DEFAULT NULL COMMENT '资金流综合评分 [0,1]',
    `created_at`                timestamp       NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`                timestamp       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`trade_date`, `industry_id`),
    KEY `idx_cn_industry_capital_flow_daily_date` (`trade_date`),
    KEY `idx_cn_industry_capital_flow_daily_score` (`capital_flow_score`),
    KEY `idx_cn_industry_capital_flow_daily_rank` (`amount_rank`, `trade_date`),
    KEY `idx_cn_industry_capital_flow_daily_industry` (`industry_id`, `trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='行业资金流日表 — P3 Unified Alpha Engine';
