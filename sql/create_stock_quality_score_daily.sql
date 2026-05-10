-- ============================================================================
-- sql/create_stock_quality_score_daily.sql
-- GrowthAlpha V8 — P3 Unified Alpha Engine
-- DDL for cn_stock_quality_score_daily
-- ============================================================================

CREATE TABLE IF NOT EXISTS `cn_stock_quality_score_daily` (
    `trade_date`                    date            NOT NULL,
    `symbol`                        varchar(10)     NOT NULL,
    `quality_score`                 decimal(18,8)   DEFAULT NULL COMMENT 'Composite quality score [0,1]',
    `growth_acceleration_score`     decimal(18,8)   DEFAULT NULL COMMENT 'Revenue/profit growth acceleration [0,1]',
    `cashflow_score`                decimal(18,8)   DEFAULT NULL COMMENT 'Operating cash flow quality [0,1]',
    `debt_control_score`            decimal(18,8)   DEFAULT NULL COMMENT 'Debt control / leverage [0,1]',
    `margin_stability_score`        decimal(18,8)   DEFAULT NULL COMMENT 'Gross margin stability [0,1]',
    `profitability_score`           decimal(18,8)   DEFAULT NULL COMMENT 'ROE / profitability [0,1]',
    `report_end_date`               date            DEFAULT NULL COMMENT 'Latest financial report end date',
    `ann_date`                      date            DEFAULT NULL COMMENT 'Announcement date of the report used',
    `fundamental_risk_flag`         varchar(32)     DEFAULT NULL COMMENT 'Risk flag: NONE / HIGH_DEBT / NEGATIVE_EARNINGS / CASHFLOW_WEAK / DATA_INSUFFICIENT',
    `reason`                        varchar(512)    DEFAULT NULL COMMENT 'Computation reason / explanation',
    `created_at`                    timestamp       NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`                    timestamp       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`trade_date`, `symbol`),
    KEY `idx_stock_quality_score_daily_symbol` (`symbol`, `trade_date`),
    KEY `idx_stock_quality_score_daily_quality` (`quality_score`),
    KEY `idx_stock_quality_score_daily_risk_flag` (`fundamental_risk_flag`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
