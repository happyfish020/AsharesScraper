-- ============================================================================
-- sql/create_unified_alpha_score_daily.sql
-- GrowthAlpha V8 — P3 Unified Alpha Engine
-- DDL for cn_unified_alpha_score_daily
-- ============================================================================

CREATE TABLE IF NOT EXISTS `cn_unified_alpha_score_daily` (
    `trade_date`                    date            NOT NULL,
    `symbol`                        varchar(10)     NOT NULL,
    `industry_id`                   varchar(32)     DEFAULT NULL COMMENT 'SW industry ID (L1)',
    `quality_score`                 decimal(18,8)   DEFAULT NULL COMMENT 'Factor 1: Quality Score [0,1]',
    `growth_acceleration_score`     decimal(18,8)   DEFAULT NULL COMMENT 'Factor 2: Growth Acceleration [0,1]',
    `mainline_strength_score`       decimal(18,8)   DEFAULT NULL COMMENT 'Factor 3: Mainline Strength [0,1]',
    `capital_concentration_score`   decimal(18,8)   DEFAULT NULL COMMENT 'Factor 4: Capital Concentration [0,1]',
    `leader_dominance_score`        decimal(18,8)   DEFAULT NULL COMMENT 'Factor 5: Leader Dominance [0,1]',
    `trend_quality_score`           decimal(18,8)   DEFAULT NULL COMMENT 'Factor 6: Trend Quality [0,1]',
    `lifecycle_position_score`      decimal(18,8)   DEFAULT NULL COMMENT 'Factor 7: Lifecycle Position [0,1]',
    `risk_crowding_score`           decimal(18,8)   DEFAULT NULL COMMENT 'Factor 8: Risk/Crowding [0,1]',
    `final_score`                   decimal(18,8)   DEFAULT NULL COMMENT 'Weighted final score [0,1]',
    `alpha_bucket`                  varchar(16)     DEFAULT NULL COMMENT 'Bucket: TOP_1/TOP_5/TOP_10/TOP_20/WATCH/NEUTRAL/AVOID',
    `lifecycle_state`               varchar(24)     DEFAULT NULL COMMENT 'Current lifecycle state of mainline',
    `mainline_name`                 varchar(128)    DEFAULT NULL COMMENT 'Mainline (industry) name',
    `explanation`                   text            DEFAULT NULL COMMENT 'Natural language explanation',
    `top_factors`                   varchar(256)    DEFAULT NULL COMMENT 'Comma-separated top contributing factors',
    `weak_factors`                  varchar(256)    DEFAULT NULL COMMENT 'Comma-separated weak/detracting factors',
    `flags`                         varchar(256)    DEFAULT NULL COMMENT 'Comma-separated flags',
    `created_at`                    timestamp       NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`                    timestamp       NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`trade_date`, `symbol`),
    KEY `idx_unified_alpha_score_daily_final_score` (`final_score`),
    KEY `idx_unified_alpha_score_daily_alpha_bucket` (`alpha_bucket`),
    KEY `idx_unified_alpha_score_daily_industry_id` (`industry_id`),
    KEY `idx_unified_alpha_score_daily_lifecycle_state` (`lifecycle_state`),
    KEY `idx_unified_alpha_score_daily_symbol` (`symbol`, `trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
