-- ============================================================================
-- sql/ddl/ga_p0_mainline_context_schema.sql
-- GrowthAlpha V8 — P0 Mainline Context Schema
--
-- This DDL extends the GA-layer tables with columns needed by the
-- Mainline Strength Engine, Market Breadth Engine, and Narrative/Context Layer.
--
-- IMPORTANT: MySQL 8 does NOT support ADD COLUMN IF NOT EXISTS natively.
-- Column additions are handled by the Python migration script:
--   scripts/apply_ga_p0_schema_migration.py
--
-- This SQL file contains only:
--   1. CREATE TABLE IF NOT EXISTS for cn_ga_market_context_daily
--   2. CREATE INDEX statements (the Python script wraps these with error handling)
--
-- The Python migration script checks INFORMATION_SCHEMA.COLUMNS before each ALTER.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. cn_ga_market_context_daily — Daily market context snapshot
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `cn_ga_market_context_daily` (
    `trade_date`              DATE          NOT NULL,
    `market_regime`           VARCHAR(64)   DEFAULT NULL,
    `mainline_phase`          VARCHAR(64)   DEFAULT NULL,
    `rotation_state`          VARCHAR(64)   DEFAULT NULL,
    `trend_confidence_score`  DECIMAL(10,4) DEFAULT NULL,
    `risk_context`            VARCHAR(128)  DEFAULT NULL,
    `narrative_summary`       TEXT          DEFAULT NULL,
    `data_quality_status`     VARCHAR(32)   DEFAULT NULL,
    `created_at`              TIMESTAMP     NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`              TIMESTAMP     NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`trade_date`),
    KEY `idx_cn_ga_market_context_daily_regime`   (`market_regime`),
    KEY `idx_cn_ga_market_context_daily_phase`    (`mainline_phase`),
    KEY `idx_cn_ga_market_context_daily_rotation` (`rotation_state`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ---------------------------------------------------------------------------
-- 2. Indexes for cn_ga_mainline_radar_daily
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS `idx_cn_ga_mainline_radar_daily_date`
    ON `cn_ga_mainline_radar_daily` (`trade_date`);
CREATE INDEX IF NOT EXISTS `idx_cn_ga_mainline_radar_daily_mainline`
    ON `cn_ga_mainline_radar_daily` (`mainline_id`, `trade_date`);
CREATE INDEX IF NOT EXISTS `idx_cn_ga_mainline_radar_daily_state`
    ON `cn_ga_mainline_radar_daily` (`mainline_state`, `trade_date`);

-- ---------------------------------------------------------------------------
-- 3. Indexes for cn_ga_market_pulse_daily
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS `idx_cn_ga_market_pulse_daily_date`
    ON `cn_ga_market_pulse_daily` (`trade_date`);
CREATE INDEX IF NOT EXISTS `idx_cn_ga_market_pulse_daily_state`
    ON `cn_ga_market_pulse_daily` (`market_state`, `trade_date`);

-- ---------------------------------------------------------------------------
-- 4. Indexes for cn_ga_stock_role_map_daily
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS `idx_cn_ga_stock_role_map_daily_date`
    ON `cn_ga_stock_role_map_daily` (`trade_date`);
CREATE INDEX IF NOT EXISTS `idx_cn_ga_stock_role_map_daily_symbol`
    ON `cn_ga_stock_role_map_daily` (`symbol`, `trade_date`);
CREATE INDEX IF NOT EXISTS `idx_cn_ga_stock_role_map_daily_role`
    ON `cn_ga_stock_role_map_daily` (`stock_role`, `trade_date`);
CREATE INDEX IF NOT EXISTS `idx_cn_ga_stock_role_map_daily_mainline`
    ON `cn_ga_stock_role_map_daily` (`mainline_id`, `trade_date`);
