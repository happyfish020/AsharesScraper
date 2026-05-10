-- ============================================================================
-- sql/ddl/ga_p2_mainline_lifecycle_schema.sql
-- GrowthAlpha V8 — P2 Mainline Lifecycle Schema
--
-- This DDL creates the cn_mainline_lifecycle_daily table used by the
-- Mainline Lifecycle Engine for lifecycle state classification,
-- rotation ranking, and risk flagging.
--
-- Safe to run multiple times via CREATE TABLE IF NOT EXISTS.
-- Column additions are handled by the Python migration script:
--   scripts/apply_ga_p0_schema_migration.py
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. cn_mainline_lifecycle_daily — Daily lifecycle state per mainline
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `cn_mainline_lifecycle_daily` (
    `trade_date`                   DATE          NOT NULL COMMENT '交易日期',
    `mainline_id`                  VARCHAR(32)   NOT NULL COMMENT '主线ID (industry_id)',
    `mainline_name`                VARCHAR(128)  DEFAULT NULL COMMENT '主线名称',
    `mainline_strength`            DECIMAL(10,4) DEFAULT NULL COMMENT '主线强度评分',
    `capital_concentration_score`  DECIMAL(10,4) DEFAULT NULL COMMENT '资本集中度评分',
    `trend_alignment_score`        DECIMAL(10,4) DEFAULT NULL COMMENT '趋势一致性评分',
    `breakout_ratio`               DECIMAL(10,4) DEFAULT NULL COMMENT '突破比例',
    `new_high_ratio`               DECIMAL(10,4) DEFAULT NULL COMMENT '新高比例',
    `leader_density`               DECIMAL(10,4) DEFAULT NULL COMMENT '龙头密度',
    `rotation_rank`                INT           DEFAULT NULL COMMENT '轮动排名 (1=best)',
    `lifecycle_state`              VARCHAR(32)   DEFAULT NULL COMMENT '生命周期状态',
    `lifecycle_score`              DECIMAL(10,4) DEFAULT NULL COMMENT '生命周期评分 [0,1]',
    `phase_reason`                 VARCHAR(256)  DEFAULT NULL COMMENT '状态判定原因',
    `risk_flag`                    VARCHAR(32)   DEFAULT 'NONE' COMMENT '风险标记',
    `created_at`                   TIMESTAMP     NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at`                   TIMESTAMP     NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`trade_date`, `mainline_id`),
    KEY `idx_mainline_lifecycle_daily_date`     (`trade_date`),
    KEY `idx_mainline_lifecycle_daily_state`    (`lifecycle_state`, `trade_date`),
    KEY `idx_mainline_lifecycle_daily_rank`     (`rotation_rank`, `trade_date`),
    KEY `idx_mainline_lifecycle_daily_strength` (`mainline_strength`, `trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='主线生命周期状态日表 — P2 Mainline Lifecycle Engine';
