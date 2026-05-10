-- ============================================================================
-- cn_market_red_p0_mainline_data_fix.sql
-- GrowthAlpha V8 — P0 Mainline Data Foundation DDL
--
-- Idempotent schema migration:
--   - ALTER ADD missing columns (never DROP TABLE, never destroy data)
--   - CREATE TABLE IF NOT EXISTS for cn_ga_data_readiness_daily
--   - CREATE INDEX IF NOT EXISTS (via stored procedure workaround)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. cn_local_industry_map_hist
-- ----------------------------------------------------------------------------
-- Ensure all required columns exist.
-- Existing DDL already has: symbol, industry_id, industry_name, industry_level,
-- in_date, out_date, is_current, updated_at.
-- We add `source` if missing.

SET @db_name = DATABASE();

-- Add `source` column if missing
SET @sql = NULL;
SELECT COUNT(*) INTO @cnt
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = @db_name
  AND TABLE_NAME = 'cn_local_industry_map_hist'
  AND COLUMN_NAME = 'source';
SET @sql = IF(@cnt = 0,
    'ALTER TABLE `cn_local_industry_map_hist` ADD COLUMN `source` VARCHAR(32) DEFAULT NULL AFTER `is_current`',
    NULL);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Ensure indexes exist on cn_local_industry_map_hist
-- idx_symbol_date(symbol, in_date, out_date)
SET @idx_exists = NULL;
SELECT COUNT(*) INTO @idx_exists
FROM INFORMATION_SCHEMA.STATISTICS
WHERE TABLE_SCHEMA = @db_name
  AND TABLE_NAME = 'cn_local_industry_map_hist'
  AND INDEX_NAME = 'idx_symbol_date';
SET @sql = IF(@idx_exists = 0,
    'CREATE INDEX `idx_symbol_date` ON `cn_local_industry_map_hist` (`symbol`, `in_date`, `out_date`)',
    NULL);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- idx_industry_date(industry_id, in_date, out_date)
SET @idx_exists = NULL;
SELECT COUNT(*) INTO @idx_exists
FROM INFORMATION_SCHEMA.STATISTICS
WHERE TABLE_SCHEMA = @db_name
  AND TABLE_NAME = 'cn_local_industry_map_hist'
  AND INDEX_NAME = 'idx_industry_date';
SET @sql = IF(@idx_exists = 0,
    'CREATE INDEX `idx_industry_date` ON `cn_local_industry_map_hist` (`industry_id`, `in_date`, `out_date`)',
    NULL);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- idx_current(is_current)
SET @idx_exists = NULL;
SELECT COUNT(*) INTO @idx_exists
FROM INFORMATION_SCHEMA.STATISTICS
WHERE TABLE_SCHEMA = @db_name
  AND TABLE_NAME = 'cn_local_industry_map_hist'
  AND INDEX_NAME = 'idx_current';
SET @sql = IF(@idx_exists = 0,
    'CREATE INDEX `idx_current` ON `cn_local_industry_map_hist` (`is_current`)',
    NULL);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- ----------------------------------------------------------------------------
-- 2. cn_local_industry_proxy_daily
-- ----------------------------------------------------------------------------
-- Ensure amount_total column exists
SET @sql = NULL;
SELECT COUNT(*) INTO @cnt
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = @db_name
  AND TABLE_NAME = 'cn_local_industry_proxy_daily'
  AND COLUMN_NAME = 'amount_total';
SET @sql = IF(@cnt = 0,
    'ALTER TABLE `cn_local_industry_proxy_daily` ADD COLUMN `amount_total` DECIMAL(24,4) DEFAULT NULL AFTER `ret_eqw`',
    NULL);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Ensure industry_name column exists
SET @sql = NULL;
SELECT COUNT(*) INTO @cnt
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = @db_name
  AND TABLE_NAME = 'cn_local_industry_proxy_daily'
  AND COLUMN_NAME = 'industry_name';
SET @sql = IF(@cnt = 0,
    'ALTER TABLE `cn_local_industry_proxy_daily` ADD COLUMN `industry_name` VARCHAR(128) DEFAULT NULL AFTER `industry_id`',
    NULL);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Ensure unique key uk_industry_trade_date exists
SET @uk_exists = NULL;
SELECT COUNT(*) INTO @uk_exists
FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
WHERE TABLE_SCHEMA = @db_name
  AND TABLE_NAME = 'cn_local_industry_proxy_daily'
  AND CONSTRAINT_NAME = 'uk_industry_trade_date'
  AND CONSTRAINT_TYPE = 'UNIQUE';
SET @sql = IF(@uk_exists = 0,
    'ALTER TABLE `cn_local_industry_proxy_daily` ADD UNIQUE KEY `uk_industry_trade_date` (`industry_id`, `trade_date`)',
    NULL);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Ensure idx_trade_date exists
SET @idx_exists = NULL;
SELECT COUNT(*) INTO @idx_exists
FROM INFORMATION_SCHEMA.STATISTICS
WHERE TABLE_SCHEMA = @db_name
  AND TABLE_NAME = 'cn_local_industry_proxy_daily'
  AND INDEX_NAME = 'idx_trade_date';
SET @sql = IF(@idx_exists = 0,
    'CREATE INDEX `idx_trade_date` ON `cn_local_industry_proxy_daily` (`trade_date`)',
    NULL);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Ensure idx_industry_id exists
SET @idx_exists = NULL;
SELECT COUNT(*) INTO @idx_exists
FROM INFORMATION_SCHEMA.STATISTICS
WHERE TABLE_SCHEMA = @db_name
  AND TABLE_NAME = 'cn_local_industry_proxy_daily'
  AND INDEX_NAME = 'idx_industry_id';
SET @sql = IF(@idx_exists = 0,
    'CREATE INDEX `idx_industry_id` ON `cn_local_industry_proxy_daily` (`industry_id`)',
    NULL);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- ----------------------------------------------------------------------------
-- 3. cn_ga_data_readiness_daily
-- ----------------------------------------------------------------------------
-- Create if not exists with the required fields
CREATE TABLE IF NOT EXISTS `cn_ga_data_readiness_daily` (
    `id` BIGINT NOT NULL AUTO_INCREMENT,
    `trade_date` DATE NOT NULL,
    `table_name` VARCHAR(128) NOT NULL,
    `status` VARCHAR(32) NOT NULL DEFAULT 'UNKNOWN',
    `severity` VARCHAR(16) DEFAULT NULL,
    `row_count` BIGINT NOT NULL DEFAULT 0,
    `max_trade_date` DATE DEFAULT NULL,
    `null_rate_summary` TEXT DEFAULT NULL,
    `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_cn_ga_data_readiness_daily` (`trade_date`, `table_name`),
    KEY `idx_cn_ga_data_readiness_daily_trade_date` (`trade_date`),
    KEY `idx_cn_ga_data_readiness_daily_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Ensure the new columns exist if table already existed
SET @sql = NULL;
SELECT COUNT(*) INTO @cnt
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = @db_name
  AND TABLE_NAME = 'cn_ga_data_readiness_daily'
  AND COLUMN_NAME = 'trade_date';
SET @sql = IF(@cnt = 0,
    'ALTER TABLE `cn_ga_data_readiness_daily` ADD COLUMN `trade_date` DATE DEFAULT NULL AFTER `id`',
    NULL);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @sql = NULL;
SELECT COUNT(*) INTO @cnt
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = @db_name
  AND TABLE_NAME = 'cn_ga_data_readiness_daily'
  AND COLUMN_NAME = 'severity';
SET @sql = IF(@cnt = 0,
    'ALTER TABLE `cn_ga_data_readiness_daily` ADD COLUMN `severity` VARCHAR(16) DEFAULT NULL AFTER `status`',
    NULL);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @sql = NULL;
SELECT COUNT(*) INTO @cnt
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = @db_name
  AND TABLE_NAME = 'cn_ga_data_readiness_daily'
  AND COLUMN_NAME = 'null_rate_summary';
SET @sql = IF(@cnt = 0,
    'ALTER TABLE `cn_ga_data_readiness_daily` ADD COLUMN `null_rate_summary` TEXT DEFAULT NULL AFTER `max_trade_date`',
    NULL);
PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- ----------------------------------------------------------------------------
-- 4. Ensure cn_mainline_backfill_job_state exists (used by all backfill scripts)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `cn_mainline_backfill_job_state` (
    `job_name` VARCHAR(64) NOT NULL,
    `chunk_key` VARCHAR(128) NOT NULL,
    `range_start` DATE DEFAULT NULL,
    `range_end` DATE DEFAULT NULL,
    `status` VARCHAR(16) NOT NULL,
    `attempts` INT NOT NULL DEFAULT 0,
    `last_rows` BIGINT NOT NULL DEFAULT 0,
    `last_error` TEXT DEFAULT NULL,
    `last_run_id` VARCHAR(64) DEFAULT NULL,
    `updated_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`job_name`, `chunk_key`),
    KEY `idx_cn_mainline_backfill_job_state_status` (`status`, `updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

SELECT 'P0 mainline data fix DDL applied successfully' AS migration_status;
