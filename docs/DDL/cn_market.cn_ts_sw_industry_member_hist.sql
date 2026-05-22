CREATE TABLE IF NOT EXISTS `cn_ts_sw_industry_member_hist` (
    `src` VARCHAR(16) NOT NULL,
    `industry_level` VARCHAR(4) NOT NULL,
    `industry_id` VARCHAR(32) NOT NULL,
    `industry_name` VARCHAR(128) DEFAULT NULL,
    `symbol` VARCHAR(16) NOT NULL,
    `ts_code` VARCHAR(16) DEFAULT NULL,
    `stock_name` VARCHAR(128) DEFAULT NULL,
    `in_date` DATE NOT NULL,
    `out_date` DATE DEFAULT NULL,
    `is_new` VARCHAR(8) DEFAULT NULL,
    `provider` VARCHAR(16) NOT NULL DEFAULT 'TUSHARE',
    `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`src`, `industry_level`, `industry_id`, `symbol`, `in_date`),
    KEY `idx_ts_sw_member_symbol_date` (`symbol`, `in_date`, `out_date`),
    KEY `idx_ts_sw_member_industry_date` (`industry_id`, `in_date`, `out_date`),
    KEY `idx_ts_sw_member_ts_code` (`ts_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
