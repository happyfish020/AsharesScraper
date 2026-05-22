CREATE TABLE IF NOT EXISTS `cn_ts_sw_industry_master` (
    `src` VARCHAR(16) NOT NULL,
    `industry_level` VARCHAR(4) NOT NULL,
    `industry_id` VARCHAR(32) NOT NULL,
    `industry_name` VARCHAR(128) NOT NULL,
    `parent_id` VARCHAR(32) DEFAULT NULL,
    `is_pub` VARCHAR(8) DEFAULT NULL,
    `provider` VARCHAR(16) NOT NULL DEFAULT 'TUSHARE',
    `asof_date` DATE NOT NULL,
    `raw_json` JSON DEFAULT NULL,
    `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`src`, `industry_level`, `industry_id`),
    KEY `idx_ts_sw_master_parent` (`parent_id`),
    KEY `idx_ts_sw_master_asof` (`asof_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
