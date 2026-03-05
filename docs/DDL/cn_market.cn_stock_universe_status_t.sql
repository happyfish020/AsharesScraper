-- cn_market.cn_stock_universe_status_t
-- Stock universe status table used by SP/statistics filters.

CREATE TABLE IF NOT EXISTS `cn_stock_universe_status_t` (
    `symbol` VARCHAR(32) NOT NULL,
    `is_active` TINYINT(1) NOT NULL DEFAULT 1,
    `inactive_reason` VARCHAR(64) NULL,
    `first_trade_date` DATE NULL,
    `last_trade_date` DATE NULL,
    `recent_trade_days` INT NOT NULL DEFAULT 0,
    `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`symbol`),
    KEY `idx_csus_active` (`is_active`),
    KEY `idx_csus_last_trade` (`last_trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
