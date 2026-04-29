-- cn_market.cn_sw_industry_daily
-- Tushare Pro `sw_daily` Shenwan industry daily行情表。

CREATE TABLE IF NOT EXISTS `cn_sw_industry_daily` (
    `ts_code` varchar(16) NOT NULL,
    `trade_date` date NOT NULL,
    `name` varchar(80) DEFAULT NULL,
    `open` decimal(18,4) DEFAULT NULL,
    `high` decimal(18,4) DEFAULT NULL,
    `low` decimal(18,4) DEFAULT NULL,
    `close` decimal(18,4) DEFAULT NULL,
    `change` decimal(18,4) DEFAULT NULL,
    `pct_change` decimal(18,6) DEFAULT NULL,
    `vol` decimal(24,4) DEFAULT NULL,
    `amount` decimal(24,4) DEFAULT NULL,
    `pe` decimal(18,6) DEFAULT NULL,
    `pb` decimal(18,6) DEFAULT NULL,
    `float_mv` decimal(24,4) DEFAULT NULL,
    `source` varchar(32) DEFAULT NULL,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`ts_code`, `trade_date`),
    KEY `idx_cn_sw_industry_daily_trade_date` (`trade_date`),
    KEY `idx_cn_sw_industry_daily_name_trade_date` (`name`, `trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
