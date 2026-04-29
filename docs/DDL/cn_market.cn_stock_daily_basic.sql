-- cn_market.cn_stock_daily_basic
-- Daily stock basic metrics loaded from Tushare `daily_basic`.

CREATE TABLE IF NOT EXISTS `cn_stock_daily_basic` (
    `symbol` varchar(10) NOT NULL,
    `trade_date` date NOT NULL,
    `total_share` decimal(24,4) DEFAULT NULL,
    `float_share` decimal(24,4) DEFAULT NULL,
    `free_share` decimal(24,4) DEFAULT NULL,
    `total_mv` decimal(24,4) DEFAULT NULL,
    `circ_mv` decimal(24,4) DEFAULT NULL,
    `pe` decimal(18,6) DEFAULT NULL,
    `pe_ttm` decimal(18,6) DEFAULT NULL,
    `pb` decimal(18,6) DEFAULT NULL,
    `ps` decimal(18,6) DEFAULT NULL,
    `ps_ttm` decimal(18,6) DEFAULT NULL,
    `dv_ratio` decimal(18,6) DEFAULT NULL,
    `dv_ttm` decimal(18,6) DEFAULT NULL,
    `turnover_rate_f` decimal(18,6) DEFAULT NULL,
    `volume_ratio` decimal(18,6) DEFAULT NULL,
    `source` varchar(32) DEFAULT NULL,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`symbol`, `trade_date`),
    KEY `idx_cn_stock_daily_basic_trade_date` (`trade_date`),
    KEY `idx_cn_stock_daily_basic_total_mv` (`trade_date`, `total_mv`),
    KEY `idx_cn_stock_daily_basic_circ_mv` (`trade_date`, `circ_mv`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
