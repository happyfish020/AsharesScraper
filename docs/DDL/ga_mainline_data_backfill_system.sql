CREATE TABLE IF NOT EXISTS `cn_local_industry_master` (
    `industry_id` varchar(32) NOT NULL,
    `industry_name` varchar(128) NOT NULL,
    `industry_level` varchar(8) NOT NULL,
    `parent_id` varchar(32) DEFAULT NULL,
    `src` varchar(32) NOT NULL,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`industry_id`),
    KEY `idx_cn_local_industry_master_level` (`industry_level`),
    KEY `idx_cn_local_industry_master_parent` (`parent_id`),
    KEY `idx_cn_local_industry_master_src` (`src`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `cn_local_industry_map_hist` (
    `symbol` varchar(10) NOT NULL,
    `industry_id` varchar(32) NOT NULL,
    `industry_name` varchar(128) NOT NULL,
    `industry_level` varchar(8) NOT NULL,
    `in_date` date NOT NULL,
    `out_date` date DEFAULT NULL,
    `is_current` tinyint(1) NOT NULL DEFAULT 0,
    `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`symbol`, `industry_id`, `in_date`),
    KEY `idx_cn_local_industry_map_hist_industry_date` (`industry_id`, `in_date`, `out_date`),
    KEY `idx_cn_local_industry_map_hist_symbol_date` (`symbol`, `in_date`, `out_date`),
    KEY `idx_cn_local_industry_map_hist_current` (`industry_id`, `is_current`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `cn_local_industry_proxy_daily` (
    `industry_id` varchar(32) NOT NULL,
    `trade_date` date NOT NULL,
    `member_count` int NOT NULL,
    `ret_eqw` decimal(18,8) DEFAULT NULL,
    `amount_total` decimal(24,4) DEFAULT NULL,
    `turnover_avg` decimal(18,6) DEFAULT NULL,
    `market_cap_total` decimal(24,4) DEFAULT NULL,
    `leader_return` decimal(18,8) DEFAULT NULL,
    `top5_concentration` decimal(18,8) DEFAULT NULL,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`industry_id`, `trade_date`),
    KEY `idx_cn_local_industry_proxy_daily_trade_date` (`trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `cn_industry_capital_flow_daily` (
    `trade_date` date NOT NULL,
    `industry_id` varchar(32) NOT NULL,
    `industry_turnover` decimal(24,4) DEFAULT NULL,
    `market_turnover_ratio` decimal(18,8) DEFAULT NULL,
    `industry_return` decimal(18,8) DEFAULT NULL,
    `relative_return` decimal(18,8) DEFAULT NULL,
    `leader_count` int NOT NULL DEFAULT 0,
    `breakout_count` int NOT NULL DEFAULT 0,
    `trend_count` int NOT NULL DEFAULT 0,
    `capital_concentration` decimal(18,8) DEFAULT NULL,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`trade_date`, `industry_id`),
    KEY `idx_cn_industry_capital_flow_daily_industry` (`industry_id`, `trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `cn_local_stock_income_q` (
    `symbol` varchar(10) NOT NULL,
    `end_date` date NOT NULL,
    `ann_date` date DEFAULT NULL,
    `f_ann_date` date DEFAULT NULL,
    `report_type` varchar(32) DEFAULT NULL,
    `total_revenue` decimal(24,4) DEFAULT NULL,
    `revenue` decimal(24,4) DEFAULT NULL,
    `n_income_attr_p` decimal(24,4) DEFAULT NULL,
    `source` varchar(64) DEFAULT NULL,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`symbol`, `end_date`),
    KEY `idx_cn_local_stock_income_q_ann_date` (`ann_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `cn_local_stock_balancesheet_q` (
    `symbol` varchar(10) NOT NULL,
    `end_date` date NOT NULL,
    `ann_date` date DEFAULT NULL,
    `f_ann_date` date DEFAULT NULL,
    `report_type` varchar(32) DEFAULT NULL,
    `inventory` decimal(24,4) DEFAULT NULL,
    `contract_liability` decimal(24,4) DEFAULT NULL,
    `fixed_assets` decimal(24,4) DEFAULT NULL,
    `total_assets` decimal(24,4) DEFAULT NULL,
    `total_liab` decimal(24,4) DEFAULT NULL,
    `source` varchar(64) DEFAULT NULL,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`symbol`, `end_date`),
    KEY `idx_cn_local_stock_balancesheet_q_ann_date` (`ann_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `cn_local_stock_fina_indicator_q` (
    `symbol` varchar(10) NOT NULL,
    `end_date` date NOT NULL,
    `ann_date` date DEFAULT NULL,
    `report_type` varchar(32) DEFAULT NULL,
    `revenue_yoy` decimal(18,6) DEFAULT NULL,
    `profit_yoy` decimal(18,6) DEFAULT NULL,
    `roe` decimal(18,6) DEFAULT NULL,
    `gross_margin` decimal(18,6) DEFAULT NULL,
    `debt_to_assets` decimal(18,6) DEFAULT NULL,
    `ocfps` decimal(18,6) DEFAULT NULL,
    `source` varchar(64) DEFAULT NULL,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`symbol`, `end_date`),
    KEY `idx_cn_local_stock_fina_indicator_q_ann_date` (`ann_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `cn_stock_fundamental_daily` (
    `symbol` varchar(10) NOT NULL,
    `trade_date` date NOT NULL,
    `report_end_date` date DEFAULT NULL,
    `ann_date` date DEFAULT NULL,
    `revenue_yoy` decimal(18,6) DEFAULT NULL,
    `profit_yoy` decimal(18,6) DEFAULT NULL,
    `roe` decimal(18,6) DEFAULT NULL,
    `gross_margin` decimal(18,6) DEFAULT NULL,
    `debt_to_assets` decimal(18,6) DEFAULT NULL,
    `ocfps` decimal(18,6) DEFAULT NULL,
    `inventory` decimal(24,4) DEFAULT NULL,
    `contract_liability` decimal(24,4) DEFAULT NULL,
    `fixed_assets` decimal(24,4) DEFAULT NULL,
    `source` varchar(64) DEFAULT NULL,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`symbol`, `trade_date`),
    KEY `idx_cn_stock_fundamental_daily_trade_date` (`trade_date`),
    KEY `idx_cn_stock_fundamental_daily_report_end_date` (`report_end_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `cn_stock_mainline_strength_daily` (
    `trade_date` date NOT NULL,
    `mainline_name` varchar(128) NOT NULL,
    `strength_score` decimal(18,6) DEFAULT NULL,
    `leader_count` int NOT NULL DEFAULT 0,
    `capital_ratio` decimal(18,8) DEFAULT NULL,
    `earnings_score` decimal(18,6) DEFAULT NULL,
    `trend_days` int NOT NULL DEFAULT 0,
    `expansion_score` decimal(18,6) DEFAULT NULL,
    `lifecycle_state` varchar(24) DEFAULT NULL,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`trade_date`, `mainline_name`),
    KEY `idx_cn_stock_mainline_strength_daily_mainline` (`mainline_name`, `trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS `cn_mainline_backfill_job_state` (
    `job_name` varchar(64) NOT NULL,
    `chunk_key` varchar(128) NOT NULL,
    `range_start` date DEFAULT NULL,
    `range_end` date DEFAULT NULL,
    `status` varchar(16) NOT NULL,
    `attempts` int NOT NULL DEFAULT 0,
    `last_rows` bigint NOT NULL DEFAULT 0,
    `last_error` text DEFAULT NULL,
    `last_run_id` varchar(64) DEFAULT NULL,
    `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`job_name`, `chunk_key`),
    KEY `idx_cn_mainline_backfill_job_state_status` (`status`, `updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
