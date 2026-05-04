-- cn_market.cn_stock_cashflow
-- Tushare Pro cashflow quarterly statements.

CREATE TABLE IF NOT EXISTS `cn_stock_cashflow` (
    `symbol` varchar(10) NOT NULL,
    `end_date` date NOT NULL,
    `ann_date` date DEFAULT NULL,
    `f_ann_date` date DEFAULT NULL,
    `report_type` varchar(32) DEFAULT NULL,
    `comp_type` varchar(8) DEFAULT NULL,
    `end_type` varchar(8) DEFAULT NULL,
    `n_cashflow_act` decimal(24,4) DEFAULT NULL,
    `n_cash_flows_inv_act` decimal(24,4) DEFAULT NULL,
    `n_cash_flows_fnc_act` decimal(24,4) DEFAULT NULL,
    `source` varchar(64) DEFAULT NULL,
    `raw_payload` longtext DEFAULT NULL,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`symbol`, `end_date`),
    KEY `idx_cn_stock_cashflow_ann_date` (`ann_date`),
    KEY `idx_cn_stock_cashflow_report_type` (`report_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
