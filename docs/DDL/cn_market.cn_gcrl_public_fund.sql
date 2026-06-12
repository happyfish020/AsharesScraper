-- GrowthAlpha / AshareScraper GCRL CN Public Fund fact tables
-- Prefix rule: every table starts with cn_
-- Scope: data production only. No theme mapping, no score, no buy/sell signal.

CREATE TABLE IF NOT EXISTS cn_gcrl_institution_registry (
    institution_id VARCHAR(64) NOT NULL,
    institution_name VARCHAR(128) NOT NULL,
    institution_short_name VARCHAR(64) NULL,
    country VARCHAR(32) NOT NULL DEFAULT 'CN',
    institution_type VARCHAR(32) NOT NULL,
    tier INT NOT NULL DEFAULT 1,
    source VARCHAR(64) NOT NULL DEFAULT 'manual_seed',
    is_active TINYINT NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (institution_id),
    UNIQUE KEY uk_cn_gcrl_inst_name_type (institution_name, institution_type),
    KEY idx_cn_gcrl_inst_type_tier (institution_type, tier, is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cn_gcrl_fund_registry (
    fund_code VARCHAR(32) NOT NULL,
    fund_name VARCHAR(255) NULL,
    management VARCHAR(128) NULL,
    custodian VARCHAR(128) NULL,
    fund_type VARCHAR(64) NULL,
    found_date DATE NULL,
    due_date DATE NULL,
    list_date DATE NULL,
    issue_date DATE NULL,
    delist_date DATE NULL,
    issue_amount DECIMAL(24,6) NULL,
    m_fee DECIMAL(18,6) NULL,
    c_fee DECIMAL(18,6) NULL,
    duration_year DECIMAL(18,6) NULL,
    p_value DECIMAL(18,6) NULL,
    min_amount DECIMAL(24,6) NULL,
    exp_return DECIMAL(18,6) NULL,
    benchmark TEXT NULL,
    status VARCHAR(32) NULL,
    invest_type VARCHAR(64) NULL,
    type VARCHAR(64) NULL,
    trustee VARCHAR(128) NULL,
    purc_startdate DATE NULL,
    redm_startdate DATE NULL,
    market VARCHAR(32) NULL,
    institution_id VARCHAR(64) NULL,
    source VARCHAR(64) NOT NULL DEFAULT 'tushare_fund_basic',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (fund_code),
    KEY idx_cn_gcrl_fund_inst (institution_id),
    KEY idx_cn_gcrl_fund_mgmt (management),
    KEY idx_cn_gcrl_fund_type_status (fund_type, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cn_gcrl_position_snapshot (
    report_period DATE NOT NULL,
    fund_code VARCHAR(32) NOT NULL,
    institution_id VARCHAR(64) NOT NULL,
    stock_code VARCHAR(16) NOT NULL,
    symbol VARCHAR(8) NOT NULL,
    stock_name VARCHAR(128) NULL,
    ann_date DATE NULL,
    shares DECIMAL(24,4) NULL,
    market_value DECIMAL(24,4) NULL,
    stock_mkv_ratio DECIMAL(18,6) NULL,
    stock_float_ratio DECIMAL(18,6) NULL,
    source VARCHAR(64) NOT NULL DEFAULT 'tushare_fund_portfolio',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (report_period, fund_code, stock_code),
    KEY idx_cn_gcrl_pos_inst_period (institution_id, report_period),
    KEY idx_cn_gcrl_pos_stock_period (stock_code, report_period),
    KEY idx_cn_gcrl_pos_symbol_period (symbol, report_period)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cn_gcrl_position_change (
    report_period DATE NOT NULL,
    prev_report_period DATE NULL,
    fund_code VARCHAR(32) NOT NULL,
    institution_id VARCHAR(64) NOT NULL,
    stock_code VARCHAR(16) NOT NULL,
    symbol VARCHAR(8) NOT NULL,
    stock_name VARCHAR(128) NULL,
    change_type VARCHAR(32) NOT NULL,
    prev_shares DECIMAL(24,4) NULL,
    current_shares DECIMAL(24,4) NULL,
    change_shares DECIMAL(24,4) NULL,
    change_ratio_pct DECIMAL(18,6) NULL,
    prev_market_value DECIMAL(24,4) NULL,
    current_market_value DECIMAL(24,4) NULL,
    market_value_change DECIMAL(24,4) NULL,
    source VARCHAR(64) NOT NULL DEFAULT 'derived_from_cn_gcrl_position_snapshot',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (report_period, fund_code, stock_code),
    KEY idx_cn_gcrl_chg_inst_period (institution_id, report_period, change_type),
    KEY idx_cn_gcrl_chg_stock_period (stock_code, report_period, change_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cn_gcrl_data_freshness (
    dataset_name VARCHAR(96) NOT NULL,
    report_period DATE NOT NULL,
    source VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL,
    row_count INT NOT NULL DEFAULT 0,
    refreshed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    message VARCHAR(512) NULL,
    PRIMARY KEY (dataset_name, report_period, source),
    KEY idx_cn_gcrl_fresh_status (status, refreshed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cn_gcrl_data_source_status (
    source_name VARCHAR(128) NOT NULL,
    source_type VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL,
    message VARCHAR(512) NULL,
    row_count INT NOT NULL DEFAULT 0,
    last_check_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_success_time TIMESTAMP NULL DEFAULT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (source_name),
    KEY idx_cn_gcrl_source_status (source_type, status, last_check_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
