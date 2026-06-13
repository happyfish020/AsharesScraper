CREATE TABLE IF NOT EXISTS cn_p1f_theme_taxonomy (
    theme_id VARCHAR(64) NOT NULL PRIMARY KEY,
    theme_name VARCHAR(128) NOT NULL,
    is_active TINYINT NOT NULL DEFAULT 1,
    source VARCHAR(64) NOT NULL DEFAULT 'P1F_AUDIT0',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cn_p1f_stock_theme_map (
    stock_code VARCHAR(32) NOT NULL,
    symbol VARCHAR(32) NULL,
    stock_name VARCHAR(128) NULL,
    theme_id VARCHAR(64) NOT NULL,
    theme_name VARCHAR(128) NULL,
    confidence DECIMAL(10,4) NULL,
    effective_start DATE NULL,
    effective_end DATE NULL,
    is_active TINYINT NOT NULL DEFAULT 1,
    source VARCHAR(64) NOT NULL DEFAULT 'P1F_MANUAL_MAPPING',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_code, theme_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cn_p1f_audit_theme_exposure (
    run_date DATE NOT NULL,
    report_period DATE NOT NULL,
    theme_id VARCHAR(64) NOT NULL,
    theme_name VARCHAR(128) NOT NULL,
    institution_count INT NOT NULL DEFAULT 0,
    fund_count INT NOT NULL DEFAULT 0,
    stock_count INT NOT NULL DEFAULT 0,
    market_value DECIMAL(24,4) NOT NULL DEFAULT 0,
    total_mapped_market_value DECIMAL(24,4) NOT NULL DEFAULT 0,
    exposure_weight_pct DECIMAL(16,8) NOT NULL DEFAULT 0,
    mapping_source VARCHAR(64) NOT NULL,
    source VARCHAR(64) NOT NULL DEFAULT 'P1F_AUDIT0',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (run_date, report_period, theme_id, mapping_source)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cn_p1f_audit_theme_delta (
    run_date DATE NOT NULL,
    report_period DATE NOT NULL,
    previous_report_period DATE NULL,
    theme_id VARCHAR(64) NOT NULL,
    theme_name VARCHAR(128) NOT NULL,
    exposure_weight_pct DECIMAL(16,8) NOT NULL DEFAULT 0,
    previous_exposure_weight_pct DECIMAL(16,8) NOT NULL DEFAULT 0,
    delta_qoq_pct DECIMAL(16,8) NOT NULL DEFAULT 0,
    institution_count INT NOT NULL DEFAULT 0,
    previous_institution_count INT NOT NULL DEFAULT 0,
    delta_institution_count INT NOT NULL DEFAULT 0,
    discovery_flag TINYINT NOT NULL DEFAULT 0,
    discovery_reason VARCHAR(255) NULL,
    mapping_source VARCHAR(64) NOT NULL,
    source VARCHAR(64) NOT NULL DEFAULT 'P1F_AUDIT0',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (run_date, report_period, theme_id, mapping_source)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS cn_p1f_audit_recall_precision (
    run_date DATE NOT NULL,
    report_period DATE NOT NULL,
    lookahead_end_period DATE NOT NULL,
    mapping_source VARCHAR(64) NOT NULL,
    discovered_theme_count INT NOT NULL DEFAULT 0,
    actual_mainline_count INT NOT NULL DEFAULT 0,
    hit_count INT NOT NULL DEFAULT 0,
    false_positive_count INT NOT NULL DEFAULT 0,
    recall_rate DECIMAL(10,4) NULL,
    precision_rate DECIMAL(10,4) NULL,
    validation_gate VARCHAR(32) NOT NULL DEFAULT 'RESEARCH_ONLY',
    gate_reason VARCHAR(255) NULL,
    source VARCHAR(64) NOT NULL DEFAULT 'P1F_AUDIT0',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (run_date, report_period, mapping_source)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
