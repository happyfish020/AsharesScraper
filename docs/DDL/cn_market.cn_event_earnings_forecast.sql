CREATE TABLE IF NOT EXISTS cn_event_earnings_forecast (
    symbol          VARCHAR(10)    NOT NULL,
    ann_date        DATE           NOT NULL,
    end_date        DATE           NOT NULL,
    report_type     VARCHAR(20),
    forecast_type   VARCHAR(50),
    p_change_min    DECIMAL(18,6),
    p_change_max    DECIMAL(18,6),
    net_profit_min  DECIMAL(20,2),
    net_profit_max  DECIMAL(20,2),
    summary         VARCHAR(2000),
    source          VARCHAR(30),
    raw_payload     JSON,
    updated_at      DATETIME       DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, ann_date, end_date, forecast_type, report_type),
    KEY idx_forecast_symbol (symbol),
    KEY idx_forecast_ann_date (ann_date),
    KEY idx_forecast_end_date (end_date)
);
