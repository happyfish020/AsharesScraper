CREATE TABLE IF NOT EXISTS cn_event_signal_daily (
    trade_date      DATE           NOT NULL,
    symbol          VARCHAR(10)    NOT NULL,
    event_type      VARCHAR(50)    NOT NULL,
    event_subtype   VARCHAR(50),
    event_score     INT,
    event_direction INT,
    anchor_date     DATE,
    raw_source_table VARCHAR(80),
    raw_event_id    VARCHAR(200),
    version         VARCHAR(20)    DEFAULT 'v1',
    created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, symbol, event_type, event_subtype, raw_event_id, version),
    KEY idx_event_symbol (symbol),
    KEY idx_event_trade_date (trade_date),
    KEY idx_event_anchor_date (anchor_date)
);
