CREATE TABLE IF NOT EXISTS cn_event_disclosure_date (
    symbol          VARCHAR(10)    NOT NULL,
    end_date        DATE           NOT NULL,
    pre_date        DATE,
    actual_date     DATE,
    modify_date     DATE,
    source          VARCHAR(30),
    raw_payload     JSON,
    updated_at      DATETIME       DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, end_date),
    KEY idx_disclosure_symbol (symbol),
    KEY idx_disclosure_end_date (end_date)
);
