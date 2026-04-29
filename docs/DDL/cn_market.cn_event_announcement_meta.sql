CREATE TABLE IF NOT EXISTS cn_event_announcement_meta (
    symbol      VARCHAR(10)    NOT NULL,
    ann_date    DATE           NOT NULL,
    title       VARCHAR(500),
    url         VARCHAR(1000),
    type        VARCHAR(100),
    source      VARCHAR(30),
    raw_payload JSON,
    updated_at  DATETIME       DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, ann_date, title),
    KEY idx_ann_symbol (symbol),
    KEY idx_ann_ann_date (ann_date)
);
