-- cn_market.cn_stock_active_universe_v
-- Unified active-stock universe for SP/statistics/audit.

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `cn_stock_active_universe_v` AS
SELECT
    s.symbol,
    s.is_active,
    s.inactive_reason,
    s.first_trade_date,
    s.last_trade_date,
    s.recent_trade_days,
    s.updated_at
FROM cn_stock_universe_status_t s
WHERE IFNULL(s.is_active, 1) = 1;
