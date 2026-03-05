-- cn_market.cn_stock_daily_price_active_v
-- Active stock price view for SP/statistics.

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `cn_stock_daily_price_active_v` AS
SELECT p.*
FROM cn_stock_daily_price p
JOIN cn_stock_universe_status_t s
  ON s.symbol = p.symbol
WHERE IFNULL(s.is_active, 1) = 1;
