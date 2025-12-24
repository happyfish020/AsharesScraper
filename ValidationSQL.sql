-- 中间交易日缺失
WITH base_calendar AS (
    SELECT trade_date
    FROM cn_index_daily_price
    WHERE index_code = 'sh000300'
      AND trade_date BETWEEN TO_DATE(:start_date,'YYYYMMDD')
                          AND TO_DATE(:end_date,'YYYYMMDD')
),
stock_universe AS (
    SELECT DISTINCT symbol, exchange
    FROM cn_stock_daily_price
),
expected AS (
    SELECT
        u.symbol,
        u.exchange,
        c.trade_date
    FROM stock_universe u
    CROSS JOIN base_calendar c
),
actual AS (
    SELECT symbol, exchange, trade_date
    FROM cn_stock_daily_price
    WHERE trade_date BETWEEN TO_DATE(:start_date,'YYYYMMDD')
                          AND TO_DATE(:end_date,'YYYYMMDD')
)
SELECT
    e.symbol,
    e.exchange,
    e.trade_date AS missing_date,
    'GAP'        AS missing_type
FROM expected e
LEFT JOIN actual a
  ON a.symbol = e.symbol
 AND a.exchange = e.exchange
 AND a.trade_date = e.trade_date
WHERE a.trade_date IS NULL
ORDER BY e.symbol, e.trade_date;
