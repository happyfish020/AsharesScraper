-- cn_market.cn_stock_sw_l1_latest_v
-- Latest stock to SW L1 mapping view using member_hist validity window.

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `cn_stock_sw_l1_latest_v` AS
WITH latest_trade_date AS (
    SELECT MAX(trade_date) AS trade_date
    FROM cn_stock_daily_price
),
sw_l1_name AS (
    SELECT board_id, board_name
    FROM (
        SELECT
            m.BOARD_ID AS board_id,
            m.BOARD_NAME AS board_name,
            ROW_NUMBER() OVER (
                PARTITION BY m.BOARD_ID
                ORDER BY m.ASOF_DATE DESC
            ) AS rn
        FROM cn_board_industry_master m
        WHERE m.SOURCE = 'TUSHARE_SW2021_L1'
    ) x
    WHERE x.rn = 1
)
SELECT
    p.trade_date,
    p.symbol,
    p.name AS stock_name,
    h.board_id AS sw_l1_id,
    n.board_name AS sw_l1_name
FROM cn_stock_daily_price p
JOIN latest_trade_date d
  ON p.trade_date = d.trade_date
JOIN cn_board_industry_member_hist h
  ON h.symbol COLLATE utf8mb4_unicode_ci = p.symbol
 AND h.source = 'tushare_sw_l1'
 AND p.trade_date >= h.valid_from
 AND p.trade_date <= COALESCE(h.valid_to, DATE('9999-12-31'))
LEFT JOIN sw_l1_name n
  ON n.board_id COLLATE utf8mb4_general_ci = h.board_id COLLATE utf8mb4_general_ci;
