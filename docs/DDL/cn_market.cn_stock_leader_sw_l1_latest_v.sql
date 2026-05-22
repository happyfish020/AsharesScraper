-- cn_market.cn_stock_leader_sw_l1_latest_v
-- Latest leader-score snapshot joined with SW L1 mapping and same-date SW daily行情.
-- Data source changed from cn_stock_leader_score_v2 (view) to
-- cn_stock_leader_score_daily (materialized table via SP).

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `cn_stock_leader_sw_l1_latest_v` AS
WITH leader_latest AS (
    SELECT MAX(trade_date) AS trade_date
    FROM cn_stock_leader_score_daily
),
leader_base AS (
    SELECT l.*
    FROM cn_stock_leader_score_daily l
    JOIN leader_latest d
      ON l.trade_date = d.trade_date
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
    l.trade_date,
    l.symbol,
    l.name AS stock_name,
    l.close AS stock_close,
    l.leader_score,
    l.leader_bucket,
    l.rs_percentile,
    l.turnover_20d_percentile,
    h.board_id AS sw_l1_id,
    n.board_name AS sw_l1_name,
    s.close AS sw_close,
    s.pct_change AS sw_pct_change,
    s.pe AS sw_pe,
    s.pb AS sw_pb,
    s.float_mv AS sw_float_mv
FROM leader_base l
LEFT JOIN cn_board_industry_member_hist h
  ON h.symbol COLLATE utf8mb4_unicode_ci = l.symbol
 AND h.source = 'tushare_sw_l1'
 AND l.trade_date >= h.valid_from
 AND l.trade_date <= COALESCE(h.valid_to, DATE('9999-12-31'))
LEFT JOIN sw_l1_name n
  ON n.board_id COLLATE utf8mb4_general_ci = h.board_id COLLATE utf8mb4_general_ci
LEFT JOIN cn_sw_industry_daily s
  ON s.ts_code COLLATE utf8mb4_general_ci = h.board_id COLLATE utf8mb4_general_ci
 AND s.trade_date = l.trade_date;
