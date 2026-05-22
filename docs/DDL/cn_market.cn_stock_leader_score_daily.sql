-- Replace the materialized table cn_stock_leader_score_daily with a view
-- that reads directly from cn_stock_leader_score_v2.
--
-- This eliminates the need for a separate materialization step in
-- StockBasicTask.run(). The v2 view already computes all leader score
-- columns from v1 + daily_basic, so querying it directly is both
-- simpler and always up-to-date.
--
-- Migration:
--   1. Run rename_cn_stock_leader_score_daily_bak.sql first
--   2. Execute this file to create the replacement view.

CREATE OR REPLACE
ALGORITHM = UNDEFINED
DEFINER = `cn_opr_red`@`localhost`
SQL SECURITY DEFINER
VIEW `cn_stock_leader_score_daily` AS
SELECT
    NULL AS id,
    v2.trade_date,
    v2.industry_id,
    v2.industry_name,
    v2.symbol,
    v2.name,
    v2.close,
    v2.amount,
    v2.market_cap,
    v2.total_mv,
    v2.circ_mv,
    v2.market_cap_rank,
    v2.market_cap_percentile,
    v2.leader_structural,
    v2.leader_structural_ready,
    v2.turnover_20d_avg,
    v2.turnover_20d_percentile,
    v2.leader_liquidity,
    v2.rs_20d_raw,
    v2.rs_percentile,
    v2.leader_trend,
    v2.breakout_strength,
    v2.breakout_ready,
    v2.leader_score,
    v2.leader_bucket,
    v2.turnover_rank_in_industry,
    v2.industry_members,
    'materialized_from_v2' AS source,
    NOW() AS created_at,
    NOW() AS updated_at
FROM cn_stock_leader_score_v2 v2;
