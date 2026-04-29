-- cn_market.cn_stock_leader_score_v2
-- Full 3-layer leader model after loading `cn_stock_daily_basic`.

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `cn_stock_leader_score_v2` AS
WITH base AS (
    SELECT
        v1.trade_date,
        v1.industry_id,
        v1.industry_name,
        v1.symbol,
        v1.name,
        v1.close,
        v1.amount,
        v1.turnover_20d_avg,
        v1.turnover_20d_percentile,
        v1.leader_liquidity,
        v1.rs_20d_raw,
        v1.rs_percentile,
        v1.leader_trend,
        v1.breakout_strength,
        v1.breakout_ready,
        v1.turnover_rank_in_industry,
        v1.industry_members,
        COALESCE(b.total_mv, b.circ_mv) AS market_cap,
        b.total_mv,
        b.circ_mv
    FROM cn_stock_leader_score_v1 v1
    LEFT JOIN cn_stock_daily_basic b
      ON (b.symbol COLLATE utf8mb4_unicode_ci) = v1.symbol
     AND b.trade_date = v1.trade_date
),
ranked AS (
    SELECT
        b.*,
        CASE
            WHEN b.market_cap IS NOT NULL
            THEN RANK() OVER (
                PARTITION BY b.trade_date, b.industry_id
                ORDER BY b.market_cap DESC, b.symbol
            )
            ELSE NULL
        END AS market_cap_rank,
        CASE
            WHEN b.market_cap IS NOT NULL
            THEN PERCENT_RANK() OVER (
                PARTITION BY b.trade_date, b.industry_id
                ORDER BY b.market_cap
            )
            ELSE NULL
        END AS market_cap_percentile
    FROM base b
)
SELECT
    r.trade_date,
    r.industry_id,
    r.industry_name,
    r.symbol,
    r.name,
    r.close,
    r.amount,
    r.market_cap,
    r.total_mv,
    r.circ_mv,
    r.market_cap_rank,
    r.market_cap_percentile,
    CASE
        WHEN r.market_cap IS NULL THEN NULL
        WHEN r.market_cap_rank <= 3 OR r.market_cap_percentile >= 0.9 THEN 1
        ELSE 0
    END AS leader_structural,
    CASE
        WHEN r.market_cap IS NULL THEN 0
        ELSE 1
    END AS leader_structural_ready,
    r.turnover_20d_avg,
    r.turnover_20d_percentile,
    r.leader_liquidity,
    r.rs_20d_raw,
    r.rs_percentile,
    r.leader_trend,
    r.breakout_strength,
    r.breakout_ready,
    (
        CASE
            WHEN r.market_cap IS NULL THEN 0
            WHEN r.market_cap_rank <= 3 OR r.market_cap_percentile >= 0.9 THEN 1
            ELSE 0
        END
        + r.leader_liquidity
        + r.leader_trend
    ) AS leader_score,
    CASE
        WHEN (
            CASE
                WHEN r.market_cap IS NULL THEN 0
                WHEN r.market_cap_rank <= 3 OR r.market_cap_percentile >= 0.9 THEN 1
                ELSE 0
            END
            + r.leader_liquidity
            + r.leader_trend
        ) = 3 THEN 'CORE_LEADER'
        WHEN (
            CASE
                WHEN r.market_cap IS NULL THEN 0
                WHEN r.market_cap_rank <= 3 OR r.market_cap_percentile >= 0.9 THEN 1
                ELSE 0
            END
            + r.leader_liquidity
            + r.leader_trend
        ) = 2 THEN 'NEAR_LEADER'
        WHEN (
            CASE
                WHEN r.market_cap IS NULL THEN 0
                WHEN r.market_cap_rank <= 3 OR r.market_cap_percentile >= 0.9 THEN 1
                ELSE 0
            END
            + r.leader_liquidity
            + r.leader_trend
        ) = 1 THEN 'EDGE_LEADER'
        ELSE 'NON_LEADER'
    END AS leader_bucket,
    r.turnover_rank_in_industry,
    r.industry_members
FROM ranked r;
