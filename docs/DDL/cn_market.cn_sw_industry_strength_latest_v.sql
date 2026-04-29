-- cn_market.cn_sw_industry_strength_latest_v
-- Latest Shenwan industry strength snapshot based on cn_sw_industry_daily.

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `cn_sw_industry_strength_latest_v` AS
WITH base AS (
    SELECT
        d.ts_code,
        d.name,
        d.trade_date,
        d.close,
        d.pct_change,
        d.amount,
        d.pe,
        d.pb,
        d.float_mv,
        AVG(d.close) OVER (
            PARTITION BY d.ts_code
            ORDER BY d.trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS ma20,
        AVG(d.close) OVER (
            PARTITION BY d.ts_code
            ORDER BY d.trade_date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ) AS ma60,
        LAG(d.close, 20) OVER (
            PARTITION BY d.ts_code
            ORDER BY d.trade_date
        ) AS close_20d_ago
    FROM cn_sw_industry_daily d
),
enriched AS (
    SELECT
        b.ts_code,
        b.name,
        b.trade_date,
        b.close,
        b.pct_change,
        b.amount,
        b.pe,
        b.pb,
        b.float_mv,
        b.ma20,
        b.ma60,
        CASE
            WHEN b.close_20d_ago IS NOT NULL AND b.close_20d_ago <> 0
            THEN b.close / b.close_20d_ago - 1
            ELSE NULL
        END AS rs_20d
    FROM base b
),
ranked AS (
    SELECT
        e.*,
        PERCENT_RANK() OVER (
            PARTITION BY e.trade_date
            ORDER BY e.rs_20d
        ) AS rs_20d_percentile,
        PERCENT_RANK() OVER (
            PARTITION BY e.trade_date
            ORDER BY e.pct_change
        ) AS pct_change_percentile,
        RANK() OVER (
            PARTITION BY e.trade_date
            ORDER BY e.rs_20d DESC, e.ts_code
        ) AS rs_rank
    FROM enriched e
)
SELECT
    r.ts_code,
    r.name,
    r.trade_date,
    r.close,
    r.pct_change,
    r.amount,
    r.pe,
    r.pb,
    r.float_mv,
    r.ma20,
    r.ma60,
    r.rs_20d,
    r.rs_20d_percentile,
    r.pct_change_percentile,
    r.rs_rank,
    CASE
        WHEN r.close > r.ma20 AND r.ma20 > r.ma60 THEN 'UPTREND'
        WHEN r.close > r.ma20 THEN 'ABOVE_MA20'
        ELSE 'WEAK'
    END AS trend_state
FROM ranked r
WHERE r.trade_date = (
    SELECT MAX(x.trade_date)
    FROM cn_sw_industry_daily x
);
