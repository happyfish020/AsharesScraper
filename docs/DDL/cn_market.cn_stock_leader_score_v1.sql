-- cn_market.cn_stock_leader_score_v1
-- Executable leader scoring view using only current local MySQL data.
--
-- Scope:
-- 1. Uses a single industry taxonomy only: EASTMONEY `BK%` or SW `801%.SI`
-- 2. Supports liquidity + trend layers now
-- 3. Structural layer is left unavailable until market-cap data is loaded
--
-- Time-safety:
-- - Output is limited to dates that exist in `cn_board_member_map_d`
-- - Industry names come from the latest available `cn_board_industry_master`
--
-- Optimization:
-- - price_enriched 直接从 cn_stock_daily_price 查询（跳过 cn_stock_daily_price_active_v 视图层）
-- - 通过 min_trade_date CTE 计算所需的最小日期，避免全表扫描
-- - 窗口函数需要前20行数据，因此将日期范围向前扩展20天
--
-- 2026-05-21: Modified to support both BK% (Eastmoney) and 801%.SI (Shenwan) industry codes.
--   The cn_board_member_map_d table stopped producing BK% data after 2026-04-23,
--   so the filter was expanded to include 801%.SI codes.

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `cn_stock_leader_score_v1` AS
WITH industry_name AS (
    SELECT board_id, board_name
    FROM (
        SELECT
            (m.board_id COLLATE utf8mb4_unicode_ci) AS board_id,
            m.board_name,
            ROW_NUMBER() OVER (
                PARTITION BY (m.board_id COLLATE utf8mb4_unicode_ci)
                ORDER BY m.asof_date DESC
            ) AS rn
        FROM cn_board_industry_master m
        WHERE (m.board_id COLLATE utf8mb4_unicode_ci) LIKE 'BK%'
           OR (m.board_id COLLATE utf8mb4_unicode_ci) LIKE '801%.SI'
    ) t
    WHERE t.rn = 1
),
industry_map AS (
    SELECT
        m.trade_date,
        (m.symbol COLLATE utf8mb4_unicode_ci) AS symbol,
        (m.sector_id COLLATE utf8mb4_unicode_ci) AS industry_id
    FROM cn_board_member_map_d m
    WHERE (m.sector_type COLLATE utf8mb4_unicode_ci) = 'INDUSTRY'
      AND (
          (m.sector_id COLLATE utf8mb4_unicode_ci) LIKE 'BK%'
          OR (m.sector_id COLLATE utf8mb4_unicode_ci) LIKE '801%.SI'
      )
),
min_trade_date AS (
    SELECT DATE_SUB(MIN(m.trade_date), INTERVAL 20 DAY) AS min_date
    FROM industry_map m
),
price_enriched AS (
    SELECT
        p.symbol,
        p.trade_date,
        p.name,
        p.close,
        p.amount,
        AVG(p.amount) OVER (
            PARTITION BY p.symbol
            ORDER BY p.trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS turnover_20d_avg,
        COUNT(p.amount) OVER (
            PARTITION BY p.symbol
            ORDER BY p.trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS turnover_20d_obs,
        LAG(p.close, 19) OVER (
            PARTITION BY p.symbol
            ORDER BY p.trade_date
        ) AS close_20d_ago
    FROM cn_stock_daily_price p
    CROSS JOIN min_trade_date m
    WHERE p.trade_date >= m.min_date
),
feature_base AS (
    SELECT
        m.trade_date,
        m.industry_id,
        n.board_name AS industry_name,
        p.symbol,
        p.name,
        p.close,
        p.amount,
        CASE
            WHEN p.turnover_20d_obs = 20 THEN p.turnover_20d_avg
            ELSE NULL
        END AS turnover_20d_avg,
        CASE
            WHEN p.close_20d_ago IS NOT NULL AND p.close_20d_ago <> 0
            THEN p.close / p.close_20d_ago - 1
            ELSE NULL
        END AS rs_20d_raw
    FROM industry_map m
    JOIN price_enriched p
      ON (p.symbol COLLATE utf8mb4_unicode_ci) = m.symbol
     AND p.trade_date = m.trade_date
    LEFT JOIN industry_name n
      ON n.board_id = m.industry_id
),
ranked AS (
    SELECT
        b.trade_date,
        b.industry_id,
        b.industry_name,
        b.symbol,
        b.name,
        b.close,
        b.amount,
        b.turnover_20d_avg,
        b.rs_20d_raw,
        CASE
            WHEN b.turnover_20d_avg IS NOT NULL
            THEN PERCENT_RANK() OVER (
                PARTITION BY b.trade_date, b.industry_id
                ORDER BY b.turnover_20d_avg
            )
            ELSE NULL
        END AS turnover_20d_percentile,
        CASE
            WHEN b.rs_20d_raw IS NOT NULL
            THEN PERCENT_RANK() OVER (
                PARTITION BY b.trade_date, b.industry_id
                ORDER BY b.rs_20d_raw
            )
            ELSE NULL
        END AS rs_percentile,
        RANK() OVER (
            PARTITION BY b.trade_date, b.industry_id
            ORDER BY b.turnover_20d_avg DESC, b.symbol
        ) AS turnover_rank_in_industry,
        COUNT(*) OVER (
            PARTITION BY b.trade_date, b.industry_id
        ) AS industry_members
    FROM feature_base b
)
SELECT
    r.trade_date,
    r.industry_id,
    r.industry_name,
    r.symbol,
    r.name,
    r.close,
    r.amount,
    CAST(NULL AS SIGNED) AS leader_structural,
    0 AS leader_structural_ready,
    r.turnover_20d_avg,
    r.turnover_20d_percentile,
    CASE
        WHEN r.turnover_20d_percentile >= 0.8 THEN 1
        ELSE 0
    END AS leader_liquidity,
    r.rs_20d_raw,
    r.rs_percentile,
    CASE
        WHEN r.rs_percentile >= 0.7 THEN 1
        ELSE 0
    END AS leader_trend,
    CAST(NULL AS CHAR(16)) AS breakout_strength,
    0 AS breakout_ready,
    (
        CASE
            WHEN r.turnover_20d_percentile >= 0.8 THEN 1
            ELSE 0
        END
        +
        CASE
            WHEN r.rs_percentile >= 0.7 THEN 1
            ELSE 0
        END
    ) AS leader_score_v1,
    2 AS leader_score_v1_max,
    CASE
        WHEN (
            CASE
                WHEN r.turnover_20d_percentile >= 0.8 THEN 1
                ELSE 0
            END
            +
            CASE
                WHEN r.rs_percentile >= 0.7 THEN 1
                ELSE 0
            END
        ) = 2 THEN 'CORE_CANDIDATE'
        WHEN (
            CASE
                WHEN r.turnover_20d_percentile >= 0.8 THEN 1
                ELSE 0
            END
            +
            CASE
                WHEN r.rs_percentile >= 0.7 THEN 1
                ELSE 0
            END
        ) = 1 THEN 'EDGE_CANDIDATE'
        ELSE 'NON_LEADER'
    END AS leader_bucket_v1,
    r.turnover_rank_in_industry,
    r.industry_members
FROM ranked r;
