-- ============================================================================
-- optimize_leader_score_perf.sql
-- Leader Score 视图链性能优化脚本
-- 
-- 问题: cn_stock_leader_score_v1 视图每次查询都全表扫描
--   - cn_board_member_map_d (30.7M 行 INDUSTRY, LIKE 过滤)
--   - cn_stock_daily_price (16.8M 行, 窗口函数)
-- 导致物化 cn_stock_leader_score_daily 时 session 极慢
--
-- 现有索引分析 (2026-05-21):
--   cn_board_member_map_d:
--     PRIMARY (trade_date, sector_type, sector_id, symbol)  ✅ 覆盖 JOIN
--     idx_map_type_sector_date (sector_type, sector_id, trade_date)  ✅ 覆盖过滤
--     idx_map_date_type_symbol_sector (trade_date, sector_type, symbol, sector_id)  ✅
--   
--   cn_stock_daily_price:
--     PRIMARY (SYMBOL, TRADE_DATE)  ✅ 覆盖窗口函数 PARTITION BY symbol ORDER BY trade_date
--     idx_price_date_symbol (TRADE_DATE, SYMBOL)  ✅
--   
--   cn_stock_daily_basic:
--     PRIMARY (symbol, trade_date)  ✅ 覆盖 LEFT JOIN
--     idx_basic_trade_date (trade_date)  ✅
--
-- 结论: 索引已经足够！性能瓶颈不在索引，而在:
--   1. ALGORITHM = UNDEFINED 阻止谓词下推
--   2. v1 视图硬编码 cn_market schema 导致跨 schema 查询
--   3. 物化脚本逐日循环导致 N 次视图解析
--   4. 30.7M 行的 industry_map + 16.8M 行的 price_enriched 无论如何都是大表扫描
--
-- 优化策略:
--   1. 将 ALGORITHM = UNDEFINED 改为 MERGE，允许谓词下推
--   2. 修复 v1 视图硬编码 cn_market schema 的问题
--   3. 优化物化脚本：批量物化 + 修复 argparse bug
--   4. (可选) 将 v1 改为物化中间表
-- ============================================================================

-- ============================================================================
-- Part 1: 重建 v1 视图 - ALGORITHM=MERGE + 修复 schema 引用
-- ============================================================================
-- 在 cn_market 库中执行

CREATE OR REPLACE
ALGORITHM = MERGE
DEFINER = CURRENT_USER
SQL SECURITY DEFINER
VIEW `cn_stock_leader_score_v1` AS
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

-- ============================================================================
-- Part 2: 重建 v2 视图 - ALGORITHM=MERGE
-- ============================================================================
-- 在 cn_market_red 库中执行
-- v1 视图引用使用 cn_market schema（由 load_sql_for_current_db 自动替换）

CREATE OR REPLACE
ALGORITHM = MERGE
DEFINER = CURRENT_USER
SQL SECURITY DEFINER
VIEW `cn_stock_leader_score_v2` AS
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
    FROM cn_market.cn_stock_leader_score_v1 v1
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
