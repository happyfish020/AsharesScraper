-- cn_market.cn_sector_rotation_transition_v
-- Builds transition signals from sector eod history facts.

DROP VIEW IF EXISTS `cn_sector_rotation_transition_v`;

CREATE VIEW `cn_sector_rotation_transition_v` AS
WITH industry_name AS (
    SELECT x.board_id, x.board_name
    FROM (
        SELECT
            m.BOARD_ID AS board_id,
            m.BOARD_NAME AS board_name,
            ROW_NUMBER() OVER (PARTITION BY m.BOARD_ID ORDER BY m.ASOF_DATE DESC) AS rn
        FROM cn_board_industry_master m
    ) x
    WHERE x.rn = 1
),
concept_name AS (
    SELECT x.concept_id, x.concept_name
    FROM (
        SELECT
            m.CONCEPT_ID AS concept_id,
            m.CONCEPT_NAME AS concept_name,
            ROW_NUMBER() OVER (PARTITION BY m.CONCEPT_ID ORDER BY m.ASOF_DATE DESC) AS rn
        FROM cn_board_concept_master m
    ) x
    WHERE x.rn = 1
),
hist0 AS (
    SELECT
        e.trade_date,
        e.sector_type,
        e.sector_id,
        COALESCE(
            CASE WHEN e.sector_type = 'INDUSTRY' THEN im.board_name END,
            CASE WHEN e.sector_type = 'CONCEPT' THEN cm.concept_name END,
            e.sector_id
        ) AS sector_name,
        e.amount_sum,
        e.score,
        e.up_ratio,
        e.cond_count,
        e.sector_pass
    FROM cn_sector_eod_hist_t e
    LEFT JOIN industry_name im
      ON e.sector_type = 'INDUSTRY'
     AND e.sector_id = im.board_id
    LEFT JOIN concept_name cm
      ON e.sector_type = 'CONCEPT'
     AND e.sector_id = cm.concept_id
),
hist1 AS (
    SELECT
        h0.*,
        AVG(h0.amount_sum) OVER (
            PARTITION BY h0.sector_type, h0.sector_id
            ORDER BY h0.trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS amt_ma20,
        AVG(h0.up_ratio) OVER (
            PARTITION BY h0.sector_type, h0.sector_id
            ORDER BY h0.trade_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS up_ma5,
        ROW_NUMBER() OVER (
            PARTITION BY h0.sector_type, h0.sector_id
            ORDER BY h0.trade_date
        ) AS rn_all,
        CASE
            WHEN h0.sector_pass = 1 THEN ROW_NUMBER() OVER (
                PARTITION BY h0.sector_type, h0.sector_id, h0.sector_pass
                ORDER BY h0.trade_date
            )
            ELSE NULL
        END AS rn_pass
    FROM hist0 h0
),
hist2 AS (
    SELECT
        h1.*,
        CASE WHEN h1.sector_pass = 1 THEN (h1.rn_all - h1.rn_pass) ELSE NULL END AS grp_pass
    FROM hist1 h1
),
hist3 AS (
    SELECT
        h2.*,
        CASE
            WHEN h2.sector_pass = 1 THEN ROW_NUMBER() OVER (
                PARTITION BY h2.sector_type, h2.sector_id, h2.grp_pass
                ORDER BY h2.trade_date
            )
            ELSE 0
        END AS confirm_streak
    FROM hist2 h2
),
hist4 AS (
    SELECT
        h3.trade_date,
        h3.sector_type,
        h3.sector_id,
        h3.sector_name,
        h3.score,
        h3.confirm_streak,
        CASE
            WHEN h3.amt_ma20 = 0 OR h3.amt_ma20 IS NULL THEN NULL
            ELSE h3.amount_sum / h3.amt_ma20
        END AS amt_impulse,
        h3.up_ma5,
        h3.up_ratio,
        CASE
            WHEN h3.sector_pass = 1 AND h3.cond_count >= 3 THEN 'CONFIRM'
            WHEN h3.sector_pass = 1 AND h3.cond_count = 2 THEN 'HOLD'
            WHEN h3.cond_count >= 2 THEN 'IGNITE'
            WHEN h3.cond_count = 1 THEN 'FADE'
            ELSE 'NEUTRAL'
        END AS state
    FROM hist3 h3
),
hist5 AS (
    SELECT
        h4.*,
        CASE
            WHEN h4.state = 'CONFIRM' AND IFNULL(h4.confirm_streak, 0) >= 2 THEN 'T1'
            WHEN h4.state = 'CONFIRM' THEN 'T2'
            WHEN h4.state = 'HOLD' THEN 'T2'
            WHEN h4.state = 'IGNITE' THEN 'T3'
            WHEN h4.state = 'FADE' THEN 'T4'
            ELSE 'T9'
        END AS tier,
        CASE
            WHEN h4.state = 'CONFIRM' AND IFNULL(h4.confirm_streak, 0) >= 2 THEN 0
            WHEN h4.state = 'CONFIRM' THEN 1
            WHEN h4.state = 'HOLD' THEN 2
            WHEN h4.state = 'IGNITE' THEN 3
            WHEN h4.state = 'FADE' THEN 4
            ELSE 9
        END AS tier_pri,
        CASE
            WHEN h4.sector_type = 'INDUSTRY' THEN 'INDUSTRY'
            WHEN h4.sector_type = 'CONCEPT' THEN 'CONCEPT'
            ELSE 'OTHER'
        END AS theme_group
    FROM hist4 h4
),
ranked AS (
    SELECT
        h5.trade_date,
        h5.sector_type,
        h5.sector_id,
        h5.sector_name,
        h5.state,
        h5.tier,
        h5.theme_group,
        ROW_NUMBER() OVER (
            PARTITION BY h5.trade_date, h5.theme_group
            ORDER BY h5.tier_pri, h5.score DESC, h5.sector_type, h5.sector_id
        ) AS theme_rank,
        h5.score,
        h5.confirm_streak,
        h5.amt_impulse,
        h5.up_ma5,
        h5.up_ratio
    FROM hist5 h5
),
x AS (
    SELECT
        r.*,
        LAG(r.state) OVER (PARTITION BY r.sector_type, r.sector_id ORDER BY r.trade_date) AS prev_state,
        LAG(r.tier)  OVER (PARTITION BY r.sector_type, r.sector_id ORDER BY r.trade_date) AS prev_tier,
        LAG(r.score) OVER (PARTITION BY r.sector_type, r.sector_id ORDER BY r.trade_date) AS prev_score
    FROM ranked r
)
SELECT
    x.trade_date,
    DATE_FORMAT(x.trade_date, '%Y-%m-%d') AS trading_date,
    x.sector_type,
    x.sector_id,
    x.sector_name,
    x.theme_group,
    x.theme_rank,
    CASE WHEN x.theme_rank = 1 THEN 'KEEP' ELSE 'DUP_THEME' END AS theme_flag,
    x.tier,
    x.state,
    x.score,
    x.confirm_streak,
    x.amt_impulse,
    x.up_ratio,
    x.up_ma5,
    x.prev_state,
    x.prev_tier,
    x.prev_score,
    (x.score - x.prev_score) AS score_delta,
    CASE
        WHEN x.prev_state IS NULL THEN 'NO_PREV'
        WHEN x.prev_state = x.state THEN 'NO_CHANGE'
        WHEN x.prev_state IN ('NEUTRAL', 'FADE') AND x.state = 'IGNITE' THEN 'START_IGNITE'
        WHEN x.prev_state = 'IGNITE' AND x.state = 'CONFIRM' THEN 'IGNITE_TO_CONFIRM'
        WHEN x.prev_state IN ('NEUTRAL', 'FADE') AND x.state = 'CONFIRM' THEN 'DIRECT_CONFIRM'
        WHEN x.prev_state = 'CONFIRM' AND x.state = 'HOLD' THEN 'CONFIRM_TO_HOLD'
        WHEN x.prev_state = 'HOLD' AND x.state = 'CONFIRM' THEN 'HOLD_TO_CONFIRM'
        WHEN x.prev_state IN ('CONFIRM', 'HOLD') AND x.state = 'FADE' THEN 'TREND_TO_FADE'
        WHEN x.prev_state IN ('CONFIRM', 'HOLD') AND x.state = 'NEUTRAL' THEN 'TREND_BREAK_TO_NEUTRAL'
        WHEN x.prev_state = 'IGNITE' AND x.state IN ('NEUTRAL', 'FADE') THEN 'IGNITE_FAIL'
        ELSE 'OTHER_CHANGE'
    END AS transition
FROM x;
