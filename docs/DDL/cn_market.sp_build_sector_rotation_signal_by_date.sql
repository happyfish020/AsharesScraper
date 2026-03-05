-- cn_market.SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE

DROP PROCEDURE IF EXISTS `SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE`;

CREATE PROCEDURE `SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE`(
    IN p_trade_date DATE
)
proc: BEGIN
    DECLARE v_trade_date DATE;
    SET v_trade_date = COALESCE(p_trade_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));

    IF v_trade_date IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE: trade_date is NULL';
    END IF;

    INSERT INTO cn_sector_rotation_signal_t (
        signal_date, sector_type, sector_id, sector_name, action,
        entry_rank, entry_cnt, weight_suggested, score, state, transition, created_at
    )
    WITH sig0 AS (
        SELECT
            t.trade_date AS signal_date,
            t.sector_type,
            t.sector_id,
            t.sector_name,
            CASE
                WHEN t.transition IN ('IGNITE_TO_CONFIRM', 'DIRECT_CONFIRM')
                     AND t.theme_rank = 1
                     AND t.tier = 'T1'
                     AND t.up_ma5 >= 0.52
                     AND t.amt_impulse >= 1.10 THEN 'ENTER'
                WHEN t.transition IN ('CONFIRM_TO_FADE', 'FADE_TO_OFF', 'CONFIRM_TO_OFF', 'T1_TO_T2', 'T2_TO_T3') THEN 'EXIT'
                ELSE 'WATCH'
            END AS action,
            t.score,
            t.state,
            t.transition
        FROM cn_sector_rotation_transition_v t
        WHERE t.trade_date = v_trade_date
    ),
    sig1 AS (
        SELECT
            s.*,
            CASE WHEN s.action = 'ENTER'
                 THEN ROW_NUMBER() OVER (PARTITION BY s.signal_date ORDER BY s.score DESC, s.sector_type, s.sector_id)
                 ELSE NULL
            END AS entry_rank
        FROM sig0 s
    ),
    sig2 AS (
        SELECT
            s.*,
            SUM(CASE WHEN s.action = 'ENTER' THEN 1 ELSE 0 END) OVER (PARTITION BY s.signal_date) AS entry_cnt
        FROM sig1 s
    )
    SELECT
        s.signal_date,
        s.sector_type,
        s.sector_id,
        s.sector_name,
        s.action,
        s.entry_rank,
        CASE WHEN s.action = 'ENTER' THEN s.entry_cnt ELSE NULL END AS entry_cnt,
        CASE WHEN s.action = 'ENTER' AND s.entry_cnt > 0 THEN 1.0 / s.entry_cnt ELSE NULL END AS weight_suggested,
        s.score,
        s.state,
        s.transition,
        NOW()
    FROM sig2 s
    ON DUPLICATE KEY UPDATE
        sector_name = VALUES(sector_name),
        action = VALUES(action),
        entry_rank = VALUES(entry_rank),
        entry_cnt = VALUES(entry_cnt),
        weight_suggested = VALUES(weight_suggested),
        score = VALUES(score),
        state = VALUES(state),
        transition = VALUES(transition),
        created_at = NOW();
END proc;
