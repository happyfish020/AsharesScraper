-- cn_market.SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE

DROP PROCEDURE IF EXISTS `SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE`;

CREATE PROCEDURE `SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE`(
    IN p_trade_date DATE
)
proc: BEGIN
    DECLARE v_trade_date DATE;
    SET v_trade_date = COALESCE(p_trade_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));

    IF v_trade_date IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE: trade_date is NULL';
    END IF;

    INSERT INTO cn_sector_rotation_ranked_t (
        trade_date, sector_type, sector_id, sector_name,
        state, tier, theme_group, theme_rank, score, confirm_streak,
        amt_impulse, up_ma5, up_ratio, created_at
    )
    SELECT
        t.trade_date,
        t.sector_type,
        t.sector_id,
        t.sector_name,
        t.state,
        t.tier,
        t.theme_group,
        t.theme_rank,
        t.score,
        t.confirm_streak,
        t.amt_impulse,
        t.up_ma5,
        t.up_ratio,
        NOW()
    FROM cn_sector_rotation_transition_v t
    WHERE t.trade_date = v_trade_date
    ON DUPLICATE KEY UPDATE
        sector_name = VALUES(sector_name),
        state = VALUES(state),
        tier = VALUES(tier),
        theme_group = VALUES(theme_group),
        theme_rank = VALUES(theme_rank),
        score = VALUES(score),
        confirm_streak = VALUES(confirm_streak),
        amt_impulse = VALUES(amt_impulse),
        up_ma5 = VALUES(up_ma5),
        up_ratio = VALUES(up_ratio),
        created_at = NOW();
END proc;
