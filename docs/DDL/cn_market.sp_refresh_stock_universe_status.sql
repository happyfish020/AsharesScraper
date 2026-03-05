-- cn_market.sp_refresh_stock_universe_status
-- Refresh stock active/inactive flags by recent trading activity.

DROP PROCEDURE IF EXISTS `sp_refresh_stock_universe_status`;

CREATE PROCEDURE `sp_refresh_stock_universe_status`(
    IN p_asof_date DATE,
    IN p_recent_days INT,
    IN p_min_trade_days INT
)
proc: BEGIN
    DECLARE v_asof DATE;
    DECLARE v_recent_days INT;
    DECLARE v_min_trade_days INT;

    SET v_asof = COALESCE(p_asof_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));
    SET v_recent_days = GREATEST(IFNULL(p_recent_days, 30), 1);
    SET v_min_trade_days = GREATEST(IFNULL(p_min_trade_days, 1), 1);

    IF v_asof IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'sp_refresh_stock_universe_status: price calendar empty';
    END IF;

    INSERT INTO cn_stock_universe_status_t (
        symbol, is_active, inactive_reason, first_trade_date, last_trade_date, recent_trade_days, updated_at
    )
    SELECT
        x.symbol,
        CASE
            WHEN x.last_trade_date >= DATE_SUB(v_asof, INTERVAL v_recent_days DAY)
                 AND x.recent_trade_days >= v_min_trade_days THEN 1
            ELSE 0
        END AS is_active,
        CASE
            WHEN x.last_trade_date >= DATE_SUB(v_asof, INTERVAL v_recent_days DAY)
                 AND x.recent_trade_days >= v_min_trade_days THEN NULL
            ELSE 'LONG_INACTIVE_OR_DELISTED'
        END AS inactive_reason,
        x.first_trade_date,
        x.last_trade_date,
        x.recent_trade_days,
        NOW()
    FROM (
        SELECT
            p.symbol,
            MIN(p.trade_date) AS first_trade_date,
            MAX(p.trade_date) AS last_trade_date,
            COUNT(DISTINCT CASE
                WHEN p.trade_date BETWEEN DATE_SUB(v_asof, INTERVAL v_recent_days DAY) AND v_asof
                THEN p.trade_date
                ELSE NULL
            END) AS recent_trade_days
        FROM cn_stock_daily_price p
        GROUP BY p.symbol
    ) x
    ON DUPLICATE KEY UPDATE
        is_active = VALUES(is_active),
        inactive_reason = VALUES(inactive_reason),
        first_trade_date = VALUES(first_trade_date),
        last_trade_date = VALUES(last_trade_date),
        recent_trade_days = VALUES(recent_trade_days),
        updated_at = NOW();
END proc;
