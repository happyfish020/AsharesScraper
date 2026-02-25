-- cn_market.SP_ROTATION_DAILY_REFRESH
-- Orchestrator procedure for MySQL migration.

DROP PROCEDURE IF EXISTS `SP_ROTATION_DAILY_REFRESH`;

CREATE PROCEDURE `SP_ROTATION_DAILY_REFRESH`(
    IN p_run_id VARCHAR(64),
    IN p_trade_date DATE,
    IN p_force TINYINT,
    IN p_refresh_energy TINYINT
)
proc: BEGIN
    DECLARE v_trade_date DATE;
    DECLARE v_run_id VARCHAR(64);

    SET v_run_id = COALESCE(NULLIF(p_run_id, ''), 'SR_LIVE_DEFAULT');
    SET v_trade_date = COALESCE(p_trade_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));

    IF v_trade_date IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'SP_ROTATION_DAILY_REFRESH: trade_date is NULL';
    END IF;

    -- reserved for compatibility; current energy pipeline is view-driven.
    IF IFNULL(p_refresh_energy, 1) = 1 THEN
        DO 1;
    END IF;

    CALL SP_BUILD_SECTOR_ROTATION_RANKED_LATEST();
    CALL SP_BUILD_SECTOR_ROTATION_SIGNAL_LATEST();
    CALL SP_BACKFILL_ROT_BT_FROM_PRICE(v_run_id, v_trade_date, IFNULL(p_force, 0));
    CALL SP_REPAIR_ROT_BT_NAV(v_run_id, 1.0000000000);
    CALL SP_REFRESH_ROTATION_SNAP_ALL(v_run_id, v_trade_date, IFNULL(p_force, 0));
END proc;
