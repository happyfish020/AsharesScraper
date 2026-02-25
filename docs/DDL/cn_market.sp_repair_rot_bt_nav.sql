-- cn_market.SP_REPAIR_ROT_BT_NAV
-- Forward-fill NULL NAV values by run_id using nearest previous non-NULL NAV.

DROP PROCEDURE IF EXISTS `SP_REPAIR_ROT_BT_NAV`;

CREATE PROCEDURE `SP_REPAIR_ROT_BT_NAV`(
    IN p_run_id VARCHAR(128),
    IN p_default_nav DECIMAL(18,10)
)
proc: BEGIN
    DECLARE v_run_id VARCHAR(128);
    DECLARE v_default_nav DECIMAL(18,10);

    SET v_run_id = NULLIF(p_run_id, '');
    SET v_default_nav = COALESCE(p_default_nav, 1.0000000000);

    DROP TEMPORARY TABLE IF EXISTS tmp_bt_nav_fix;
    CREATE TEMPORARY TABLE tmp_bt_nav_fix (
        run_id VARCHAR(64) NOT NULL,
        trade_date DATE NOT NULL,
        nav_new DECIMAL(18,10) NOT NULL,
        PRIMARY KEY (run_id, trade_date)
    ) ENGINE=MEMORY;

    INSERT INTO tmp_bt_nav_fix (run_id, trade_date, nav_new)
    SELECT
        b.run_id,
        b.trade_date,
        COALESCE((
            SELECT b2.nav
            FROM cn_sector_rot_bt_daily_t b2
            WHERE b2.run_id = b.run_id
              AND b2.trade_date < b.trade_date
              AND b2.nav IS NOT NULL
            ORDER BY b2.trade_date DESC
            LIMIT 1
        ), v_default_nav) AS nav_new
    FROM cn_sector_rot_bt_daily_t b
    WHERE b.nav IS NULL
      AND (v_run_id IS NULL OR b.run_id = v_run_id);

    UPDATE cn_sector_rot_bt_daily_t b
    JOIN tmp_bt_nav_fix t
      ON t.run_id = b.run_id
     AND t.trade_date = b.trade_date
    SET b.nav = t.nav_new
    WHERE b.nav IS NULL;

    DROP TEMPORARY TABLE IF EXISTS tmp_bt_nav_fix;
END proc;
