-- cn_market.SP_VALIDATE_SECTOR_ROT_RUN

DROP PROCEDURE IF EXISTS `SP_VALIDATE_SECTOR_ROT_RUN`;

CREATE PROCEDURE `SP_VALIDATE_SECTOR_ROT_RUN`(
    IN p_run_id VARCHAR(128)
)
proc: BEGIN
    DECLARE v_run_id VARCHAR(128);
    DECLARE v_rows BIGINT DEFAULT 0;
    DECLARE v_dt_min DATE;
    DECLARE v_dt_max DATE;
    DECLARE v_nav_min DECIMAL(18,10);
    DECLARE v_nav_max DECIMAL(18,10);
    DECLARE v_nav_last DECIMAL(18,10);
    DECLARE v_nav_null BIGINT DEFAULT 0;
    DECLARE v_neg_nav BIGINT DEFAULT 0;
    DECLARE v_sig_days BIGINT DEFAULT 0;
    DECLARE v_bt_days BIGINT DEFAULT 0;
    DECLARE v_status VARCHAR(16);

    SET v_run_id = COALESCE(NULLIF(p_run_id, ''), 'SR_LIVE_DEFAULT');

    SELECT
        COUNT(*),
        MIN(trade_date),
        MAX(trade_date),
        MIN(nav),
        MAX(nav),
        SUM(CASE WHEN nav IS NULL THEN 1 ELSE 0 END),
        SUM(CASE WHEN IFNULL(nav, 0) <= 0 THEN 1 ELSE 0 END)
    INTO
        v_rows, v_dt_min, v_dt_max, v_nav_min, v_nav_max, v_nav_null, v_neg_nav
    FROM cn_sector_rot_bt_daily_t
    WHERE run_id = v_run_id;

    SELECT b.nav
    INTO v_nav_last
    FROM cn_sector_rot_bt_daily_t b
    WHERE b.run_id = v_run_id
    ORDER BY b.trade_date DESC
    LIMIT 1;

    SELECT COUNT(DISTINCT signal_date)
    INTO v_sig_days
    FROM cn_sector_rotation_signal_t;

    SET v_bt_days = IFNULL(v_rows, 0);

    IF v_rows = 0 THEN
        SET v_status = 'FAIL';
    ELSEIF IFNULL(v_nav_null, 0) > 0 OR IFNULL(v_neg_nav, 0) > 0 THEN
        SET v_status = 'FAIL';
    ELSE
        SET v_status = 'PASS';
    END IF;

    SELECT
        v_run_id AS run_id,
        v_status AS validation_status,
        v_rows AS bt_rows,
        v_dt_min AS bt_start_date,
        v_dt_max AS bt_end_date,
        v_nav_last AS nav_last,
        v_nav_min AS nav_min,
        v_nav_max AS nav_max,
        v_nav_null AS nav_null_rows,
        v_neg_nav AS nav_nonpositive_rows,
        v_sig_days AS signal_days,
        v_bt_days AS bt_days,
        CASE
            WHEN v_status = 'PASS' THEN 'OK'
            WHEN v_rows = 0 THEN 'BT_EMPTY'
            WHEN IFNULL(v_nav_null, 0) > 0 THEN 'BT_NAV_NULL'
            ELSE 'BT_NAV_NONPOSITIVE'
        END AS reason_code;
END proc;
