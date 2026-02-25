-- cn_market.SP_BACKFILL_ROT_BT_FROM_PRICE

DROP PROCEDURE IF EXISTS `SP_BACKFILL_ROT_BT_FROM_PRICE`;

CREATE PROCEDURE `SP_BACKFILL_ROT_BT_FROM_PRICE`(
    IN p_run_id VARCHAR(64),
    IN p_end_date DATE,
    IN p_force TINYINT
)
proc: BEGIN
    DECLARE v_run_id VARCHAR(64);
    DECLARE v_end_date DATE;
    DECLARE v_min_px_date DATE;
    DECLARE v_last_bt_date DATE;
    DECLARE v_start_date DATE;
    DECLARE v_base_nav DECIMAL(18,10);

    SET v_run_id = COALESCE(NULLIF(p_run_id, ''), 'SR_LIVE_DEFAULT');
    SET v_end_date = COALESCE(p_end_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));
    SET v_min_px_date = (SELECT MIN(trade_date) FROM cn_stock_daily_price);

    IF v_end_date IS NULL OR v_min_px_date IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'SP_BACKFILL_ROT_BT_FROM_PRICE: price calendar is empty';
    END IF;

    SET v_last_bt_date = (
        SELECT MAX(b.trade_date)
        FROM cn_sector_rot_bt_daily_t b
        WHERE b.run_id = v_run_id
          AND b.trade_date <= v_end_date
    );

    IF IFNULL(p_force, 0) = 1 THEN
        SET v_start_date = COALESCE(v_last_bt_date, v_min_px_date);
        DELETE FROM cn_sector_rot_bt_daily_t
        WHERE run_id = v_run_id
          AND trade_date BETWEEN v_start_date AND v_end_date;
        SET v_base_nav = (
            SELECT b.nav
            FROM cn_sector_rot_bt_daily_t b
            WHERE b.run_id = v_run_id
              AND b.trade_date < v_start_date
              AND b.nav IS NOT NULL
            ORDER BY b.trade_date DESC
            LIMIT 1
        );
    ELSE
        IF v_last_bt_date IS NULL THEN
            SET v_start_date = v_min_px_date;
            SET v_base_nav = 1.0000000000;
        ELSE
            SET v_start_date = DATE_ADD(v_last_bt_date, INTERVAL 1 DAY);
            SET v_base_nav = (
                SELECT b.nav
                FROM cn_sector_rot_bt_daily_t b
                WHERE b.run_id = v_run_id
                  AND b.trade_date = v_last_bt_date
                  AND b.nav IS NOT NULL
                LIMIT 1
            );
            IF v_base_nav IS NULL THEN
                SET v_base_nav = (
                    SELECT b.nav
                    FROM cn_sector_rot_bt_daily_t b
                    WHERE b.run_id = v_run_id
                      AND b.trade_date <= v_last_bt_date
                      AND b.nav IS NOT NULL
                    ORDER BY b.trade_date DESC
                    LIMIT 1
                );
            END IF;
        END IF;
    END IF;

    IF v_start_date > v_end_date THEN
        LEAVE proc;
    END IF;

    SET v_base_nav = COALESCE(v_base_nav, 1.0000000000);

    INSERT INTO cn_sector_rot_bt_daily_t (
        trade_date, run_id, n_pos, k_used, port_ret_1, turnover, cost, net_ret, nav, exposed_flag, created_at
    )
    SELECT
        d.trade_date,
        v_run_id,
        IFNULL(pz.n_pos, 0) AS n_pos,
        IFNULL(sg.k_used, 0) AS k_used,
        CAST(0 AS DECIMAL(18,10)) AS port_ret_1,
        CAST(0 AS DECIMAL(18,10)) AS turnover,
        CAST(0 AS DECIMAL(18,10)) AS cost,
        CAST(0 AS DECIMAL(18,10)) AS net_ret,
        v_base_nav AS nav,
        CASE WHEN IFNULL(pz.n_pos, 0) > 0 THEN 1 ELSE 0 END AS exposed_flag,
        NOW()
    FROM (
        SELECT DISTINCT p.trade_date
        FROM cn_stock_daily_price p
        WHERE p.trade_date BETWEEN v_start_date AND v_end_date
    ) d
    LEFT JOIN (
        SELECT trade_date, COUNT(*) AS n_pos
        FROM cn_sector_rot_pos_daily_t
        WHERE run_id = v_run_id
          AND w > 0
          AND trade_date BETWEEN v_start_date AND v_end_date
        GROUP BY trade_date
    ) pz
      ON pz.trade_date = d.trade_date
    LEFT JOIN (
        SELECT signal_date AS trade_date, MAX(CAST(IFNULL(entry_cnt, 0) AS SIGNED)) AS k_used
        FROM cn_sector_rotation_signal_t
        WHERE signal_date BETWEEN v_start_date AND v_end_date
          AND action = 'ENTER'
        GROUP BY signal_date
    ) sg
      ON sg.trade_date = d.trade_date
    WHERE NOT EXISTS (
        SELECT 1
        FROM cn_sector_rot_bt_daily_t b
        WHERE b.run_id = v_run_id
          AND b.trade_date = d.trade_date
    )
    ORDER BY d.trade_date;
END proc;
