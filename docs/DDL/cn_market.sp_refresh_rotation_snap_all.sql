-- cn_market.SP_REFRESH_ROTATION_SNAP_ALL

DROP PROCEDURE IF EXISTS `SP_REFRESH_ROTATION_SNAP_ALL`;

CREATE PROCEDURE `SP_REFRESH_ROTATION_SNAP_ALL`(
    IN p_run_id VARCHAR(64),
    IN p_trade_date DATE,
    IN p_force TINYINT
)
proc: BEGIN
    DECLARE v_trade_date DATE;
    DECLARE v_run_id VARCHAR(64);
    DECLARE v_has_entry BIGINT DEFAULT 0;
    DECLARE v_has_holding BIGINT DEFAULT 0;
    DECLARE v_has_exit BIGINT DEFAULT 0;

    SET v_run_id = COALESCE(NULLIF(p_run_id, ''), 'SR_LIVE_DEFAULT');
    SET v_trade_date = COALESCE(p_trade_date, (SELECT MAX(trade_date) FROM cn_stock_daily_price));

    IF v_trade_date IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'SP_REFRESH_ROTATION_SNAP_ALL: trade_date is NULL';
    END IF;

    IF IFNULL(p_force, 0) = 1 THEN
        DELETE FROM cn_rotation_entry_snap_t WHERE run_id = v_run_id AND trade_date = v_trade_date;
        DELETE FROM cn_rotation_holding_snap_t WHERE run_id = v_run_id AND trade_date = v_trade_date;
        DELETE FROM cn_rotation_exit_snap_t WHERE run_id = v_run_id AND trade_date = v_trade_date;
    END IF;

    INSERT INTO cn_rotation_entry_snap_t (
        run_id, trade_date, sector_type, sector_id, sector_name,
        entry_rank, entry_cnt, weight_suggested, signal_score,
        energy_score, energy_pct, energy_tier, state, transition,
        source_json, created_at
    )
    SELECT
        v_run_id,
        s.signal_date,
        s.sector_type,
        s.sector_id,
        s.sector_name,
        CAST(s.entry_rank AS SIGNED),
        CAST(s.entry_cnt AS SIGNED),
        CAST(s.weight_suggested AS DECIMAL(18,8)),
        CAST(s.score AS DECIMAL(38,16)),
        CAST(e.energy_score AS DECIMAL(38,16)),
        CAST(e.energy_pct AS DECIMAL(38,16)),
        CASE
            WHEN e.energy_pct >= 0.80 THEN 'T1'
            WHEN e.energy_pct >= 0.50 THEN 'T2'
            WHEN e.energy_pct IS NULL THEN NULL
            ELSE 'T3'
        END,
        s.state,
        s.transition,
        JSON_OBJECT('source', 'SP_REFRESH_ROTATION_SNAP_ALL', 'trade_date', CAST(v_trade_date AS CHAR)),
        NOW(6)
    FROM cn_sector_rotation_signal_t s
    LEFT JOIN cn_sector_energy_v e
      ON e.trade_date = s.signal_date
     AND e.sector_type = s.sector_type
     AND e.sector_id = s.sector_id
    WHERE s.signal_date = v_trade_date
      AND s.action = 'ENTER'
    ON DUPLICATE KEY UPDATE
        sector_name = VALUES(sector_name),
        entry_rank = VALUES(entry_rank),
        entry_cnt = VALUES(entry_cnt),
        weight_suggested = VALUES(weight_suggested),
        signal_score = VALUES(signal_score),
        energy_score = VALUES(energy_score),
        energy_pct = VALUES(energy_pct),
        energy_tier = VALUES(energy_tier),
        state = VALUES(state),
        transition = VALUES(transition),
        source_json = VALUES(source_json),
        created_at = NOW(6);

    SELECT COUNT(*) INTO v_has_entry
    FROM cn_rotation_entry_snap_t
    WHERE run_id = v_run_id AND trade_date = v_trade_date;

    IF v_has_entry = 0 THEN
        INSERT INTO cn_rotation_entry_snap_t (
            run_id, trade_date, sector_type, sector_id, sector_name,
            entry_rank, entry_cnt, weight_suggested, signal_score,
            energy_score, energy_pct, energy_tier, state, transition,
            source_json, created_at
        )
        VALUES (
            v_run_id, v_trade_date, 'ALL', '-1', 'NO_ENTRY_TODAY',
            NULL, 0, NULL, NULL,
            NULL, NULL, NULL, NULL, NULL,
            JSON_OBJECT('summary', 'NO_ENTRY_TODAY'),
            NOW(6)
        )
        ON DUPLICATE KEY UPDATE
            sector_name = VALUES(sector_name),
            entry_cnt = VALUES(entry_cnt),
            source_json = VALUES(source_json),
            created_at = NOW(6);
    END IF;

    INSERT INTO cn_rotation_holding_snap_t (
        run_id, trade_date, sector_type, sector_id, sector_name,
        enter_signal_date, exec_enter_date, hold_days, min_hold_days,
        exit_signal_today, exit_transition, exit_exec_status, next_exit_eligible_date,
        source_json, created_at
    )
    SELECT
        p.run_id,
        p.trade_date,
        p.sector_type,
        p.sector_id,
        p.sector_name,
        NULL,
        NULL,
        CAST(p.hold_days AS SIGNED),
        CAST(p.min_hold AS SIGNED),
        CAST(IFNULL(p.exit_flag, 0) AS SIGNED),
        p.exit_reason,
        CASE WHEN IFNULL(p.exit_flag, 0) = 1 THEN 'PENDING' ELSE 'HOLD' END,
        DATE_ADD(p.trade_date, INTERVAL GREATEST(IFNULL(p.min_hold, 0) - IFNULL(p.hold_days, 0), 0) DAY),
        JSON_OBJECT('source', 'CN_SECTOR_ROT_POS_DAILY_T'),
        NOW(6)
    FROM cn_sector_rot_pos_daily_t p
    WHERE p.run_id = v_run_id
      AND p.trade_date = v_trade_date
      AND p.w > 0
    ON DUPLICATE KEY UPDATE
        sector_name = VALUES(sector_name),
        hold_days = VALUES(hold_days),
        min_hold_days = VALUES(min_hold_days),
        exit_signal_today = VALUES(exit_signal_today),
        exit_transition = VALUES(exit_transition),
        exit_exec_status = VALUES(exit_exec_status),
        next_exit_eligible_date = VALUES(next_exit_eligible_date),
        source_json = VALUES(source_json),
        created_at = NOW(6);

    SELECT COUNT(*) INTO v_has_holding
    FROM cn_rotation_holding_snap_t
    WHERE run_id = v_run_id AND trade_date = v_trade_date;

    IF v_has_holding = 0 THEN
        INSERT INTO cn_rotation_holding_snap_t (
            run_id, trade_date, sector_type, sector_id, sector_name,
            enter_signal_date, exec_enter_date, hold_days, min_hold_days,
            exit_signal_today, exit_transition, exit_exec_status, next_exit_eligible_date,
            source_json, created_at
        )
        VALUES (
            v_run_id, v_trade_date, 'ALL', '-1', 'NO_HOLDING_TODAY',
            NULL, NULL, 0, 0,
            0, NULL, 'NO_HOLDING', NULL,
            JSON_OBJECT('summary', 'NO_HOLDING_TODAY'),
            NOW(6)
        )
        ON DUPLICATE KEY UPDATE
            sector_name = VALUES(sector_name),
            hold_days = VALUES(hold_days),
            min_hold_days = VALUES(min_hold_days),
            exit_exec_status = VALUES(exit_exec_status),
            source_json = VALUES(source_json),
            created_at = NOW(6);
    END IF;

    INSERT INTO cn_rotation_exit_snap_t (
        run_id, trade_date, exec_exit_date, sector_type, sector_id, sector_name,
        state, transition, entry_rank, signal_score,
        enter_signal_date, exec_enter_date, hold_days, min_hold_days,
        exit_exec_status, source_json, created_at
    )
    SELECT
        v_run_id,
        s.signal_date,
        s.signal_date,
        s.sector_type,
        s.sector_id,
        s.sector_name,
        s.state,
        s.transition,
        CAST(s.entry_rank AS SIGNED),
        CAST(s.score AS DECIMAL(38,16)),
        NULL,
        NULL,
        CAST(IFNULL(p.hold_days, 0) AS SIGNED),
        CAST(IFNULL(p.min_hold, 0) AS SIGNED),
        'EXIT_TODAY',
        JSON_OBJECT('source', 'SP_REFRESH_ROTATION_SNAP_ALL', 'action', 'EXIT'),
        NOW(6)
    FROM cn_sector_rotation_signal_t s
    LEFT JOIN cn_sector_rot_pos_daily_t p
      ON p.trade_date = s.signal_date
     AND p.run_id = v_run_id
     AND p.sector_type = s.sector_type
     AND p.sector_id = s.sector_id
    WHERE s.signal_date = v_trade_date
      AND s.action = 'EXIT'
    ON DUPLICATE KEY UPDATE
        sector_name = VALUES(sector_name),
        state = VALUES(state),
        transition = VALUES(transition),
        entry_rank = VALUES(entry_rank),
        signal_score = VALUES(signal_score),
        hold_days = VALUES(hold_days),
        min_hold_days = VALUES(min_hold_days),
        exit_exec_status = VALUES(exit_exec_status),
        source_json = VALUES(source_json),
        created_at = NOW(6);

    SELECT COUNT(*) INTO v_has_exit
    FROM cn_rotation_exit_snap_t
    WHERE run_id = v_run_id AND trade_date = v_trade_date;

    IF v_has_exit = 0 THEN
        INSERT INTO cn_rotation_exit_snap_t (
            run_id, trade_date, exec_exit_date, sector_type, sector_id, sector_name,
            state, transition, entry_rank, signal_score,
            enter_signal_date, exec_enter_date, hold_days, min_hold_days,
            exit_exec_status, source_json, created_at
        )
        VALUES (
            v_run_id, v_trade_date, NULL, 'ALL', '-1', 'NO_EXIT_TODAY',
            NULL, NULL, NULL, NULL,
            NULL, NULL, 0, 0,
            'NO_EXIT',
            JSON_OBJECT('summary', 'NO_EXIT_TODAY'),
            NOW(6)
        )
        ON DUPLICATE KEY UPDATE
            sector_name = VALUES(sector_name),
            hold_days = VALUES(hold_days),
            min_hold_days = VALUES(min_hold_days),
            exit_exec_status = VALUES(exit_exec_status),
            source_json = VALUES(source_json),
            created_at = NOW(6);
    END IF;
END proc;
