-- cn_market.SP_VALIDATE_AGAINST_BASELINE

DROP PROCEDURE IF EXISTS `SP_VALIDATE_AGAINST_BASELINE`;

CREATE PROCEDURE `SP_VALIDATE_AGAINST_BASELINE`(
    IN p_run_id VARCHAR(128),
    IN p_baseline_id VARCHAR(64),
    IN p_min_alpha DECIMAL(18,10)
)
proc: BEGIN
    DECLARE v_run_id VARCHAR(128);
    DECLARE v_baseline_key VARCHAR(64);
    DECLARE v_baseline_run_id VARCHAR(128);
    DECLARE v_asof DATE;
    DECLARE v_nav_run DECIMAL(18,10);
    DECLARE v_nav_base DECIMAL(18,10);
    DECLARE v_alpha DECIMAL(18,10);
    DECLARE v_decision VARCHAR(16);
    DECLARE v_reason VARCHAR(4000);
    DECLARE v_min_alpha DECIMAL(18,10);

    SET v_run_id = COALESCE(NULLIF(p_run_id, ''), 'SR_LIVE_DEFAULT');
    SET v_baseline_key = COALESCE(NULLIF(p_baseline_id, ''), 'DEFAULT_BASELINE');
    SET v_min_alpha = COALESCE(p_min_alpha, 0);

    SELECT r.run_id
    INTO v_baseline_run_id
    FROM cn_baseline_registry_t r
    WHERE r.baseline_key = v_baseline_key
      AND IFNULL(r.is_active, 1) = 1
    LIMIT 1;

    IF v_baseline_run_id IS NULL THEN
        SELECT b.run_id
        INTO v_baseline_run_id
        FROM cn_sector_rot_baseline_t b
        WHERE b.baseline_key = v_baseline_key
          AND IFNULL(b.is_active, 1) = 1
        LIMIT 1;
    END IF;

    IF v_baseline_run_id IS NULL THEN
        SET v_decision = 'FAIL';
        SET v_reason = CONCAT('BASELINE_NOT_FOUND:', v_baseline_key);
        SET v_asof = CURRENT_DATE();
        SET v_nav_run = NULL;
        SET v_nav_base = NULL;
        SET v_alpha = NULL;
    ELSE
        SET v_asof = (
            SELECT LEAST(
                COALESCE((SELECT MAX(trade_date) FROM cn_sector_rot_bt_daily_t WHERE run_id = v_run_id), CURRENT_DATE()),
                COALESCE((SELECT MAX(trade_date) FROM cn_sector_rot_bt_daily_t WHERE run_id = v_baseline_run_id), CURRENT_DATE())
            )
        );

        SELECT b.nav
        INTO v_nav_run
        FROM cn_sector_rot_bt_daily_t b
        WHERE b.run_id = v_run_id
          AND b.trade_date = v_asof
        LIMIT 1;

        SELECT b.nav
        INTO v_nav_base
        FROM cn_sector_rot_bt_daily_t b
        WHERE b.run_id = v_baseline_run_id
          AND b.trade_date = v_asof
        LIMIT 1;

        IF v_nav_run IS NULL THEN
            SET v_decision = 'FAIL';
            SET v_reason = CONCAT('RUN_NAV_MISSING@', CAST(v_asof AS CHAR));
            SET v_alpha = NULL;
        ELSEIF v_nav_base IS NULL OR v_nav_base = 0 THEN
            SET v_decision = 'FAIL';
            SET v_reason = CONCAT('BASE_NAV_MISSING_OR_ZERO@', CAST(v_asof AS CHAR));
            SET v_alpha = NULL;
        ELSE
            SET v_alpha = (v_nav_run / v_nav_base) - 1;
            IF v_alpha >= v_min_alpha THEN
                SET v_decision = 'PASS';
                SET v_reason = 'ALPHA_OK';
            ELSE
                SET v_decision = 'FAIL';
                SET v_reason = 'ALPHA_BELOW_THRESHOLD';
            END IF;
        END IF;
    END IF;

    INSERT INTO cn_baseline_decision_t (
        run_id, baseline_key, baseline_run_id, decision, reason_code,
        metrics_json, compare_asof, created_at, updated_at, created_by
    )
    VALUES (
        v_run_id,
        v_baseline_key,
        COALESCE(v_baseline_run_id, ''),
        v_decision,
        v_reason,
        JSON_OBJECT(
            'alpha', v_alpha,
            'min_alpha', v_min_alpha,
            'nav_run', v_nav_run,
            'nav_baseline', v_nav_base,
            'asof', CAST(v_asof AS CHAR)
        ),
        NOW(6),
        NOW(6),
        NOW(6),
        'SP_VALIDATE_AGAINST_BASELINE'
    )
    ON DUPLICATE KEY UPDATE
        baseline_run_id = VALUES(baseline_run_id),
        decision = VALUES(decision),
        reason_code = VALUES(reason_code),
        metrics_json = VALUES(metrics_json),
        compare_asof = NOW(6),
        updated_at = NOW(6),
        created_by = 'SP_VALIDATE_AGAINST_BASELINE';

    SELECT
        v_run_id AS run_id,
        v_baseline_key AS baseline_key,
        v_baseline_run_id AS baseline_run_id,
        v_decision AS decision,
        v_reason AS reason_code,
        v_alpha AS alpha,
        v_min_alpha AS min_alpha,
        v_nav_run AS nav_run,
        v_nav_base AS nav_baseline,
        v_asof AS asof_date;
END proc;
