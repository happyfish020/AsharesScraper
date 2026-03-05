-- cn_market.sp_refresh_sector_eod_hist
-- History-safe refresh using daily board-member mapping table (cn_board_member_map_d).

DROP PROCEDURE IF EXISTS `sp_refresh_sector_eod_hist`;

CREATE PROCEDURE `sp_refresh_sector_eod_hist`(
    IN p_from DATE,
    IN p_to DATE,
    IN p_top_pct DECIMAL(10, 6),
    IN p_breadth_min DECIMAL(10, 6)
)
BEGIN
    IF p_from IS NULL OR p_to IS NULL OR p_from > p_to THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'invalid date range';
    END IF;

    DELETE FROM cn_sector_eod_hist_t
    WHERE trade_date BETWEEN p_from AND p_to;

    DROP TEMPORARY TABLE IF EXISTS tmp_price;
    CREATE TEMPORARY TABLE tmp_price (
        trade_date DATE NOT NULL,
        symbol VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
        amount DOUBLE NULL,
        pre_close DOUBLE NULL,
        close DOUBLE NULL,
        chg DOUBLE NULL,
        chg_pct DOUBLE NULL,
        KEY idx_tp_date_symbol (trade_date, symbol),
        KEY idx_tp_symbol_date (symbol, trade_date)
    ) ENGINE=InnoDB;

    INSERT INTO tmp_price (trade_date, symbol, amount, pre_close, close, chg, chg_pct)
    SELECT p.trade_date, p.symbol, p.amount, p.pre_close, p.close, p.change, p.chg_pct
    FROM cn_stock_daily_price_active_v p
    WHERE p.trade_date BETWEEN p_from AND p_to;

    DROP TEMPORARY TABLE IF EXISTS tmp_map;
    CREATE TEMPORARY TABLE tmp_map (
        trade_date DATE NOT NULL,
        sector_type VARCHAR(16) NOT NULL,
        sector_id VARCHAR(64) NOT NULL,
        symbol VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
        KEY idx_tm_date_symbol_type_sector (trade_date, symbol, sector_type, sector_id),
        KEY idx_tm_symbol_date (symbol, trade_date, sector_type, sector_id)
    ) ENGINE=InnoDB;

    INSERT INTO tmp_map (trade_date, sector_type, sector_id, symbol)
    SELECT m.trade_date, m.sector_type, m.sector_id, (m.symbol COLLATE utf8mb4_unicode_ci)
    FROM cn_board_member_map_d m
    WHERE m.trade_date BETWEEN p_from AND p_to
      AND m.sector_type IN ('INDUSTRY', 'CONCEPT');

    DROP TEMPORARY TABLE IF EXISTS tmp_sector_eod_raw;
    CREATE TEMPORARY TABLE tmp_sector_eod_raw (
        trade_date DATE NOT NULL,
        sector_type VARCHAR(16) NOT NULL,
        sector_id VARCHAR(64) NOT NULL,
        members INT NOT NULL,
        amount_sum DOUBLE NULL,
        eqw_ret DOUBLE NULL,
        up_ratio DOUBLE NULL,
        KEY idx_tmp1 (trade_date, sector_type, sector_id),
        KEY idx_tmp2 (sector_type, sector_id, trade_date)
    ) ENGINE=InnoDB;

    INSERT INTO tmp_sector_eod_raw (trade_date, sector_type, sector_id, members, amount_sum, eqw_ret, up_ratio)
    SELECT z.trade_date, z.sector_type, z.sector_id,
           COUNT(*) AS members,
           SUM(IFNULL(z.amount, 0)) AS amount_sum,
           AVG(z.ret_eff) AS eqw_ret,
           AVG(CASE WHEN z.ret_eff > 0 THEN 1 ELSE 0 END) AS up_ratio
    FROM (
        SELECT p.trade_date, m.sector_type, m.sector_id, p.amount,
               CASE
                   WHEN COALESCE(p.pre_close, p.close - p.chg) IS NOT NULL
                        AND COALESCE(p.pre_close, p.close - p.chg) <> 0
                       THEN (p.close / COALESCE(p.pre_close, p.close - p.chg)) - 1
                   WHEN p.chg_pct IS NOT NULL
                       THEN CASE WHEN ABS(p.chg_pct) > 1 THEN p.chg_pct / 100 ELSE p.chg_pct END
                   ELSE NULL
               END AS ret_eff
        FROM tmp_price p
        JOIN tmp_map m ON m.trade_date = p.trade_date AND m.symbol = p.symbol
    ) z
    GROUP BY z.trade_date, z.sector_type, z.sector_id;

    IF NOT EXISTS (SELECT 1 FROM tmp_sector_eod_raw LIMIT 1) THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = 'no source rows found for requested date range (check price/mapping tables)';
    END IF;

    INSERT INTO cn_sector_eod_hist_t (
        trade_date, sector_type, sector_id,
        members, amount_sum,
        sector_close, sector_ma20, sector_ret20, up_ratio,
        score, rank_pct,
        cond1_trend, cond2_rank, cond3_breadth, cond_count, sector_pass
    )
    WITH base AS (
        SELECT r.trade_date, r.sector_type, r.sector_id, r.members, r.amount_sum,
               NULL AS sector_close, NULL AS sector_ma20, NULL AS sector_ret20,
               CAST(CASE WHEN r.up_ratio IS NULL OR r.up_ratio <> r.up_ratio THEN NULL WHEN r.up_ratio < 0 THEN 0 WHEN r.up_ratio > 1 THEN 1 ELSE r.up_ratio END AS DECIMAL(10, 6)) AS up_ratio,
               CAST(CASE WHEN r.eqw_ret IS NULL OR r.eqw_ret <> r.eqw_ret THEN 0 WHEN r.eqw_ret > 99999 THEN 99999 WHEN r.eqw_ret < -99999 THEN -99999 ELSE r.eqw_ret END * 100 AS DECIMAL(20, 8)) AS score,
               CASE WHEN IFNULL(r.eqw_ret, 0) > 0 THEN 1 ELSE 0 END AS cond1_trend,
               CASE WHEN IFNULL(r.up_ratio, 0) > p_breadth_min THEN 1 ELSE 0 END AS cond3_breadth
        FROM tmp_sector_eod_raw r
    ), ranked AS (
        SELECT b.*, CAST(CASE WHEN pr IS NULL THEN 1 WHEN pr < 0 THEN 0 WHEN pr > 1 THEN 1 ELSE pr END AS DECIMAL(10,6)) AS rank_pct
        FROM (
            SELECT b.*, PERCENT_RANK() OVER (PARTITION BY b.trade_date ORDER BY b.score DESC) AS pr
            FROM base b
        ) b
    )
    SELECT
        trade_date, sector_type, sector_id, members, amount_sum,
        sector_close, sector_ma20, sector_ret20, up_ratio,
        CAST(CASE WHEN score > 999999999999.99999999 THEN 999999999999.99999999 WHEN score < -999999999999.99999999 THEN -999999999999.99999999 ELSE score END AS DECIMAL(20,8)) AS score,
        rank_pct,
        cond1_trend,
        CASE WHEN rank_pct <= p_top_pct THEN 1 ELSE 0 END AS cond2_rank,
        cond3_breadth,
        cond1_trend + (CASE WHEN rank_pct <= p_top_pct THEN 1 ELSE 0 END) + cond3_breadth AS cond_count,
        CASE WHEN cond1_trend + (CASE WHEN rank_pct <= p_top_pct THEN 1 ELSE 0 END) + cond3_breadth >= 2 THEN 1 ELSE 0 END AS sector_pass
    FROM ranked;
END;
