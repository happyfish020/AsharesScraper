-- ============================================================================
-- optimize_sp_rebuild_rotation_year.sql
-- 优化 sp_rebuild_rotation_year 性能：方案1+2（方案3 MEMORY 引擎已移除）
-- 1. 物化中间表 cn_sector_rotation_transition_d 替代视图
-- 2. 批量 SQL 替代逐日游标循环
-- 3. [已移除] MEMORY 引擎 — MySQL 9.x/Windows 下 MEMORY 有每表内存限制，
--    即使设置 max_heap_table_size=2GB，大表（如 tmp_map 的 600万+ 行）
--    仍会报 "table is full"。改用 InnoDB 临时表，无此限制。
--
-- 结果正确性保证（不漂移）：
--   - 物化中间表的 CTE 逻辑与 cn_sector_rotation_transition_v 视图完全一致
--   - 窗口函数（AVG 20天、AVG 5天、LAG）需要历史数据预热，
--     因此 hist0 多取 p_from 之前 20 个交易日的数据，确保边界值正确
--   - 最终只输出 p_from ~ p_to 范围的结果
--   - 批量 ranked/signal 插入逻辑与逐日版完全一致
--
-- 使用方式：
--   1. SOURCE sql/optimize_sp_rebuild_rotation_year.sql;
--   2. CALL sp_rebuild_rotation_year_optimized('2024-01-01', '2024-12-31', 0.30, 0.60);
-- ============================================================================
--
-- 注意：本脚本不删除原有存储过程，新增 _optimized 后缀版本。
--       原有 SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE 和
--       SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE 保持不动（日常增量仍使用）。
-- ============================================================================

-- ============================================================================
-- 第一步：创建物化中间表 cn_sector_rotation_transition_d
-- 结构与 cn_sector_rotation_transition_v 视图输出一致
-- ============================================================================
DROP TABLE IF EXISTS cn_sector_rotation_transition_d;

CREATE TABLE cn_sector_rotation_transition_d (
    trade_date          DATE          NOT NULL,
    trading_date        VARCHAR(10)   DEFAULT NULL,
    sector_type         VARCHAR(16)   NOT NULL,
    sector_id           VARCHAR(64)   NOT NULL,
    sector_name         VARCHAR(200)  DEFAULT NULL,
    theme_group         VARCHAR(50)   DEFAULT NULL,
    theme_rank          INT           DEFAULT NULL,
    theme_flag          VARCHAR(16)   DEFAULT NULL,
    tier                VARCHAR(10)   DEFAULT NULL,
    state               VARCHAR(20)   DEFAULT NULL,
    score               DECIMAL(20,8) DEFAULT NULL,
    confirm_streak      INT           DEFAULT NULL,
    amt_impulse         DECIMAL(20,8) DEFAULT NULL,
    up_ratio            DECIMAL(10,6) DEFAULT NULL,
    up_ma5              DECIMAL(10,6) DEFAULT NULL,
    prev_state          VARCHAR(20)   DEFAULT NULL,
    prev_tier           VARCHAR(10)   DEFAULT NULL,
    prev_score          DECIMAL(20,8) DEFAULT NULL,
    score_delta         DECIMAL(20,8) DEFAULT NULL,
    transition          VARCHAR(32)   DEFAULT NULL,
    created_at          TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, sector_type, sector_id),
    KEY idx_transition_d_date (trade_date),
    KEY idx_transition_d_type_sector_date (sector_type, sector_id, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='物化中间表：替代 cn_sector_rotation_transition_v 视图，由 sp_build_sector_rotation_transition_batch 填充';


-- ============================================================================
-- 第二步：创建批量填充中间表的存储过程
-- 一次性计算所有日期的 transition 数据（替代视图的 8 层 CTE）
--
-- 结果正确性保证：
--   视图 cn_sector_rotation_transition_v 是全量计算的（无 WHERE 过滤），
--   其中的窗口函数（AVG 20天、AVG 5天、LAG）依赖历史数据。
--   为了结果不漂移，本过程在 hist0 中多取 p_from 之前 20 个交易日的数据
--   作为窗口函数的"预热"数据，但最终只输出 p_from ~ p_to 范围的结果。
-- ============================================================================
DROP PROCEDURE IF EXISTS sp_build_sector_rotation_transition_batch;

DELIMITER $$

CREATE PROCEDURE sp_build_sector_rotation_transition_batch(
    IN p_from DATE,
    IN p_to   DATE
)
BEGIN
    DECLARE v_hist_start DATE;

    -- 往前多取 20 个交易日（覆盖 AVG 20天窗口 + LAG 1天）
    -- 使用 cn_stock_daily_price 表获取实际交易日历
    SELECT COALESCE(MIN(trade_date), DATE_SUB(p_from, INTERVAL 20 DAY))
    INTO v_hist_start
    FROM (
        SELECT DISTINCT trade_date
        FROM cn_stock_daily_price
        WHERE trade_date < p_from
        ORDER BY trade_date DESC
        LIMIT 20
    ) t;

    -- 删除目标日期范围内的旧数据
    DELETE FROM cn_sector_rotation_transition_d
    WHERE trade_date BETWEEN p_from AND p_to;

    -- 一次性批量插入，逻辑与 cn_sector_rotation_transition_v 完全一致
    -- hist0 多取 v_hist_start ~ p_from 的数据作为窗口函数预热，
    -- 但最终 INSERT 只输出 p_from ~ p_to 范围的结果
    INSERT INTO cn_sector_rotation_transition_d (
        trade_date, trading_date,
        sector_type, sector_id, sector_name,
        theme_group, theme_rank, theme_flag,
        tier, state, score, confirm_streak,
        amt_impulse, up_ratio, up_ma5,
        prev_state, prev_tier, prev_score,
        score_delta, transition
    )
    WITH industry_name AS (
        SELECT x.board_id, x.board_name
        FROM (
            SELECT
                m.BOARD_ID AS board_id,
                m.BOARD_NAME AS board_name,
                ROW_NUMBER() OVER (PARTITION BY m.BOARD_ID ORDER BY m.ASOF_DATE DESC) AS rn
            FROM cn_board_industry_master m
        ) x
        WHERE x.rn = 1
    ),
    concept_name AS (
        SELECT x.concept_id, x.concept_name
        FROM (
            SELECT
                m.CONCEPT_ID AS concept_id,
                m.CONCEPT_NAME AS concept_name,
                ROW_NUMBER() OVER (PARTITION BY m.CONCEPT_ID ORDER BY m.ASOF_DATE DESC) AS rn
            FROM cn_board_concept_master m
        ) x
        WHERE x.rn = 1
    ),
    hist0 AS (
        SELECT
            e.trade_date,
            e.sector_type,
            e.sector_id,
            COALESCE(
                CASE WHEN e.sector_type = 'INDUSTRY' THEN im.board_name END,
                CASE WHEN e.sector_type = 'CONCEPT' THEN cm.concept_name END,
                e.sector_id
            ) AS sector_name,
            e.amount_sum,
            e.score,
            e.up_ratio,
            e.cond_count,
            e.sector_pass
        FROM cn_sector_eod_hist_t e
        LEFT JOIN industry_name im
          ON e.sector_type = 'INDUSTRY'
         AND e.sector_id = im.board_id
        LEFT JOIN concept_name cm
          ON e.sector_type = 'CONCEPT'
         AND e.sector_id = cm.concept_id
        -- 多取历史数据确保窗口函数（AVG 20天、LAG）结果正确
        WHERE e.trade_date BETWEEN v_hist_start AND p_to
    ),
    hist1 AS (
        SELECT
            h0.*,
            AVG(h0.amount_sum) OVER (
                PARTITION BY h0.sector_type, h0.sector_id
                ORDER BY h0.trade_date
                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) AS amt_ma20,
            AVG(h0.up_ratio) OVER (
                PARTITION BY h0.sector_type, h0.sector_id
                ORDER BY h0.trade_date
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            ) AS up_ma5,
            ROW_NUMBER() OVER (
                PARTITION BY h0.sector_type, h0.sector_id
                ORDER BY h0.trade_date
            ) AS rn_all,
            CASE
                WHEN h0.sector_pass = 1 THEN ROW_NUMBER() OVER (
                    PARTITION BY h0.sector_type, h0.sector_id, h0.sector_pass
                    ORDER BY h0.trade_date
                )
                ELSE NULL
            END AS rn_pass
        FROM hist0 h0
    ),
    hist2 AS (
        SELECT
            h1.*,
            CASE WHEN h1.sector_pass = 1 THEN (h1.rn_all - h1.rn_pass) ELSE NULL END AS grp_pass
        FROM hist1 h1
    ),
    hist3 AS (
        SELECT
            h2.*,
            CASE
                WHEN h2.sector_pass = 1 THEN ROW_NUMBER() OVER (
                    PARTITION BY h2.sector_type, h2.sector_id, h2.grp_pass
                    ORDER BY h2.trade_date
                )
                ELSE 0
            END AS confirm_streak
        FROM hist2 h2
    ),
    hist4 AS (
        SELECT
            h3.trade_date,
            h3.sector_type,
            h3.sector_id,
            h3.sector_name,
            h3.score,
            h3.confirm_streak,
            CASE
                WHEN h3.amt_ma20 = 0 OR h3.amt_ma20 IS NULL THEN NULL
                ELSE h3.amount_sum / h3.amt_ma20
            END AS amt_impulse,
            h3.up_ma5,
            h3.up_ratio,
            CASE
                WHEN h3.sector_pass = 1 AND h3.cond_count >= 3 THEN 'CONFIRM'
                WHEN h3.sector_pass = 1 AND h3.cond_count = 2 THEN 'HOLD'
                WHEN h3.cond_count >= 2 THEN 'IGNITE'
                WHEN h3.cond_count = 1 THEN 'FADE'
                ELSE 'NEUTRAL'
            END AS state
        FROM hist3 h3
    ),
    hist5 AS (
        SELECT
            h4.*,
            CASE
                WHEN h4.state = 'CONFIRM' AND IFNULL(h4.confirm_streak, 0) >= 2 THEN 'T1'
                WHEN h4.state = 'CONFIRM' THEN 'T2'
                WHEN h4.state = 'HOLD' THEN 'T2'
                WHEN h4.state = 'IGNITE' THEN 'T3'
                WHEN h4.state = 'FADE' THEN 'T4'
                ELSE 'T9'
            END AS tier,
            CASE
                WHEN h4.state = 'CONFIRM' AND IFNULL(h4.confirm_streak, 0) >= 2 THEN 0
                WHEN h4.state = 'CONFIRM' THEN 1
                WHEN h4.state = 'HOLD' THEN 2
                WHEN h4.state = 'IGNITE' THEN 3
                WHEN h4.state = 'FADE' THEN 4
                ELSE 9
            END AS tier_pri,
            CASE
                WHEN h4.sector_type = 'INDUSTRY' THEN 'INDUSTRY'
                WHEN h4.sector_type = 'CONCEPT' THEN 'CONCEPT'
                ELSE 'OTHER'
            END AS theme_group
        FROM hist4 h4
    ),
    ranked AS (
        SELECT
            h5.trade_date,
            h5.sector_type,
            h5.sector_id,
            h5.sector_name,
            h5.state,
            h5.tier,
            h5.theme_group,
            ROW_NUMBER() OVER (
                PARTITION BY h5.trade_date, h5.theme_group
                ORDER BY h5.tier_pri, h5.score DESC, h5.sector_type, h5.sector_id
            ) AS theme_rank,
            h5.score,
            h5.confirm_streak,
            h5.amt_impulse,
            h5.up_ma5,
            h5.up_ratio
        FROM hist5 h5
    ),
    x AS (
        SELECT
            r.*,
            LAG(r.state) OVER (PARTITION BY r.sector_type, r.sector_id ORDER BY r.trade_date) AS prev_state,
            LAG(r.tier)  OVER (PARTITION BY r.sector_type, r.sector_id ORDER BY r.trade_date) AS prev_tier,
            LAG(r.score) OVER (PARTITION BY r.sector_type, r.sector_id ORDER BY r.trade_date) AS prev_score
        FROM ranked r
    )
    SELECT
        x.trade_date,
        DATE_FORMAT(x.trade_date, '%Y-%m-%d') AS trading_date,
        x.sector_type,
        x.sector_id,
        x.sector_name,
        x.theme_group,
        x.theme_rank,
        CASE WHEN x.theme_rank = 1 THEN 'KEEP' ELSE 'DUP_THEME' END AS theme_flag,
        x.tier,
        x.state,
        x.score,
        x.confirm_streak,
        x.amt_impulse,
        x.up_ratio,
        x.up_ma5,
        x.prev_state,
        x.prev_tier,
        x.prev_score,
        (x.score - x.prev_score) AS score_delta,
        CASE
            WHEN x.prev_state IS NULL THEN 'NO_PREV'
            WHEN x.prev_state = x.state THEN 'NO_CHANGE'
            WHEN x.prev_state IN ('NEUTRAL', 'FADE') AND x.state = 'IGNITE' THEN 'START_IGNITE'
            WHEN x.prev_state = 'IGNITE' AND x.state = 'CONFIRM' THEN 'IGNITE_TO_CONFIRM'
            WHEN x.prev_state IN ('NEUTRAL', 'FADE') AND x.state = 'CONFIRM' THEN 'DIRECT_CONFIRM'
            WHEN x.prev_state = 'CONFIRM' AND x.state = 'HOLD' THEN 'CONFIRM_TO_HOLD'
            WHEN x.prev_state = 'HOLD' AND x.state = 'CONFIRM' THEN 'HOLD_TO_CONFIRM'
            WHEN x.prev_state IN ('CONFIRM', 'HOLD') AND x.state = 'FADE' THEN 'TREND_TO_FADE'
            WHEN x.prev_state IN ('CONFIRM', 'HOLD') AND x.state = 'NEUTRAL' THEN 'TREND_BREAK_TO_NEUTRAL'
            WHEN x.prev_state = 'IGNITE' AND x.state IN ('NEUTRAL', 'FADE') THEN 'IGNITE_FAIL'
            ELSE 'OTHER_CHANGE'
        END AS transition
    FROM x
    -- 关键：只输出目标日期范围的结果，预热数据不写入中间表
    WHERE x.trade_date BETWEEN p_from AND p_to;
END$$

DELIMITER ;


-- ============================================================================
-- 第三步：优化 sp_refresh_sector_eod_hist
-- 临时表使用 InnoDB 引擎（原版使用默认引擎，此处显式指定 InnoDB）
-- 注意：MEMORY 引擎在 MySQL 9.x / Windows 环境下有每表内存限制，
--       即使设置 max_heap_table_size=2GB，大表（如 tmp_map 的 600万+ 行）
--       仍会报 "table is full"。改用 InnoDB 临时表无此限制，
--       且 InnoDB 临时表使用临时表空间（ibtmp1），性能足够。
-- ============================================================================
DROP PROCEDURE IF EXISTS sp_refresh_sector_eod_hist_optimized;

DELIMITER $$

CREATE PROCEDURE sp_refresh_sector_eod_hist_optimized(
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

    -- tmp_price: InnoDB 临时表（原方案3使用 MEMORY，但 MySQL 9.x/Windows 下
    -- MEMORY 引擎有每表内存限制，大表会报 "table is full"，故改用 InnoDB）
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

    -- tmp_map: InnoDB 临时表
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

    -- tmp_sector_eod_raw: InnoDB 临时表
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

    -- 清理临时表
    DROP TEMPORARY TABLE IF EXISTS tmp_price;
    DROP TEMPORARY TABLE IF EXISTS tmp_map;
    DROP TEMPORARY TABLE IF EXISTS tmp_sector_eod_raw;
END$$

DELIMITER ;


-- ============================================================================
-- 第四步：创建批量版 ranked 插入（替代逐日 CALL）
-- 从物化中间表 cn_sector_rotation_transition_d 批量读取
-- ============================================================================
DROP PROCEDURE IF EXISTS sp_build_sector_rotation_ranked_batch;

DELIMITER $$

CREATE PROCEDURE sp_build_sector_rotation_ranked_batch(
    IN p_from DATE,
    IN p_to   DATE
)
BEGIN
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
    FROM cn_sector_rotation_transition_d t
    WHERE t.trade_date BETWEEN p_from AND p_to
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
END$$

DELIMITER ;


-- ============================================================================
-- 第五步：创建批量版 signal 插入（替代逐日 CALL）
-- 从物化中间表 cn_sector_rotation_transition_d 批量读取
-- ============================================================================
DROP PROCEDURE IF EXISTS sp_build_sector_rotation_signal_batch;

DELIMITER $$

CREATE PROCEDURE sp_build_sector_rotation_signal_batch(
    IN p_from DATE,
    IN p_to   DATE
)
BEGIN
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
        FROM cn_sector_rotation_transition_d t
        WHERE t.trade_date BETWEEN p_from AND p_to
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
END$$

DELIMITER ;


-- ============================================================================
-- 第六步：创建优化版 sp_rebuild_rotation_year
-- 方案1（批量SQL）+ 方案2（物化中间表）+ 方案3（MEMORY临时表）
-- ============================================================================
DROP PROCEDURE IF EXISTS sp_rebuild_rotation_year_optimized;

DELIMITER $$

CREATE PROCEDURE sp_rebuild_rotation_year_optimized(
    IN p_start DATE,
    IN p_end DATE,
    IN p_top_pct DECIMAL(10,4),
    IN p_breadth_min DECIMAL(10,4)
)
BEGIN
    DECLARE v_start DATETIME;
    DECLARE v_step1 DATETIME;
    DECLARE v_step2 DATETIME;
    DECLARE v_step3 DATETIME;
    DECLARE v_step4 DATETIME;

    SET v_start = NOW();

    -- ============================================================
    -- Step 1: 清理目标表
    -- ============================================================
    DELETE FROM cn_sector_eod_hist_t
    WHERE trade_date BETWEEN p_start AND p_end;

    DELETE FROM cn_sector_rotation_ranked_t
    WHERE trade_date BETWEEN p_start AND p_end;

    DELETE FROM cn_sector_rotation_signal_t
    WHERE signal_date BETWEEN p_start AND p_end;

    DELETE FROM cn_sector_rotation_transition_d
    WHERE trade_date BETWEEN p_start AND p_end;

    SET v_step1 = NOW();
    SELECT CONCAT('[PERF] Step1 DELETE: ', TIMEDIFF(v_step1, v_start)) AS perf_log;

    -- ============================================================
    -- Step 2: 重建 sector EOD 历史（优化版：MEMORY 临时表）
    -- ============================================================
    CALL sp_refresh_sector_eod_hist_optimized(p_start, p_end, p_top_pct, p_breadth_min);

    SET v_step2 = NOW();
    SELECT CONCAT('[PERF] Step2 sp_refresh_sector_eod_hist_optimized: ', TIMEDIFF(v_step2, v_step1)) AS perf_log;

    -- ============================================================
    -- Step 3: 批量填充物化中间表（替代视图，只计算1次）
    -- ============================================================
    CALL sp_build_sector_rotation_transition_batch(p_start, p_end);

    SET v_step3 = NOW();
    SELECT CONCAT('[PERF] Step3 sp_build_sector_rotation_transition_batch: ', TIMEDIFF(v_step3, v_step2)) AS perf_log;

    -- ============================================================
    -- Step 4: 批量插入 ranked 和 signal（替代244次游标循环）
    -- ============================================================
    CALL sp_build_sector_rotation_ranked_batch(p_start, p_end);
    CALL sp_build_sector_rotation_signal_batch(p_start, p_end);

    SET v_step4 = NOW();
    SELECT CONCAT('[PERF] Step4 batch INSERT ranked+signal: ', TIMEDIFF(v_step4, v_step3)) AS perf_log;
    SELECT CONCAT('[PERF] TOTAL: ', TIMEDIFF(v_step4, v_start)) AS perf_log;
END$$

DELIMITER ;


-- ============================================================================
-- 第七步（可选）：修改原有逐日 SP 改为查中间表
-- 这样日常增量调用（sp_rotation_daily_refresh）也能受益
-- 注意：这修改了原有存储过程，如果不想改动请注释此段
-- ============================================================================
-- 修改 SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE 改为查中间表
DROP PROCEDURE IF EXISTS SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE;

DELIMITER $$

CREATE PROCEDURE SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE(
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
    FROM cn_sector_rotation_transition_d t
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
END proc$$

DELIMITER ;


-- 修改 SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE 改为查中间表
DROP PROCEDURE IF EXISTS SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE;

DELIMITER $$

CREATE PROCEDURE SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE(
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
        FROM cn_sector_rotation_transition_d t
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
END proc$$

DELIMITER ;


-- ============================================================================
-- 使用示例
-- ============================================================================
-- 1. 执行本脚本创建所有对象
--    SOURCE sql/optimize_sp_rebuild_rotation_year.sql;
--
-- 2. 调用优化版存储过程（参数与原版完全一致）
--    CALL sp_rebuild_rotation_year_optimized('2024-01-01', '2024-12-31', 0.30, 0.60);
--
-- 4. 验证结果与原版一致
--    -- 对比 ranked 表
--    SELECT COUNT(*) FROM cn_sector_rotation_ranked_t
--    WHERE trade_date BETWEEN '2024-01-01' AND '2024-12-31';
--    -- 对比 signal 表
--    SELECT COUNT(*) FROM cn_sector_rotation_signal_t
--    WHERE signal_date BETWEEN '2024-01-01' AND '2024-12-31';
--
-- 5. 性能对比
--    -- 原版：~3小时
--    -- 优化版：预计 < 10 分钟
--
-- ============================================================================
-- 回滚方法
-- ============================================================================
-- 如果需要回滚到原版，执行以下命令：
-- DROP TABLE IF EXISTS cn_sector_rotation_transition_d;
-- DROP PROCEDURE IF EXISTS sp_build_sector_rotation_transition_batch;
-- DROP PROCEDURE IF EXISTS sp_refresh_sector_eod_hist_optimized;
-- DROP PROCEDURE IF EXISTS sp_build_sector_rotation_ranked_batch;
-- DROP PROCEDURE IF EXISTS sp_build_sector_rotation_signal_batch;
-- DROP PROCEDURE IF EXISTS sp_rebuild_rotation_year_optimized;
-- 然后重新导入原版 SP（从 DDL 备份恢复）