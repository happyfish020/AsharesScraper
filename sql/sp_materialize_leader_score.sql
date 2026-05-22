-- ============================================================================
-- sp_materialize_leader_score.sql
-- 存储过程: 使用临时表分步物化 leader score，替代 v1/v2 视图链
--
-- 设计思路:
--   原 v1/v2 视图链每次查询都全表扫描 cn_board_member_map_d (30.7M) +
--   cn_stock_daily_price (16.8M)，即使只查一天也如此。
--   本 SP 使用临时表分步计算，只处理目标日期范围的数据，大幅减少扫描量。
--
-- 执行流程:
--   1. 创建临时表 #tmp_v1: 物化 v1 层逻辑（industry_map + price_enriched + ranked）
--   2. 创建临时表 #tmp_v2: 在 #tmp_v1 基础上计算 v2 层逻辑（market_cap + leader_score）
--   3. INSERT ... ON DUPLICATE KEY UPDATE 写入 cn_stock_leader_score_daily
--   4. DROP TEMPORARY TABLE 自动清理
--
-- 用法:
--   CALL sp_materialize_leader_score('2026-05-20', '2026-05-21');
--   CALL sp_materialize_leader_score('2026-01-01', '2026-05-21');
--
-- 注意:
--   使用 MySQL 8.0 的 AS alias 语法替代已废弃的 VALUES() 函数
-- ============================================================================

DELIMITER $$

DROP PROCEDURE IF EXISTS sp_materialize_leader_score $$

CREATE PROCEDURE sp_materialize_leader_score(
    IN p_start_date DATE,
    IN p_end_date   DATE
)
sp_main: BEGIN
    DECLARE v_start_time       DATETIME(6);
    DECLARE v_step1_start      DATETIME(6);
    DECLARE v_step2_start      DATETIME(6);
    DECLARE v_step3_start      DATETIME(6);
    DECLARE v_price_lookback   DATE;
    DECLARE v_row_count        INT DEFAULT 0;
    DECLARE v_duration         DECIMAL(12,2);

    DECLARE CONTINUE HANDLER FOR SQLEXCEPTION 
    BEGIN
        GET DIAGNOSTICS CONDITION 1 @err_msg = MESSAGE_TEXT;
        SELECT CONCAT('[sp_materialize_leader_score] ERROR: ', @err_msg) AS error_msg;
        ROLLBACK;
    END;

    SET v_start_time = NOW(6);

    -- ======================================================================
    -- Step 0: 计算 price_enriched 所需的最小日期（前推 20 个交易日）
    -- ======================================================================
    SELECT DATE_SUB(MIN(trade_date), INTERVAL 20 DAY) INTO v_price_lookback
    FROM cn_board_member_map_d 
    WHERE trade_date BETWEEN p_start_date AND p_end_date
      AND sector_type = 'INDUSTRY'
      AND (sector_id LIKE 'BK%' OR sector_id LIKE '801%.SI');

    -- 如果目标范围内无 industry_map 数据，直接跳过
    IF v_price_lookback IS NULL THEN
        SELECT CONCAT('[sp_materialize_leader_score] 范围 ', p_start_date, ' ~ ', p_end_date,
                      ' 无 industry_map 数据，跳过') AS msg;
        SELECT 0 AS inserted_rows, 0.0 AS duration_sec;
        LEAVE sp_main;
    END IF;

    -- ======================================================================
    -- Step 1: #tmp_v1 — 物化 v1 层逻辑
    -- ======================================================================
    SET v_step1_start = NOW(6);

    DROP TEMPORARY TABLE IF EXISTS `#tmp_v1`;

    CREATE TEMPORARY TABLE `#tmp_v1` (
        `trade_date`                DATE        NOT NULL,
        `industry_id`               VARCHAR(32) NOT NULL,
        `industry_name`             VARCHAR(64) DEFAULT NULL,
        `symbol`                    VARCHAR(16) NOT NULL,
        `name`                      VARCHAR(64) DEFAULT NULL,
        `close`                     DECIMAL(18,4) DEFAULT NULL,
        `amount`                    DECIMAL(24,4) DEFAULT NULL,
        `turnover_20d_avg`          DECIMAL(24,4) DEFAULT NULL,
        `turnover_20d_percentile`   DOUBLE DEFAULT NULL,
        `leader_liquidity`          INT DEFAULT 0,
        `rs_20d_raw`                DECIMAL(18,6) DEFAULT NULL,
        `rs_percentile`             DOUBLE DEFAULT NULL,
        `leader_trend`              INT DEFAULT 0,
        `turnover_rank_in_industry` INT DEFAULT NULL,
        `industry_members`          INT DEFAULT NULL,
        PRIMARY KEY (`trade_date`, `industry_id`, `symbol`),
        INDEX `idx_tmp_v1_date` (`trade_date`)
    ) ENGINE=InnoDB ROW_FORMAT=DYNAMIC;

    INSERT INTO `#tmp_v1`
    WITH 
    industry_name AS (
        SELECT board_id, board_name
        FROM (
            SELECT 
                board_id,
                board_name,
                ROW_NUMBER() OVER (PARTITION BY board_id ORDER BY asof_date DESC) AS rn
            FROM cn_board_industry_master
            WHERE board_id LIKE 'BK%' OR board_id LIKE '801%.SI'
        ) t WHERE rn = 1
    ),
    industry_map AS (
        SELECT 
            trade_date,
            symbol,
            sector_id AS industry_id
        FROM cn_board_member_map_d
        WHERE trade_date BETWEEN p_start_date AND p_end_date
          AND sector_type = 'INDUSTRY'
          AND (sector_id LIKE 'BK%' OR sector_id LIKE '801%.SI')
    ),
    price_enriched AS (
        SELECT
            symbol,
            trade_date,
            name,
            close,
            amount,
            AVG(amount) OVER (PARTITION BY symbol ORDER BY trade_date 
                              ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS turnover_20d_avg,
            COUNT(amount) OVER (PARTITION BY symbol ORDER BY trade_date 
                                ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS turnover_20d_obs,
            LAG(close, 19) OVER (PARTITION BY symbol ORDER BY trade_date) AS close_20d_ago
        FROM cn_stock_daily_price
        WHERE trade_date >= v_price_lookback 
          AND trade_date <= p_end_date
    ),
    feature_base AS (
        SELECT
            m.trade_date,
            m.industry_id,
            n.board_name AS industry_name,
            p.symbol,
            p.name,
            p.close,
            p.amount,
            CASE WHEN p.turnover_20d_obs = 20 THEN p.turnover_20d_avg END AS turnover_20d_avg,
            CASE WHEN p.close_20d_ago IS NOT NULL AND p.close_20d_ago <> 0 
                 THEN p.close / p.close_20d_ago - 1 
            END AS rs_20d_raw
        FROM industry_map m
        JOIN price_enriched p ON p.symbol = m.symbol AND p.trade_date = m.trade_date
        LEFT JOIN industry_name n ON n.board_id = m.industry_id
    ),
    ranked AS (
        SELECT
            *,
            PERCENT_RANK() OVER (PARTITION BY trade_date, industry_id ORDER BY turnover_20d_avg) AS turnover_20d_percentile,
            PERCENT_RANK() OVER (PARTITION BY trade_date, industry_id ORDER BY rs_20d_raw) AS rs_percentile,
            RANK() OVER (PARTITION BY trade_date, industry_id ORDER BY turnover_20d_avg DESC, symbol) AS turnover_rank_in_industry,
            COUNT(*) OVER (PARTITION BY trade_date, industry_id) AS industry_members
        FROM feature_base
    )
    SELECT
        trade_date, industry_id, industry_name, symbol, name,
        close, amount,
        turnover_20d_avg,
        turnover_20d_percentile,
        IF(turnover_20d_percentile >= 0.8, 1, 0) AS leader_liquidity,
        rs_20d_raw,
        rs_percentile,
        IF(rs_percentile >= 0.7, 1, 0) AS leader_trend,
        turnover_rank_in_industry,
        industry_members
    FROM ranked;

    SET v_row_count = ROW_COUNT();
    SELECT CONCAT('[sp_materialize_leader_score] Step1 #tmp_v1: ', v_row_count, ' 行, 耗时 ',
                  TIMEDIFF(NOW(6), v_step1_start)) AS progress;

    -- ======================================================================
    -- Step 2: #tmp_v2 — 在 #tmp_v1 基础上计算 v2 层逻辑
    -- ======================================================================
    SET v_step2_start = NOW(6);

    DROP TEMPORARY TABLE IF EXISTS `#tmp_v2`;

    CREATE TEMPORARY TABLE `#tmp_v2` (
        `trade_date`                DATE            NOT NULL,
        `industry_id`               VARCHAR(32)     NOT NULL,
        `industry_name`             VARCHAR(64)     DEFAULT NULL,
        `symbol`                    VARCHAR(16)     NOT NULL,
        `name`                      VARCHAR(64)     DEFAULT NULL,
        `close`                     DECIMAL(18,4)   DEFAULT NULL,
        `amount`                    DECIMAL(24,4)   DEFAULT NULL,
        `market_cap`                DECIMAL(24,4)   DEFAULT NULL,
        `total_mv`                  DECIMAL(24,4)   DEFAULT NULL,
        `circ_mv`                   DECIMAL(24,4)   DEFAULT NULL,
        `market_cap_rank`           INT             DEFAULT NULL,
        `market_cap_percentile`     DOUBLE          DEFAULT NULL,
        `leader_structural`         INT             DEFAULT NULL,
        `leader_structural_ready`   INT             DEFAULT 0,
        `turnover_20d_avg`          DECIMAL(24,4)   DEFAULT NULL,
        `turnover_20d_percentile`   DOUBLE          DEFAULT NULL,
        `leader_liquidity`          INT             DEFAULT 0,
        `rs_20d_raw`                DECIMAL(18,6)   DEFAULT NULL,
        `rs_percentile`             DOUBLE          DEFAULT NULL,
        `leader_trend`              INT             DEFAULT 0,
        `turnover_rank_in_industry` INT             DEFAULT NULL,
        `industry_members`          INT             DEFAULT NULL,
        `leader_score`              INT             DEFAULT 0,
        `leader_bucket`             VARCHAR(32)     DEFAULT 'NON_LEADER',
        PRIMARY KEY (`trade_date`, `industry_id`, `symbol`),
        INDEX `idx_tmp_v2_date` (`trade_date`)
    ) ENGINE=InnoDB ROW_FORMAT=DYNAMIC;

    INSERT INTO `#tmp_v2` (
        trade_date, industry_id, industry_name, symbol, name,
        close, amount,
        market_cap, total_mv, circ_mv,
        market_cap_rank, market_cap_percentile,
        leader_structural, leader_structural_ready,
        turnover_20d_avg, turnover_20d_percentile,
        leader_liquidity, rs_20d_raw, rs_percentile,
        leader_trend,
        turnover_rank_in_industry, industry_members,
        leader_score, leader_bucket
    )
    WITH base AS (
        SELECT
            v1.trade_date,
            v1.industry_id,
            v1.industry_name,
            v1.symbol,
            v1.name,
            v1.close,
            v1.amount,
            v1.turnover_20d_avg,
            v1.turnover_20d_percentile,
            v1.leader_liquidity,
            v1.rs_20d_raw,
            v1.rs_percentile,
            v1.leader_trend,
            v1.turnover_rank_in_industry,
            v1.industry_members,
            COALESCE(b.total_mv, b.circ_mv) AS market_cap,
            b.total_mv,
            b.circ_mv
        FROM `#tmp_v1` v1
        LEFT JOIN cn_stock_daily_basic b
          ON b.symbol = v1.symbol AND b.trade_date = v1.trade_date
    ),
    ranked AS (
        SELECT
            base.*,
            RANK() OVER (PARTITION BY trade_date, industry_id
                         ORDER BY market_cap DESC, symbol) AS market_cap_rank,
            PERCENT_RANK() OVER (PARTITION BY trade_date, industry_id
                                 ORDER BY market_cap) AS market_cap_percentile
        FROM base
    )
    SELECT
        trade_date, industry_id, industry_name, symbol, name,
        close, amount,
        market_cap, total_mv, circ_mv,
        market_cap_rank, market_cap_percentile,
        IF(market_cap IS NULL, 0,
           IF(market_cap_rank <= 3 OR market_cap_percentile >= 0.9, 1, 0)) AS leader_structural,
        IF(market_cap IS NULL, 0, 1) AS leader_structural_ready,
        turnover_20d_avg, turnover_20d_percentile,
        leader_liquidity, rs_20d_raw, rs_percentile,
        leader_trend,
        turnover_rank_in_industry, industry_members,
        IF(market_cap IS NULL, 0, IF(market_cap_rank <= 3 OR market_cap_percentile >= 0.9, 1, 0))
        + leader_liquidity + leader_trend AS leader_score,
        CASE
            WHEN (IF(market_cap IS NULL, 0, IF(market_cap_rank <= 3 OR market_cap_percentile >= 0.9, 1, 0))
                 + leader_liquidity + leader_trend) = 3 THEN 'CORE_LEADER'
            WHEN (IF(market_cap IS NULL, 0, IF(market_cap_rank <= 3 OR market_cap_percentile >= 0.9, 1, 0))
                 + leader_liquidity + leader_trend) = 2 THEN 'NEAR_LEADER'
            WHEN (IF(market_cap IS NULL, 0, IF(market_cap_rank <= 3 OR market_cap_percentile >= 0.9, 1, 0))
                 + leader_liquidity + leader_trend) = 1 THEN 'EDGE_LEADER'
            ELSE 'NON_LEADER'
        END AS leader_bucket
    FROM ranked;

    SET v_row_count = ROW_COUNT();
    SELECT CONCAT('[sp_materialize_leader_score] Step2 #tmp_v2: ', v_row_count, ' 行, 耗时 ',
                  TIMEDIFF(NOW(6), v_step2_start)) AS progress;

    -- ======================================================================
    -- Step 3: 写入目标表（使用 AS t2 别名语法替代已废弃的 VALUES()）
    -- ======================================================================
    SET v_step3_start = NOW(6);

    INSERT INTO cn_stock_leader_score_daily (
        trade_date, industry_id, industry_name, symbol, name,
        close, amount, market_cap, total_mv, circ_mv,
        market_cap_rank, market_cap_percentile,
        leader_structural, leader_structural_ready,
        turnover_20d_avg, turnover_20d_percentile,
        leader_liquidity, rs_20d_raw, rs_percentile,
        leader_trend, breakout_strength, breakout_ready,
        leader_score, leader_bucket,
        turnover_rank_in_industry, industry_members,
        source, created_at, updated_at
    )
    SELECT
        trade_date, industry_id, industry_name, symbol, name,
        close, amount, market_cap, total_mv, circ_mv,
        market_cap_rank,
        CAST(market_cap_percentile AS DECIMAL(12,8)) AS market_cap_percentile,
        leader_structural, leader_structural_ready,
        turnover_20d_avg,
        CAST(turnover_20d_percentile AS DECIMAL(12,8)) AS turnover_20d_percentile,
        leader_liquidity,
        CAST(rs_20d_raw AS DECIMAL(16,8)) AS rs_20d_raw,
        CAST(rs_percentile AS DECIMAL(12,8)) AS rs_percentile,
        leader_trend,
        NULL AS breakout_strength,
        0 AS breakout_ready,
        leader_score, leader_bucket,
        turnover_rank_in_industry, industry_members,
        'sp_materialize', NOW(), NOW()
    FROM `#tmp_v2` AS t2
    ON DUPLICATE KEY UPDATE
        industry_id            = t2.industry_id,
        industry_name          = t2.industry_name,
        name                   = t2.name,
        close                  = t2.close,
        amount                 = t2.amount,
        market_cap             = t2.market_cap,
        total_mv               = t2.total_mv,
        circ_mv                = t2.circ_mv,
        market_cap_rank        = t2.market_cap_rank,
        market_cap_percentile  = t2.market_cap_percentile,
        leader_structural      = t2.leader_structural,
        leader_structural_ready = t2.leader_structural_ready,
        turnover_20d_avg       = t2.turnover_20d_avg,
        turnover_20d_percentile = t2.turnover_20d_percentile,
        leader_liquidity       = t2.leader_liquidity,
        rs_20d_raw             = t2.rs_20d_raw,
        rs_percentile          = t2.rs_percentile,
        leader_trend           = t2.leader_trend,
        leader_score           = t2.leader_score,
        leader_bucket          = t2.leader_bucket,
        turnover_rank_in_industry = t2.turnover_rank_in_industry,
        industry_members       = t2.industry_members,
        updated_at             = NOW();

    SET v_row_count = ROW_COUNT();
    SET v_duration = TIMESTAMPDIFF(MICROSECOND, v_start_time, NOW(6)) / 1000000.0;

    -- 清理
    DROP TEMPORARY TABLE IF EXISTS `#tmp_v1`;
    DROP TEMPORARY TABLE IF EXISTS `#tmp_v2`;

    -- 输出结果
    SELECT v_row_count AS inserted_rows, v_duration AS duration_sec;
    
    SELECT CONCAT('[sp_materialize_leader_score] 完成: ', p_start_date, ' ~ ', p_end_date,
                  ' | 写入 ', v_row_count, ' 行 | 总耗时 ', ROUND(v_duration, 2), 's') AS summary;

END sp_main $$

DELIMITER ;
