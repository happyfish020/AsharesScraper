from __future__ import annotations

import argparse
import calendar
import time
from datetime import date, datetime

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.settings import build_engine


def _parse_ymd(s: str) -> date:
    s = (s or "").strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return datetime.strptime(s, "%Y-%m-%d").date()
    return datetime.strptime(s, "%Y%m%d").date()


def _month_chunks(start: date, end: date, months_per_chunk: int = 1):
    if months_per_chunk < 1:
        raise ValueError("months_per_chunk must be >= 1")
    cur = date(start.year, start.month, 1)
    while cur <= end:
        y = cur.year
        m = cur.month
        for _ in range(months_per_chunk - 1):
            if m == 12:
                y += 1
                m = 1
            else:
                m += 1
        last_day = calendar.monthrange(y, m)[1]
        m_end = date(y, m, last_day)
        if cur < start:
            c_start = start
        else:
            c_start = cur
        if m_end > end:
            c_end = end
        else:
            c_end = m_end
        yield c_start, c_end
        y2 = y
        m2 = m
        if m2 == 12:
            cur = date(y2 + 1, 1, 1)
        else:
            cur = date(y2, m2 + 1, 1)


def _is_retryable_mysql_error(exc: Exception) -> bool:
    if not isinstance(exc, DBAPIError):
        return False
    o = getattr(exc, "orig", None)
    if not o:
        return False
    # PyMySQL usually keeps mysql error code at orig.args[0]
    code = None
    if hasattr(o, "args") and o.args:
        try:
            code = int(o.args[0])
        except Exception:
            code = None
    # 1205 lock wait timeout, 1213 deadlock
    return code in (1205, 1213)


def main():
    p = argparse.ArgumentParser(description="Rebuild eod/ranked/signal from history mapping table")
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    p.add_argument("--top-pct", type=float, default=0.30)
    p.add_argument("--breadth-min", type=float, default=0.60)
    p.add_argument("--months-per-chunk", type=int, default=1, help="months per transaction chunk")
    p.add_argument("--clear-first", type=int, default=1, help="1=delete target date range first; 0=keep existing rows and upsert")
    p.add_argument("--resume-after-max", type=int, default=0, help="1=auto set start to max(common_built_date)+1")
    p.add_argument("--retries", type=int, default=5, help="retry count for deadlock/lock wait timeout")
    p.add_argument("--retry-sleep-sec", type=float, default=2.0, help="base sleep seconds between retries")
    p.add_argument(
        "--rank-signal-mode",
        default="hybrid_base",
        choices=["sp_daily", "bulk_sql", "hybrid_temp", "hybrid_base"],
        help=(
            "sp_daily uses per-trade-date SP calls; "
            "bulk_sql writes from transition_v directly; "
            "hybrid_temp stages transition rows to temp table then writes signal; "
            "hybrid_base builds ranked/transition from base table without ranked_v/transition_v"
        ),
    )
    args = p.parse_args()

    start = _parse_ymd(args.start)
    end = _parse_ymd(args.end)
    engine = build_engine()
    if args.resume_after_max == 1:
        with engine.connect() as conn:
            mx = conn.execute(
                text(
                    """
                    SELECT LEAST(
                        COALESCE((SELECT MAX(trade_date) FROM cn_sector_eod_hist_t), DATE('1900-01-01')),
                        COALESCE((SELECT MAX(trade_date) FROM cn_sector_rotation_ranked_t), DATE('1900-01-01')),
                        COALESCE((SELECT MAX(signal_date) FROM cn_sector_rotation_signal_t), DATE('1900-01-01'))
                    )
                    """
                )
            ).scalar()
        if mx and mx > date(1900, 1, 1):
            start = date.fromordinal(mx.toordinal() + 1)
            print(f"resume-after-max enabled; start reset to {start} (max built date={mx})")

    if start > end:
        raise SystemExit(f"invalid range: {start}>{end}")

    ranked_sql = text(
        """
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
        FROM cn_sector_rotation_transition_v t
        WHERE t.trade_date BETWEEN :d1 AND :d2
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
            created_at = NOW()
        """
    )
    ranked_from_ranked_view_sql = text(
        """
        INSERT INTO cn_sector_rotation_ranked_t (
            trade_date, sector_type, sector_id, sector_name,
            state, tier, theme_group, theme_rank, score, confirm_streak,
            amt_impulse, up_ma5, up_ratio, created_at
        )
        SELECT
            r.trade_date,
            r.sector_type,
            r.sector_id,
            r.sector_name,
            r.state,
            r.tier,
            r.theme_group,
            r.theme_rank,
            r.score,
            r.confirm_streak,
            r.amt_impulse,
            r.up_ma5,
            r.up_ratio,
            NOW()
        FROM cn_sector_rotation_ranked_v r
        WHERE r.trade_date BETWEEN :d1 AND :d2
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
            created_at = NOW()
        """
    )
    create_tmp_ranked_base_sql = text(
        """
        CREATE TEMPORARY TABLE tmp_ranked_month ENGINE=InnoDB AS
        WITH target AS (
            SELECT DISTINCT e.sector_type, e.sector_id
            FROM cn_sector_eod_hist_t e
            WHERE e.trade_date BETWEEN :d1 AND :d2
        ),
        industry_name AS (
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
                e.members,
                e.amount_sum,
                e.score,
                e.up_ratio,
                e.cond_count,
                e.sector_pass
            FROM cn_sector_eod_hist_t e
            JOIN target t
              ON t.sector_type = e.sector_type
             AND t.sector_id = e.sector_id
            LEFT JOIN industry_name im
              ON e.sector_type = 'INDUSTRY'
             AND e.sector_id = im.board_id
            LEFT JOIN concept_name cm
              ON e.sector_type = 'CONCEPT'
             AND e.sector_id = cm.concept_id
            WHERE e.trade_date BETWEEN DATE_SUB(:d1, INTERVAL 40 DAY) AND :d2
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
                    WHEN h4.sector_name LIKE '%流感%' OR h4.sector_name LIKE '%肝炎%' OR h4.sector_name LIKE '%病毒%' OR h4.sector_name LIKE '%防治%' OR h4.sector_name LIKE '%疫苗%' THEN 'MEDICAL_EVENT'
                    WHEN h4.sector_name LIKE '%中药%' OR h4.sector_name LIKE '%中医%' THEN 'TCM'
                    WHEN h4.sector_name LIKE '%汽车%' OR h4.sector_name LIKE '%零部件%' OR h4.sector_name LIKE '%智能驾驶%' THEN 'AUTO_CHAIN'
                    WHEN h4.sector_name LIKE '%有机硅%' OR h4.sector_name LIKE '%碳纤维%' OR h4.sector_name LIKE '%小金属%' OR h4.sector_name LIKE '%稀土%' OR h4.sector_name LIKE '%新材料%' THEN 'MATERIALS'
                    WHEN h4.sector_name LIKE '%核聚变%' THEN 'NUCLEAR_FUSION'
                    WHEN h4.sector_name LIKE '%合成生物%' THEN 'SYNBIO'
                    ELSE 'OTHER'
                END AS theme_group
            FROM hist4 h4
        )
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
        WHERE h5.trade_date BETWEEN :d1 AND :d2
        """
    )
    add_tmp_ranked_idx_sql = text(
        """
        ALTER TABLE tmp_ranked_month
            ADD KEY idx_rm_1 (trade_date, sector_type, sector_id),
            ADD KEY idx_rm_2 (trade_date, theme_group, theme_rank)
        """
    )
    ranked_from_tmp_sql = text(
        """
        INSERT INTO cn_sector_rotation_ranked_t (
            trade_date, sector_type, sector_id, sector_name,
            state, tier, theme_group, theme_rank, score, confirm_streak,
            amt_impulse, up_ma5, up_ratio, created_at
        )
        SELECT
            r.trade_date,
            r.sector_type,
            r.sector_id,
            r.sector_name,
            r.state,
            r.tier,
            r.theme_group,
            r.theme_rank,
            r.score,
            r.confirm_streak,
            r.amt_impulse,
            r.up_ma5,
            r.up_ratio,
            NOW()
        FROM tmp_ranked_month r
        WHERE r.trade_date BETWEEN :d1 AND :d2
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
            created_at = NOW()
        """
    )

    signal_sql = text(
        """
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
            FROM cn_sector_rotation_transition_v t
            WHERE t.trade_date BETWEEN :d1 AND :d2
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
            created_at = NOW()
        """
    )
    create_tmp_transition_sql = text(
        """
        CREATE TEMPORARY TABLE tmp_transition_month ENGINE=InnoDB AS
        WITH target AS (
            SELECT DISTINCT r.sector_type, r.sector_id
            FROM cn_sector_rotation_ranked_t r
            WHERE r.trade_date BETWEEN :d1 AND :d2
        ),
        prev_point AS (
            SELECT
                r.sector_type,
                r.sector_id,
                MAX(r.trade_date) AS prev_trade_date
            FROM cn_sector_rotation_ranked_t r
            JOIN target t
              ON t.sector_type = r.sector_type
             AND t.sector_id = r.sector_id
            WHERE r.trade_date < :d1
            GROUP BY r.sector_type, r.sector_id
        ),
        hist AS (
            SELECT
                r.trade_date,
                r.sector_type,
                r.sector_id,
                r.sector_name,
                r.theme_group,
                r.theme_rank,
                r.tier,
                r.state,
                r.score,
                r.confirm_streak,
                r.amt_impulse,
                r.up_ratio,
                r.up_ma5
            FROM cn_sector_rotation_ranked_t r
            JOIN target t
              ON t.sector_type = r.sector_type
             AND t.sector_id = r.sector_id
            WHERE r.trade_date BETWEEN :d1 AND :d2
            UNION ALL
            SELECT
                r.trade_date,
                r.sector_type,
                r.sector_id,
                r.sector_name,
                r.theme_group,
                r.theme_rank,
                r.tier,
                r.state,
                r.score,
                r.confirm_streak,
                r.amt_impulse,
                r.up_ratio,
                r.up_ma5
            FROM cn_sector_rotation_ranked_t r
            JOIN prev_point p
              ON p.sector_type = r.sector_type
             AND p.sector_id = r.sector_id
             AND p.prev_trade_date = r.trade_date
        ),
        x AS (
            SELECT
                h.*,
                LAG(h.state) OVER (PARTITION BY h.sector_type, h.sector_id ORDER BY h.trade_date) AS prev_state,
                LAG(h.tier)  OVER (PARTITION BY h.sector_type, h.sector_id ORDER BY h.trade_date) AS prev_tier,
                LAG(h.score) OVER (PARTITION BY h.sector_type, h.sector_id ORDER BY h.trade_date) AS prev_score
            FROM hist h
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
                WHEN x.prev_state IN ('NEUTRAL','FADE') AND x.state = 'IGNITE' THEN 'START_IGNITE'
                WHEN x.prev_state = 'IGNITE' AND x.state = 'CONFIRM' THEN 'IGNITE_TO_CONFIRM'
                WHEN x.prev_state IN ('NEUTRAL','FADE') AND x.state = 'CONFIRM' THEN 'DIRECT_CONFIRM'
                WHEN x.prev_state = 'CONFIRM' AND x.state = 'HOLD' THEN 'CONFIRM_TO_HOLD'
                WHEN x.prev_state = 'HOLD' AND x.state = 'CONFIRM' THEN 'HOLD_TO_CONFIRM'
                WHEN x.prev_state IN ('CONFIRM','HOLD') AND x.state = 'FADE' THEN 'TREND_TO_FADE'
                WHEN x.prev_state IN ('CONFIRM','HOLD') AND x.state = 'NEUTRAL' THEN 'TREND_BREAK_TO_NEUTRAL'
                WHEN x.prev_state = 'IGNITE' AND x.state IN ('NEUTRAL','FADE') THEN 'IGNITE_FAIL'
                ELSE 'OTHER_CHANGE'
            END AS transition
        FROM x
        WHERE x.trade_date BETWEEN :d1 AND :d2
        """
    )
    add_tmp_transition_idx_sql = text(
        """
        ALTER TABLE tmp_transition_month
            ADD KEY idx_tm_1 (trade_date, sector_type, sector_id),
            ADD KEY idx_tm_2 (trade_date, score)
        """
    )
    signal_tmp_sql = text(
        """
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
            FROM tmp_transition_month t
            WHERE t.trade_date BETWEEN :d1 AND :d2
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
            created_at = NOW()
        """
    )

    print(f"rebuild range: {start}..{end}")
    if args.clear_first == 1:
        with engine.begin() as conn:
            # clear target range to avoid stale rows from old mapping logic
            conn.execute(text("DELETE FROM cn_sector_eod_hist_t WHERE trade_date BETWEEN :d1 AND :d2"), {"d1": start, "d2": end})
            conn.execute(text("DELETE FROM cn_sector_rotation_ranked_t WHERE trade_date BETWEEN :d1 AND :d2"), {"d1": start, "d2": end})
            conn.execute(text("DELETE FROM cn_sector_rotation_signal_t WHERE signal_date BETWEEN :d1 AND :d2"), {"d1": start, "d2": end})
            print("clear-first done")
    else:
        print("clear-first skipped (upsert mode)")

    for i, (d1, d2) in enumerate(_month_chunks(start, end, args.months_per_chunk), start=1):
        ok = False
        last_exc = None
        for attempt in range(1, args.retries + 2):
            try:
                with engine.begin() as conn:
                    conn.execute(
                        text("CALL sp_refresh_sector_eod_hist(:d1, :d2, :tp, :bm)"),
                        {"d1": d1, "d2": d2, "tp": args.top_pct, "bm": args.breadth_min},
                    )
                    if args.rank_signal_mode == "bulk_sql":
                        conn.execute(ranked_sql, {"d1": d1, "d2": d2})
                        conn.execute(signal_sql, {"d1": d1, "d2": d2})
                    elif args.rank_signal_mode == "hybrid_temp":
                        conn.execute(ranked_from_ranked_view_sql, {"d1": d1, "d2": d2})
                        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_transition_month"))
                        conn.execute(create_tmp_transition_sql, {"d1": d1, "d2": d2})
                        conn.execute(add_tmp_transition_idx_sql)
                        conn.execute(signal_tmp_sql, {"d1": d1, "d2": d2})
                        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_transition_month"))
                    elif args.rank_signal_mode == "hybrid_base":
                        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_ranked_month"))
                        conn.execute(create_tmp_ranked_base_sql, {"d1": d1, "d2": d2})
                        conn.execute(add_tmp_ranked_idx_sql)
                        conn.execute(ranked_from_tmp_sql, {"d1": d1, "d2": d2})
                        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_transition_month"))
                        conn.execute(create_tmp_transition_sql, {"d1": d1, "d2": d2})
                        conn.execute(add_tmp_transition_idx_sql)
                        conn.execute(signal_tmp_sql, {"d1": d1, "d2": d2})
                        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_transition_month"))
                        conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_ranked_month"))
                    else:
                        days = conn.execute(
                            text(
                                """
                                SELECT DISTINCT trade_date
                                FROM cn_stock_daily_price
                                WHERE trade_date BETWEEN :d1 AND :d2
                                ORDER BY trade_date
                                """
                            ),
                            {"d1": d1, "d2": d2},
                        ).fetchall()
                        for row in days:
                            td = row[0]
                            conn.execute(text("CALL SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE(:d)"), {"d": td})
                            conn.execute(text("CALL SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE(:d)"), {"d": td})
                ok = True
                break
            except Exception as e:
                last_exc = e
                if _is_retryable_mysql_error(e) and attempt <= args.retries:
                    sleep_s = args.retry_sleep_sec * attempt
                    print(
                        f"retry chunk={i} range={d1}..{d2} attempt={attempt}/{args.retries} "
                        f"due_to_lock sleep={sleep_s:.1f}s"
                    )
                    time.sleep(sleep_s)
                    continue
                break
        if not ok:
            raise SystemExit(f"chunk failed: {i} {d1}..{d2} err={last_exc}")

        with engine.connect() as conn:
            progress = conn.execute(
                text(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM cn_sector_eod_hist_t WHERE trade_date BETWEEN :d1 AND :d2) AS eod_cnt,
                        (SELECT COUNT(*) FROM cn_sector_rotation_ranked_t WHERE trade_date BETWEEN :d1 AND :d2) AS ranked_cnt,
                        (SELECT COUNT(*) FROM cn_sector_rotation_signal_t WHERE signal_date BETWEEN :d1 AND :d2) AS signal_cnt
                    """
                ),
                {"d1": d1, "d2": d2},
            ).one()
        print(
            f"processed chunk={i} range={d1}..{d2} mode={args.rank_signal_mode} "
            f"eod={progress[0]} ranked={progress[1]} signal={progress[2]}"
        )

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT 'eod_hist' AS t, COUNT(*) AS c
                FROM cn_sector_eod_hist_t
                WHERE trade_date BETWEEN :d1 AND :d2
                UNION ALL
                SELECT 'ranked', COUNT(*)
                FROM cn_sector_rotation_ranked_t
                WHERE trade_date BETWEEN :d1 AND :d2
                UNION ALL
                SELECT 'signal', COUNT(*)
                FROM cn_sector_rotation_signal_t
                WHERE signal_date BETWEEN :d1 AND :d2
                """
            ),
            {"d1": start, "d2": end},
        ).fetchall()
        print("final_counts=", rows)


if __name__ == "__main__":
    main()
