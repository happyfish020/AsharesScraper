from __future__ import annotations

import argparse
import builtins
from datetime import date, datetime

from sqlalchemy import text

from app.settings import build_engine, load_sql_for_current_db


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return builtins.print(*args, **kwargs)


def _parse_ymd(s: str) -> date:
    v = (s or "").strip()
    if len(v) == 10 and v[4] == "-" and v[7] == "-":
        return datetime.strptime(v, "%Y-%m-%d").date()
    return datetime.strptime(v, "%Y%m%d").date()


def ensure_table(engine) -> None:
    ddl = load_sql_for_current_db("docs/DDL/cn_market.cn_stock_leader_sw_l1_latest_snap.sql")
    with engine.begin() as conn:
        conn.execute(text(ddl))


def ensure_dependencies(engine) -> None:
    ddl_paths = [
        "docs/DDL/cn_market.cn_stock_universe_status_t.sql",
        "docs/DDL/cn_market.cn_stock_active_universe_v.sql",
        "docs/DDL/cn_market.cn_stock_non_active_universe_v.sql",
        "docs/DDL/cn_market.cn_stock_daily_price_active_v.sql",
    ]
    with engine.begin() as conn:
        for ddl_path in ddl_paths:
            sql = load_sql_for_current_db(ddl_path)
            conn.execute(text(sql))


def resolve_effective_trade_date(engine) -> date:
    sql = text(
        """
        SELECT LEAST(
            COALESCE((SELECT MAX(trade_date) FROM cn_stock_daily_basic), DATE('1900-01-01')),
            COALESCE((SELECT MAX(trade_date) FROM cn_board_member_map_d WHERE sector_type = 'INDUSTRY'), DATE('1900-01-01')),
            COALESCE((SELECT MAX(trade_date) FROM cn_sw_industry_daily), DATE('1900-01-01'))
        ) AS effective_trade_date
        """
    )
    with engine.connect() as conn:
        dt = conn.execute(sql).scalar()
    if dt is None:
        raise RuntimeError("Unable to resolve effective trade_date for latest snap.")
    return dt


def build_snapshot(engine, trade_date: date) -> int:
    delete_sql = text("DELETE FROM cn_stock_leader_sw_l1_latest_snap WHERE trade_date = :trade_date")
    insert_sql = text(
        """
        INSERT INTO cn_stock_leader_sw_l1_latest_snap (
            trade_date,
            symbol,
            stock_name,
            stock_close,
            bk_industry_id,
            bk_industry_name,
            market_cap,
            total_mv,
            circ_mv,
            market_cap_rank,
            market_cap_percentile,
            leader_structural,
            leader_structural_ready,
            turnover_20d_avg,
            turnover_20d_percentile,
            leader_liquidity,
            rs_20d_raw,
            rs_percentile,
            leader_trend,
            breakout_strength,
            breakout_ready,
            leader_score,
            leader_bucket,
            turnover_rank_in_industry,
            industry_members,
            sw_l1_id,
            sw_l1_name,
            sw_close,
            sw_pct_change,
            sw_pe,
            sw_pb,
            sw_float_mv
        )
        WITH target_date AS (
            SELECT :trade_date AS trade_date
        ),
        price_window AS (
            SELECT
                p.symbol,
                p.trade_date,
                p.name,
                p.close,
                p.amount,
                ROW_NUMBER() OVER (
                    PARTITION BY p.symbol
                    ORDER BY p.trade_date DESC
                ) AS rn_desc
            FROM cn_stock_daily_price_active_v p
            JOIN target_date t
              ON p.trade_date <= t.trade_date
             AND p.trade_date >= DATE_SUB(t.trade_date, INTERVAL 60 DAY)
        ),
        latest_20d AS (
            SELECT
                w.symbol,
                MAX(CASE WHEN w.rn_desc = 1 THEN w.trade_date END) AS trade_date,
                MAX(CASE WHEN w.rn_desc = 1 THEN w.name END) AS stock_name,
                MAX(CASE WHEN w.rn_desc = 1 THEN w.close END) AS stock_close,
                MAX(CASE WHEN w.rn_desc = 20 THEN w.close END) AS close_20d_ago,
                AVG(CASE WHEN w.rn_desc <= 20 THEN w.amount END) AS turnover_20d_avg,
                SUM(CASE WHEN w.rn_desc <= 20 AND w.amount IS NOT NULL THEN 1 ELSE 0 END) AS turnover_20d_obs
            FROM price_window w
            GROUP BY w.symbol
        ),
        sw_map_ranked AS (
            SELECT
                t.trade_date,
                h.symbol,
                h.board_id AS sw_l1_id,
                ROW_NUMBER() OVER (
                    PARTITION BY t.trade_date, h.symbol
                    ORDER BY h.valid_from DESC, COALESCE(h.valid_to, DATE('9999-12-31')) DESC, h.board_id
                ) AS rn
            FROM target_date t
            JOIN cn_board_industry_member_hist h
              ON h.source = 'tushare_sw_l1'
             AND t.trade_date >= h.valid_from
             AND t.trade_date <= COALESCE(h.valid_to, DATE('9999-12-31'))
        ),
        sw_map AS (
            SELECT trade_date, symbol, sw_l1_id
            FROM sw_map_ranked
            WHERE rn = 1
        ),
        sw_l1_name AS (
            SELECT board_id, board_name
            FROM (
                SELECT
                    m.BOARD_ID AS board_id,
                    m.BOARD_NAME AS board_name,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.BOARD_ID
                        ORDER BY m.ASOF_DATE DESC
                    ) AS rn
                FROM cn_board_industry_master m
                WHERE m.SOURCE = 'TUSHARE_SW2021_L1'
            ) x
            WHERE x.rn = 1
        ),
        feature_base AS (
            SELECT
                m.trade_date,
                m.symbol,
                p.stock_name,
                p.stock_close,
                m.sw_l1_id,
                n.board_name AS sw_l1_name,
                CASE
                    WHEN p.turnover_20d_obs = 20 THEN p.turnover_20d_avg
                    ELSE NULL
                END AS turnover_20d_avg,
                CASE
                    WHEN p.close_20d_ago IS NOT NULL AND p.close_20d_ago <> 0
                    THEN p.stock_close / p.close_20d_ago - 1
                    ELSE NULL
                END AS rs_20d_raw
            FROM sw_map m
            JOIN latest_20d p
              ON p.symbol COLLATE utf8mb4_unicode_ci = m.symbol COLLATE utf8mb4_unicode_ci
             AND p.trade_date = m.trade_date
            LEFT JOIN sw_l1_name n
              ON n.board_id COLLATE utf8mb4_general_ci = m.sw_l1_id COLLATE utf8mb4_general_ci
        ),
        ranked_base AS (
            SELECT
                b.trade_date,
                b.symbol,
                b.stock_name,
                b.stock_close,
                b.sw_l1_id,
                b.sw_l1_name,
                b.turnover_20d_avg,
                b.rs_20d_raw,
                CASE
                    WHEN b.turnover_20d_avg IS NOT NULL
                    THEN PERCENT_RANK() OVER (
                        PARTITION BY b.trade_date, b.sw_l1_id
                        ORDER BY b.turnover_20d_avg
                    )
                    ELSE NULL
                END AS turnover_20d_percentile,
                CASE
                    WHEN b.rs_20d_raw IS NOT NULL
                    THEN PERCENT_RANK() OVER (
                        PARTITION BY b.trade_date, b.sw_l1_id
                        ORDER BY b.rs_20d_raw
                    )
                    ELSE NULL
                END AS rs_percentile,
                RANK() OVER (
                    PARTITION BY b.trade_date, b.sw_l1_id
                    ORDER BY b.turnover_20d_avg DESC, b.symbol
                ) AS turnover_rank_in_industry,
                COUNT(*) OVER (
                    PARTITION BY b.trade_date, b.sw_l1_id
                ) AS industry_members
            FROM feature_base b
        ),
        leader_base AS (
            SELECT
                r.trade_date,
                r.symbol,
                r.stock_name,
                r.stock_close,
                CAST(NULL AS CHAR(32)) AS bk_industry_id,
                CAST(NULL AS CHAR(80)) AS bk_industry_name,
                r.sw_l1_id,
                r.sw_l1_name,
                db.total_mv,
                db.circ_mv,
                COALESCE(db.total_mv, db.circ_mv) AS market_cap,
                CASE
                    WHEN COALESCE(db.total_mv, db.circ_mv) IS NOT NULL
                    THEN RANK() OVER (
                        PARTITION BY r.trade_date, r.sw_l1_id
                        ORDER BY COALESCE(db.total_mv, db.circ_mv) DESC, r.symbol
                    )
                    ELSE NULL
                END AS market_cap_rank,
                CASE
                    WHEN COALESCE(db.total_mv, db.circ_mv) IS NOT NULL
                    THEN PERCENT_RANK() OVER (
                        PARTITION BY r.trade_date, r.sw_l1_id
                        ORDER BY COALESCE(db.total_mv, db.circ_mv)
                    )
                    ELSE NULL
                END AS market_cap_percentile,
                CASE
                    WHEN COALESCE(db.total_mv, db.circ_mv) IS NULL THEN NULL
                    WHEN (
                        RANK() OVER (
                            PARTITION BY r.trade_date, r.sw_l1_id
                            ORDER BY COALESCE(db.total_mv, db.circ_mv) DESC, r.symbol
                        ) <= 3
                    )
                    OR (
                        PERCENT_RANK() OVER (
                            PARTITION BY r.trade_date, r.sw_l1_id
                            ORDER BY COALESCE(db.total_mv, db.circ_mv)
                        ) >= 0.9
                    )
                    THEN 1
                    ELSE 0
                END AS leader_structural,
                CASE
                    WHEN COALESCE(db.total_mv, db.circ_mv) IS NULL THEN 0
                    ELSE 1
                END AS leader_structural_ready,
                r.turnover_20d_avg,
                r.turnover_20d_percentile,
                CASE
                    WHEN r.turnover_20d_percentile >= 0.8 THEN 1
                    ELSE 0
                END AS leader_liquidity,
                r.rs_20d_raw,
                r.rs_percentile,
                CASE
                    WHEN r.rs_percentile >= 0.7 THEN 1
                    ELSE 0
                END AS leader_trend,
                CAST(NULL AS CHAR(16)) AS breakout_strength,
                0 AS breakout_ready,
                (
                    CASE
                        WHEN COALESCE(db.total_mv, db.circ_mv) IS NULL THEN 0
                        WHEN (
                            RANK() OVER (
                                PARTITION BY r.trade_date, r.sw_l1_id
                                ORDER BY COALESCE(db.total_mv, db.circ_mv) DESC, r.symbol
                            ) <= 3
                        )
                        OR (
                            PERCENT_RANK() OVER (
                                PARTITION BY r.trade_date, r.sw_l1_id
                                ORDER BY COALESCE(db.total_mv, db.circ_mv)
                            ) >= 0.9
                        )
                        THEN 1
                        ELSE 0
                    END
                    +
                    CASE
                        WHEN r.turnover_20d_percentile >= 0.8 THEN 1
                        ELSE 0
                    END
                    +
                    CASE
                        WHEN r.rs_percentile >= 0.7 THEN 1
                        ELSE 0
                    END
                ) AS leader_score,
                CASE
                    WHEN (
                        (
                            CASE
                                WHEN COALESCE(db.total_mv, db.circ_mv) IS NULL THEN 0
                                WHEN (
                                    RANK() OVER (
                                        PARTITION BY r.trade_date, r.sw_l1_id
                                        ORDER BY COALESCE(db.total_mv, db.circ_mv) DESC, r.symbol
                                    ) <= 3
                                )
                                OR (
                                    PERCENT_RANK() OVER (
                                        PARTITION BY r.trade_date, r.sw_l1_id
                                        ORDER BY COALESCE(db.total_mv, db.circ_mv)
                                    ) >= 0.9
                                )
                                THEN 1
                                ELSE 0
                            END
                        )
                        +
                        CASE
                            WHEN r.turnover_20d_percentile >= 0.8 THEN 1
                            ELSE 0
                        END
                        +
                        CASE
                            WHEN r.rs_percentile >= 0.7 THEN 1
                            ELSE 0
                        END
                    ) = 3 THEN 'CORE_LEADER'
                    WHEN (
                        (
                            CASE
                                WHEN COALESCE(db.total_mv, db.circ_mv) IS NULL THEN 0
                                WHEN (
                                    RANK() OVER (
                                        PARTITION BY r.trade_date, r.sw_l1_id
                                        ORDER BY COALESCE(db.total_mv, db.circ_mv) DESC, r.symbol
                                    ) <= 3
                                )
                                OR (
                                    PERCENT_RANK() OVER (
                                        PARTITION BY r.trade_date, r.sw_l1_id
                                        ORDER BY COALESCE(db.total_mv, db.circ_mv)
                                    ) >= 0.9
                                )
                                THEN 1
                                ELSE 0
                            END
                        )
                        +
                        CASE
                            WHEN r.turnover_20d_percentile >= 0.8 THEN 1
                            ELSE 0
                        END
                        +
                        CASE
                            WHEN r.rs_percentile >= 0.7 THEN 1
                            ELSE 0
                        END
                    ) = 2 THEN 'NEAR_LEADER'
                    WHEN (
                        (
                            CASE
                                WHEN COALESCE(db.total_mv, db.circ_mv) IS NULL THEN 0
                                WHEN (
                                    RANK() OVER (
                                        PARTITION BY r.trade_date, r.sw_l1_id
                                        ORDER BY COALESCE(db.total_mv, db.circ_mv) DESC, r.symbol
                                    ) <= 3
                                )
                                OR (
                                    PERCENT_RANK() OVER (
                                        PARTITION BY r.trade_date, r.sw_l1_id
                                        ORDER BY COALESCE(db.total_mv, db.circ_mv)
                                    ) >= 0.9
                                )
                                THEN 1
                                ELSE 0
                            END
                        )
                        +
                        CASE
                            WHEN r.turnover_20d_percentile >= 0.8 THEN 1
                            ELSE 0
                        END
                        +
                        CASE
                            WHEN r.rs_percentile >= 0.7 THEN 1
                            ELSE 0
                        END
                    ) = 1 THEN 'EDGE_LEADER'
                    ELSE 'NON_LEADER'
                END AS leader_bucket,
                r.turnover_rank_in_industry,
                r.industry_members
            FROM ranked_base r
            LEFT JOIN cn_stock_daily_basic db
              ON db.symbol COLLATE utf8mb4_unicode_ci = r.symbol COLLATE utf8mb4_unicode_ci
             AND db.trade_date = r.trade_date
        )
        SELECT
            l.trade_date,
            l.symbol,
            l.stock_name,
            l.stock_close,
            l.bk_industry_id,
            l.bk_industry_name,
            l.market_cap,
            l.total_mv,
            l.circ_mv,
            l.market_cap_rank,
            l.market_cap_percentile,
            l.leader_structural,
            l.leader_structural_ready,
            l.turnover_20d_avg,
            l.turnover_20d_percentile,
            l.leader_liquidity,
            l.rs_20d_raw,
            l.rs_percentile,
            l.leader_trend,
            l.breakout_strength,
            l.breakout_ready,
            l.leader_score,
            l.leader_bucket,
            l.turnover_rank_in_industry,
            l.industry_members,
            l.sw_l1_id,
            l.sw_l1_name,
            s.close AS sw_close,
            s.pct_change AS sw_pct_change,
            s.pe AS sw_pe,
            s.pb AS sw_pb,
            s.float_mv AS sw_float_mv
        FROM leader_base l
        LEFT JOIN cn_sw_industry_daily s
          ON s.ts_code COLLATE utf8mb4_general_ci = l.sw_l1_id COLLATE utf8mb4_general_ci
         AND s.trade_date = l.trade_date
        """
    )
    with engine.begin() as conn:
        conn.execute(delete_sql, {"trade_date": trade_date})
        ret = conn.execute(insert_sql, {"trade_date": trade_date})
        return int(ret.rowcount or 0)


def main() -> int:
    p = argparse.ArgumentParser(description="Build materialized latest snapshot for leader_score + SW L1 + sw_daily.")
    p.add_argument("--trade-date", default="", help="YYYY-MM-DD or YYYYMMDD; default auto-resolve latest safe date")
    args = p.parse_args()

    engine = build_engine()
    ensure_dependencies(engine)
    ensure_table(engine)

    trade_date = _parse_ymd(args.trade_date) if (args.trade_date or "").strip() else resolve_effective_trade_date(engine)
    print(f"building latest snap for trade_date={trade_date}")
    affected = build_snapshot(engine, trade_date)

    with engine.connect() as conn:
        stats = conn.execute(
            text(
                """
                SELECT COUNT(*) AS row_cnt,
                       COUNT(DISTINCT symbol) AS symbol_cnt,
                       COUNT(DISTINCT sw_l1_id) AS sw_cnt,
                       SUM(CASE WHEN leader_score >= 2 THEN 1 ELSE 0 END) AS leader_ge_2_cnt
                FROM cn_stock_leader_sw_l1_latest_snap
                WHERE trade_date = :trade_date
                """
            ),
            {"trade_date": trade_date},
        ).one()
        print(
            f"done affected={affected} row_cnt={stats[0]} "
            f"symbol_cnt={stats[1]} sw_cnt={stats[2]} leader_ge_2_cnt={stats[3]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
