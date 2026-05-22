from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import Set

from sqlalchemy import text
from app.settings import get_db_name

_BIND_RE = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")

import datetime as _dt


def _ensure_date(x) -> _dt.date:
    # already a date (but not datetime)
    if isinstance(x, _dt.date) and not isinstance(x, _dt.datetime):
        return x
    # datetime -> date
    if isinstance(x, _dt.datetime):
        return x.date()
    # common string formats
    if isinstance(x, str):
        s = x.strip()
        # 'YYYYMMDD'
        if len(s) == 8 and s.isdigit():
            return _dt.date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
        # 'YYYY-MM-DD'
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return _dt.date(int(s[0:4]), int(s[5:7]), int(s[8:10]))
        raise ValueError(f"Unsupported date string format: {x!r}")
    raise TypeError(f"Unsupported trade_date type: {type(x)} value={x!r}")


def _parse_yyyymmdd(s: str) -> date:
    return datetime.strptime(s, "%Y%m%d").date()


def _needed_binds(plsql_block: str) -> Set[str]:
    return set(_BIND_RE.findall(plsql_block or ""))


def _resolve_trade_date_for_rotation(conn, requested_trade_date: _dt.date) -> _dt.date:
    """Clamp requested trade date to latest day that has both price and board-member mapping."""
    max_dt = conn.execute(
        text(
            """
            SELECT MAX(p.trade_date)
            FROM cn_stock_daily_price p
            WHERE p.trade_date <= :req_dt
              AND EXISTS (
                  SELECT 1
                  FROM cn_board_member_map_d m
                  WHERE m.trade_date = p.trade_date
                  LIMIT 1
              )
            """
        ),
        {"req_dt": requested_trade_date},
    ).scalar()
    if max_dt is None:
        max_dt = conn.execute(text("SELECT MAX(trade_date) FROM cn_stock_daily_price")).scalar()
    if max_dt is None:
        return requested_trade_date
    max_dt = _ensure_date(max_dt)
    if requested_trade_date > max_dt:
        return max_dt
    return requested_trade_date


def _map_proc_exists(conn) -> bool:
    """Check map builder SP existence for daily map incremental step."""
    n = conn.execute(
        text(
            """
            SELECT COUNT(*)
            FROM information_schema.routines
            WHERE routine_schema = DATABASE()
              AND routine_type = 'PROCEDURE'
              AND routine_name = 'sp_build_board_member_map'
            """
        )
    ).scalar()
    return int(n or 0) > 0

def _latest_price_trade_date_on_or_before(conn, requested_trade_date: _dt.date) -> _dt.date | None:
    """Return latest cn_stock_daily_price trade_date <= requested date."""
    max_dt = conn.execute(
        text(
            """
            SELECT MAX(trade_date)
            FROM cn_stock_daily_price
            WHERE trade_date <= :req_dt
            """
        ),
        {"req_dt": requested_trade_date},
    ).scalar()
    if max_dt is None:
        return None
    return _ensure_date(max_dt)


def _find_missing_board_member_map_dates(conn, end_date: _dt.date):
    """Trade dates with stock price but no board-member map row."""
    rows = conn.execute(
        text(
            """
            SELECT p.trade_date
            FROM (
                SELECT DISTINCT trade_date
                FROM cn_stock_daily_price
                WHERE trade_date <= :end_date
            ) p
            WHERE NOT EXISTS (
                SELECT 1
                FROM cn_board_member_map_d m
                WHERE m.trade_date = p.trade_date
                LIMIT 1
            )
            ORDER BY p.trade_date
            """
        ),
        {"end_date": end_date},
    ).fetchall()
    return [_ensure_date(r[0]) for r in rows]


def _is_missing_transition_view_error(exc: Exception) -> bool:
    # MySQL 1146: table/view does not exist.
    msg = str(exc).lower()
    return "1146" in msg and "cn_sector_rotation_transition_v" in msg


def _active_universe_refresh_proc_exists(conn) -> bool:
    n = conn.execute(
        text(
            """
            SELECT COUNT(*)
            FROM information_schema.routines
            WHERE routine_schema = DATABASE()
              AND routine_type = 'PROCEDURE'
              AND routine_name = 'sp_refresh_stock_universe_status'
            """
        )
    ).scalar()
    return int(n or 0) > 0


@dataclass
class SectorRotationSnapshotTask:
    """Generate sector rotation ENTER/HOLD/EXIT snapshots by calling a stored procedure (SP).

    This task is read-write by design (it generates snapshot tables) but it never patches SPs.
    It only executes a user-provided SQL block that calls the already-existing SP.
    """

    name: str = "SectorRotationSnapshotTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        self.engine = getattr(ctx, "engine", None)
        if self.engine is None:
            raise RuntimeError("[rotation_snapshot] task='rotation' requires MySQL engine.")
        if str(getattr(self.engine.dialect, "name", "")).lower() != "mysql":
            raise RuntimeError("[rotation_snapshot] task='rotation' must run on MySQL.")

        cfg.finalize_dates()
        start = str(cfg.start_date)
        end = str(cfg.end_date)

        sp_sql = os.getenv(
            "ROTATION_SNAPSHOT_SQL",
            f"CALL {get_db_name()}.SP_ROTATION_DAILY_REFRESH(:p_run_id, :p_trade_date, :p_force, :p_refresh_energy)",
        ).strip()
        if not sp_sql:
            raise RuntimeError(
                "ROTATION_SNAPSHOT_SQL is required. "
                "Provide a MySQL CALL statement for SP_ROTATION_DAILY_REFRESH."
            )

        run_id = (
            os.getenv("ROTATION_SNAPSHOT_RUN_ID")
            or getattr(ctx.config, "rotation_run_id", None)
            or getattr(ctx.config, "run_id", None)
            or "SR_BASE_V535_EP90_XP55_XC2_MH5_RF5_K2_COST5BPS"
        )
        needed = _needed_binds(sp_sql)

        # Sanity: if SQL references run_id but none is provided -> fail fast
        if ("run_id" in needed or "p_run_id" in needed) and not run_id:
            raise RuntimeError(
                "Your ROTATION_SNAPSHOT_SQL references :run_id (or :p_run_id) but ROTATION_SNAPSHOT_RUN_ID is not set."
            )

        # As-of semantics: use end_date for p_trade_date in daily refresh.
        payload = self._build_bind_payload(ctx=ctx, run_id=run_id, trade_date=end)
        map_sql = os.getenv(
            "ROTATION_MAP_SQL",
            "CALL sp_build_board_member_map(:p_trade_date, :p_trade_date)",
        ).strip()
        energy_sql = os.getenv("ROTATION_ENERGY_SQL", "").strip()
        refresh_mode = str(os.getenv("ROTATION_REFRESH_MODE", "v8_core")).strip().lower()
        if refresh_mode not in {"v8_core", "base_chain", "sp"}:
            raise RuntimeError(
                "[rotation_snapshot] invalid ROTATION_REFRESH_MODE=%r. Expected 'v8_core', 'base_chain' or 'sp'."
                % refresh_mode
            )

        ctx.log.info(
            f"[rotation_snapshot] refresh start; mode={refresh_mode} start={start} end={end} "
            f"run_id={run_id} binds={sorted(payload.keys())}"
        )

        try:
            # Execute with transaction
            with self.engine.begin() as conn:
                req_dt = _ensure_date(payload["p_trade_date"])

                # Historical yearly wrappers often pass calendar year-end dates such as
                # 2011-12-31. If that date is not a trading day, use the latest price
                # trade date on or before it (e.g. 2011-12-30) instead of failing as a
                # false stale-data error.
                price_trade_dt = _latest_price_trade_date_on_or_before(conn, req_dt)
                if price_trade_dt is not None and price_trade_dt < req_dt:
                    ctx.log.info(
                        "[rotation_snapshot] requested trade_date=%s is not an available price trade date; "
                        "using latest price trade_date=%s",
                        req_dt,
                        price_trade_dt,
                    )
                    payload["p_trade_date"] = price_trade_dt
                    payload["trade_date"] = price_trade_dt
                    req_dt = price_trade_dt

                # Ensure cn_board_member_map_d is available up to the effective target
                # date before computing latest valid rotation date. Without this,
                # rotation can get stuck at the old map max date even when price data is
                # already refreshed to newer years.
                auto_map_backfill = str(os.getenv("ROTATION_AUTO_BACKFILL_BOARD_MEMBER_MAP", "1")).strip().lower()
                if auto_map_backfill not in {"0", "false", "no", "off"}:
                    self._backfill_missing_board_member_map(conn=conn, end_date=req_dt)

                eff_dt = _resolve_trade_date_for_rotation(conn, req_dt)
                if eff_dt != req_dt:
                    msg = (
                        "[rotation_snapshot] requested trade_date exceeds latest valid rotation date "
                        f"(requested={req_dt}, effective={eff_dt}). cn_board_member_map_d may still be incomplete."
                    )
                    if str(os.getenv("ROTATION_FAIL_IF_STALE", "1")).strip().lower() in {"0", "false", "no", "off"}:
                        ctx.log.warning(msg + " Skip by ROTATION_FAIL_IF_STALE=0.")
                        return
                    raise RuntimeError(msg + " Refuse silent fallback.")

                self._refresh_active_universe_before_daily(conn=conn, payload=payload)

                auto_backfill = int(os.getenv("ROTATION_AUTO_BACKFILL_MISSING", "1"))
                if auto_backfill == 1:
                    self._backfill_missing_rotation_history(conn=conn, payload=payload)

                # Daily map incremental step (by trade_date) to ensure latest symbol->sector mapping is present.
                if map_sql:
                    if "sp_build_board_member_map" in map_sql.lower() and not _map_proc_exists(conn):
                        raise RuntimeError("[rotation_snapshot] map SP missing: sp_build_board_member_map. Refuse skip.")
                    else:
                        ctx.log.info("[rotation_snapshot] pre-step map refresh enabled; executing ROTATION_MAP_SQL")
                        try:
                            conn.execute(text(map_sql), payload)
                        except Exception:
                            ctx.log.exception(
                                "[rotation_snapshot] ROTATION_MAP_SQL failed; sql=%s payload=%s",
                                map_sql,
                                payload,
                            )
                            raise

                if energy_sql:
                    ctx.log.info("[rotation_snapshot] pre-step energy refresh enabled; executing ROTATION_ENERGY_SQL")
                    try:
                        conn.execute(text(energy_sql), payload)
                    except Exception:
                        ctx.log.exception(
                            "[rotation_snapshot] ROTATION_ENERGY_SQL failed; sql=%s payload=%s",
                            energy_sql,
                            payload,
                        )
                        raise

                if refresh_mode == "v8_core":
                    # V8 production/backfill core-only mode:
                    # keep only the required board-member map and optional active-universe refresh.
                    # Do not run legacy V7 sector-rotation ranked/signal/BT/snapshot chains.
                    ctx.log.info(
                        "[rotation_snapshot] v8_core done; trade_date=%s run_id=%s; "
                        "skip legacy sector rotation chain",
                        payload.get("p_trade_date"),
                        run_id,
                    )
                    return

                if refresh_mode == "base_chain":
                    # Legacy/research sector rotation chain. Disabled by default because it writes
                    # cn_sector_eod_hist_t, cn_sector_rotation_ranked_t, cn_sector_rotation_signal_t,
                    # cn_sector_rot_bt_daily_t and rotation snapshot tables.
                    self._run_daily_refresh_base_chain(conn=conn, payload=payload, include_downstream=True)
                else:
                    try:
                        conn.execute(text(sp_sql), payload)
                    except Exception as e:
                        if _is_missing_transition_view_error(e):
                            raise RuntimeError(
                                "[rotation_snapshot] ROTATION_REFRESH_MODE=sp requires cn_sector_rotation_transition_v, "
                                "but the view is missing. Use ROTATION_REFRESH_MODE=base_chain or deploy the SP/view DDL."
                            ) from e
                        ctx.log.exception(
                            "[rotation_snapshot] ROTATION_SNAPSHOT_SQL failed; sql=%s payload=%s",
                            sp_sql,
                            payload,
                        )
                        raise

                    auto_repair_downstream = int(os.getenv("ROTATION_AUTO_REPAIR_DOWNSTREAM", "1"))
                    if auto_repair_downstream == 1:
                        self._repair_missing_rotation_downstream(conn=conn, payload=payload)
        except Exception:
            ctx.log.exception(
                "[rotation_snapshot] run failed; start=%s end=%s run_id=%s trade_date=%s",
                start,
                end,
                run_id,
                payload.get("p_trade_date"),
            )
            raise

        ctx.log.info("[rotation_snapshot] done")

    def audit_coverage(self, ctx, run_id: str | None = None, end_date=None) -> dict[str, list[_dt.date]]:
        engine = getattr(ctx, "engine", None)
        if engine is None:
            raise RuntimeError("[rotation_snapshot] audit requires MySQL engine.")
        cfg = ctx.config
        cfg.finalize_dates()
        resolved_run_id = (
            run_id
            or os.getenv("ROTATION_SNAPSHOT_RUN_ID")
            or getattr(ctx.config, "rotation_run_id", None)
            or getattr(ctx.config, "run_id", None)
            or "SR_BASE_V535_EP90_XP55_XC2_MH5_RF5_K2_COST5BPS"
        )
        requested_end = _ensure_date(end_date or cfg.end_date)
        with engine.begin() as conn:
            effective_end = _resolve_trade_date_for_rotation(conn, requested_end)
            upstream_missing = self._find_missing_rotation_dates(conn=conn, end_date=effective_end)
            bt_missing = self._find_missing_rotation_bt_dates(conn=conn, run_id=resolved_run_id, end_date=effective_end)
            snap_missing = self._find_missing_rotation_snap_dates(conn=conn, run_id=resolved_run_id, end_date=effective_end)
        return {
            "effective_end_date": [effective_end],
            "upstream_missing": upstream_missing,
            "bt_missing": bt_missing,
            "snap_missing": snap_missing,
        }

    def _backfill_missing_board_member_map(self, conn, end_date: _dt.date) -> None:
        """Backfill cn_board_member_map_d for missing price trade dates up to end_date."""
        if not _map_proc_exists(conn):
            raise RuntimeError("[rotation_snapshot] map SP missing: sp_build_board_member_map. Refuse skip.")

        missing_days = _find_missing_board_member_map_dates(conn=conn, end_date=end_date)
        if not missing_days:
            self.log.info("[rotation_snapshot] board-member map backfill check: no missing map days up to %s", end_date)
            return

        self.log.info(
            "[rotation_snapshot] board-member map auto backfill enabled: missing_days=%s range=%s..%s",
            len(missing_days),
            missing_days[0],
            missing_days[-1],
        )

        progress_every = max(1, int(os.getenv("ROTATION_BOARD_MAP_PROGRESS_EVERY", "50")))
        for i, d in enumerate(missing_days, start=1):
            conn.execute(
                text("CALL sp_build_board_member_map(:d1, :d2)"),
                {"d1": d, "d2": d},
            )
            if i % progress_every == 0 or i == len(missing_days):
                self.log.info(
                    "[rotation_snapshot] board-member map backfill progress: %s/%s latest=%s",
                    i,
                    len(missing_days),
                    d,
                )

        remaining_days = _find_missing_board_member_map_dates(conn=conn, end_date=end_date)
        if remaining_days:
            raise RuntimeError(
                "[rotation_snapshot] board-member map auto backfill incomplete "
                f"(remaining_days={len(remaining_days)} first={remaining_days[0]} last={remaining_days[-1]})."
            )

        self.log.info(
            "[rotation_snapshot] board-member map auto backfill complete: filled_days=%s end=%s",
            len(missing_days),
            end_date,
        )

    def _refresh_active_universe_before_daily(self, conn, payload) -> None:
        enabled = str(os.getenv("ROTATION_AUTO_REFRESH_ACTIVE_UNIVERSE", "1")).strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            self.log.info("[rotation_snapshot] skip active-universe refresh by env")
            return

        if not _active_universe_refresh_proc_exists(conn):
            raise RuntimeError("[rotation_snapshot] required procedure missing: sp_refresh_stock_universe_status. Refuse skip.")

        recent_days = int(os.getenv("ROTATION_ACTIVE_RECENT_DAYS", "30"))
        min_trade_days = int(os.getenv("ROTATION_ACTIVE_MIN_TRADE_DAYS", "1"))
        asof = _ensure_date(payload["p_trade_date"])
        conn.execute(
            text(
                """
                CALL sp_refresh_stock_universe_status(
                    :asof_date,
                    :recent_days,
                    :min_trade_days
                )
                """
            ),
            {
                "asof_date": asof,
                "recent_days": recent_days,
                "min_trade_days": min_trade_days,
            },
        )
        self.log.info(
            "[rotation_snapshot] active-universe refreshed before daily: asof=%s recent_days=%s min_trade_days=%s",
            asof,
            recent_days,
            min_trade_days,
        )

    def _run_daily_refresh_base_chain(self, conn, payload, include_downstream: bool = True) -> None:
        trade_date = _ensure_date(payload["p_trade_date"])
        run_id = payload.get("p_run_id") or payload.get("run_id")
        force = int(payload.get("p_force", 0))

        # 1) eod_hist daily incremental
        conn.execute(
            text("CALL sp_refresh_sector_eod_hist(:d1, :d2, :tp, :bm)"),
            {"d1": trade_date, "d2": trade_date, "tp": 0.30, "bm": 0.60},
        )

        # 2) ranked_t daily upsert from base table (no transition view dependency)
        ranked_sql = text(
            """
            INSERT INTO cn_sector_rotation_ranked_t (
                trade_date, sector_type, sector_id, sector_name,
                state, tier, theme_group, theme_rank, score, confirm_streak,
                amt_impulse, up_ma5, up_ratio, created_at
            )
            WITH hist0 AS (
                SELECT
                    e.trade_date,
                    e.sector_type,
                    e.sector_id,
                    e.sector_id AS sector_name,
                    e.amount_sum,
                    e.score,
                    e.up_ratio,
                    e.cond_count,
                    e.sector_pass
                FROM cn_sector_eod_hist_t e
                WHERE e.trade_date BETWEEN DATE_SUB(:d, INTERVAL 40 DAY) AND :d
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
                    END AS tier_pri
                FROM hist4 h4
            )
            SELECT
                h5.trade_date,
                h5.sector_type,
                h5.sector_id,
                h5.sector_name,
                h5.state,
                h5.tier,
                'OTHER' AS theme_group,
                ROW_NUMBER() OVER (
                    PARTITION BY h5.trade_date, 'OTHER'
                    ORDER BY h5.tier_pri, h5.score DESC, h5.sector_type, h5.sector_id
                ) AS theme_rank,
                h5.score,
                h5.confirm_streak,
                h5.amt_impulse,
                h5.up_ma5,
                h5.up_ratio,
                NOW()
            FROM hist5 h5
            WHERE h5.trade_date = :d
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
        conn.execute(ranked_sql, {"d": trade_date})

        # 3) signal_t daily upsert from ranked_t transition-on-the-fly
        signal_sql = text(
            """
            INSERT INTO cn_sector_rotation_signal_t (
                signal_date, sector_type, sector_id, sector_name, action,
                entry_rank, entry_cnt, weight_suggested, score, state, transition, created_at
            )
            WITH target AS (
                SELECT DISTINCT r.sector_type, r.sector_id
                FROM cn_sector_rotation_ranked_t r
                WHERE r.trade_date = :d
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
                WHERE r.trade_date < :d
                GROUP BY r.sector_type, r.sector_id
            ),
            hist AS (
                SELECT
                    r.trade_date,
                    r.sector_type,
                    r.sector_id,
                    r.sector_name,
                    r.theme_rank,
                    r.tier,
                    r.state,
                    r.score,
                    r.amt_impulse,
                    r.up_ma5
                FROM cn_sector_rotation_ranked_t r
                JOIN target t
                  ON t.sector_type = r.sector_type
                 AND t.sector_id = r.sector_id
                WHERE r.trade_date = :d
                UNION ALL
                SELECT
                    r.trade_date,
                    r.sector_type,
                    r.sector_id,
                    r.sector_name,
                    r.theme_rank,
                    r.tier,
                    r.state,
                    r.score,
                    r.amt_impulse,
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
                    LAG(h.tier)  OVER (PARTITION BY h.sector_type, h.sector_id ORDER BY h.trade_date) AS prev_tier
                FROM hist h
            ),
            sig0 AS (
                SELECT
                    x.trade_date AS signal_date,
                    x.sector_type,
                    x.sector_id,
                    x.sector_name,
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
                    END AS transition,
                    CASE
                        WHEN (
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
                            END
                        ) IN ('IGNITE_TO_CONFIRM', 'DIRECT_CONFIRM')
                             AND x.theme_rank = 1
                             AND x.tier = 'T1'
                             AND x.up_ma5 >= 0.52
                             AND x.amt_impulse >= 1.10 THEN 'ENTER'
                        WHEN (
                            CASE
                                WHEN x.prev_tier = 'T1' AND x.tier = 'T2' THEN 'T1_TO_T2'
                                WHEN x.prev_tier = 'T2' AND x.tier = 'T3' THEN 'T2_TO_T3'
                                ELSE ''
                            END
                        ) IN ('T1_TO_T2', 'T2_TO_T3') THEN 'EXIT'
                        ELSE 'WATCH'
                    END AS action,
                    x.score,
                    x.state
                FROM x
                WHERE x.trade_date = :d
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
        conn.execute(signal_sql, {"d": trade_date})

        # 4) keep the same downstream daily chain behavior as SP_ROTATION_DAILY_REFRESH
        if include_downstream:
            conn.execute(
                text("CALL SP_BACKFILL_ROT_BT_FROM_PRICE(:run_id, :d, :force)"),
                {"run_id": run_id, "d": trade_date, "force": force},
            )
            conn.execute(
                text("CALL SP_REPAIR_ROT_BT_NAV(:run_id, :nav_base)"),
                {"run_id": run_id, "nav_base": 1.0},
            )
            conn.execute(
                text("CALL SP_REFRESH_ROTATION_SNAP_ALL(:run_id, :d, :force)"),
                {"run_id": run_id, "d": trade_date, "force": force},
            )
        self.log.info(
            "[rotation_snapshot] base-chain done; trade_date=%s run_id=%s force=%s include_downstream=%s",
            trade_date,
            run_id,
            force,
            include_downstream,
        )

    def _find_missing_rotation_dates(self, conn, end_date: _dt.date):
        sql = text(
            """
            SELECT p.trade_date
            FROM (
                SELECT DISTINCT trade_date
                FROM cn_stock_daily_price
                WHERE trade_date <= :d
            ) p
            WHERE EXISTS (
                SELECT 1
                FROM cn_board_member_map_d m
                WHERE m.trade_date = p.trade_date
                LIMIT 1
            )
            AND (
                NOT EXISTS (
                    SELECT 1
                    FROM cn_sector_eod_hist_t e
                    WHERE e.trade_date = p.trade_date
                    LIMIT 1
                )
                OR NOT EXISTS (
                    SELECT 1
                    FROM cn_sector_rotation_ranked_t r
                    WHERE r.trade_date = p.trade_date
                    LIMIT 1
                )
                OR NOT EXISTS (
                    SELECT 1
                    FROM cn_sector_rotation_signal_t s
                    WHERE s.signal_date = p.trade_date
                    LIMIT 1
                )
            )
            ORDER BY p.trade_date
            """
        )
        rows = conn.execute(sql, {"d": end_date}).fetchall()
        return [_ensure_date(r[0]) for r in rows]

    def _backfill_missing_rotation_history(self, conn, payload) -> None:
        end_date = _ensure_date(payload["p_trade_date"])
        missing_days = self._find_missing_rotation_dates(conn=conn, end_date=end_date)
        if not missing_days:
            self.log.info("[rotation_snapshot] auto backfill check: no missing historical board days up to %s", end_date)
            return

        self.log.info(
            "[rotation_snapshot] auto backfill enabled: missing_days=%s range=%s..%s",
            len(missing_days),
            missing_days[0],
            missing_days[-1],
        )

        # For history补齐，先补三张板块日表，不跑BT/snapshot，避免重复放大开销。
        for i, d in enumerate(missing_days, start=1):
            hist_payload = dict(payload)
            hist_payload["p_trade_date"] = d
            hist_payload["trade_date"] = d
            self._run_daily_refresh_base_chain(conn=conn, payload=hist_payload, include_downstream=False)
            if i % 20 == 0 or i == len(missing_days):
                self.log.info(
                    "[rotation_snapshot] auto backfill progress: %s/%s (latest=%s)",
                    i,
                    len(missing_days),
                    d,
                )

    def _find_missing_rotation_bt_dates(self, conn, run_id: str, end_date: _dt.date):
        sql = text(
            """
            SELECT p.trade_date
            FROM (
                SELECT DISTINCT trade_date
                FROM cn_stock_daily_price
                WHERE trade_date <= :d
            ) p
            WHERE EXISTS (
                SELECT 1
                FROM cn_board_member_map_d m
                WHERE m.trade_date = p.trade_date
                LIMIT 1
            )
            AND NOT EXISTS (
                SELECT 1
                FROM cn_sector_rot_bt_daily_t b
                WHERE b.run_id = (CAST(:run_id AS CHAR CHARACTER SET utf8mb4) COLLATE utf8mb4_unicode_ci)
                  AND b.trade_date = p.trade_date
                LIMIT 1
            )
            ORDER BY p.trade_date
            """
        )
        rows = conn.execute(sql, {"d": end_date, "run_id": run_id}).fetchall()
        return [_ensure_date(r[0]) for r in rows]

    def _find_missing_rotation_snap_dates(self, conn, run_id: str, end_date: _dt.date):
        sql = text(
            """
            SELECT b.trade_date
            FROM cn_sector_rot_bt_daily_t b
            WHERE b.run_id = (CAST(:run_id AS CHAR CHARACTER SET utf8mb4) COLLATE utf8mb4_unicode_ci)
              AND b.trade_date <= :d
              AND (
                    IFNULL(b.exposed_flag, 0) > 0
                    OR EXISTS (
                        SELECT 1
                        FROM cn_sector_rotation_signal_t s
                        WHERE s.signal_date = b.trade_date
                        LIMIT 1
                    )
                  )
              AND NOT EXISTS (
                    SELECT 1
                    FROM cn_rotation_entry_snap_t e
                    WHERE e.run_id = b.run_id
                      AND e.trade_date = b.trade_date
                    LIMIT 1
                  )
              AND NOT EXISTS (
                    SELECT 1
                    FROM cn_rotation_holding_snap_t h
                    WHERE h.run_id = b.run_id
                      AND h.trade_date = b.trade_date
                    LIMIT 1
                  )
              AND NOT EXISTS (
                    SELECT 1
                    FROM cn_rotation_exit_snap_t x
                    WHERE x.run_id = b.run_id
                      AND x.trade_date = b.trade_date
                    LIMIT 1
                  )
            ORDER BY b.trade_date
            """
        )
        rows = conn.execute(sql, {"d": end_date, "run_id": run_id}).fetchall()
        return [_ensure_date(r[0]) for r in rows]

    def _load_rotation_bt_dates(self, conn, run_id: str, start_date: _dt.date, end_date: _dt.date):
        rows = conn.execute(
            text(
                """
                SELECT trade_date
                FROM cn_sector_rot_bt_daily_t
                WHERE run_id = (CAST(:run_id AS CHAR CHARACTER SET utf8mb4) COLLATE utf8mb4_unicode_ci)
                  AND trade_date BETWEEN :start_date AND :end_date
                ORDER BY trade_date
                """
            ),
            {"run_id": run_id, "start_date": start_date, "end_date": end_date},
        ).fetchall()
        return [_ensure_date(r[0]) for r in rows]

    def _repair_missing_rotation_downstream(self, conn, payload) -> None:
        end_date = _ensure_date(payload["p_trade_date"])
        run_id = str(payload.get("p_run_id") or payload.get("run_id") or "").strip()
        if not run_id:
            raise RuntimeError("[rotation_snapshot] run_id is empty. Refuse skip downstream repair.")

        bt_missing = self._find_missing_rotation_bt_dates(conn=conn, run_id=run_id, end_date=end_date)
        snap_missing = self._find_missing_rotation_snap_dates(conn=conn, run_id=run_id, end_date=end_date)
        if not bt_missing and not snap_missing:
            self.log.info("[rotation_snapshot] downstream repair check: no BT/snap gaps up to %s for run_id=%s", end_date, run_id)
            return

        rebuild_start = min(bt_missing + snap_missing)
        self.log.info(
            "[rotation_snapshot] downstream repair enabled: bt_missing=%s snap_missing=%s rebuild_start=%s end=%s run_id=%s",
            len(bt_missing),
            len(snap_missing),
            rebuild_start,
            end_date,
            run_id,
        )

        if bt_missing:
            conn.execute(
                text(
                    """
                    DELETE FROM cn_sector_rot_bt_daily_t
                    WHERE run_id = (CAST(:run_id AS CHAR CHARACTER SET utf8mb4) COLLATE utf8mb4_unicode_ci)
                      AND trade_date BETWEEN :start_date AND :end_date
                    """
                ),
                {"run_id": run_id, "start_date": rebuild_start, "end_date": end_date},
            )
            conn.execute(
                text("CALL SP_BACKFILL_ROT_BT_FROM_PRICE(:run_id, :d, :force)"),
                {"run_id": run_id, "d": end_date, "force": 1},
            )
            conn.execute(
                text("CALL SP_REPAIR_ROT_BT_NAV(:run_id, :nav_base)"),
                {"run_id": run_id, "nav_base": 1.0},
            )

        bt_dates = self._load_rotation_bt_dates(conn=conn, run_id=run_id, start_date=rebuild_start, end_date=end_date)
        if not bt_dates:
            raise RuntimeError(
                "[rotation_snapshot] downstream repair cannot rebuild snapshots because no BT dates exist "
                f"in {rebuild_start}..{end_date} for run_id={run_id}. Refuse skip."
            )

        progress_every = max(1, int(os.getenv("ROTATION_DOWNSTREAM_PROGRESS_EVERY", "20")))
        for i, d in enumerate(bt_dates, start=1):
            conn.execute(
                text("CALL SP_REFRESH_ROTATION_SNAP_ALL(:run_id, :d, :force)"),
                {"run_id": run_id, "d": d, "force": 1},
            )
            if i % progress_every == 0 or i == len(bt_dates):
                self.log.info(
                    "[rotation_snapshot] downstream repair progress: %s/%s (latest=%s run_id=%s)",
                    i,
                    len(bt_dates),
                    d,
                    run_id,
                )

        remaining_bt = self._find_missing_rotation_bt_dates(conn=conn, run_id=run_id, end_date=end_date)
        remaining_snap = self._find_missing_rotation_snap_dates(conn=conn, run_id=run_id, end_date=end_date)
        if remaining_bt or remaining_snap:
            raise RuntimeError(
                "[rotation_snapshot] downstream repair incomplete "
                f"(remaining_bt={len(remaining_bt)} remaining_snap={len(remaining_snap)} "
                f"run_id={run_id} end_date={end_date})"
            )

        self.log.info(
            "[rotation_snapshot] downstream repair complete: rebuilt_dates=%s run_id=%s range=%s..%s",
            len(bt_dates),
            run_id,
            rebuild_start,
            end_date,
        )

    def _build_bind_payload(self, ctx, run_id, trade_date):
        force = getattr(ctx.config, "rotation_force", None)
        if force is None:
            force = 0

        refresh_energy = int(os.getenv("ROTATION_REFRESH_ENERGY", "0"))

        dt = _ensure_date(trade_date)

        # Provide both prefixed and plain bind names to be compatible with SQL variants.
        payload = {
            "p_run_id": run_id,
            "run_id": run_id,
            "p_trade_date": dt,
            "trade_date": dt,
            "p_force": int(force),
            "force": int(force),
            "p_refresh_energy": int(refresh_energy),
            "refresh_energy": int(refresh_energy),
        }
        return payload
