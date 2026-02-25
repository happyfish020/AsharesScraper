from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, Any, Set, Optional

from sqlalchemy import text

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



@dataclass
class SectorRotationSnapshotTask:
    """Generate sector rotation ENTER/HOLD/EXIT snapshots by calling a stored procedure (SP).

    This task is **read-write** by design (it generates snapshot tables) but it never patches SPs.
    It only executes a user-provided PL/SQL block that calls the already-existing SP.

    Configuration (environment variables):
    - ROTATION_SNAPSHOT_SQL (optional):
        Example:
          CALL cn_market.SP_ROTATION_DAILY_REFRESH(:p_run_id, :p_trade_date, :p_force, :p_refresh_energy);

    - ROTATION_SNAPSHOT_RUN_ID (optional):
        If your SQL needs a run_id bind, set it here. If not referenced, it's ignored.

    Behavior:
    - If the block references any of :start_date/:end_date (or *_yyyymmdd variants), we call it once.
    - Otherwise, we call it once using the as-of (end_date) binds.
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
            "CALL cn_market.SP_ROTATION_DAILY_REFRESH(:p_run_id, :p_trade_date, :p_force, :p_refresh_energy)",
        ).strip()
        
        
        
        if not sp_sql:
            raise RuntimeError(
                "ROTATION_SNAPSHOT_SQL is required. "
                "Provide a MySQL CALL statement for SP_ROTATION_DAILY_REFRESH."
            )

        run_id = 'SR_BASE_V535_EP90_XP55_XC2_MH5_RF5_K2_COST5BPS' #
        #os.getenv("ROTATION_SNAPSHOT_RUN_ID")
        needed = _needed_binds(sp_sql)

        # Sanity: if block references run_id but env missing -> fail fast
        if ("run_id" in needed or "p_run_id" in needed) and not run_id:
            raise RuntimeError(
                "Your ROTATION_SNAPSHOT_SQL references :run_id (or :p_run_id) but ROTATION_SNAPSHOT_RUN_ID is not set."
            )

        payload = self._build_bind_payload(ctx=ctx , trade_date=start)

        ctx.log.info(
            f"[rotation_snapshot] calling mysql SP; start={start} end={end} "
            f"binds={sorted(payload.keys())}"
        )

        # Execute with transaction
        with self.engine.begin() as conn:
            conn.execute(text(sp_sql), payload)

        ctx.log.info("[rotation_snapshot] done")


    def _build_bind_payload(self, ctx, trade_date):
        run_id = getattr(ctx.config, "rotation_run_id", None) or \
                 getattr(ctx.config, "run_id", None) or \
                 "SR_BASE_V535_EP90_XP55_XC2_MH5_RF5_K2_COST5BPS"
    
        force = getattr(ctx.config, "rotation_force", None)
        if force is None:
            force = 0
    
        refresh_energy = getattr(ctx.config, "rotation_refresh_energy", None)
        if refresh_energy is None:
            refresh_energy = 1
    
        dt = _ensure_date(trade_date)
    
        payload = {
            "p_run_id": run_id,
            "p_trade_date": dt,                 # ✅ must be date/datetime, NOT string
            "p_force": int(force),
            "p_refresh_energy": int(refresh_energy),
        }
        return payload
    
    
