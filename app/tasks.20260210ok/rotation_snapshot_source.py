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
    - ROTATION_SNAPSHOT_SP_BLOCK (required):
        Example:
          BEGIN SECOPR.SP_REFRESH_SECTOR_ROTATION_DAILY(p_run_id=>:run_id, p_trade_date=>:trade_date); END;
        or (range mode):
          BEGIN SECOPR.SP_BUILD_SECTOR_ROT_SIGNAL(p_run_id=>:run_id, p_start_date=>:start_date, p_end_date=>:end_date); END;

    - ROTATION_SNAPSHOT_RUN_ID (optional):
        If your SP needs a run_id bind, set it here. If not referenced in the block, it's ignored.

    Behavior:
    - If the block references any of :start_date/:end_date (or *_yyyymmdd variants), we call it once.
    - Otherwise, we call it once using the as-of (end_date) binds.
    """

    name: str = "SectorRotationSnapshotTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        self.engine=ctx.engine
        cfg.finalize_dates()
        start = str(cfg.start_date)
        end = str(cfg.end_date)

        #plsql_block = os.getenv("ROTATION_SNAPSHOT_SP_BLOCK", "").strip()
        #plsql_block = "SP_ROTATION_DAILY_REFRESH"
        plsql_block = """
            BEGIN
              SECOPR.SP_ROTATION_DAILY_REFRESH(
                :p_run_id,
                :p_trade_date,
                :p_force,
                :p_refresh_energy
              );
            END;
            """
        
        
        
        if not plsql_block:
            raise RuntimeError(
                "ROTATION_SNAPSHOT_SP_BLOCK is required. "
                "Provide a PL/SQL block that calls your existing SECOPR SP to generate ENTER/HOLD/EXIT snapshots."
            )

        run_id = 'SR_BASE_V535_EP90_XP55_XC2_MH5_RF5_K2_COST5BPS' #
        #os.getenv("ROTATION_SNAPSHOT_RUN_ID")
        needed = _needed_binds(plsql_block)

        # Sanity: if block references run_id but env missing -> fail fast
        if ("run_id" in needed or "p_run_id" in needed) and not run_id:
            raise RuntimeError(
                "Your ROTATION_SNAPSHOT_SP_BLOCK references :run_id (or :p_run_id) but ROTATION_SNAPSHOT_RUN_ID is not set."
            )

        payload = self._build_bind_payload(ctx=ctx , trade_date=start)

        ctx.log.info(
            f"[rotation_snapshot] calling SP block; start={start} end={end} "
            f"binds={sorted(payload.keys())}"
        )

        # Execute with transaction
        with ctx.engine.begin() as conn:
            conn.execute(text(plsql_block), payload)

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
    
    