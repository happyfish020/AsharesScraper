from __future__ import annotations

import argparse
import builtins
import os
import time
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional

import requests
from sqlalchemy import text

from app.settings import build_engine
from app.tools.sync_cn_stock_daily_price_from_tushare import resolve_tushare_token
from app.utils.progress import ProgressLogger


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return builtins.print(*args, **kwargs)


TS_URL = "https://api.tushare.pro"
_SESSION = requests.Session()
_SESSION.trust_env = False


def _parse_ymd(s: str) -> date:
    s = (s or "").strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return datetime.strptime(s, "%Y-%m-%d").date()
    return datetime.strptime(s, "%Y%m%d").date()


def _to_db_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return datetime.strptime(s, "%Y%m%d").date()


def _year_chunks(start: date, end: date, years_per_chunk: int) -> Iterable[tuple[date, date]]:
    if years_per_chunk < 1:
        yield start, end
        return

    cur_year = start.year
    while cur_year <= end.year:
        chunk_start = date(cur_year, 1, 1)
        chunk_end = date(min(cur_year + years_per_chunk - 1, end.year), 12, 31)
        if chunk_start < start:
            chunk_start = start
        if chunk_end > end:
            chunk_end = end
        yield chunk_start, chunk_end
        cur_year += years_per_chunk


def _norm_symbol(ts_code: str) -> str:
    c = (ts_code or "").strip()
    if "." in c:
        return c.split(".", 1)[0]
    return c


def ts_call(token: str, api_name: str, params: Dict, fields: str = "", retries: int = 3):
    payload = {
        "api_name": api_name,
        "token": token,
        "params": params or {},
        "fields": fields or "",
    }
    for i in range(retries):
        try:
            resp = _SESSION.post(TS_URL, json=payload, timeout=30)
            resp.raise_for_status()
            obj = resp.json()
            if int(obj.get("code", -1)) != 0:
                raise RuntimeError(f"tushare api error: {obj.get('msg')}")
            data = obj.get("data") or {}
            return data.get("fields") or [], data.get("items") or []
        except Exception:
            if i + 1 == retries:
                raise
            time.sleep(1.5 * (i + 1))
    return [], []


def iter_concepts(token: str) -> List[Dict[str, str]]:
    cols, items = ts_call(token, "concept", {}, "code,name,src")
    idx = {c: i for i, c in enumerate(cols)}
    out: List[Dict[str, str]] = []
    for it in items:
        code = str(it[idx["code"]]).strip() if it[idx["code"]] else ""
        if code:
            out.append(
                {
                    "concept_id": code,
                    "concept_name": str(it[idx["name"]]).strip() if idx.get("name") is not None and it[idx["name"]] else code,
                    "src": str(it[idx["src"]]).strip() if idx.get("src") is not None and it[idx["src"]] else "",
                }
            )
    dedup: dict[str, Dict[str, str]] = {}
    for row in out:
        dedup[row["concept_id"]] = row
    return [dedup[k] for k in sorted(dedup.keys())]


def iter_concept_members(token: str, concept_id: str) -> Iterable[Dict]:
    cols, items = ts_call(
        token,
        "concept_detail",
        {"id": concept_id},
        "id,concept_name,ts_code,name,in_date,out_date",
    )
    idx = {c: i for i, c in enumerate(cols)}
    for it in items:
        yield {
            "concept_id": str(it[idx["id"]]),
            "ts_code": str(it[idx["ts_code"]]),
            "in_date": it[idx["in_date"]],
            "out_date": it[idx["out_date"]],
        }


def intersects(vf: date, vt: Optional[date], start: date, end: date) -> bool:
    if vf > end:
        return False
    if vt is not None and vt < start:
        return False
    return True


def _max_date(a: Optional[date], b: Optional[date]) -> Optional[date]:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _min_date(a: Optional[date], b: Optional[date]) -> Optional[date]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def main():
    p = argparse.ArgumentParser(description="Backfill concept membership history from Tushare")
    p.add_argument("--start", default="20000101", help="YYYYMMDD or YYYY-MM-DD")
    p.add_argument("--end", default=date.today().strftime("%Y%m%d"), help="YYYYMMDD or YYYY-MM-DD")
    p.add_argument("--source-label", default="tushare_concept", help="source label")
    p.add_argument("--token", default="", help="Tushare token")
    p.add_argument("--config", default="", help="Config file path for Tushare token")
    p.add_argument("--sleep-ms", type=int, default=120, help="sleep milliseconds between concept calls")
    p.add_argument("--skip-map", action="store_true", help="skip calling sp_build_board_member_map at the end")
    p.add_argument(
        "--map-chunk-years",
        type=int,
        default=1,
        help="rebuild cn_board_member_map_d in year chunks; use 0 for one single sp_build_board_member_map call",
    )
    p.add_argument("--no-reset", action="store_true", help="do not delete old source rows before upsert")
    args = p.parse_args()

    token, tried_files = resolve_tushare_token(args.token, args.config)
    if not token:
        msg = "Tushare token is required. Use --token, env TUSHARE_TOKEN/TS_TOKEN, or --config."
        if tried_files:
            msg += f" Tried files: {', '.join(str(p) for p in tried_files)}"
        raise SystemExit(msg)

    start = _parse_ymd(args.start)
    end = _parse_ymd(args.end)
    if start > end:
        raise SystemExit(f"Invalid range: {start}>{end}")

    print(f"loading concept history from tushare: {start}..{end}")
    concepts = iter_concepts(token)
    print(f"concept_ids={len(concepts)}")
    if not concepts:
        raise SystemExit("No concept ids returned.")

    engine = build_engine()
    concept_master_rows = [
        {
            "concept_id": row["concept_id"],
            "concept_name": row["concept_name"],
            "provider": "TUSHARE",
            "asof_date": end,
            "source": "TUSHARE_CONCEPT_TS",
            "raw_json": str({"src": row["src"]}),
        }
        for row in concepts
    ]
    concept_master_sql = text(
        """
        INSERT INTO cn_board_concept_master
            (CONCEPT_ID, CONCEPT_NAME, PROVIDER, ASOF_DATE, SOURCE, CREATED_AT, RAW_JSON)
        VALUES
            (:concept_id, :concept_name, :provider, :asof_date, :source, NOW(), :raw_json)
        ON DUPLICATE KEY UPDATE
            CONCEPT_NAME = VALUES(CONCEPT_NAME),
            PROVIDER = VALUES(PROVIDER),
            SOURCE = VALUES(SOURCE),
            RAW_JSON = VALUES(RAW_JSON)
        """
    )
    with engine.begin() as conn:
        conn.execute(concept_master_sql, concept_master_rows)
    print(f"concept_master_upserted={len(concept_master_rows)} source=TUSHARE_CONCEPT_TS")

    insert_sql = text(
        """
        INSERT INTO cn_board_concept_member_hist (concept_id, symbol, valid_from, valid_to, source)
        VALUES (:concept_id, :symbol, :valid_from, :valid_to, :source)
        ON DUPLICATE KEY UPDATE
            valid_to = VALUES(valid_to),
            source = VALUES(source),
            updated_at = CURRENT_TIMESTAMP(6)
        """
    )

    total_rows = 0
    touched_ids = 0
    buf: List[Dict] = []
    batch_size = 5000
    earliest_raw_valid_from: Optional[date] = None
    latest_raw_valid_from: Optional[date] = None
    effective_load_start: Optional[date] = None
    effective_load_end: Optional[date] = None
    progress = ProgressLogger(name="concept_history", total=len(concepts), unit="concept_ids", every=1, min_interval_seconds=5.0)

    for i, concept_row in enumerate(concepts, start=1):
        cid = concept_row["concept_id"]
        cid_rows = 0
        progress.note(f"[concept_history] fetching {i}/{len(concepts)} concept_id={cid}")
        for m in iter_concept_members(token, cid):
            vf = _to_db_date(m["in_date"])
            vt = _to_db_date(m["out_date"])
            if vf is None:
                continue
            earliest_raw_valid_from = _min_date(earliest_raw_valid_from, vf)
            latest_raw_valid_from = _max_date(latest_raw_valid_from, vf)
            if not intersects(vf, vt, start, end):
                continue
            effective_load_start = _min_date(effective_load_start, vf)
            effective_load_end = _max_date(effective_load_end, _max_date(vf, vt))
            buf.append(
                {
                    "concept_id": cid,
                    "symbol": _norm_symbol(m["ts_code"]),
                    "valid_from": vf,
                    "valid_to": vt,
                    "source": args.source_label,
                }
            )
            cid_rows += 1
            if len(buf) >= batch_size:
                with engine.begin() as conn:
                    conn.execute(insert_sql, buf)
                total_rows += len(buf)
                buf.clear()

        if cid_rows > 0:
            touched_ids += 1
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)
        progress.update(current_item=cid, rows=cid_rows, extra=f"touched_ids={touched_ids} buffered={len(buf)} inserted={total_rows}")

    pending_rows = len(buf)
    if total_rows + pending_rows <= 0:
        progress.finish(
            extra=(
                f"touched_ids={touched_ids} inserted=0 "
                f"raw_valid_from={earliest_raw_valid_from}..{latest_raw_valid_from} "
                f"requested={start}..{end} skipped=no_supported_overlap"
            )
        )
        print(
            f"concept_hist_upserted=0 raw_valid_from={earliest_raw_valid_from}..{latest_raw_valid_from} "
            f"requested={start}..{end} skipped=no_supported_overlap"
        )
        print("skip_map=1 reason=no_supported_overlap")
        return

    if not args.no_reset and effective_load_start is not None and effective_load_end is not None:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM cn_board_concept_member_hist
                    WHERE source IN ('seed_cons', :src)
                      AND valid_from <= :effective_end
                      AND COALESCE(valid_to, DATE('9999-12-31')) >= :effective_start
                    """
                ),
                {
                    "src": args.source_label,
                    "effective_start": effective_load_start,
                    "effective_end": effective_load_end,
                },
            )

    if buf:
        with engine.begin() as conn:
            conn.execute(insert_sql, buf)
        total_rows += len(buf)
        buf.clear()
    progress.finish(
        extra=(
            f"touched_ids={touched_ids} inserted={total_rows} "
            f"raw_valid_from={earliest_raw_valid_from}..{latest_raw_valid_from} "
            f"effective_load={effective_load_start}..{effective_load_end}"
        )
    )

    print(
        f"concept_hist_upserted={total_rows} raw_valid_from={earliest_raw_valid_from}..{latest_raw_valid_from} "
        f"effective_load={effective_load_start}..{effective_load_end}"
    )

    if not args.skip_map:
        map_start = effective_load_start or start
        map_end = min(end, effective_load_end) if effective_load_end is not None else end
        map_chunk_years = int(getattr(args, "map_chunk_years", 1) or 0)
        map_ranges = list(_year_chunks(map_start, map_end, map_chunk_years))
        print(
            f"map_rebuild_start ranges={len(map_ranges)} "
            f"chunk_years={map_chunk_years} range={map_start}..{map_end}"
        )
        for i, (d1, d2) in enumerate(map_ranges, start=1):
            t0 = time.time()
            print(f"map_rebuild_chunk_start {i}/{len(map_ranges)} range={d1}..{d2}")
            with engine.begin() as conn:
                conn.execute(
                    text("CALL sp_build_board_member_map(:d1, :d2)"),
                    {
                        "d1": d1,
                        "d2": d2,
                    },
                )
            elapsed = round(time.time() - t0, 1)
            print(f"map_rebuild_chunk_done {i}/{len(map_ranges)} range={d1}..{d2} elapsed={elapsed}s")

    with engine.connect() as conn:
        stats = conn.execute(
            text(
                """
                SELECT
                  (SELECT COUNT(*) FROM cn_board_concept_member_hist WHERE source=:src) AS hist_rows,
                  (SELECT MIN(valid_from) FROM cn_board_concept_member_hist WHERE source=:src) AS min_valid_from,
                  (SELECT MAX(COALESCE(valid_to, DATE('9999-12-31'))) FROM cn_board_concept_member_hist WHERE source=:src) AS max_valid_to,
                  (SELECT COUNT(*) FROM cn_board_member_map_d WHERE trade_date BETWEEN :d1 AND :d2 AND sector_type='CONCEPT') AS map_rows
                """
            ),
            {
                "src": args.source_label,
                "d1": effective_load_start or start,
                "d2": min(end, effective_load_end) if effective_load_end is not None else end,
            },
        ).one()
        print(
            f"done hist_rows={stats[0]} min_valid_from={stats[1]} max_valid_to={stats[2]} "
            f"map_rows={stats[3]} skip_map={args.skip_map} effective_load={effective_load_start}..{effective_load_end}"
        )


if __name__ == "__main__":
    main()
