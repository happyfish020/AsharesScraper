from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional

import requests
from sqlalchemy import text

from app.settings import build_engine


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


def iter_concept_ids(token: str) -> List[str]:
    cols, items = ts_call(token, "concept", {}, "code,name,src")
    idx = {c: i for i, c in enumerate(cols)}
    out: List[str] = []
    for it in items:
        code = it[idx["code"]]
        if code:
            out.append(str(code))
    return sorted(set(out))


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


def main():
    p = argparse.ArgumentParser(description="Backfill concept membership history from Tushare")
    p.add_argument("--start", default="20000101", help="YYYYMMDD or YYYY-MM-DD")
    p.add_argument("--end", default=date.today().strftime("%Y%m%d"), help="YYYYMMDD or YYYY-MM-DD")
    p.add_argument("--source-label", default="tushare_concept", help="source label")
    p.add_argument("--token", default=os.getenv("TUSHARE_TOKEN", "").strip(), help="Tushare token")
    p.add_argument("--sleep-ms", type=int, default=120, help="sleep milliseconds between concept calls")
    p.add_argument("--skip-map", action="store_true", help="skip calling sp_build_board_member_map at the end")
    p.add_argument("--no-reset", action="store_true", help="do not delete old source rows before upsert")
    args = p.parse_args()

    token = (args.token or "").strip()
    if not token:
        raise SystemExit("Tushare token required. Use --token or TUSHARE_TOKEN.")

    start = _parse_ymd(args.start)
    end = _parse_ymd(args.end)
    if start > end:
        raise SystemExit(f"Invalid range: {start}>{end}")

    print(f"loading concept history from tushare: {start}..{end}")
    ids = iter_concept_ids(token)
    print(f"concept_ids={len(ids)}")
    if not ids:
        raise SystemExit("No concept ids returned.")

    engine = build_engine()
    if not args.no_reset:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM cn_board_concept_member_hist
                    WHERE source IN ('seed_cons', :src)
                    """
                ),
                {"src": args.source_label},
            )

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

    for i, cid in enumerate(ids, start=1):
        cid_rows = 0
        for m in iter_concept_members(token, cid):
            vf = _to_db_date(m["in_date"])
            vt = _to_db_date(m["out_date"])
            if vf is None:
                continue
            if not intersects(vf, vt, start, end):
                continue
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
        if i % 100 == 0 or i == len(ids):
            print(f"progress {i}/{len(ids)} touched_ids={touched_ids} buffered={len(buf)} inserted={total_rows}")

    if buf:
        with engine.begin() as conn:
            conn.execute(insert_sql, buf)
        total_rows += len(buf)
        buf.clear()

    print(f"concept_hist_upserted={total_rows}")

    if not args.skip_map:
        with engine.begin() as conn:
            conn.execute(
                text("CALL sp_build_board_member_map(:d1, :d2)"),
                {"d1": start, "d2": end},
            )

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
            {"src": args.source_label, "d1": start, "d2": end},
        ).one()
        print(f"done hist_rows={stats[0]} min_valid_from={stats[1]} max_valid_to={stats[2]} map_rows={stats[3]} skip_map={args.skip_map}")


if __name__ == "__main__":
    main()
