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


def _to_ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _to_db_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return datetime.strptime(s, "%Y%m%d").date()


def _norm_symbol(con_code: str) -> str:
    c = (con_code or "").strip()
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
            cols = data.get("fields") or []
            items = data.get("items") or []
            return cols, items
        except Exception:
            if i + 1 == retries:
                raise
            time.sleep(1.5 * (i + 1))
    return [], []


def iter_sw_l3_codes(token: str, src: str) -> List[str]:
    fields = "index_code,industry_name,level,src"
    cols, items = ts_call(
        token=token,
        api_name="index_classify",
        params={"src": src, "level": "L3"},
        fields=fields,
    )
    idx = {c: i for i, c in enumerate(cols)}
    codes: List[str] = []
    for it in items:
        code = it[idx["index_code"]]
        if code:
            codes.append(str(code))
    return sorted(set(codes))


def iter_index_members(token: str, index_code: str, start: date, end: date) -> Iterable[Dict]:
    fields = "index_code,con_code,in_date,out_date,is_new"
    # Prefer index_member_all when available, fallback to index_member for tokens without that dataset.
    cols, items = ts_call(
        token=token,
        api_name="index_member_all",
        params={"ts_code": index_code, "offset": 0, "limit": 5000},
        fields=fields,
    )
    if items:
        idx = {c: i for i, c in enumerate(cols)}
        for it in items:
            yield {
                "index_code": str(it[idx["index_code"]]),
                "con_code": str(it[idx["con_code"]]),
                "in_date": it[idx["in_date"]],
                "out_date": it[idx["out_date"]],
            }
        return

    cols, items = ts_call(
        token=token,
        api_name="index_member",
        params={"index_code": index_code, "start_date": _to_ymd(start), "end_date": _to_ymd(end)},
        fields=fields,
    )
    idx = {c: i for i, c in enumerate(cols)}
    for it in items:
        yield {
            "index_code": str(it[idx["index_code"]]),
            "con_code": str(it[idx["con_code"]]),
            "in_date": it[idx["in_date"]],
            "out_date": it[idx["out_date"]],
        }


def intersects(d1: date, d2: Optional[date], start: date, end: date) -> bool:
    if d1 > end:
        return False
    if d2 is not None and d2 < start:
        return False
    return True


def main():
    p = argparse.ArgumentParser(description="Backfill SW L3 industry membership history from Tushare")
    p.add_argument("--start", default="20000101", help="YYYYMMDD or YYYY-MM-DD")
    p.add_argument("--end", default=date.today().strftime("%Y%m%d"), help="YYYYMMDD or YYYY-MM-DD")
    p.add_argument("--src", default="SW2021", help="index_classify src, e.g. SW2021")
    p.add_argument("--source-label", default="tushare_sw_l3", help="source label to write into hist table")
    p.add_argument("--token", default=os.getenv("TUSHARE_TOKEN", "").strip(), help="Tushare token, or set env TUSHARE_TOKEN")
    p.add_argument("--sleep-ms", type=int, default=120, help="sleep milliseconds between index calls")
    p.add_argument("--skip-map", action="store_true", help="skip calling sp_build_board_member_map at the end")
    args = p.parse_args()

    token = (args.token or "").strip()
    if not token:
        raise SystemExit("Tushare token is required. Use --token or env TUSHARE_TOKEN.")

    start = _parse_ymd(args.start)
    end = _parse_ymd(args.end)
    if start > end:
        raise SystemExit(f"Invalid range: start {start} > end {end}")

    print(f"loading sw-l3 history from tushare: {start}..{end}, src={args.src}")
    codes = iter_sw_l3_codes(token, src=args.src)
    if not codes:
        raise SystemExit("No SW L3 index codes returned from Tushare.")
    print(f"l3_codes={len(codes)}")

    engine = build_engine()

    # Replace previous industry history loaded by seed/tushare to avoid overlap.
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DELETE FROM cn_board_industry_member_hist
                WHERE source IN ('seed_cons', :src)
                """
            ),
            {"src": args.source_label},
        )

    insert_sql = text(
        """
        INSERT INTO cn_board_industry_member_hist (board_id, symbol, valid_from, valid_to, source)
        VALUES (:board_id, :symbol, :valid_from, :valid_to, :source)
        ON DUPLICATE KEY UPDATE
            valid_to = VALUES(valid_to),
            source = VALUES(source),
            updated_at = CURRENT_TIMESTAMP(6)
        """
    )

    total_rows = 0
    touched_codes = 0
    buf: List[Dict] = []
    batch_size = 5000

    for i, code in enumerate(codes, start=1):
        code_rows = 0
        for m in iter_index_members(token, code, start, end):
            vf = _to_db_date(m["in_date"])
            vt = _to_db_date(m["out_date"])
            if vf is None:
                continue
            if not intersects(vf, vt, start, end):
                continue
            buf.append(
                {
                    "board_id": code,
                    "symbol": _norm_symbol(m["con_code"]),
                    "valid_from": vf,
                    "valid_to": vt,
                    "source": args.source_label,
                }
            )
            code_rows += 1
            if len(buf) >= batch_size:
                with engine.begin() as conn:
                    conn.execute(insert_sql, buf)
                total_rows += len(buf)
                buf.clear()
        if code_rows > 0:
            touched_codes += 1
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)
        if i % 50 == 0 or i == len(codes):
            print(f"progress {i}/{len(codes)} touched_codes={touched_codes} buffered={len(buf)} inserted={total_rows}")

    if buf:
        with engine.begin() as conn:
            conn.execute(insert_sql, buf)
        total_rows += len(buf)
        buf.clear()

    print(f"industry_hist_upserted={total_rows}")

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
                  (SELECT COUNT(*) FROM cn_board_industry_member_hist WHERE source=:src) AS hist_rows,
                  (SELECT MIN(valid_from) FROM cn_board_industry_member_hist WHERE source=:src) AS min_valid_from,
                  (SELECT MAX(COALESCE(valid_to, DATE('9999-12-31'))) FROM cn_board_industry_member_hist WHERE source=:src) AS max_valid_to,
                  (SELECT COUNT(*) FROM cn_board_member_map_d WHERE trade_date BETWEEN :d1 AND :d2 AND sector_type='INDUSTRY') AS map_rows
                """
            ),
            {"src": args.source_label, "d1": start, "d2": end},
        ).one()
        print(f"done hist_rows={stats[0]} min_valid_from={stats[1]} max_valid_to={stats[2]} map_rows={stats[3]} skip_map={args.skip_map}")


if __name__ == "__main__":
    main()
