from __future__ import annotations

import argparse
import json
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


def iter_sw_codes(token: str, src: str, level: str) -> List[Dict[str, str]]:
    requested_fields = "index_code,industry_name,level,industry_code,is_pub,parent_code,src"
    cols, items = ts_call(
        token=token,
        api_name="index_classify",
        params={"src": src, "level": level},
        fields=requested_fields,
    )
    if not cols:
        return []

    idx = {c: i for i, c in enumerate(cols)}
    rows: List[Dict[str, str]] = []
    for it in items:
        index_code = str(it[idx["index_code"]]).strip() if idx.get("index_code") is not None and it[idx["index_code"]] else ""
        if not index_code:
            continue
        row = {
            "index_code": index_code,
            "industry_name": str(it[idx["industry_name"]]).strip() if idx.get("industry_name") is not None and it[idx["industry_name"]] else index_code,
            "level": str(it[idx["level"]]).strip() if idx.get("level") is not None and it[idx["level"]] else level,
            "industry_code": str(it[idx["industry_code"]]).strip() if idx.get("industry_code") is not None and it[idx["industry_code"]] else "",
            "is_pub": str(it[idx["is_pub"]]).strip() if idx.get("is_pub") is not None and it[idx["is_pub"]] is not None else "",
            "parent_code": str(it[idx["parent_code"]]).strip() if idx.get("parent_code") is not None and it[idx["parent_code"]] else "",
            "src": str(it[idx["src"]]).strip() if idx.get("src") is not None and it[idx["src"]] else src,
        }
        rows.append(row)
    rows.sort(key=lambda x: x["index_code"])
    return rows


def iter_index_members(token: str, index_code: str, start: date, end: date) -> Iterable[Dict]:
    fields = "index_code,con_code,in_date,out_date,is_new"
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


def _default_master_source(src: str, level: str) -> str:
    return f"TUSHARE_{src}_{level}".upper()


def _default_member_source(level: str) -> str:
    return f"tushare_sw_{level.lower()}"


def _master_payload(code_row: Dict[str, str], asof_date: date, master_source: str) -> Dict[str, object]:
    raw_json = json.dumps(code_row, ensure_ascii=True, separators=(",", ":"))
    return {
        "board_id": code_row["index_code"],
        "board_name": code_row["industry_name"],
        "provider": "TUSHARE",
        "asof_date": asof_date,
        "source": master_source,
        "raw_json": raw_json,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backfill SW industry membership history from Tushare")
    p.add_argument("--start", default="20000101", help="YYYYMMDD or YYYY-MM-DD")
    p.add_argument("--end", default=date.today().strftime("%Y%m%d"), help="YYYYMMDD or YYYY-MM-DD")
    p.add_argument("--asof-date", default="", help="master snapshot date; default uses --end")
    p.add_argument("--src", default="SW2021", help="index_classify src, e.g. SW2021")
    p.add_argument("--level", default="L1", choices=["L1", "L2", "L3"], help="Shenwan classification level")
    p.add_argument("--master-source", default="", help="source label to write into cn_board_industry_master")
    p.add_argument("--member-source", default="", help="source label to write into cn_board_industry_member_hist")
    p.add_argument("--token", default=os.getenv("TUSHARE_TOKEN", "").strip(), help="Tushare token, or set env TUSHARE_TOKEN")
    p.add_argument("--sleep-ms", type=int, default=120, help="sleep milliseconds between index calls")
    p.add_argument("--skip-master", action="store_true", help="skip upserting cn_board_industry_master")
    p.add_argument("--skip-map", action="store_true", help="skip calling sp_build_board_member_map at the end")
    p.add_argument("--keep-existing-member-source", action="store_true", help="do not delete existing rows for the selected member source before load")
    return p


def run(args: argparse.Namespace) -> None:
    token = (args.token or "").strip()
    if not token:
        raise SystemExit("Tushare token is required. Use --token or env TUSHARE_TOKEN.")

    start = _parse_ymd(args.start)
    end = _parse_ymd(args.end)
    if start > end:
        raise SystemExit(f"Invalid range: start {start} > end {end}")

    asof_date = _parse_ymd(args.asof_date) if (args.asof_date or "").strip() else end
    level = str(args.level).upper()
    master_source = (args.master_source or "").strip() or _default_master_source(args.src, level)
    member_source = (args.member_source or "").strip() or _default_member_source(level)

    print(
        f"loading sw industry history from tushare: level={level} src={args.src} "
        f"range={start}..{end} asof_date={asof_date}"
    )
    code_rows = iter_sw_codes(token=token, src=args.src, level=level)
    if not code_rows:
        raise SystemExit(f"No SW {level} index codes returned from Tushare.")
    print(f"{level.lower()}_codes={len(code_rows)} master_source={master_source} member_source={member_source}")

    engine = build_engine()

    if not args.skip_master:
        master_sql = text(
            """
            INSERT INTO cn_board_industry_master
                (BOARD_ID, BOARD_NAME, PROVIDER, ASOF_DATE, SOURCE, CREATED_AT, RAW_JSON)
            VALUES
                (:board_id, :board_name, :provider, :asof_date, :source, NOW(), :raw_json)
            ON DUPLICATE KEY UPDATE
                BOARD_NAME = VALUES(BOARD_NAME),
                PROVIDER = VALUES(PROVIDER),
                SOURCE = VALUES(SOURCE),
                RAW_JSON = VALUES(RAW_JSON)
            """
        )
        master_rows = [_master_payload(code_row, asof_date=asof_date, master_source=master_source) for code_row in code_rows]
        with engine.begin() as conn:
            conn.execute(master_sql, master_rows)
        print(f"master_upserted={len(master_rows)}")

    if not args.keep_existing_member_source:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM cn_board_industry_member_hist
                    WHERE source = :src
                    """
                ),
                {"src": member_source},
            )
        print(f"member_source_cleared={member_source}")

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
    buf: List[Dict[str, object]] = []
    batch_size = 5000

    for i, code_row in enumerate(code_rows, start=1):
        code = code_row["index_code"]
        code_rows_inserted = 0
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
                    "source": member_source,
                }
            )
            code_rows_inserted += 1
            if len(buf) >= batch_size:
                with engine.begin() as conn:
                    conn.execute(insert_sql, buf)
                total_rows += len(buf)
                buf.clear()
        if code_rows_inserted > 0:
            touched_codes += 1
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)
        if i % 20 == 0 or i == len(code_rows):
            print(
                f"progress {i}/{len(code_rows)} touched_codes={touched_codes} "
                f"buffered={len(buf)} inserted={total_rows}"
            )

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
                  (SELECT COUNT(*) FROM cn_board_industry_master WHERE SOURCE = :master_src) AS master_rows,
                  (SELECT COUNT(DISTINCT BOARD_ID) FROM cn_board_industry_master WHERE SOURCE = :master_src) AS master_board_cnt,
                  (SELECT COUNT(*) FROM cn_board_industry_member_hist WHERE source = :member_src) AS hist_rows,
                  (SELECT COUNT(DISTINCT board_id) FROM cn_board_industry_member_hist WHERE source = :member_src) AS hist_board_cnt,
                  (SELECT MIN(valid_from) FROM cn_board_industry_member_hist WHERE source = :member_src) AS min_valid_from,
                  (SELECT MAX(COALESCE(valid_to, DATE('9999-12-31'))) FROM cn_board_industry_member_hist WHERE source = :member_src) AS max_valid_to,
                  (SELECT COUNT(*) FROM cn_board_member_map_d WHERE trade_date BETWEEN :d1 AND :d2 AND sector_type = 'INDUSTRY') AS map_rows
                """
            ),
            {"master_src": master_source, "member_src": member_source, "d1": start, "d2": end},
        ).one()
        print(
            "done "
            f"master_rows={stats[0]} master_board_cnt={stats[1]} "
            f"hist_rows={stats[2]} hist_board_cnt={stats[3]} "
            f"min_valid_from={stats[4]} max_valid_to={stats[5]} "
            f"map_rows={stats[6]} skip_map={args.skip_map}"
        )


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
