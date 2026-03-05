from __future__ import annotations

import argparse
import configparser
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Semaphore
from threading import local
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import tushare as ts
from sqlalchemy import text

from app.settings import build_engine

DEFAULT_TUSHARE_TOKEN = "16acec7cb9250190ecd69818107d7e4307a097235248830225a432fa"
_PANDAS_FILLNA_PATCHED = False


def patch_pandas_fillna_method_compat() -> None:
    """
    Compat shim for pandas>=3 where NDFrame.fillna(method=...) is removed.
    Some tushare internals still call fillna(method='ffill'/'bfill').
    """
    global _PANDAS_FILLNA_PATCHED
    if _PANDAS_FILLNA_PATCHED:
        return

    ndframe_cls = getattr(pd.core.generic, "NDFrame", None)
    if ndframe_cls is None:
        return
    old_fillna = ndframe_cls.fillna

    def _fillna_compat(self, *args, **kwargs):
        method = kwargs.pop("method", None)
        axis = kwargs.pop("axis", None)
        inplace = bool(kwargs.get("inplace", False))
        limit = kwargs.pop("limit", None)

        if method is not None:
            m = str(method).lower()
            # pandas old behavior: value+method cannot be both set
            if kwargs.get("value", None) is not None or args:
                raise ValueError("Cannot specify both 'value' and 'method'.")
            if m in {"ffill", "pad"}:
                return self.ffill(axis=axis, inplace=inplace, limit=limit)
            if m in {"bfill", "backfill"}:
                return self.bfill(axis=axis, inplace=inplace, limit=limit)
            raise ValueError(f"Invalid fillna method: {method}")

        return old_fillna(self, *args, **kwargs)

    ndframe_cls.fillna = _fillna_compat
    _PANDAS_FILLNA_PATCHED = True


def _parse_month(s: str) -> date:
    v = (s or "").strip()
    if len(v) != 7 or v[4] != "-":
        raise ValueError(f"Invalid month format: {s!r}, expected YYYY-MM")
    return datetime.strptime(v, "%Y-%m").date().replace(day=1)


def _month_end(d: date) -> date:
    if d.month == 12:
        return d.replace(month=12, day=31)
    return d.replace(month=d.month + 1, day=1) - timedelta(days=1)


def _iter_months(start_month: date, end_month: date) -> Iterable[Tuple[date, date]]:
    cur = start_month
    while cur <= end_month:
        yield cur, _month_end(cur)
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            cur = cur.replace(month=cur.month + 1, day=1)


def _to_ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _exchange_from_ts_code(ts_code: str) -> str | None:
    parts = (ts_code or "").split(".")
    if len(parts) != 2:
        return None
    sfx = parts[1].upper()
    if sfx == "SH":
        return "SSE"
    if sfx == "SZ":
        return "SZSE"
    if sfx == "BJ":
        return "BJSE"
    return None


def _read_token_from_env_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip().upper()
        if key in {"TUSHARE_TOKEN", "TS_TOKEN"}:
            return v.strip().strip("'\"")
    return ""


def _read_token_from_ini(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    cp = configparser.ConfigParser()
    cp.read(path, encoding="utf-8")

    key_candidates = [
        "TUSHARE_TOKEN",
        "tushare_token",
        "ts_token",
        "token",
    ]
    section_candidates = [
        "tushare",
        "api",
        "auth",
        "default",
        "DEFAULT",
    ]

    for sec in section_candidates:
        if sec in cp:
            for k in key_candidates:
                if k in cp[sec]:
                    return str(cp[sec][k]).strip().strip("'\"")
    for sec in cp.sections():
        for k in key_candidates:
            if k in cp[sec]:
                return str(cp[sec][k]).strip().strip("'\"")
    if "DEFAULT" in cp:
        for k in key_candidates:
            if k in cp["DEFAULT"]:
                return str(cp["DEFAULT"][k]).strip().strip("'\"")
    return ""


def _read_token_from_json(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""

    direct_keys = ["TUSHARE_TOKEN", "tushare_token", "ts_token", "token"]
    for k in direct_keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    tushare_obj = obj.get("tushare")
    if isinstance(tushare_obj, dict):
        for k in ("token", "TUSHARE_TOKEN", "tushare_token", "ts_token"):
            v = tushare_obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def resolve_tushare_token(cli_token: str, config_path: str) -> tuple[str, List[Path]]:
    token = (cli_token or "").strip()
    if token:
        return token, []

    token = os.getenv("TUSHARE_TOKEN", "").strip() or os.getenv("TS_TOKEN", "").strip()
    if token:
        return token, []

    root = Path.cwd()
    candidates: List[Path] = []

    user_cfg = (config_path or "").strip()
    if user_cfg:
        candidates.append(Path(user_cfg).expanduser())

    env_cfg = os.getenv("ASHARE_CONFIG", "").strip()
    if env_cfg:
        candidates.append(Path(env_cfg).expanduser())

    candidates.extend(
        [
            root / ".env",
            root / "config.ini",
            root / "settings.ini",
            root / "app.ini",
            root / "config" / "config.ini",
            root / "config" / "settings.ini",
            root / "config" / "app.ini",
            root / "config.json",
            root / "settings.json",
        ]
    )

    seen = set()
    tried: List[Path] = []
    for p in candidates:
        pp = p.resolve()
        if pp in seen:
            continue
        seen.add(pp)
        tried.append(pp)
        suffix = pp.suffix.lower()
        if pp.name.lower() == ".env":
            token = _read_token_from_env_file(pp)
        elif suffix in {".ini", ".cfg", ".conf"}:
            token = _read_token_from_ini(pp)
        elif suffix == ".json":
            token = _read_token_from_json(pp)
        else:
            token = _read_token_from_env_file(pp) or _read_token_from_ini(pp) or _read_token_from_json(pp)
        if token:
            return token, tried
    if DEFAULT_TUSHARE_TOKEN:
        return DEFAULT_TUSHARE_TOKEN, tried
    return "", tried


@dataclass
class TushareStockDailySync:
    token: str
    sleep_ms: int = 100
    source_label: str = "tushare_qfq"
    fetch_retries: int = 5
    api_concurrency: int = 1

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("Tushare token is required, use --token or TUSHARE_TOKEN env")
        patch_pandas_fillna_method_compat()
        ts.set_token(self.token)
        self._tls = local()
        self._api_sem = Semaphore(max(1, int(self.api_concurrency)))
        self.engine = build_engine()
        self._upsert_cols = [
            "symbol",
            "trade_date",
            "open",
            "close",
            "pre_close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "chg_pct",
            "change",
            "turnover_rate",
            "source",
            "window_start",
            "exchange",
            "name",
        ]

    def _get_thread_pro(self):
        pro = getattr(self._tls, "pro", None)
        if pro is None:
            ts.set_token(self.token)
            pro = ts.pro_api()
            self._tls.pro = pro
        return pro

    def list_a_share_universe(self) -> pd.DataFrame:
        pro = self._get_thread_pro()
        frames: List[pd.DataFrame] = []
        for st in ("L", "D", "P"):
            with self._api_sem:
                df = pro.stock_basic(
                    exchange="",
                    list_status=st,
                    fields="ts_code,symbol,name,exchange,list_status",
                )
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["ts_code", "symbol", "name", "exchange", "list_status"])
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["ts_code"]).sort_values("ts_code").reset_index(drop=True)
        return out

    def fetch_qfq_daily(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        last_err: Exception | None = None
        for i in range(max(1, int(self.fetch_retries))):
            try:
                with self._api_sem:
                    df = ts.pro_bar(
                        ts_code=ts_code,
                        adj="qfq",
                        start_date=_to_ymd(start),
                        end_date=_to_ymd(end),
                        factors=["tor"],
                        asset="E",
                        freq="D",
                    )
                break
            except Exception as e:
                last_err = e
                # Recreate thread-local client after network/protocol errors.
                self._tls.pro = None
                wait_s = min(8.0, 0.8 * (2**i))
                time.sleep(wait_s)
        else:
            # Exhausted retries.
            print(f"[WARN] fetch failed ts_code={ts_code} start={start} end={end} err={last_err}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce").dt.date
        out = out.dropna(subset=["trade_date"])
        out["symbol"] = out["ts_code"].astype(str).str.split(".").str[0]
        out["exchange"] = out["ts_code"].map(_exchange_from_ts_code)
        if "pct_chg" in out.columns:
            out["chg_pct"] = pd.to_numeric(out["pct_chg"], errors="coerce")
        else:
            out["chg_pct"] = None
        if "turnover_rate" in out.columns:
            out["turnover_rate"] = pd.to_numeric(out["turnover_rate"], errors="coerce")
        elif "tor" in out.columns:
            out["turnover_rate"] = pd.to_numeric(out["tor"], errors="coerce")
        else:
            out["turnover_rate"] = None

        for c in ("open", "close", "pre_close", "high", "low", "vol", "amount", "change", "chg_pct", "turnover_rate"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        out["volume"] = out.get("vol")
        out["amplitude"] = ((out["high"] - out["low"]) / out["pre_close"] * 100.0).where(out["pre_close"] > 0)
        return out

    def upsert_rows(self, rows: List[Dict]) -> int:
        if not rows:
            return 0
        sql = text(
            """
            INSERT INTO cn_stock_daily_price
            (`symbol`, `trade_date`, `open`, `close`, `pre_close`, `high`, `low`, `volume`, `amount`,
             `amplitude`, `chg_pct`, `change`, `turnover_rate`, `source`, `window_start`, `exchange`, `name`)
            VALUES
            (:symbol, :trade_date, :open, :close, :pre_close, :high, :low, :volume, :amount,
             :amplitude, :chg_pct, :change, :turnover_rate, :source, :window_start, :exchange, :name)
            ON DUPLICATE KEY UPDATE
                `open` = VALUES(`open`),
                `close` = VALUES(`close`),
                `pre_close` = VALUES(`pre_close`),
                `high` = VALUES(`high`),
                `low` = VALUES(`low`),
                `volume` = VALUES(`volume`),
                `amount` = VALUES(`amount`),
                `amplitude` = VALUES(`amplitude`),
                `chg_pct` = VALUES(`chg_pct`),
                `change` = VALUES(`change`),
                `turnover_rate` = VALUES(`turnover_rate`),
                `source` = VALUES(`source`),
                `window_start` = VALUES(`window_start`),
                `exchange` = COALESCE(VALUES(`exchange`), `exchange`),
                `name` = COALESCE(VALUES(`name`), `name`)
            """
        )
        with self.engine.begin() as conn:
            ret = conn.execute(sql, rows)
            return int(ret.rowcount or 0)

    def _build_month_frame(self, month_start: date, month_end: date, universe: pd.DataFrame, symbol_limit: int = 0) -> pd.DataFrame:
        parts: List[pd.DataFrame] = []
        if symbol_limit > 0:
            universe = universe.head(symbol_limit).copy()

        month_str = month_start.strftime("%Y-%m")
        for i, rec in enumerate(universe.itertuples(index=False), start=1):
            ts_code = str(rec.ts_code)
            name = str(rec.name) if rec.name is not None else None
            try:
                df = self.fetch_qfq_daily(ts_code=ts_code, start=month_start, end=month_end)
            except Exception as e:
                print(f"[WARN] [{month_str}] ts_code={ts_code} unexpected error={e}")
                df = pd.DataFrame()
            if df.empty:
                if self.sleep_ms > 0:
                    time.sleep(self.sleep_ms / 1000.0)
                continue

            df["source"] = self.source_label
            df["window_start"] = month_start
            df["name"] = name
            for c in self._upsert_cols:
                if c not in df.columns:
                    df[c] = None
            parts.append(df[self._upsert_cols].copy())
            if i % 200 == 0 or i == len(universe):
                print(f"[{month_str}] fetch progress {i}/{len(universe)} parts={len(parts)}")
            if self.sleep_ms > 0:
                time.sleep(self.sleep_ms / 1000.0)

        if not parts:
            return pd.DataFrame(columns=self._upsert_cols)
        out = pd.concat(parts, ignore_index=True)
        out = out.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
        return out

    def _upsert_dataframe(self, df: pd.DataFrame, chunk_size: int = 5000) -> int:
        if df is None or df.empty:
            return 0
        work = df.copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.date
        work["window_start"] = pd.to_datetime(work["window_start"], errors="coerce").dt.date
        work = work.dropna(subset=["symbol", "trade_date"])
        if work.empty:
            return 0

        total = 0
        # PyMySQL does not accept float('nan'); normalize all missing values to None.
        records: List[Dict] = []
        for rec in work[self._upsert_cols].to_dict(orient="records"):
            clean = {}
            for k, v in rec.items():
                clean[k] = None if pd.isna(v) else v
            records.append(clean)
        for i in range(0, len(records), chunk_size):
            total += self.upsert_rows(records[i : i + chunk_size])
        return total

    def sync_month(self, month_start: date, month_end: date, universe: pd.DataFrame, csv_path: Path, symbol_limit: int = 0, reuse_csv: bool = True) -> Dict[str, int]:
        touched_symbols = 0
        touched_rows = 0
        total_exec_rowcount = 0

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        month_str = month_start.strftime("%Y-%m")
        use_existing = reuse_csv and csv_path.exists()

        if use_existing:
            month_df = pd.read_csv(csv_path)
            for c in self._upsert_cols:
                if c not in month_df.columns:
                    month_df[c] = None
            month_df = month_df[self._upsert_cols]
            print(f"[{month_str}] reuse csv={csv_path} rows={len(month_df)}")
        else:
            month_df = self._build_month_frame(month_start, month_end, universe, symbol_limit=symbol_limit)
            month_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"[{month_str}] wrote csv={csv_path} rows={len(month_df)}")

        if not month_df.empty:
            touched_symbols = int(month_df["symbol"].nunique(dropna=True))
            touched_rows = int(len(month_df))
            total_exec_rowcount = self._upsert_dataframe(month_df)

        return {
            "month": month_str,
            "csv_path": str(csv_path),
            "symbols_in_universe": int(len(universe.head(symbol_limit)) if symbol_limit > 0 else len(universe)),
            "touched_symbols": int(touched_symbols),
            "touched_rows": int(touched_rows),
            "exec_rowcount": int(total_exec_rowcount),
        }

    @staticmethod
    def _load_state(state_file: Path) -> Dict:
        if not state_file.exists():
            return {"completed_symbols": [], "failed_symbols": {}}
        try:
            obj = json.loads(state_file.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                return {"completed_symbols": [], "failed_symbols": {}}
            completed = obj.get("completed_symbols") or []
            failed = obj.get("failed_symbols") or {}
            if not isinstance(completed, list):
                completed = []
            if not isinstance(failed, dict):
                failed = {}
            return {"completed_symbols": completed, "failed_symbols": failed}
        except Exception:
            return {"completed_symbols": [], "failed_symbols": {}}

    @staticmethod
    def _save_state(state_file: Path, completed_symbols: set[str], failed_symbols: Dict[str, str]) -> None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "completed_symbols": sorted(completed_symbols),
            "failed_symbols": failed_symbols,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def sync_by_symbol_batches(
        self,
        universe: pd.DataFrame,
        start: date,
        end: date,
        csv_dir: Path,
        batch_size: int,
        reuse_csv: bool,
        state_file: Path,
        symbol_limit: int = 0,
    ) -> Dict[str, int]:
        csv_dir.mkdir(parents=True, exist_ok=True)
        state = self._load_state(state_file)
        completed_symbols = set(str(x) for x in state.get("completed_symbols", []))
        failed_symbols = dict(state.get("failed_symbols", {}))

        work = universe.sort_values(["symbol", "ts_code"]).reset_index(drop=True)
        if symbol_limit > 0:
            work = work.head(symbol_limit).copy()

        total = len(work)
        done_this_run = 0
        rows_this_run = 0
        db_rowcount = 0
        processed_since_flush = 0

        start_ymd = _to_ymd(start)
        end_ymd = _to_ymd(end)

        for i, rec in enumerate(work.itertuples(index=False), start=1):
            ts_code = str(rec.ts_code)
            symbol = str(rec.symbol)
            name = str(rec.name) if rec.name is not None else None
            if symbol in completed_symbols:
                if i % 500 == 0 or i == total:
                    print(f"[resume] progress {i}/{total} skip_completed={len(completed_symbols)}")
                continue

            csv_path = csv_dir / f"{symbol}.csv"
            try:
                use_existing = reuse_csv and csv_path.exists()
                if use_existing:
                    df = pd.read_csv(csv_path)
                    for c in self._upsert_cols:
                        if c not in df.columns:
                            df[c] = None
                    df = df[self._upsert_cols]
                else:
                    raw = self.fetch_qfq_daily(ts_code=ts_code, start=start, end=end)
                    if raw.empty:
                        df = pd.DataFrame(columns=self._upsert_cols)
                    else:
                        raw["source"] = self.source_label
                        raw["window_start"] = start
                        raw["name"] = name
                        for c in self._upsert_cols:
                            if c not in raw.columns:
                                raw[c] = None
                        df = raw[self._upsert_cols].copy()
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

                # Optional date-range filter in case reused CSV contains broader range.
                if not df.empty:
                    td = pd.to_datetime(df["trade_date"], errors="coerce")
                    mask = (td >= pd.to_datetime(start_ymd)) & (td <= pd.to_datetime(end_ymd))
                    df = df.loc[mask].copy()

                affected = self._upsert_dataframe(df)
                completed_symbols.add(symbol)
                failed_symbols.pop(symbol, None)
                done_this_run += 1
                rows_this_run += int(len(df))
                db_rowcount += int(affected)
                processed_since_flush += 1
                print(
                    f"[{i}/{total}] {symbol} ok rows={len(df)} db_rowcount={affected} "
                    f"csv={csv_path.name} reuse={use_existing}"
                )
            except Exception as e:
                failed_symbols[symbol] = str(e)
                processed_since_flush += 1
                print(f"[{i}/{total}] {symbol} failed err={e}")

            if self.sleep_ms > 0:
                time.sleep(self.sleep_ms / 1000.0)

            if processed_since_flush >= max(1, int(batch_size)):
                self._save_state(state_file, completed_symbols, failed_symbols)
                print(
                    f"[checkpoint] saved batch={processed_since_flush} completed={len(completed_symbols)} "
                    f"failed={len(failed_symbols)} state={state_file}"
                )
                processed_since_flush = 0

        self._save_state(state_file, completed_symbols, failed_symbols)
        return {
            "total_symbols": total,
            "done_this_run": done_this_run,
            "rows_this_run": rows_this_run,
            "db_rowcount": db_rowcount,
            "completed_total": len(completed_symbols),
            "failed_total": len(failed_symbols),
            "state_file": str(state_file),
            "csv_dir": str(csv_dir),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync cn_stock_daily_price from Tushare (qfq) by sorted stock pool. Save per-symbol CSV and support checkpoint resume."
    )
    parser.add_argument("--start-month", default="2026-01", help="YYYY-MM")
    parser.add_argument("--end-month", default="2026-12", help="YYYY-MM")
    parser.add_argument("--token", default="", help="Tushare token")
    parser.add_argument("--config", default="", help="Config file path (.env/.ini/.cfg/.json) to read token")
    parser.add_argument("--sleep-ms", type=int, default=100, help="Sleep milliseconds between symbols")
    parser.add_argument("--symbol-limit", type=int, default=0, help="For trial run: only process first N symbols")
    parser.add_argument("--source-label", default="tushare_qfq", help="source value written to cn_stock_daily_price")
    parser.add_argument("--api-concurrency", type=int, default=1, help="Global concurrent Tushare API calls")
    parser.add_argument("--csv-dir", default="data/tushare_by_symbol", help="Directory to save per-symbol CSV files")
    parser.add_argument("--reuse-csv", dest="reuse_csv", action="store_true", help="Reuse existing per-symbol CSV (default)")
    parser.add_argument("--no-reuse-csv", dest="reuse_csv", action="store_false", help="Ignore existing CSV and fetch again")
    parser.add_argument("--fetch-retries", type=int, default=5, help="Retry times for per-symbol tushare fetch")
    parser.add_argument("--batch-size", type=int, default=30, help="Checkpoint flush interval by symbols")
    parser.add_argument("--state-file", default="state/tushare_stock_qfq_state.json", help="Resume state JSON file")
    parser.set_defaults(reuse_csv=True)
    args = parser.parse_args()

    start_month = _parse_month(args.start_month)
    end_month = _parse_month(args.end_month)
    if start_month > end_month:
        raise SystemExit(f"Invalid month range: {start_month} > {end_month}")

    token, tried_files = resolve_tushare_token(args.token, args.config)
    if not token:
        msg = "Tushare token is required. Use --token, env TUSHARE_TOKEN/TS_TOKEN, or --config."
        if tried_files:
            msg += f" Tried files: {', '.join(str(p) for p in tried_files)}"
        raise SystemExit(msg)

    syncer = TushareStockDailySync(
        token=token,
        sleep_ms=max(0, int(args.sleep_ms)),
        source_label=(args.source_label or "tushare_qfq").strip(),
        fetch_retries=max(1, int(args.fetch_retries)),
        api_concurrency=max(1, int(args.api_concurrency)),
    )

    universe = syncer.list_a_share_universe()
    if universe.empty:
        raise SystemExit("No A-share symbols from Tushare stock_basic.")
    print(f"universe_size={len(universe)} month_range={start_month.strftime('%Y-%m')}..{end_month.strftime('%Y-%m')}")

    reuse_csv = bool(args.reuse_csv)
    csv_dir = Path(args.csv_dir)
    state_file = Path(args.state_file)
    start_date = start_month
    end_date = _month_end(end_month)

    print(
        f"by-symbol mode csv_dir={csv_dir} reuse_csv={reuse_csv} "
        f"batch_size={max(1, int(args.batch_size))} state_file={state_file}"
    )
    stats = syncer.sync_by_symbol_batches(
        universe=universe,
        start=start_date,
        end=end_date,
        csv_dir=csv_dir,
        batch_size=max(1, int(args.batch_size)),
        reuse_csv=reuse_csv,
        state_file=state_file,
        symbol_limit=max(0, int(args.symbol_limit)),
    )
    print(
        f"all_done total_symbols={stats['total_symbols']} done_this_run={stats['done_this_run']} "
        f"rows_this_run={stats['rows_this_run']} db_rowcount={stats['db_rowcount']} "
        f"completed_total={stats['completed_total']} failed_total={stats['failed_total']} "
        f"state_file={stats['state_file']} csv_dir={stats['csv_dir']}"
    )


if __name__ == "__main__":
    main()
