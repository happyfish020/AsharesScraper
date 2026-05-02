from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Callable

import akshare as ak
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout, RequestException
from sqlalchemy import text

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    sys.path.append(str(Path(__file__).resolve().parent))
    from app.settings import build_engine
    from config import (
        BOARD_CONCEPT_TABLE,
        BOARD_FAILURE_TABLE,
        BOARD_INDUSTRY_TABLE,
        BOARD_PATHS,
        DEFAULT_START,
        LOG_FILE,
        RETRY_DELAYS,
        SOURCE_NAME,
    )
else:
    from app.settings import build_engine
    from .config import (
        BOARD_CONCEPT_TABLE,
        BOARD_FAILURE_TABLE,
        BOARD_INDUSTRY_TABLE,
        BOARD_PATHS,
        DEFAULT_START,
        LOG_FILE,
        RETRY_DELAYS,
        SOURCE_NAME,
    )


DATE_FMT = "%Y%m%d"
OUTPUT_DATE_FMT = "%Y-%m-%d"
STOP_MESSAGE = "东方财富接口疑似限流/断连，请稍后使用 --resume-failed 继续。"
SUPPORTED_EXCEPTIONS = (
    RemoteDisconnected,
    ConnectionError,
    ReadTimeout,
    ChunkedEncodingError,
    RequestException,
)
STANDARD_COLUMNS = [
    "board_type",
    "board_code",
    "board_name",
    "trade_date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "pct_change",
    "source",
    "update_time",
]
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
}


@dataclass(frozen=True)
class BoardApi:
    board_type: str
    list_fetcher: Callable[[], pd.DataFrame]
    hist_fetcher: Callable[..., pd.DataFrame]
    list_name_candidates: tuple[str, ...]
    list_code_candidates: tuple[str, ...]
    hist_period_value: str
    list_url: str
    list_params: dict[str, str]
    hist_url: str
    table_name: str


@dataclass(frozen=True)
class RunOptions:
    start_date: str
    end_date: str
    force_refresh: bool
    resume_failed: bool
    limit: int | None
    board_name: str | None
    sleep_min: int
    sleep_max: int
    batch_cooldown_every: int
    batch_cooldown_min: int
    batch_cooldown_max: int
    load_csv: bool


@dataclass
class RetryFailure(Exception):
    action: str
    retry_count: int
    original_error: Exception

    def __str__(self) -> str:
        return f"{self.action} failed after {self.retry_count} attempts: {self.original_error}"


BOARD_APIS = {
    "industry": BoardApi(
        board_type="industry",
        list_fetcher=ak.stock_board_industry_name_em,
        hist_fetcher=ak.stock_board_industry_hist_em,
        list_name_candidates=("板块名称", "名称", "board_name"),
        list_code_candidates=("板块代码", "代码", "board_code"),
        hist_period_value="日k",
        list_url="https://17.push2.eastmoney.com/api/qt/clist/get",
        list_params={
            "pn": "1",
            "pz": "200",
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f3",
            "fs": "m:90 t:2 f:!50",
            "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f11,f62,f128,f136,f115,f152,f124,f107,f104,f105,f140,f141,f207,f208,f209,f222",
        },
        hist_url="https://7.push2his.eastmoney.com/api/qt/stock/kline/get",
        table_name=BOARD_INDUSTRY_TABLE,
    ),
    "concept": BoardApi(
        board_type="concept",
        list_fetcher=ak.stock_board_concept_name_em,
        hist_fetcher=ak.stock_board_concept_hist_em,
        list_name_candidates=("板块名称", "名称", "board_name"),
        list_code_candidates=("板块代码", "代码", "board_code"),
        hist_period_value="daily",
        list_url="https://79.push2.eastmoney.com/api/qt/clist/get",
        list_params={
            "pn": "1",
            "pz": "200",
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f12",
            "fs": "m:90 t:3 f:!50",
            "fields": "f2,f3,f4,f8,f12,f14,f15,f16,f17,f18,f20,f21,f24,f25,f22,f33,f11,f62,f128,f124,f107,f104,f105,f136",
        },
        hist_url="https://91.push2his.eastmoney.com/api/qt/stock/kline/get",
        table_name=BOARD_CONCEPT_TABLE,
    ),
}


def setup_logger() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fetch_akshare_boards")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch A-share board daily data and write directly into MySQL.")
    parser.add_argument("--type", choices=["industry", "concept", "all"], default="all")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=datetime.today().strftime(DATE_FMT))
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--resume-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-min", type=int, default=3)
    parser.add_argument("--sleep-max", type=int, default=8)
    parser.add_argument("--batch-cooldown-every", type=int, default=10)
    parser.add_argument("--batch-cooldown-min", type=int, default=30)
    parser.add_argument("--batch-cooldown-max", type=int, default=90)
    parser.add_argument("--board-name", default=None)
    parser.add_argument("--load-csv", action="store_true", help="Load existing normalized CSVs into MySQL once, without fetching")
    return parser.parse_args()


def validate_options(args: argparse.Namespace) -> RunOptions:
    datetime.strptime(args.start, DATE_FMT)
    datetime.strptime(args.end, DATE_FMT)
    if args.start > args.end:
        raise ValueError("start date cannot be greater than end date")
    if args.sleep_min < 0 or args.sleep_max < args.sleep_min:
        raise ValueError("sleep range is invalid")
    if args.batch_cooldown_min < 0 or args.batch_cooldown_max < args.batch_cooldown_min:
        raise ValueError("batch cooldown range is invalid")
    if args.batch_cooldown_every < 0:
        raise ValueError("batch cooldown every must be >= 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be > 0")
    return RunOptions(
        start_date=args.start,
        end_date=args.end,
        force_refresh=args.force_refresh,
        resume_failed=args.resume_failed,
        limit=args.limit,
        board_name=args.board_name.strip() if args.board_name else None,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        batch_cooldown_every=args.batch_cooldown_every,
        batch_cooldown_min=args.batch_cooldown_min,
        batch_cooldown_max=args.batch_cooldown_max,
        load_csv=args.load_csv,
    )


def ensure_schema(engine, logger: logging.Logger) -> None:
    logger.info("ensuring board daily tables in MySQL")
    ddls = [
        f"""
        CREATE TABLE IF NOT EXISTS {BOARD_INDUSTRY_TABLE} (
            board_code VARCHAR(32) DEFAULT NULL,
            board_name VARCHAR(80) NOT NULL,
            trade_date DATE NOT NULL,
            open DOUBLE DEFAULT NULL,
            close DOUBLE DEFAULT NULL,
            high DOUBLE DEFAULT NULL,
            low DOUBLE DEFAULT NULL,
            volume DOUBLE DEFAULT NULL,
            amount DOUBLE DEFAULT NULL,
            pct_change DOUBLE DEFAULT NULL,
            source VARCHAR(64) NOT NULL,
            update_time DATETIME NOT NULL,
            PRIMARY KEY (board_name, trade_date),
            KEY idx_{BOARD_INDUSTRY_TABLE}_date (trade_date),
            KEY idx_{BOARD_INDUSTRY_TABLE}_code (board_code)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {BOARD_CONCEPT_TABLE} (
            board_code VARCHAR(32) DEFAULT NULL,
            board_name VARCHAR(80) NOT NULL,
            trade_date DATE NOT NULL,
            open DOUBLE DEFAULT NULL,
            close DOUBLE DEFAULT NULL,
            high DOUBLE DEFAULT NULL,
            low DOUBLE DEFAULT NULL,
            volume DOUBLE DEFAULT NULL,
            amount DOUBLE DEFAULT NULL,
            pct_change DOUBLE DEFAULT NULL,
            source VARCHAR(64) NOT NULL,
            update_time DATETIME NOT NULL,
            PRIMARY KEY (board_name, trade_date),
            KEY idx_{BOARD_CONCEPT_TABLE}_date (trade_date),
            KEY idx_{BOARD_CONCEPT_TABLE}_code (board_code)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {BOARD_FAILURE_TABLE} (
            board_type VARCHAR(16) NOT NULL,
            board_name VARCHAR(80) NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            error_type VARCHAR(64) NOT NULL,
            error_message VARCHAR(500) DEFAULT NULL,
            retry_count INT NOT NULL DEFAULT 0,
            update_time DATETIME NOT NULL,
            PRIMARY KEY (board_type, board_name),
            KEY idx_{BOARD_FAILURE_TABLE}_type_time (board_type, update_time)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
        """,
    ]
    with engine.begin() as conn:
        for ddl in ddls:
            conn.execute(text(ddl))


def sleep_with_log(logger: logging.Logger, label: str, minimum: int, maximum: int) -> None:
    seconds = minimum if minimum == maximum else random.randint(minimum, maximum)
    logger.info("sleep label=%s seconds=%s", label, seconds)
    time.sleep(seconds)


def retry_call(func: Callable, logger: logging.Logger, action: str, **kwargs):
    total_attempts = len(RETRY_DELAYS) + 1
    last_error: Exception | None = None
    for attempt in range(1, total_attempts + 1):
        try:
            return func(**kwargs)
        except SUPPORTED_EXCEPTIONS as exc:
            last_error = exc
            if attempt == total_attempts:
                break
            sleep_seconds = RETRY_DELAYS[attempt - 1]
            logger.warning("retry action=%s attempt=%s/%s error_type=%s error=%s next_wait=%ss", action, attempt, total_attempts, type(exc).__name__, exc, sleep_seconds)
            logger.info("sleep label=retry action=%s seconds=%s", action, sleep_seconds)
            time.sleep(sleep_seconds)
    raise RetryFailure(action=action, retry_count=len(RETRY_DELAYS), original_error=last_error or RuntimeError("unknown retry failure"))


def find_first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def upsert_failed_item(engine, board_type: str, board_name: str, start_date: str, end_date: str, error: Exception, retry_count: int, logger: logging.Logger) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                INSERT INTO {BOARD_FAILURE_TABLE} (
                    board_type, board_name, start_date, end_date, error_type, error_message, retry_count, update_time
                )
                VALUES (:board_type, :board_name, :start_date, :end_date, :error_type, :error_message, :retry_count, :update_time)
                ON DUPLICATE KEY UPDATE
                    start_date = VALUES(start_date),
                    end_date = VALUES(end_date),
                    error_type = VALUES(error_type),
                    error_message = VALUES(error_message),
                    retry_count = VALUES(retry_count),
                    update_time = VALUES(update_time)
                """
            ),
            {
                "board_type": board_type,
                "board_name": board_name,
                "start_date": datetime.strptime(start_date, DATE_FMT).date(),
                "end_date": datetime.strptime(end_date, DATE_FMT).date(),
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
                "retry_count": retry_count,
                "update_time": datetime.now(),
            },
        )
    logger.warning("failed_saved board_type=%s board_name=%s table=%s", board_type, board_name, BOARD_FAILURE_TABLE)


def remove_failed_item(engine, board_type: str, board_name: str, logger: logging.Logger) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(f"DELETE FROM {BOARD_FAILURE_TABLE} WHERE board_type=:board_type AND board_name=:board_name"),
            {"board_type": board_type, "board_name": board_name},
        )
    logger.info("failed_cleared board_type=%s board_name=%s table=%s", board_type, board_name, BOARD_FAILURE_TABLE)


def fetch_board_list(api: BoardApi, logger: logging.Logger) -> list[dict[str, str]]:
    try:
        raw_df = retry_call(api.list_fetcher, logger=logger, action=f"fetch {api.board_type} board list via akshare")
    except RetryFailure as exc:
        logger.warning("%s board list via akshare failed: %s; fallback to direct Eastmoney API", api.board_type, exc)
        raw_df = fetch_board_list_direct(api, logger)
    if raw_df is None or raw_df.empty:
        return []
    name_col = find_first_column(raw_df, api.list_name_candidates)
    code_col = find_first_column(raw_df, api.list_code_candidates)
    if not name_col:
        raise KeyError(f"missing board name column for {api.board_type}: {list(raw_df.columns)}")
    boards: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in raw_df.iterrows():
        board_name = str(row.get(name_col, "")).strip()
        if not board_name or board_name in seen:
            continue
        seen.add(board_name)
        boards.append({"board_name": board_name, "board_code": str(row.get(code_col, "")).strip() if code_col else ""})
    return boards


def fetch_board_list_direct(api: BoardApi, logger: logging.Logger) -> pd.DataFrame:
    all_rows: list[dict] = []
    page = 1
    while True:
        params = dict(api.list_params)
        params["pn"] = str(page)
        response = retry_call(requests.get, logger=logger, action=f"fetch {api.board_type} board list direct page={page}", url=api.list_url, params=params, headers=REQUEST_HEADERS, timeout=20)
        response.raise_for_status()
        diff = ((response.json().get("data") or {}).get("diff")) or []
        if not diff:
            break
        all_rows.extend(diff)
        if len(diff) < int(params["pz"]):
            break
        page += 1
    return pd.DataFrame(all_rows).rename(columns={"f12": "板块代码", "f14": "板块名称"}) if all_rows else pd.DataFrame()


def fetch_board_history(api: BoardApi, board_name: str, board_code: str, start_date: str, end_date: str, logger: logging.Logger) -> pd.DataFrame:
    try:
        return retry_call(
            api.hist_fetcher,
            logger=logger,
            action=f"fetch {api.board_type} history via akshare board={board_name}",
            symbol=board_name,
            start_date=start_date,
            end_date=end_date,
            period=api.hist_period_value,
        )
    except RetryFailure as exc:
        logger.warning("%s history via akshare failed for %s: %s; fallback to direct Eastmoney API", api.board_type, board_name, exc)
        if not board_code:
            raise exc
        response = retry_call(
            requests.get,
            logger=logger,
            action=f"fetch {api.board_type} history direct board={board_name}",
            url=api.hist_url,
            params={
                "secid": f"90.{board_code}",
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": "101",
                "fqt": "0",
                "beg": start_date,
                "end": end_date,
                "smplmt": "10000",
                "lmt": "1000000",
            },
            headers=REQUEST_HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        klines = ((response.json().get("data") or {}).get("klines")) or []
        if not klines:
            return pd.DataFrame()
        df = pd.DataFrame([item.split(",") for item in klines])
        df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
        return df


def standardize_history(raw_df: pd.DataFrame, board_type: str, board_code: str, board_name: str, update_time: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    renamed = raw_df.rename(columns={"日期": "trade_date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_change"}).copy()
    for column in ("trade_date", "open", "close", "high", "low", "volume", "amount", "pct_change"):
        if column not in renamed.columns:
            renamed[column] = pd.NA
    out = renamed[["trade_date", "open", "close", "high", "low", "volume", "amount", "pct_change"]].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.strftime(OUTPUT_DATE_FMT)
    out = out[out["trade_date"].notna()].copy()
    for column in ("open", "close", "high", "low", "volume", "amount", "pct_change"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out.insert(0, "board_name", board_name)
    out.insert(0, "board_code", board_code)
    out.insert(0, "board_type", board_type)
    out["source"] = SOURCE_NAME
    out["update_time"] = update_time
    return out[STANDARD_COLUMNS]


def resolve_db_start_date(engine, api: BoardApi, board_name: str, cli_start: str, force_refresh: bool) -> str:
    if force_refresh:
        return cli_start
    with engine.connect() as conn:
        max_date = conn.execute(text(f"SELECT MAX(trade_date) FROM {api.table_name} WHERE board_name=:board_name"), {"board_name": board_name}).scalar()
    if max_date is None:
        return cli_start
    return max(cli_start, datetime.strftime(max_date, DATE_FMT))


def save_board_to_db(engine, api: BoardApi, board_df: pd.DataFrame, logger: logging.Logger) -> int:
    if board_df.empty:
        return 0
    rows = []
    for row in board_df.to_dict(orient="records"):
        rows.append(
            {
                "board_code": row["board_code"] or None,
                "board_name": row["board_name"],
                "trade_date": datetime.strptime(str(row["trade_date"]), OUTPUT_DATE_FMT).date(),
                "open": None if pd.isna(row["open"]) else float(row["open"]),
                "close": None if pd.isna(row["close"]) else float(row["close"]),
                "high": None if pd.isna(row["high"]) else float(row["high"]),
                "low": None if pd.isna(row["low"]) else float(row["low"]),
                "volume": None if pd.isna(row["volume"]) else float(row["volume"]),
                "amount": None if pd.isna(row["amount"]) else float(row["amount"]),
                "pct_change": None if pd.isna(row["pct_change"]) else float(row["pct_change"]),
                "source": row["source"],
                "update_time": datetime.strptime(str(row["update_time"]), "%Y-%m-%d %H:%M:%S"),
            }
        )
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                INSERT INTO {api.table_name} (
                    board_code, board_name, trade_date, open, close, high, low, volume, amount, pct_change, source, update_time
                )
                VALUES (
                    :board_code, :board_name, :trade_date, :open, :close, :high, :low, :volume, :amount, :pct_change, :source, :update_time
                )
                ON DUPLICATE KEY UPDATE
                    board_code = VALUES(board_code),
                    open = VALUES(open),
                    close = VALUES(close),
                    high = VALUES(high),
                    low = VALUES(low),
                    volume = VALUES(volume),
                    amount = VALUES(amount),
                    pct_change = VALUES(pct_change),
                    source = VALUES(source),
                    update_time = VALUES(update_time)
                """
            ),
            rows,
        )
    logger.info("db_upsert table=%s rows=%s board_name=%s", api.table_name, len(rows), board_df.iloc[0]["board_name"])
    return len(rows)


def load_existing_csvs(engine, api: BoardApi, logger: logging.Logger) -> int:
    csv_dir = BOARD_PATHS[api.board_type].normalized_dir
    if not csv_dir.exists():
        logger.info("csv_load skipped board_type=%s dir_missing=%s", api.board_type, csv_dir)
        return 0
    total_rows = 0
    for path in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "board_type" not in df.columns:
            df["board_type"] = api.board_type
        if "board_code" not in df.columns:
            df["board_code"] = None
        if "source" not in df.columns:
            df["source"] = SOURCE_NAME
        if "update_time" not in df.columns:
            df["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = df[[col for col in STANDARD_COLUMNS if col in df.columns]].copy()
        missing_cols = [col for col in STANDARD_COLUMNS if col not in df.columns]
        for col in missing_cols:
            df[col] = None
        df = df[STANDARD_COLUMNS]
        total_rows += save_board_to_db(engine, api, df, logger)
        logger.info("csv_loaded table=%s file=%s rows=%s", api.table_name, path, len(df))
    return total_rows


def build_board_queue(api: BoardApi, options: RunOptions, engine, logger: logging.Logger) -> list[dict[str, str]]:
    boards = fetch_board_list(api, logger)
    board_map = {item["board_name"]: item for item in boards}
    if options.resume_failed:
        with engine.connect() as conn:
            failed_rows = conn.execute(text(f"SELECT board_name FROM {BOARD_FAILURE_TABLE} WHERE board_type=:board_type ORDER BY update_time DESC"), {"board_type": api.board_type}).fetchall()
        failed_names = [row[0] for row in failed_rows]
        logger.info("resume_failed board_type=%s failed_count=%s", api.board_type, len(failed_names))
        boards = [board_map[name] for name in failed_names if name in board_map]
    if options.board_name:
        boards = [item for item in boards if item["board_name"] == options.board_name]
        logger.info("board_name_filter board_type=%s board_name=%s matched=%s", api.board_type, options.board_name, len(boards))
    if options.limit is not None:
        boards = boards[: options.limit]
        logger.info("limit_applied board_type=%s limit=%s remaining=%s", api.board_type, options.limit, len(boards))
    return boards


def process_board(engine, api: BoardApi, board: dict[str, str], options: RunOptions, logger: logging.Logger) -> tuple[str, int]:
    board_name = board["board_name"]
    board_code = board.get("board_code", "")
    effective_start = resolve_db_start_date(engine, api, board_name, options.start_date, options.force_refresh)
    if not options.force_refresh and effective_start >= options.end_date:
        logger.info("[%s] %s skipped reason=no_new_dates table=%s", api.board_type, board_name, api.table_name)
        remove_failed_item(engine, api.board_type, board_name, logger)
        return "skipped", 0
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        raw_df = fetch_board_history(api, board_name, board_code, effective_start, options.end_date, logger)
    except RetryFailure as exc:
        upsert_failed_item(engine, api.board_type, board_name, effective_start, options.end_date, exc.original_error, exc.retry_count, logger)
        return "failed", exc.retry_count
    except Exception as exc:
        upsert_failed_item(engine, api.board_type, board_name, effective_start, options.end_date, exc, 0, logger)
        return "failed", 0
    if raw_df is None or raw_df.empty:
        upsert_failed_item(engine, api.board_type, board_name, effective_start, options.end_date, ValueError("empty_result"), 0, logger)
        return "failed", 0
    normalized_df = standardize_history(raw_df, api.board_type, board_code, board_name, update_time)
    if normalized_df.empty:
        upsert_failed_item(engine, api.board_type, board_name, effective_start, options.end_date, ValueError("normalization_empty"), 0, logger)
        return "failed", 0
    row_count = save_board_to_db(engine, api, normalized_df, logger)
    remove_failed_item(engine, api.board_type, board_name, logger)
    logger.info("[%s] %s saved rows=%s table=%s", api.board_type, board_name, row_count, api.table_name)
    return "saved", row_count


def run_for_type(engine, board_type: str, options: RunOptions, logger: logging.Logger) -> bool:
    api = BOARD_APIS[board_type]
    if options.load_csv:
        loaded_rows = load_existing_csvs(engine, api, logger)
        logger.info("csv_load_finished board_type=%s loaded_rows=%s table=%s", board_type, loaded_rows, api.table_name)
        return True
    boards = build_board_queue(api, options, engine, logger)
    logger.info("start board_type=%s source=%s start_date=%s end_date=%s table=%s total_boards=%s", board_type, SOURCE_NAME, options.start_date, options.end_date, api.table_name, len(boards))
    saved_count = 0
    skipped_count = 0
    failed_count = 0
    consecutive_failures = 0
    for board in boards:
        try:
            status, _ = process_board(engine, api, board, options, logger)
        except Exception as exc:
            logger.exception("[%s] %s unexpected_error=%s", board_type, board["board_name"], exc)
            upsert_failed_item(engine, board_type, board["board_name"], options.start_date, options.end_date, exc, 0, logger)
            status = "failed"
        if status == "saved":
            saved_count += 1
            consecutive_failures = 0
            sleep_with_log(logger, f"{board_type}_per_board", options.sleep_min, options.sleep_max)
            if options.batch_cooldown_every > 0 and saved_count % options.batch_cooldown_every == 0:
                sleep_with_log(logger, f"{board_type}_batch_cooldown", options.batch_cooldown_min, options.batch_cooldown_max)
        elif status == "skipped":
            skipped_count += 1
            consecutive_failures = 0
        else:
            failed_count += 1
            consecutive_failures += 1
            logger.warning("failure_streak board_type=%s current=%s threshold=5", board_type, consecutive_failures)
            if consecutive_failures >= 5:
                logger.error("stop_early board_type=%s reason=consecutive_failures message=%s", board_type, STOP_MESSAGE)
                break
    logger.info("finish board_type=%s saved=%s skipped=%s failed=%s", board_type, saved_count, skipped_count, failed_count)
    return consecutive_failures < 5


def main() -> int:
    logger = setup_logger()
    args = parse_args()
    try:
        options = validate_options(args)
    except ValueError as exc:
        logger.error("invalid_parameter error=%s", exc)
        return 1
    engine = build_engine()
    ensure_schema(engine, logger)
    logger.info("job_started source=%s akshare_version=%s load_csv=%s", SOURCE_NAME, ak.__version__, options.load_csv)
    run_types = ["industry", "concept"] if args.type == "all" else [args.type]
    exit_code = 0
    for board_type in run_types:
        try:
            keep_running = run_for_type(engine, board_type, options, logger)
        except Exception as exc:
            logger.exception("board_type=%s terminated_unexpectedly error=%s", board_type, exc)
            logger.error(STOP_MESSAGE)
            exit_code = 1
            break
        if not keep_running:
            logger.error(STOP_MESSAGE)
            exit_code = 1
            break
    logger.info("job_finished exit_code=%s", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
