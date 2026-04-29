from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Callable

import akshare as ak
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout, RequestException

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from config import BOARD_PATHS, DEFAULT_START, FAILED_ROOT, LOG_FILE, RETRY_DELAYS, SOURCE_NAME
else:
    from .config import BOARD_PATHS, DEFAULT_START, FAILED_ROOT, LOG_FILE, RETRY_DELAYS, SOURCE_NAME


INVALID_FILENAME = re.compile(r'[\\/:*?"<>|]')
DATE_FMT = "%Y%m%d"
OUTPUT_DATE_FMT = "%Y-%m-%d"
SUPPORTED_EXCEPTIONS = (
    RemoteDisconnected,
    ConnectionError,
    ReadTimeout,
    ChunkedEncodingError,
    RequestException,
)
STANDARD_COLUMNS = [
    "board_type",
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
FAILED_COLUMNS = [
    "board_type",
    "board_name",
    "start_date",
    "end_date",
    "error_type",
    "error_message",
    "retry_count",
    "update_time",
]
STOP_MESSAGE = "东方财富接口疑似限流/断连，请稍后使用 --resume-failed 继续。"
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
    parser = argparse.ArgumentParser(description="Fetch A-share board daily data in low-frequency recoverable mode.")
    parser.add_argument("--type", choices=["industry", "concept", "all"], default="all")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date in YYYYMMDD, default: 20200101")
    parser.add_argument("--end", default=datetime.today().strftime(DATE_FMT), help="End date in YYYYMMDD, default: today")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore local cache and refetch full history")
    parser.add_argument("--resume-failed", action="store_true", help="Only rerun boards listed in failed csv")
    parser.add_argument("--limit", type=int, default=None, help="Only fetch first N boards after filtering")
    parser.add_argument("--sleep-min", type=int, default=3, help="Sleep minimum seconds after each successful board")
    parser.add_argument("--sleep-max", type=int, default=8, help="Sleep maximum seconds after each successful board")
    parser.add_argument("--batch-cooldown-every", type=int, default=10, help="Extra cooldown after every N successful boards")
    parser.add_argument("--batch-cooldown-min", type=int, default=30, help="Batch cooldown minimum seconds")
    parser.add_argument("--batch-cooldown-max", type=int, default=90, help="Batch cooldown maximum seconds")
    parser.add_argument("--board-name", default=None, help="Only fetch a single board by exact board name")
    return parser.parse_args()


def validate_date(text: str) -> datetime:
    return datetime.strptime(text, DATE_FMT)


def validate_options(args: argparse.Namespace) -> RunOptions:
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
    )


def sanitize_filename(name: str) -> str:
    cleaned = INVALID_FILENAME.sub("_", str(name)).strip().strip(".")
    return cleaned or "unnamed_board"


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
            logger.warning(
                "retry action=%s attempt=%s/%s error_type=%s error=%s next_wait=%ss",
                action,
                attempt,
                total_attempts,
                type(exc).__name__,
                exc,
                sleep_seconds,
            )
            logger.info("sleep label=retry action=%s seconds=%s", action, sleep_seconds)
            time.sleep(sleep_seconds)
        except Exception:
            raise
    raise RetryFailure(action=action, retry_count=len(RETRY_DELAYS), original_error=last_error or RuntimeError("unknown retry failure"))


def find_first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def failed_csv_path(board_type: str) -> Path:
    FAILED_ROOT.mkdir(parents=True, exist_ok=True)
    return FAILED_ROOT / f"{board_type}_failed.csv"


def load_failed_items(board_type: str) -> pd.DataFrame:
    path = failed_csv_path(board_type)
    if not path.exists():
        return pd.DataFrame(columns=FAILED_COLUMNS)
    return pd.read_csv(path, dtype=str)


def upsert_failed_item(
    board_type: str,
    board_name: str,
    start_date: str,
    end_date: str,
    error: Exception,
    retry_count: int,
    logger: logging.Logger,
) -> None:
    path = failed_csv_path(board_type)
    failed_df = load_failed_items(board_type)
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = pd.DataFrame(
        [
            {
                "board_type": board_type,
                "board_name": board_name,
                "start_date": start_date,
                "end_date": end_date,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "retry_count": str(retry_count),
                "update_time": update_time,
            }
        ]
    )
    merged = pd.concat([failed_df, record], ignore_index=True)
    merged = merged.drop_duplicates(subset=["board_name"], keep="last")
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    logger.warning("failed_saved board_type=%s board_name=%s failed_csv=%s", board_type, board_name, path)


def remove_failed_item(board_type: str, board_name: str, logger: logging.Logger) -> None:
    path = failed_csv_path(board_type)
    if not path.exists():
        return
    failed_df = pd.read_csv(path, dtype=str)
    updated = failed_df[failed_df["board_name"] != board_name].copy()
    updated.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("failed_cleared board_type=%s board_name=%s failed_csv=%s", board_type, board_name, path)


def fetch_board_list(api: BoardApi, logger: logging.Logger) -> list[dict[str, str]]:
    try:
        raw_df = retry_call(api.list_fetcher, logger=logger, action=f"fetch {api.board_type} board list via akshare")
    except RetryFailure as exc:
        logger.warning("%s board list via akshare failed: %s; fallback to direct Eastmoney API", api.board_type, exc)
        raw_df = fetch_board_list_direct(api, logger)
    if raw_df is None or raw_df.empty:
        logger.warning("%s board list is empty", api.board_type)
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
        board_code = str(row.get(code_col, "")).strip() if code_col else ""
        boards.append({"board_name": board_name, "board_code": board_code})
    return boards


def fetch_board_list_direct(api: BoardApi, logger: logging.Logger) -> pd.DataFrame:
    all_rows: list[dict] = []
    page = 1
    while True:
        params = dict(api.list_params)
        params["pn"] = str(page)
        response = retry_call(
            requests.get,
            logger=logger,
            action=f"fetch {api.board_type} board list direct page={page}",
            url=api.list_url,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        diff = ((payload.get("data") or {}).get("diff")) or []
        if not diff:
            break
        all_rows.extend(diff)
        if len(diff) < int(params["pz"]):
            break
        page += 1

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows).rename(
        columns={
            "f12": "板块代码",
            "f14": "板块名称",
            "f2": "最新价",
            "f3": "涨跌幅",
            "f4": "涨跌额",
            "f8": "换手率",
            "f20": "总市值",
            "f104": "上涨家数",
            "f105": "下跌家数",
            "f128": "领涨股票",
            "f136": "领涨股票-涨跌幅",
        }
    )


def next_start_date(existing_path: Path, cli_start: str, force_refresh: bool, logger: logging.Logger) -> str:
    if force_refresh or not existing_path.exists():
        return cli_start
    try:
        existing_df = pd.read_csv(existing_path, usecols=["trade_date"])
    except ValueError:
        existing_df = pd.read_csv(existing_path)
    except Exception as exc:
        logger.warning("existing_file_read_failed path=%s error=%s fallback_start=%s", existing_path, exc, cli_start)
        return cli_start

    if "trade_date" not in existing_df.columns or existing_df.empty:
        return cli_start

    trade_dates = pd.to_datetime(existing_df["trade_date"], errors="coerce")
    max_date = trade_dates.max()
    if pd.isna(max_date):
        return cli_start
    return max(cli_start, (max_date + timedelta(days=1)).strftime(DATE_FMT))


def fetch_board_history(api: BoardApi, board_name: str, board_code: str, start_date: str, end_date: str, logger: logging.Logger) -> pd.DataFrame:
    kwargs = {"symbol": board_name, "start_date": start_date, "end_date": end_date}
    if api.board_type == "industry":
        kwargs["period"] = api.hist_period_value
    else:
        kwargs["period"] = api.hist_period_value
    try:
        return retry_call(api.hist_fetcher, logger=logger, action=f"fetch {api.board_type} history via akshare board={board_name}", **kwargs)
    except RetryFailure as exc:
        logger.warning("%s history via akshare failed for %s: %s; fallback to direct Eastmoney API", api.board_type, board_name, exc)
        return fetch_board_history_direct(api, board_name=board_name, board_code=board_code, start_date=start_date, end_date=end_date, logger=logger)


def fetch_board_history_direct(api: BoardApi, board_name: str, board_code: str, start_date: str, end_date: str, logger: logging.Logger) -> pd.DataFrame:
    if not board_code:
        raise ValueError(f"missing board code for {board_name}")
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
    payload = response.json()
    klines = ((payload.get("data") or {}).get("klines")) or []
    if not klines:
        return pd.DataFrame()
    temp_df = pd.DataFrame([item.split(",") for item in klines])
    temp_df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
    return temp_df


def standardize_history(raw_df: pd.DataFrame, board_type: str, board_name: str, update_time: str) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    renamed = raw_df.rename(
        columns={
            "日期": "trade_date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_change",
        }
    ).copy()
    for column in ("trade_date", "open", "close", "high", "low", "volume", "amount", "pct_change"):
        if column not in renamed.columns:
            renamed[column] = pd.NA

    out = renamed[["trade_date", "open", "close", "high", "low", "volume", "amount", "pct_change"]].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.strftime(OUTPUT_DATE_FMT)
    out = out[out["trade_date"].notna()].copy()
    for column in ("open", "close", "high", "low", "volume", "amount", "pct_change"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out.insert(0, "board_name", board_name)
    out.insert(0, "board_type", board_type)
    out["source"] = SOURCE_NAME
    out["update_time"] = update_time
    return out[STANDARD_COLUMNS]


def merge_normalized(existing_path: Path, new_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    existing_df = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame(columns=STANDARD_COLUMNS)
    merged = pd.concat([existing_df, new_df], ignore_index=True)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce").dt.strftime(OUTPUT_DATE_FMT)
    merged = merged[merged["trade_date"].notna()].copy()
    existing_dates = set(existing_df["trade_date"].dropna().astype(str)) if "trade_date" in existing_df.columns else set()
    merged = merged.drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date").reset_index(drop=True)
    new_rows = sum(1 for trade_date in merged["trade_date"].astype(str) if trade_date not in existing_dates)
    return merged, new_rows


def merge_raw(existing_path: Path, new_df: pd.DataFrame, board_type: str, board_name: str, update_time: str) -> pd.DataFrame:
    raw = new_df.copy()
    raw["board_type"] = board_type
    raw["board_name"] = board_name
    raw["source"] = SOURCE_NAME
    raw["update_time"] = update_time
    if "日期" in raw.columns and "trade_date" not in raw.columns:
        raw["trade_date"] = pd.to_datetime(raw["日期"], errors="coerce").dt.strftime(OUTPUT_DATE_FMT)
    elif "trade_date" in raw.columns:
        raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.strftime(OUTPUT_DATE_FMT)

    if existing_path.exists():
        existing_raw = pd.read_csv(existing_path)
        raw = pd.concat([existing_raw, raw], ignore_index=True, sort=False)

    if "trade_date" in raw.columns:
        raw = raw[raw["trade_date"].notna()].copy()
        raw = raw.drop_duplicates(subset=["trade_date"], keep="last").sort_values("trade_date").reset_index(drop=True)
    return raw


def ensure_output_dirs(board_type: str) -> None:
    paths = BOARD_PATHS[board_type]
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.normalized_dir.mkdir(parents=True, exist_ok=True)
    FAILED_ROOT.mkdir(parents=True, exist_ok=True)


def build_board_queue(api: BoardApi, options: RunOptions, logger: logging.Logger) -> list[dict[str, str]]:
    boards = fetch_board_list(api, logger)
    board_map = {item["board_name"]: item for item in boards}

    if options.resume_failed:
        failed_df = load_failed_items(api.board_type)
        failed_names = [str(name).strip() for name in failed_df.get("board_name", pd.Series(dtype=str)).tolist() if str(name).strip()]
        logger.info("resume_failed board_type=%s failed_count=%s", api.board_type, len(failed_names))
        boards = [board_map[name] for name in failed_names if name in board_map]

    if options.board_name:
        boards = [item for item in boards if item["board_name"] == options.board_name]
        logger.info("board_name_filter board_type=%s board_name=%s matched=%s", api.board_type, options.board_name, len(boards))

    if options.limit is not None:
        boards = boards[: options.limit]
        logger.info("limit_applied board_type=%s limit=%s remaining=%s", api.board_type, options.limit, len(boards))

    return boards


def process_board(api: BoardApi, board: dict[str, str], options: RunOptions, logger: logging.Logger) -> tuple[str, int]:
    board_name = board["board_name"]
    board_code = board.get("board_code", "")
    safe_name = sanitize_filename(board_name)
    paths = BOARD_PATHS[api.board_type]
    raw_path = paths.raw_dir / f"{safe_name}.csv"
    normalized_path = paths.normalized_dir / f"{safe_name}.csv"
    effective_start = next_start_date(normalized_path, options.start_date, options.force_refresh, logger)

    if effective_start > options.end_date:
        logger.info("[%s] %s skipped reason=no_new_dates normalized_path=%s", api.board_type, board_name, normalized_path)
        remove_failed_item(api.board_type, board_name, logger)
        return "skipped", 0

    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        raw_df = fetch_board_history(
            api,
            board_name=board_name,
            board_code=board_code,
            start_date=effective_start,
            end_date=options.end_date,
            logger=logger,
        )
    except RetryFailure as exc:
        upsert_failed_item(api.board_type, board_name, effective_start, options.end_date, exc.original_error, exc.retry_count, logger)
        return "failed", exc.retry_count
    except Exception as exc:
        upsert_failed_item(api.board_type, board_name, effective_start, options.end_date, exc, 0, logger)
        return "failed", 0

    if raw_df is None or raw_df.empty:
        logger.warning("[%s] %s empty_result range=%s-%s", api.board_type, board_name, effective_start, options.end_date)
        upsert_failed_item(api.board_type, board_name, effective_start, options.end_date, ValueError("empty_result"), 0, logger)
        return "failed", 0

    normalized_df = standardize_history(raw_df, board_type=api.board_type, board_name=board_name, update_time=update_time)
    if normalized_df.empty:
        logger.warning("[%s] %s normalization_empty", api.board_type, board_name)
        upsert_failed_item(api.board_type, board_name, effective_start, options.end_date, ValueError("normalization_empty"), 0, logger)
        return "failed", 0

    merged_normalized, new_rows = merge_normalized(normalized_path, normalized_df)
    merged_raw = merge_raw(raw_path, raw_df, board_type=api.board_type, board_name=board_name, update_time=update_time)
    merged_raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    merged_normalized.to_csv(normalized_path, index=False, encoding="utf-8-sig")
    remove_failed_item(api.board_type, board_name, logger)
    logger.info("[%s] %s saved new_rows=%s raw_path=%s normalized_path=%s", api.board_type, board_name, new_rows, raw_path, normalized_path)
    return "saved", new_rows


def run_for_type(board_type: str, options: RunOptions, logger: logging.Logger) -> bool:
    ensure_output_dirs(board_type)
    api = BOARD_APIS[board_type]
    logger.info(
        "start board_type=%s source=%s start_date=%s end_date=%s force_refresh=%s resume_failed=%s",
        board_type,
        SOURCE_NAME,
        options.start_date,
        options.end_date,
        options.force_refresh,
        options.resume_failed,
    )
    boards = build_board_queue(api, options, logger)
    logger.info("board_type=%s total_boards=%s", board_type, len(boards))

    saved_count = 0
    skipped_count = 0
    failed_count = 0
    consecutive_failures = 0
    stop_early = False

    for board in boards:
        try:
            status, _ = process_board(api, board, options, logger)
        except Exception as exc:
            logger.exception("[%s] %s unexpected_error=%s", board_type, board["board_name"], exc)
            upsert_failed_item(board_type, board["board_name"], options.start_date, options.end_date, exc, 0, logger)
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
                stop_early = True
                logger.error("stop_early board_type=%s reason=consecutive_failures message=%s", board_type, STOP_MESSAGE)
                break

    logger.info(
        "finish board_type=%s saved=%s skipped=%s failed=%s stop_early=%s",
        board_type,
        saved_count,
        skipped_count,
        failed_count,
        stop_early,
    )
    return not stop_early


def main() -> int:
    args = parse_args()
    logger = setup_logger()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("job_started start_time=%s source=%s akshare_version=%s", started_at, SOURCE_NAME, ak.__version__)

    try:
        options = validate_options(args)
        start_dt = validate_date(options.start_date)
        end_dt = validate_date(options.end_date)
    except ValueError as exc:
        logger.error("invalid_parameter error=%s", exc)
        return 1

    if start_dt > end_dt:
        logger.error("invalid_date_range start=%s end=%s", options.start_date, options.end_date)
        return 1

    board_types = ["industry", "concept"] if args.type == "all" else [args.type]
    exit_code = 0
    for board_type in board_types:
        try:
            keep_running = run_for_type(board_type, options, logger)
        except Exception as exc:
            logger.exception("board_type=%s terminated_unexpectedly error=%s", board_type, exc)
            logger.error(STOP_MESSAGE)
            exit_code = 1
            break
        if not keep_running:
            logger.error(STOP_MESSAGE)
            exit_code = 1
            break

    ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("job_finished end_time=%s exit_code=%s", ended_at, exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
