from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
LOG_ROOT = PROJECT_ROOT / "logs"
FAILED_ROOT = PROJECT_ROOT / "data_fetcher" / "data" / "failed"


@dataclass(frozen=True)
class BoardPathConfig:
    board_type: str
    raw_dir: Path
    normalized_dir: Path


BOARD_PATHS = {
    "industry": BoardPathConfig(
        board_type="industry",
        raw_dir=DATA_ROOT / "raw" / "industry_daily",
        normalized_dir=DATA_ROOT / "normalized" / "industry_daily",
    ),
    "concept": BoardPathConfig(
        board_type="concept",
        raw_dir=DATA_ROOT / "raw" / "concept_daily",
        normalized_dir=DATA_ROOT / "normalized" / "concept_daily",
    ),
}

DEFAULT_START = "20200101"
LOG_FILE = LOG_ROOT / "fetch_akshare_boards.log"
SOURCE_NAME = "akshare_eastmoney_board"
RETRY_DELAYS = (10, 30, 60, 120, 300)

LOCAL_INDUSTRY_PROXY_LOG_FILE = LOG_ROOT / "local_industry_proxy.log"
LOCAL_INDUSTRY_MAP_TABLE = "cn_local_industry_map_hist"
LOCAL_INDUSTRY_PROXY_TABLE = "cn_local_industry_proxy_daily"
LOCAL_TASK_STATUS_TABLE = "cn_local_task_status"
LOCAL_INDUSTRY_LEVEL = "SW_L1"
LOCAL_MAP_MEMBER_SOURCE = "tushare_sw_l1"
LOCAL_MAP_MASTER_SOURCE = "TUSHARE_SW2021_L1"
LOCAL_PROXY_SOURCE = "local_proxy_from_stock_daily"
