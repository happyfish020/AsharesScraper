from __future__ import annotations

import os
import re
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL
import pymysql 

_CN_MARKET_RE = re.compile(r"(?<![A-Za-z0-9_])cn_market(?![A-Za-z0-9_])")


def get_db_name() -> str:
    return os.getenv("ASHARE_MYSQL_DB", "cn_market_red").strip() or "cn_market_red"


def load_sql_for_current_db(path: str) -> str:
    sql = Path(path).read_text(encoding="utf-8")
    db_name = get_db_name()
    if db_name != "cn_market":
        sql = _CN_MARKET_RE.sub(db_name, sql)
    return sql


def build_engine() -> Engine:
    """Build MySQL engine (default write path)."""
    db_user = os.getenv("ASHARE_MYSQL_USER", "cn_opr_red")
    db_password = os.getenv("ASHARE_MYSQL_PASSWORD", "sec_Bobo123")
    db_host = os.getenv("ASHARE_MYSQL_HOST", "localhost")
    db_port = os.getenv("ASHARE_MYSQL_PORT", "3306")
    db_name = get_db_name()

    conn = URL.create(
        drivername="mysql+pymysql",
        username=db_user,
        password=db_password,
        host=db_host,
        port=int(db_port),
        database=db_name,
        query={"charset": "utf8mb4"},
    )
    return create_engine(conn, pool_pre_ping=True, future=True)
