from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL
import pymysql 

def build_engine() -> Engine:
    """Build MySQL engine (default write path)."""
    db_user = os.getenv("ASHARE_MYSQL_USER", "cn_opr")
    db_password = os.getenv("ASHARE_MYSQL_PASSWORD", "sec@Bobo123")
    db_host = os.getenv("ASHARE_MYSQL_HOST", "localhost")
    db_port = os.getenv("ASHARE_MYSQL_PORT", "3306")
    db_name = "cn_market"

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
