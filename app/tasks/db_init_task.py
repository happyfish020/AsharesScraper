from __future__ import annotations
from dataclasses import dataclass

@dataclass
class DbInitTask:
    name: str = "DbInit"

    def run(self, ctx) -> None:
        # MySQL only: tables are created lazily by loaders/to_sql.
        ctx.log.info("[DB_INIT] mysql-only mode; skip explicit DDL init")
