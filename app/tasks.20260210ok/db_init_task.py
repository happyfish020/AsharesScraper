from __future__ import annotations
from dataclasses import dataclass
from app.utils.oracle_utils import create_tables_if_not_exists

@dataclass
class DbInitTask:
    name: str = "DbInit"

    def run(self, ctx) -> None:
        with ctx.engine.begin() as conn:
            create_tables_if_not_exists(conn)
