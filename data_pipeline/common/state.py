from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from uuid import uuid4

from sqlalchemy import text


@dataclass(slots=True)
class BackfillState:
    engine: object
    job_name: str
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            object.__setattr__(self, "run_id", uuid4().hex[:16])

    def is_completed(self, chunk_key: str) -> bool:
        sql = """
        SELECT status
        FROM cn_mainline_backfill_job_state
        WHERE job_name = :job_name
          AND chunk_key = :chunk_key
        """
        with self.engine.connect() as conn:
            value = conn.execute(text(sql), {"job_name": self.job_name, "chunk_key": chunk_key}).scalar()
        return str(value or "").lower() == "completed"

    def start(self, chunk_key: str, range_start: date | None, range_end: date | None) -> None:
        sql = """
        INSERT INTO cn_mainline_backfill_job_state
            (job_name, chunk_key, range_start, range_end, status, attempts, last_rows, last_error, last_run_id)
        VALUES
            (:job_name, :chunk_key, :range_start, :range_end, 'running', 1, 0, NULL, :last_run_id)
        ON DUPLICATE KEY UPDATE
            range_start = VALUES(range_start),
            range_end = VALUES(range_end),
            status = 'running',
            attempts = attempts + 1,
            last_error = NULL,
            last_run_id = VALUES(last_run_id)
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(sql),
                {
                    "job_name": self.job_name,
                    "chunk_key": chunk_key,
                    "range_start": range_start,
                    "range_end": range_end,
                    "last_run_id": self.run_id,
                },
            )

    def complete(self, chunk_key: str, rows: int) -> None:
        sql = """
        UPDATE cn_mainline_backfill_job_state
        SET status = 'completed',
            last_rows = :rows,
            last_error = NULL,
            last_run_id = :last_run_id
        WHERE job_name = :job_name
          AND chunk_key = :chunk_key
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(sql),
                {"job_name": self.job_name, "chunk_key": chunk_key, "rows": rows, "last_run_id": self.run_id},
            )

    def fail(self, chunk_key: str, error: Exception) -> None:
        sql = """
        UPDATE cn_mainline_backfill_job_state
        SET status = 'failed',
            last_error = :last_error,
            last_run_id = :last_run_id
        WHERE job_name = :job_name
          AND chunk_key = :chunk_key
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(sql),
                {
                    "job_name": self.job_name,
                    "chunk_key": chunk_key,
                    "last_error": str(error)[:4000],
                    "last_run_id": self.run_id,
                },
            )
