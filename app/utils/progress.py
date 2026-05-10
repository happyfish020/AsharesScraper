from __future__ import annotations

import time


class ProgressLogger:
    def __init__(
        self,
        *,
        name: str,
        total: int | None = None,
        unit: str = "items",
        log=None,
        every: int = 50,
        min_interval_seconds: float = 30.0,
    ) -> None:
        self.name = str(name).strip() or "progress"
        self.total = int(total) if total is not None else None
        self.unit = str(unit).strip() or "items"
        self.log = log
        self.every = max(1, int(every))
        self.min_interval_seconds = max(1.0, float(min_interval_seconds))
        self.started_at = time.monotonic()
        self.last_emit_at = 0.0
        self.completed = 0
        self.rows = 0
        self.affected = 0

    def _emit(self, message: str, *args) -> None:
        if args:
            try:
                message = message % args
            except Exception:
                message = f"{message} {' '.join(str(a) for a in args)}"
        if self.log is not None and hasattr(self.log, "info"):
            self.log.info(message)
            return
        print(message, flush=True)

    def note(self, message: str, *args) -> None:
        self._emit(message, *args)

    def _format_eta(self) -> str:
        if not self.total or self.completed <= 0 or self.completed >= self.total:
            return "NA"
        elapsed = max(0.001, time.monotonic() - self.started_at)
        remaining = self.total - self.completed
        eta_seconds = elapsed / self.completed * remaining
        return f"{eta_seconds:.1f}s"

    def update(
        self,
        *,
        current_item: str | None = None,
        rows: int = 0,
        affected: int = 0,
        extra: str | None = None,
        force: bool = False,
    ) -> None:
        self.completed += 1
        self.rows += int(rows or 0)
        self.affected += int(affected or 0)

        now = time.monotonic()
        should_emit = force
        if not should_emit:
            if self.completed == 1:
                should_emit = True
            elif self.total is not None and self.completed >= self.total:
                should_emit = True
            elif self.completed % self.every == 0:
                should_emit = True
            elif now - self.last_emit_at >= self.min_interval_seconds:
                should_emit = True
        if not should_emit:
            return

        elapsed = now - self.started_at
        if self.total is not None and self.total > 0:
            progress_text = f"{self.completed}/{self.total} ({self.completed / self.total * 100:.1f}%)"
        else:
            progress_text = str(self.completed)
        item_text = f" current={current_item}" if current_item else ""
        row_text = f" rows={self.rows}" if self.rows else ""
        affected_text = f" affected={self.affected}" if self.affected else ""
        extra_text = f" {extra}" if extra else ""
        self._emit(
            "[progress][%s] %s %s elapsed=%.1fs eta=%s%s%s%s%s",
            self.name,
            progress_text,
            self.unit,
            elapsed,
            self._format_eta(),
            item_text,
            row_text,
            affected_text,
            extra_text,
        )
        self.last_emit_at = now

    def finish(self, *, extra: str | None = None) -> None:
        elapsed = time.monotonic() - self.started_at
        if self.total is not None and self.total > 0:
            progress_text = f"{self.completed}/{self.total} ({self.completed / self.total * 100:.1f}%)"
        else:
            progress_text = str(self.completed)
        row_text = f" rows={self.rows}" if self.rows else ""
        affected_text = f" affected={self.affected}" if self.affected else ""
        extra_text = f" {extra}" if extra else ""
        self._emit(
            "[progress][%s] done %s %s elapsed=%.1fs%s%s%s",
            self.name,
            progress_text,
            self.unit,
            elapsed,
            row_text,
            affected_text,
            extra_text,
        )
