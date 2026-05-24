from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import text

from app.tasks.board_membership_refresh_task import BoardMembershipRefreshTask
from app.tasks.coverage_audit_task import CoverageAuditTask
from app.tasks.event_loader_task import EventLoaderTask
from app.tasks.his_stocks_loader_task import HisStocksLoaderTask
from app.tasks.index_loader_task import IndexLoaderTask
from app.tasks.rotation_sector_snapshot_task import SectorRotationSnapshotTask
from app.tasks.stock_basic_weekly_task import StockBasicWeeklyTask
from app.tasks.stock_fundamental_monthly_task import StockFundamentalMonthlyTask
from app.tasks.stock_loader_task import StockLoaderTask
from app.tasks.sw_industry_daily_task import SwIndustryDailyTask
from app.utils.wireguard_helper import activate_tunnel


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(str(raw).strip())


def _env_text(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value if value else default


def _parse_yyyymmdd(value: str) -> datetime:
    return datetime.strptime(str(value), "%Y%m%d")


def _shift_yyyymmdd(value: str, days: int) -> str:
    return (_parse_yyyymmdd(value) + timedelta(days=days)).strftime("%Y%m%d")


def _to_iso_date(value: str) -> str:
    return _parse_yyyymmdd(value).strftime("%Y-%m-%d")


@contextmanager
def _temporary_cfg_dates(ctx, start_date: str, end_date: str):
    cfg = ctx.config
    old_start = getattr(cfg, "start_date", None)
    old_end = getattr(cfg, "end_date", None)
    cfg.start_date = start_date
    cfg.end_date = end_date
    try:
        yield
    finally:
        cfg.start_date = old_start
        cfg.end_date = old_end


@contextmanager
def _temporary_cfg_history(ctx, start_date: str, end_date: str, frequency: str):
    cfg = ctx.config
    old_start = getattr(cfg, "start_date", None)
    old_end = getattr(cfg, "end_date", None)
    old_his_start = getattr(cfg, "his_start_date", None)
    old_his_end = getattr(cfg, "his_end_date", None)
    old_freq = getattr(cfg, "his_universe_frequency", "monthly")
    cfg.start_date = start_date
    cfg.end_date = end_date
    cfg.his_start_date = start_date
    cfg.his_end_date = end_date
    cfg.his_universe_frequency = frequency
    try:
        yield
    finally:
        cfg.start_date = old_start
        cfg.end_date = old_end
        cfg.his_start_date = old_his_start
        cfg.his_end_date = old_his_end
        cfg.his_universe_frequency = old_freq


@contextmanager
def _temporary_env(overrides: dict[str, str | None]):
    old_values: dict[str, str | None] = {}
    for key, value in overrides.items():
        old_values[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _V8OpsMixin:
    def _root_dir(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _run_tasks(self, ctx, tasks: Iterable[object]) -> None:
        task_list = list(tasks)
        for i, task in enumerate(task_list, start=1):
            ctx.log.info("[V8] subtask %s/%s %s - START", i, len(task_list), getattr(task, "name", type(task).__name__))
            task.run(ctx)
            ctx.log.info("[V8] subtask %s/%s %s - DONE", i, len(task_list), getattr(task, "name", type(task).__name__))

    def _read_audit_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size <= 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def _run_coverage_audit(self, ctx) -> dict[str, pd.DataFrame]:
        audit_task = CoverageAuditTask()
        audit_task.run(ctx)
        root_dir = Path(getattr(ctx.config, "audit_reports_dir", "audit_reports"))
        return {
            "stock_missing": self._read_audit_csv(root_dir / "audit_stock_missing.csv"),
            "index_gap": self._read_audit_csv(root_dir / "audit_index_gap.csv"),
            "index_window_start": self._read_audit_csv(root_dir / "audit_index_window_start.csv"),
        }

    def _run_coverage_audit_for_window(self, ctx, start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
        with _temporary_cfg_dates(ctx, start_date, end_date):
            return self._run_coverage_audit(ctx)

    def _audit_has_issues(self, audit_frames: dict[str, pd.DataFrame]) -> bool:
        # Fatal coverage issues are limited to true stock gaps and true index GAPs.
        # index_window_start is a diagnostic boundary condition: it means the audit
        # window starts before an index series begins, not that an in-window trading
        # day is missing. Keep strict behavior only when explicitly requested.
        if self._audit_has_stock_issues(audit_frames):
            return True
        if self._audit_has_index_gap_issues(audit_frames):
            return True
        if _env_flag("V8_FAIL_ON_INDEX_WINDOW_START", False):
            return self._audit_has_index_window_start_issues(audit_frames)
        return False

    def _log_audit_summary(self, ctx, audit_frames: dict[str, pd.DataFrame], prefix: str) -> None:
        counts = {name: int(len(df)) for name, df in audit_frames.items()}
        ctx.log.info(
            "[V8] %s audit summary stock_missing=%s index_gap=%s index_window_start=%s",
            prefix,
            counts.get("stock_missing", 0),
            counts.get("index_gap", 0),
            counts.get("index_window_start", 0),
        )

        report_dir = Path(getattr(ctx.config, "audit_reports_dir", "audit_reports"))
        report_files = {
            "stock_missing": report_dir / "audit_stock_missing.csv",
            "index_gap": report_dir / "audit_index_gap.csv",
            "index_window_start": report_dir / "audit_index_window_start.csv",
        }
        for name, path in report_files.items():
            if counts.get(name, 0) > 0:
                ctx.log.info("[V8] %s detail report %s: %s", prefix, name, path)

    def _audit_has_stock_issues(self, audit_frames: dict[str, pd.DataFrame]) -> bool:
        df = audit_frames.get("stock_missing")
        return df is not None and not df.empty

    def _audit_has_index_gap_issues(self, audit_frames: dict[str, pd.DataFrame]) -> bool:
        df = audit_frames.get("index_gap")
        return df is not None and not df.empty

    def _audit_has_index_window_start_issues(self, audit_frames: dict[str, pd.DataFrame]) -> bool:
        df = audit_frames.get("index_window_start")
        return df is not None and not df.empty

    def _audit_has_index_issues(self, audit_frames: dict[str, pd.DataFrame]) -> bool:
        # WINDOW_START means the requested audit window starts before a series has
        # data in cn_index_daily_price. It is diagnostically useful, but it is not
        # an in-window index GAP and must not raise the misleading
        # "found index gaps" RuntimeError. Keep the historical strict behavior only
        # when explicitly requested.
        if self._audit_has_index_gap_issues(audit_frames):
            return True
        if _env_flag("V8_FAIL_ON_INDEX_WINDOW_START", False):
            return self._audit_has_index_window_start_issues(audit_frames)
        return False


    def _rotation_audit_report_dirs(self, ctx) -> list[Path]:
        # Write rotation audit details to both the configured audit dir and a
        # stable V8 reports dir. Some .bat runs start from different working
        # directories, so relying on only audit_reports_dir made the CSV hard to
        # find even when it was written successfully.
        root = self._root_dir()
        candidates = [
            Path(getattr(ctx.config, "audit_reports_dir", "audit_reports")),
            root / "reports" / "v8_dataset_audit",
        ]
        out: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            resolved_key = str(path.resolve()) if path.is_absolute() else str((root / path).resolve())
            if resolved_key in seen:
                continue
            seen.add(resolved_key)
            out.append(Path(resolved_key))
        return out

    def _write_rotation_audit_detail(self, ctx, audit: dict, prefix: str) -> dict[str, Path]:
        report_dirs = self._rotation_audit_report_dirs(ctx)
        for report_dir in report_dirs:
            report_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        effective_end = audit.get("effective_end_date", [getattr(ctx.config, "end_date", "")])[0]
        files: dict[str, Path] = {}
        manifest_rows: list[dict[str, str | int]] = []
        groups = (
            ("upstream_missing", "ROTATION_UPSTREAM_MISSING", "cn_stock_daily_price/cn_board_member_map_d"),
            ("bt_missing", "ROTATION_BT_MISSING", "cn_sector_rot_bt_daily_t"),
            ("snap_missing", "ROTATION_SNAPSHOT_MISSING", "cn_rotation_entry_snap_t/cn_rotation_holding_snap_t/cn_rotation_exit_snap_t"),
        )
        for key, reason, source_table in groups:
            dates = audit.get(key, []) or []
            if not dates:
                continue
            rows = [
                {
                    "trade_date": str(d),
                    "missing_group": key,
                    "missing_reason": reason,
                    "effective_end_date": str(effective_end),
                    "source_table": source_table,
                    "audit_version": "v8_rotation_audit_v3_detail",
                }
                for d in dates
            ]
            df = pd.DataFrame(rows)
            sample = ",".join(str(d) for d in list(dates)[:20])
            for report_dir in report_dirs:
                path = report_dir / f"{prefix}_{key}_detail_{ts}.csv"
                df.to_csv(path, index=False, encoding="utf-8-sig")
                files[f"{key}:{report_dir}"] = path
                manifest_rows.append(
                    {
                        "prefix": prefix,
                        "missing_group": key,
                        "rows": len(rows),
                        "path": str(path),
                        "sample": sample,
                    }
                )
                ctx.log.info("[V8] %s detail report %s rows=%s path=%s sample=%s", prefix, key, len(rows), path, sample)

        if manifest_rows:
            manifest_df = pd.DataFrame(manifest_rows)
            for report_dir in report_dirs:
                manifest_path = report_dir / f"{prefix}_missing_detail_manifest_{ts}.csv"
                manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
                ctx.log.info("[V8] %s detail manifest: %s", prefix, manifest_path)
        return files

    def _rotation_snap_missing_is_fatal(self) -> bool:
        # Snapshot gaps are downstream derived diagnostic gaps. In daily runs, they
        # should not kill the whole refresh unless explicitly requested; upstream
        # source gaps and BT gaps remain fatal.
        return _env_flag("V8_ROTATION_FAIL_ON_SNAP_MISSING", False)

    def _assert_weekly_reference_tables_ready(self, ctx) -> None:
        required_tables = [
            "cn_board_concept_member_hist",
            "cn_board_industry_member_hist",
            "cn_board_member_map_d",
        ]
        missing_or_empty = [name for name in required_tables if not self._table_has_rows(ctx, name)]
        if missing_or_empty:
            raise RuntimeError("[V8] weekly reference audit failed; missing/empty tables: " + ", ".join(missing_or_empty))
        ctx.log.info("[V8] weekly reference audit passed tables=%s", ",".join(required_tables))

    def _resolve_daily_audit_window(self, end_date: str) -> tuple[str, str]:
        lookback_days = max(1, _env_int("V8_DAILY_AUDIT_LOOKBACK_DAYS", 90))
        start_date = _shift_yyyymmdd(end_date, -(lookback_days - 1))
        return start_date, end_date

    def _resolve_audit_window(self, prefix: str, end_date: str, default_days: int) -> tuple[str, str]:
        lookback_days = max(1, _env_int(f"{prefix}_AUDIT_LOOKBACK_DAYS", default_days))
        start_date = _shift_yyyymmdd(end_date, -(lookback_days - 1))
        return start_date, end_date

    def _extract_earliest_missing_date(self, audit_frames: dict[str, pd.DataFrame]) -> str | None:
        candidates: list[datetime] = []
        for frame in audit_frames.values():
            if frame is None or frame.empty or "missing_date" not in frame.columns:
                continue
            parsed = pd.to_datetime(frame["missing_date"], errors="coerce").dropna()
            if parsed.empty:
                continue
            candidates.append(parsed.min().to_pydatetime())
        if not candidates:
            return None
        return min(candidates).strftime("%Y%m%d")

    def _resolve_repair_start(self, end_date: str, audit_frames: dict[str, pd.DataFrame]) -> str:
        fallback_days = max(2, _env_int("V8_DAILY_REPAIR_LOOKBACK_DAYS", 15))
        max_days = max(fallback_days, _env_int("V8_DAILY_REPAIR_MAX_LOOKBACK_DAYS", 365))
        fallback_start = _shift_yyyymmdd(end_date, -(fallback_days - 1))
        floor_start = _shift_yyyymmdd(end_date, -(max_days - 1))
        earliest_missing = self._extract_earliest_missing_date(audit_frames)
        if not earliest_missing:
            return fallback_start
        return max(floor_start, min(earliest_missing, end_date))

    def _resolve_repair_start_for_prefix(
        self,
        prefix: str,
        end_date: str,
        audit_frames: dict[str, pd.DataFrame],
        default_lookback_days: int,
        default_max_days: int,
    ) -> str:
        fallback_days = max(2, _env_int(f"{prefix}_REPAIR_LOOKBACK_DAYS", default_lookback_days))
        max_days = max(fallback_days, _env_int(f"{prefix}_REPAIR_MAX_LOOKBACK_DAYS", default_max_days))
        fallback_start = _shift_yyyymmdd(end_date, -(fallback_days - 1))
        floor_start = _shift_yyyymmdd(end_date, -(max_days - 1))
        earliest_missing = self._extract_earliest_missing_date(audit_frames)
        if not earliest_missing:
            return fallback_start
        return max(floor_start, min(earliest_missing, end_date))

    def _db_cli_args(self, ctx) -> list[str]:
        url = ctx.engine.url
        args = [
            "--db-host",
            str(url.host or "127.0.0.1"),
            "--db-port",
            str(url.port or 3306),
            "--db-user",
            str(url.username or "root"),
            "--db-name",
            str(url.database or ""),
        ]
        password = url.password or os.getenv("ASHARE_MYSQL_PASSWORD") or ""
        if password:
            args.extend(["--db-password", str(password)])
        return args

    def _redact_cmd(self, cmd: list[str]) -> str:
        redacted: list[str] = []
        hide_next = False
        for token in cmd:
            if hide_next:
                redacted.append("******")
                hide_next = False
                continue
            redacted.append(token)
            if token == "--db-password":
                hide_next = True
        return " ".join(redacted)

    def _run_python_module(self, ctx, module_name: str, args: list[str]) -> None:
        root_dir = self._root_dir()
        cmd = [sys.executable, "-m", module_name] + args
        ctx.log.info("[V8] run module: %s", self._redact_cmd(cmd))
        subprocess.run(cmd, cwd=str(root_dir), check=True)

    def _run_python_script(self, ctx, script_relative_path: str, args: list[str], env_overrides: dict[str, str] | None = None) -> None:
        root_dir = self._root_dir()
        cmd = [sys.executable, script_relative_path] + args
        run_env = os.environ.copy()
        if env_overrides:
            run_env.update({key: str(value) for key, value in env_overrides.items()})
        ctx.log.info("[V8] run script: %s", self._redact_cmd(cmd))
        subprocess.run(cmd, cwd=str(root_dir), env=run_env, check=True)

    def _run_optional_python_script(
        self,
        ctx,
        script_relative_path: str,
        args: list[str],
        *,
        strict_env: str = "V8_DERIVED_FOUNDATION_STRICT",
        env_overrides: dict[str, str] | None = None,
    ) -> bool:
        """Run a derived/research builder without blocking raw daily history refresh by default.

        Several derived foundation builders require long lookback source data (for example
        financial statements before the requested start date). Historical market-data rebuilds
        should not fail only because optional factor inputs are unavailable. Set the strict_env
        variable to 0/false/off to allow warning + skip behavior.
        """
        root_dir = self._root_dir()
        cmd = [sys.executable, script_relative_path] + args
        run_env = os.environ.copy()
        if env_overrides:
            run_env.update({key: str(value) for key, value in env_overrides.items()})
        ctx.log.info("[V8] run optional script: %s", self._redact_cmd(cmd))
        try:
            subprocess.run(cmd, cwd=str(root_dir), env=run_env, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            if _env_flag(strict_env, True):
                raise
            ctx.log.warning(
                "[V8] optional derived script skipped after failure; script=%s exit_code=%s strict_env=%s. "
                "This usually means source fundamentals/optional inputs do not cover the requested lookback window.",
                script_relative_path,
                exc.returncode,
                strict_env,
            )
            return False

    def _run_external_script(self, ctx, root_dir: Path, script_relative_path: str, args: list[str], env_overrides: dict[str, str] | None = None) -> None:
        cmd = [sys.executable, script_relative_path] + args
        run_env = os.environ.copy()
        if env_overrides:
            run_env.update({key: str(value) for key, value in env_overrides.items()})
        ctx.log.info("[V8] run external script: %s", self._redact_cmd(cmd))
        subprocess.run(cmd, cwd=str(root_dir), env=run_env, check=True)

    def _table_exists(self, ctx, table_name: str) -> bool:
        with ctx.engine.connect() as conn:
            cnt = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = :table_name
                    """
                ),
                {"table_name": table_name},
            ).scalar()
        return bool(cnt)

    def _table_has_rows(self, ctx, table_name: str) -> bool:
        if not self._table_exists(ctx, table_name):
            return False
        with ctx.engine.connect() as conn:
            cnt = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        return bool(cnt)

    def _run_latest_snap(self, ctx, trade_date: str) -> None:
        if not _env_flag("V8_ENABLE_LEADER_SW_L1_LATEST_SNAP", True):
            return
        self._run_python_module(
            ctx,
            "app.tools.build_cn_stock_leader_sw_l1_latest_snap",
            ["--trade-date", trade_date],
        )

    def _run_crosswalk_latest(self, ctx) -> None:
        if not _env_flag("V8_ENABLE_CROSSWALK_LATEST", True):
            return
        if not self._table_has_rows(ctx, "cn_v7_v8_industry_crosswalk"):
            ctx.log.info("[V8] skip build_v7_v8_crosswalk_latest.py because cn_v7_v8_industry_crosswalk is empty or missing")
            return
        self._run_python_script(
            ctx,
            "scripts/build_v7_v8_crosswalk_latest.py",
            ["--replace"],
        )

    def _run_optional_leader_recall(self, ctx, start_date: str, end_date: str) -> None:
        if not _env_flag("V8_ENABLE_LEADER_RECALL_VALIDATION", False):
            return
        ga_root_text = _env_text("V8_GROWTHALPHA_V7_ROOT", "")
        cfg_path_text = _env_text("V8_LEADER_RECALL_CONFIG", "")
        if not ga_root_text or not cfg_path_text:
            raise RuntimeError(
                "[V8] leader recall validation enabled but V8_GROWTHALPHA_V7_ROOT / V8_LEADER_RECALL_CONFIG is not configured"
            )
        ga_root = Path(ga_root_text).resolve()
        cfg_path = Path(cfg_path_text)
        if not cfg_path.is_absolute():
            cfg_path = (ga_root / cfg_path).resolve()
        self._run_external_script(
            ctx,
            ga_root,
            "scripts/validate_unified_alpha_leader_recall.py",
            [
                "--config",
                str(cfg_path),
                "--start",
                _to_iso_date(start_date),
                "--end",
                _to_iso_date(end_date),
            ],
        )

    def _run_derived_foundation_chain(self, ctx, start_date: str, end_date: str) -> None:
        start_iso = _to_iso_date(start_date)
        end_iso = _to_iso_date(end_date)
        db_args = self._db_cli_args(ctx)
        chunk_small = str(_env_int("V8_DERIVED_CHUNK_MONTHS_SMALL", 1))
        chunk_medium = str(_env_int("V8_DERIVED_CHUNK_MONTHS_MEDIUM", 3))

        # build_stock_fundamental_daily.py delegates to data_pipeline.builders.stock_fundamental_daily,
        # whose CLI intentionally does not accept --db-host/--db-user/--db-name/--db-password.
        # It reads DB settings from the shared project config/environment instead.
        # Keep DB args for the later derived builders that do support them, but do not pass
        # them here, otherwise yearly refresh fails with "unrecognized arguments: --db-host ...".
        if not _env_flag("V8_SKIP_STOCK_FUNDAMENTAL_DAILY", False):
            self._run_python_script(
                ctx,
                "scripts/build_stock_fundamental_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace"],
            )
        else:
            ctx.log.info("[V8] V8_SKIP_STOCK_FUNDAMENTAL_DAILY=1; skipped build_stock_fundamental_daily.py")
        # The remaining foundation builders are derived/optional for raw daily history rebuilds.
        # They may legitimately fail in early years when financial source tables do not have
        # the required lookback window. Keep them strict only when explicitly requested.
        if not _env_flag("V8_SKIP_STOCK_QUALITY_SCORE", False):
            self._run_optional_python_script(
                ctx,
                "scripts/build_stock_quality_score_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_medium] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_STOCK_QUALITY_SCORE=1; skipped build_stock_quality_score_daily.py")
        self._run_latest_snap(ctx, end_date)
        if not _env_flag("V8_SKIP_INDUSTRY_CAPITAL_FLOW", False):
            self._run_optional_python_script(
                ctx,
                "scripts/build_industry_capital_flow_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_small] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_INDUSTRY_CAPITAL_FLOW=1; skipped build_industry_capital_flow_daily.py")
        if not _env_flag("V8_SKIP_GA_STOCK_ROLE_MAP", False):
            self._run_optional_python_script(
                ctx,
                "scripts/build_ga_stock_role_map_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_medium] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_GA_STOCK_ROLE_MAP=1; skipped build_ga_stock_role_map_daily.py")

    def _run_derived_mainline_chain(self, ctx, start_date: str, end_date: str, *, include_validations: bool) -> None:
        start_iso = _to_iso_date(start_date)
        end_iso = _to_iso_date(end_date)
        db_args = self._db_cli_args(ctx)
        chunk_small = str(_env_int("V8_DERIVED_CHUNK_MONTHS_SMALL", 1))
        chunk_medium = str(_env_int("V8_DERIVED_CHUNK_MONTHS_MEDIUM", 3))
        validate_min_rows = str(_env_int("V8_DERIVED_VALIDATE_MIN_ROWS", 1))

        skip_mainline_strength = _env_flag("V8_SKIP_MAINLINE_STRENGTH", False)
        skip_mainline_radar = _env_flag("V8_SKIP_MAINLINE_RADAR", False)
        skip_market_pulse = _env_flag("V8_SKIP_MARKET_PULSE", False)
        skip_local_industry_proxy = _env_flag("V8_SKIP_LOCAL_INDUSTRY_PROXY", False)
        skip_mainline_lifecycle = _env_flag("V8_SKIP_MAINLINE_LIFECYCLE", False)

        if not skip_mainline_strength:
            self._run_python_script(
                ctx,
                "scripts/build_cn_stock_mainline_strength_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_small] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_MAINLINE_STRENGTH=1; skipped build_cn_stock_mainline_strength_daily.py (first pass)")

        if not skip_mainline_radar:
            self._run_python_script(
                ctx,
                "scripts/build_ga_mainline_radar_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_small] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_MAINLINE_RADAR=1; skipped build_ga_mainline_radar_daily.py (first pass)")

        if not skip_market_pulse:
            self._run_python_script(
                ctx,
                "scripts/build_ga_market_pulse_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_medium] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_MARKET_PULSE=1; skipped build_ga_market_pulse_daily.py (first pass)")

        # build_local_industry_proxy_daily.py builds cn_local_industry_proxy_daily,
        # which is a REQUIRED source table for build_mainline_lifecycle_daily.py.
        # It must run before the lifecycle builder to ensure member_count data
        # is available for the industry membership filter.
        #
        # NOTE: build_local_industry_proxy_daily.py reads DB settings from the
        # shared project config/environment (build_engine() in app/settings.py),
        # NOT from CLI args. Do NOT pass db_args here, otherwise the script
        # fails with "unrecognized arguments: --db-host ...".
        if not skip_local_industry_proxy:
            self._run_optional_python_script(
                ctx,
                "scripts/build_local_industry_proxy_daily.py",
                ["--start", start_iso, "--end", end_iso, "--chunk-days", "5"],
            )
        else:
            ctx.log.info("[V8] V8_SKIP_LOCAL_INDUSTRY_PROXY=1; skipped build_local_industry_proxy_daily.py")

        if not skip_mainline_lifecycle:
            self._run_python_script(
                ctx,
                "scripts/build_mainline_lifecycle_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_small] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_MAINLINE_LIFECYCLE=1; skipped build_mainline_lifecycle_daily.py")

        if include_validations and not skip_mainline_lifecycle:
            self._run_python_script(
                ctx,
                "scripts/validate_mainline_lifecycle_daily.py",
                ["--start", start_iso, "--end", end_iso, "--min-rows", validate_min_rows] + db_args,
            )

        if not skip_mainline_strength:
            self._run_python_script(
                ctx,
                "scripts/build_cn_stock_mainline_strength_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_small] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_MAINLINE_STRENGTH=1; skipped build_cn_stock_mainline_strength_daily.py (second pass)")

        if include_validations and not skip_mainline_strength:
            self._run_python_script(
                ctx,
                "scripts/validate_cn_mainline_strength_daily.py",
                ["--start", start_iso, "--end", end_iso, "--min-rows", validate_min_rows] + db_args,
            )

        if not skip_mainline_radar:
            self._run_python_script(
                ctx,
                "scripts/build_ga_mainline_radar_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_small] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_MAINLINE_RADAR=1; skipped build_ga_mainline_radar_daily.py (second pass)")

        if not skip_market_pulse:
            self._run_python_script(
                ctx,
                "scripts/build_ga_market_pulse_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_medium] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_MARKET_PULSE=1; skipped build_ga_market_pulse_daily.py (second pass)")

    def _run_derived_alpha_chain(self, ctx, start_date: str, end_date: str, *, include_validations: bool, include_crosswalk: bool) -> None:
        start_iso = _to_iso_date(start_date)
        end_iso = _to_iso_date(end_date)
        db_args = self._db_cli_args(ctx)
        chunk_medium = str(_env_int("V8_DERIVED_CHUNK_MONTHS_MEDIUM", 3))
        validate_min_rows = str(_env_int("V8_DERIVED_VALIDATE_MIN_ROWS", 1))

        self._run_python_script(
            ctx,
            "scripts/build_unified_alpha_score_daily.py",
            ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_medium, "--no-report"] + db_args,
        )

        if include_validations:
            self._run_python_script(
                ctx,
                "scripts/validate_unified_alpha_score_daily.py",
                ["--start", start_iso, "--end", end_iso, "--min-rows", validate_min_rows] + db_args,
            )

        if include_crosswalk:
            self._run_crosswalk_latest(ctx)

        self._run_optional_leader_recall(ctx, start_date, end_date)

    def _run_derived_chain(self, ctx, start_date: str, end_date: str, *, include_validations: bool, include_crosswalk: bool) -> None:
        """Legacy: run full derived chain (all sub-chains).

        Kept for backward compatibility; new code should call
        _run_daily_derived_chain() or _run_monthly_derived_chain() instead.
        """
        self._run_derived_foundation_chain(ctx, start_date, end_date)
        self._run_derived_mainline_chain(ctx, start_date, end_date, include_validations=include_validations)
        self._run_derived_alpha_chain(
            ctx,
            start_date,
            end_date,
            include_validations=include_validations,
            include_crosswalk=include_crosswalk,
        )

    def _run_daily_derived_chain(self, ctx, start_date: str, end_date: str, *, include_validations: bool, include_crosswalk: bool) -> None:
        """Daily derived chain: market, rotation, mainline, alpha.

        Owned by V8DailyOpsTask. Covers:
          - mainline strength, radar, market pulse (two passes)
          - local industry proxy, mainline lifecycle
          - industry capital flow, stock role map
          - unified alpha score
        """
        self._run_derived_mainline_chain(ctx, start_date, end_date, include_validations=include_validations)
        start_iso = _to_iso_date(start_date)
        end_iso = _to_iso_date(end_date)
        db_args = self._db_cli_args(ctx)
        chunk_small = str(_env_int("V8_DERIVED_CHUNK_MONTHS_SMALL", 1))
        chunk_medium = str(_env_int("V8_DERIVED_CHUNK_MONTHS_MEDIUM", 3))
        if not _env_flag("V8_SKIP_INDUSTRY_CAPITAL_FLOW", False):
            self._run_optional_python_script(
                ctx,
                "scripts/build_industry_capital_flow_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_small] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_INDUSTRY_CAPITAL_FLOW=1; skipped build_industry_capital_flow_daily.py")
        if not _env_flag("V8_SKIP_GA_STOCK_ROLE_MAP", False):
            self._run_optional_python_script(
                ctx,
                "scripts/build_ga_stock_role_map_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_medium] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_GA_STOCK_ROLE_MAP=1; skipped build_ga_stock_role_map_daily.py")
        self._run_derived_alpha_chain(
            ctx,
            start_date,
            end_date,
            include_validations=include_validations,
            include_crosswalk=include_crosswalk,
        )

    def _run_monthly_derived_chain(self, ctx, start_date: str, end_date: str, *, include_validations: bool, include_crosswalk: bool) -> None:
        """Monthly derived chain: financial statements & quality factors.

        Owned by V8MonthlyOpsTask / V8MonthlyDerivedTask. Covers:
          - stock fundamental daily (financial snapshot)
          - stock quality score
          - unified alpha score (optional, disabled by default)

        NOTE: Monthly pipeline does NOT backfill daily-frequency data
        (mainline/rotation/alpha). Those are owned by the Daily pipeline.
        Unified alpha score refresh here is optional (V8_MONTHLY_INCLUDE_ALPHA=1).
        """
        start_iso = _to_iso_date(start_date)
        end_iso = _to_iso_date(end_date)
        db_args = self._db_cli_args(ctx)
        chunk_medium = str(_env_int("V8_DERIVED_CHUNK_MONTHS_MEDIUM", 3))
        if not _env_flag("V8_SKIP_STOCK_FUNDAMENTAL_DAILY", False):
            self._run_python_script(
                ctx,
                "scripts/build_stock_fundamental_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace"],
            )
        else:
            ctx.log.info("[V8] V8_SKIP_STOCK_FUNDAMENTAL_DAILY=1; skipped build_stock_fundamental_daily.py")
        if not _env_flag("V8_SKIP_STOCK_QUALITY_SCORE", False):
            self._run_optional_python_script(
                ctx,
                "scripts/build_stock_quality_score_daily.py",
                ["--start", start_iso, "--end", end_iso, "--replace", "--chunk-months", chunk_medium] + db_args,
            )
        else:
            ctx.log.info("[V8] V8_SKIP_STOCK_QUALITY_SCORE=1; skipped build_stock_quality_score_daily.py")
        # Monthly pipeline does NOT run _run_latest_snap() — that is a daily-frequency
        # leader snapshot owned by the Weekly pipeline (v8_weekly_finalize).
        # Unified alpha score is optional here; daily pipeline handles it by default.
        if _env_flag("V8_MONTHLY_INCLUDE_ALPHA", False):
            self._run_derived_alpha_chain(
                ctx,
                start_date,
                end_date,
                include_validations=include_validations,
                include_crosswalk=include_crosswalk,
            )
        else:
            ctx.log.info("[V8] V8_MONTHLY_INCLUDE_ALPHA=0; skipped unified alpha score (daily pipeline handles it)")

    def _repair_index_gaps_from_audit(self, ctx, audit_frames: dict[str, pd.DataFrame]) -> int:
        repair_dates_by_index: dict[str, set] = {}
        gap_df = audit_frames.get("index_gap")
        if gap_df is not None and not gap_df.empty:
            for _, row in gap_df.iterrows():
                index_code = str(row.get("index_code") or "").strip()
                missing_date = pd.to_datetime(row.get("missing_date"), errors="coerce")
                if not index_code or pd.isna(missing_date):
                    continue
                repair_dates_by_index.setdefault(index_code, set()).add(missing_date.date())

        inserted_total = 0
        if repair_dates_by_index:
            task = IndexLoaderTask()
            for index_code, dates in repair_dates_by_index.items():
                inserted = task.repair_specific_dates(ctx, index_code=index_code, dates=sorted(dates))
                inserted_total += inserted
                ctx.log.info("[V8] targeted index repair index_code=%s dates=%s inserted=%s", index_code, len(dates), inserted)
        return inserted_total

    def _run_daily_market_raw_chain(self, ctx) -> None:
        self._run_tasks(
            ctx,
            [
                StockLoaderTask(),
                IndexLoaderTask(),
            ],
        )

    def _run_daily_reference_chain(self, ctx) -> None:
        ref_tasks: list[object] = [
            SwIndustryDailyTask(),
            SectorRotationSnapshotTask(),
            EventLoaderTask(name="EventLoaderDaily", frequency_tag="daily"),
        ]

        if _env_flag("V8_SKIP_ROTATION_SNAPSHOT", False):
            before = len(ref_tasks)
            ref_tasks = [
                t for t in ref_tasks
                if not isinstance(t, SectorRotationSnapshotTask)
                and getattr(t, "name", type(t).__name__) != "SectorRotationSnapshotTask"
            ]
            skipped = before - len(ref_tasks)
            if skipped:
                ctx.log.info("[V8] V8_SKIP_ROTATION_SNAPSHOT=1; skipped SectorRotationSnapshotTask")

        if _env_flag("V8_SKIP_EVENT_LOADER", False):
            before = len(ref_tasks)
            ref_tasks = [
                t for t in ref_tasks
                if not isinstance(t, EventLoaderTask)
                and getattr(t, "name", type(t).__name__) != "EventLoaderDaily"
            ]
            skipped = before - len(ref_tasks)
            if skipped:
                ctx.log.info("[V8] V8_SKIP_EVENT_LOADER=1; skipped EventLoaderDaily")

        self._run_tasks(ctx, ref_tasks)


    def _run_daily_raw_chain(self, ctx) -> None:
        self._run_daily_market_raw_chain(ctx)
        self._run_daily_reference_chain(ctx)

    def _run_weekly_refresh_chain(self, ctx) -> None:
        self._run_tasks(
            ctx,
            [
                BoardMembershipRefreshTask(),
                StockBasicWeeklyTask(),
                EventLoaderTask(name="EventLoaderPeriodicWeekly", frequency_tag="periodic"),
            ],
        )

    def _run_weekly_finalize_chain(self, ctx, trade_date: str) -> None:
        if _env_flag("V8_WEEKLY_REFRESH_LATEST_SNAPSHOT", True):
            self._run_latest_snap(ctx, trade_date)
        if _env_flag("V8_WEEKLY_BUILD_CROSSWALK_LATEST", True):
            self._run_crosswalk_latest(ctx)

    def _run_monthly_refresh_chain(self, ctx) -> None:
        with _temporary_env(
            {
                "STOCK_FUNDAMENTAL_MONTHLY_FORCE": "1",
                "STOCK_FUNDAMENTAL_MODE": os.getenv("V8_MONTHLY_FUNDAMENTAL_MODE", "all"),
            }
        ):
            self._run_tasks(
                ctx,
                [
                    StockFundamentalMonthlyTask(),
                    EventLoaderTask(name="EventLoaderPeriodicMonthly", frequency_tag="periodic"),
                ],
            )


@dataclass
class V8StockTask(_V8OpsMixin):
    name: str = "V8StockTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] stock refresh start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_tasks(ctx, [StockLoaderTask()])
        ctx.log.info("[V8] stock refresh complete")


@dataclass
class V8IndexTask(_V8OpsMixin):
    name: str = "V8IndexTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] index refresh start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_tasks(ctx, [IndexLoaderTask()])
        ctx.log.info("[V8] index refresh complete")


@dataclass
class V8BoardRefreshTask(_V8OpsMixin):
    name: str = "V8BoardRefreshTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] board refresh start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_tasks(ctx, [BoardMembershipRefreshTask()])
        ctx.log.info("[V8] board refresh complete")


@dataclass
class V8StockBasicTask(_V8OpsMixin):
    name: str = "V8StockBasicTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] stock basic refresh start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_tasks(ctx, [StockBasicWeeklyTask()])
        ctx.log.info("[V8] stock basic refresh complete")


@dataclass
class V8SwIndustryDailyTask(_V8OpsMixin):
    name: str = "V8SwIndustryDailyTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] sw industry daily start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_tasks(ctx, [SwIndustryDailyTask()])
        ctx.log.info("[V8] sw industry daily complete")


@dataclass
class V8RotationTask(_V8OpsMixin):
    name: str = "V8RotationTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] rotation refresh start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_tasks(ctx, [SectorRotationSnapshotTask()])
        ctx.log.info("[V8] rotation refresh complete")


@dataclass
class V8RotationAuditTask(_V8OpsMixin):
    name: str = "V8RotationAuditTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        task = SectorRotationSnapshotTask()
        audit = task.audit_coverage(ctx)
        effective_end = audit["effective_end_date"][0] if audit.get("effective_end_date") else cfg.end_date
        upstream_missing = audit.get("upstream_missing", [])
        bt_missing = audit.get("bt_missing", [])
        snap_missing = audit.get("snap_missing", [])
        ctx.log.info(
            "[V8] rotation audit end=%s upstream_missing=%s bt_missing=%s snap_missing=%s",
            effective_end,
            len(upstream_missing),
            len(bt_missing),
            len(snap_missing),
        )
        if upstream_missing or bt_missing or snap_missing:
            self._write_rotation_audit_detail(ctx, audit, "rotation_audit")
        fatal_snap_missing = bool(snap_missing) and self._rotation_snap_missing_is_fatal()
        if upstream_missing or bt_missing or fatal_snap_missing:
            raise RuntimeError(
                "[V8] rotation audit found fatal gaps "
                f"(upstream_missing={len(upstream_missing)} bt_missing={len(bt_missing)} "
                f"snap_missing={len(snap_missing)} snap_fatal={fatal_snap_missing})"
            )
        if snap_missing:
            ctx.log.warning(
                "[V8] rotation audit found non-fatal snapshot gaps "
                "(snap_missing=%s; set V8_ROTATION_FAIL_ON_SNAP_MISSING=1 to fail)",
                len(snap_missing),
            )
        else:
            ctx.log.info("[V8] rotation audit complete without gaps")


@dataclass
class V8RotationRepairTask(_V8OpsMixin):
    name: str = "V8RotationRepairTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        with _temporary_env(
            {
                "ROTATION_AUTO_BACKFILL_MISSING": "1",
                "ROTATION_AUTO_REPAIR_DOWNSTREAM": "1",
            }
        ):
            task = SectorRotationSnapshotTask()
            ctx.log.info("[V8] rotation repair start window=%s..%s", cfg.start_date, cfg.end_date)
            task.run(ctx)
            audit = task.audit_coverage(ctx)
        upstream_missing = audit.get("upstream_missing", [])
        bt_missing = audit.get("bt_missing", [])
        snap_missing = audit.get("snap_missing", [])
        if upstream_missing or bt_missing or snap_missing:
            self._write_rotation_audit_detail(ctx, audit, "rotation_repair")
        fatal_snap_missing = bool(snap_missing) and self._rotation_snap_missing_is_fatal()
        if upstream_missing or bt_missing or fatal_snap_missing:
            raise RuntimeError(
                "[V8] rotation repair completed but fatal gaps remain "
                f"(upstream_missing={len(upstream_missing)} bt_missing={len(bt_missing)} "
                f"snap_missing={len(snap_missing)} snap_fatal={fatal_snap_missing})"
            )
        if snap_missing:
            ctx.log.warning(
                "[V8] rotation repair completed with non-fatal snapshot gaps "
                "(snap_missing=%s; detail CSV written; set V8_ROTATION_FAIL_ON_SNAP_MISSING=1 to fail)",
                len(snap_missing),
            )
        else:
            ctx.log.info("[V8] rotation repair complete without remaining gaps")


@dataclass
class V8EventDailyTask(_V8OpsMixin):
    name: str = "V8EventDailyTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] event daily refresh start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_tasks(ctx, [EventLoaderTask(name="EventLoaderDaily", frequency_tag="daily")])
        ctx.log.info("[V8] event daily refresh complete")


@dataclass
class V8EventPeriodicTask(_V8OpsMixin):
    name: str = "V8EventPeriodicTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] event periodic refresh start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_tasks(ctx, [EventLoaderTask(name="EventLoaderPeriodic", frequency_tag="periodic")])
        ctx.log.info("[V8] event periodic refresh complete")


@dataclass
class V8StockFundamentalRefreshTask(_V8OpsMixin):
    name: str = "V8StockFundamentalRefreshTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] stock fundamental monthly refresh start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_monthly_refresh_chain(ctx)
        ctx.log.info("[V8] stock fundamental monthly refresh complete")


@dataclass
class V8DailyMarketRawTask(_V8OpsMixin):
    name: str = "V8DailyMarketRawTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] daily market raw start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_daily_market_raw_chain(ctx)
        ctx.log.info("[V8] daily market raw complete")


@dataclass
class V8DailyReferenceRefreshTask(_V8OpsMixin):
    name: str = "V8DailyReferenceRefreshTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] daily reference refresh start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_daily_reference_chain(ctx)
        ctx.log.info("[V8] daily reference refresh complete")


@dataclass
class V8DailyAuditTask(_V8OpsMixin):
    name: str = "V8DailyAuditTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        audit_start, audit_end = self._resolve_daily_audit_window(cfg.end_date)
        ctx.log.info("[V8] daily audit start window=%s..%s", audit_start, audit_end)
        audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
        self._log_audit_summary(ctx, audit_frames, prefix="daily-standalone")
        inserted = self._repair_index_gaps_from_audit(ctx, audit_frames)
        if inserted > 0:
            audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
            self._log_audit_summary(ctx, audit_frames, prefix="daily-standalone-post-index-repair")
        if self._audit_has_issues(audit_frames):
            raise RuntimeError("[V8] daily standalone audit found coverage gaps")
        ctx.log.info("[V8] daily audit complete without gaps")


@dataclass
class V8DailyDerivedFoundationTask(_V8OpsMixin):
    name: str = "V8DailyDerivedFoundationTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] daily derived foundation start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_derived_foundation_chain(ctx, cfg.start_date, cfg.end_date)
        ctx.log.info("[V8] daily derived foundation complete")


@dataclass
class V8DailyDerivedMainlineTask(_V8OpsMixin):
    name: str = "V8DailyDerivedMainlineTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        include_validations = _env_flag("V8_DAILY_INCLUDE_VALIDATIONS", True)
        ctx.log.info(
            "[V8] daily derived mainline start window=%s..%s include_validations=%s",
            cfg.start_date,
            cfg.end_date,
            include_validations,
        )
        self._run_derived_mainline_chain(ctx, cfg.start_date, cfg.end_date, include_validations=include_validations)
        ctx.log.info("[V8] daily derived mainline complete")


@dataclass
class V8DailyDerivedAlphaTask(_V8OpsMixin):
    name: str = "V8DailyDerivedAlphaTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        include_validations = _env_flag("V8_DAILY_INCLUDE_VALIDATIONS", True)
        include_crosswalk = _env_flag("V8_DAILY_INCLUDE_CROSSWALK_LATEST", True)
        ctx.log.info(
            "[V8] daily derived alpha start window=%s..%s include_validations=%s include_crosswalk=%s",
            cfg.start_date,
            cfg.end_date,
            include_validations,
            include_crosswalk,
        )
        self._run_derived_alpha_chain(
            ctx,
            cfg.start_date,
            cfg.end_date,
            include_validations=include_validations,
            include_crosswalk=include_crosswalk,
        )
        ctx.log.info("[V8] daily derived alpha complete")


@dataclass
class V8WeeklyRefreshTask(_V8OpsMixin):
    name: str = "V8WeeklyRefreshTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] weekly refresh start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_weekly_refresh_chain(ctx)
        ctx.log.info("[V8] weekly refresh complete")


@dataclass
class V8WeeklyAuditTask(_V8OpsMixin):
    name: str = "V8WeeklyAuditTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] weekly reference audit start window=%s..%s", cfg.start_date, cfg.end_date)
        self._assert_weekly_reference_tables_ready(ctx)
        ctx.log.info("[V8] weekly reference audit complete")


@dataclass
class V8WeeklyAuditStockTask(_V8OpsMixin):
    name: str = "V8WeeklyAuditStockTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        audit_start, audit_end = self._resolve_audit_window("V8_WEEKLY", cfg.end_date, 180)
        ctx.log.info("[V8] weekly stock coverage audit start window=%s..%s", audit_start, audit_end)
        audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
        self._log_audit_summary(ctx, audit_frames, prefix="weekly-stock-standalone")
        if self._audit_has_stock_issues(audit_frames):
            raise RuntimeError("[V8] weekly stock coverage audit found stock gaps")
        ctx.log.info("[V8] weekly stock coverage audit complete without gaps")


@dataclass
class V8WeeklyAuditIndexTask(_V8OpsMixin):
    name: str = "V8WeeklyAuditIndexTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        audit_start, audit_end = self._resolve_audit_window("V8_WEEKLY", cfg.end_date, 180)
        ctx.log.info("[V8] weekly index coverage audit start window=%s..%s", audit_start, audit_end)
        audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
        self._log_audit_summary(ctx, audit_frames, prefix="weekly-index-standalone")
        inserted = self._repair_index_gaps_from_audit(ctx, audit_frames)
        if inserted > 0:
            audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
            self._log_audit_summary(ctx, audit_frames, prefix="weekly-index-standalone-post-index-repair")
        if self._audit_has_index_gap_issues(audit_frames):
            raise RuntimeError("[V8] weekly index coverage audit found index gaps")
        if _env_flag("V8_FAIL_ON_INDEX_WINDOW_START", False) and self._audit_has_index_window_start_issues(audit_frames):
            raise RuntimeError("[V8] weekly index coverage audit found index window-start gaps")
        ctx.log.info("[V8] weekly index coverage audit complete without gaps")


@dataclass
class V8WeeklyAuditMarketTask(_V8OpsMixin):
    name: str = "V8WeeklyAuditMarketTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        audit_start, audit_end = self._resolve_audit_window("V8_WEEKLY", cfg.end_date, 180)
        ctx.log.info("[V8] weekly market coverage audit start window=%s..%s", audit_start, audit_end)
        audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
        self._log_audit_summary(ctx, audit_frames, prefix="weekly-market-standalone")
        inserted = self._repair_index_gaps_from_audit(ctx, audit_frames)
        if inserted > 0:
            audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
            self._log_audit_summary(ctx, audit_frames, prefix="weekly-market-standalone-post-index-repair")
        if self._audit_has_stock_issues(audit_frames) or self._audit_has_index_gap_issues(audit_frames):
            raise RuntimeError("[V8] weekly market coverage audit found stock/index gaps")
        if _env_flag("V8_FAIL_ON_INDEX_WINDOW_START", False) and self._audit_has_index_window_start_issues(audit_frames):
            raise RuntimeError("[V8] weekly market coverage audit found index window-start gaps")
        ctx.log.info("[V8] weekly market coverage audit complete without gaps")


@dataclass
class V8WeeklyFinalizeTask(_V8OpsMixin):
    name: str = "V8WeeklyFinalizeTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] weekly finalize start trade_date=%s", cfg.end_date)
        self._run_weekly_finalize_chain(ctx, cfg.end_date)
        ctx.log.info("[V8] weekly finalize complete")


@dataclass
class V8MonthlyRefreshTask(_V8OpsMixin):
    name: str = "V8MonthlyRefreshTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        ctx.log.info("[V8] monthly refresh start window=%s..%s", cfg.start_date, cfg.end_date)
        self._run_monthly_refresh_chain(ctx)
        ctx.log.info("[V8] monthly refresh complete")


@dataclass
class V8MonthlyAuditTask(_V8OpsMixin):
    name: str = "V8MonthlyAuditTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        audit_start, audit_end = self._resolve_audit_window("V8_MONTHLY", cfg.end_date, 365)
        ctx.log.info("[V8] monthly audit start window=%s..%s", audit_start, audit_end)
        audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
        self._log_audit_summary(ctx, audit_frames, prefix="monthly-standalone")
        inserted = self._repair_index_gaps_from_audit(ctx, audit_frames)
        if inserted > 0:
            audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
            self._log_audit_summary(ctx, audit_frames, prefix="monthly-standalone-post-index-repair")
        if self._audit_has_issues(audit_frames):
            raise RuntimeError("[V8] monthly standalone audit found coverage gaps")
        ctx.log.info("[V8] monthly audit complete without gaps")


@dataclass
class V8MonthlyDerivedTask(_V8OpsMixin):
    name: str = "V8MonthlyDerivedTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        include_validations = _env_flag("V8_MONTHLY_INCLUDE_VALIDATIONS", True)
        include_crosswalk = _env_flag("V8_MONTHLY_INCLUDE_CROSSWALK_LATEST", True)
        ctx.log.info(
            "[V8] monthly derived start window=%s..%s include_validations=%s include_crosswalk=%s",
            cfg.start_date,
            cfg.end_date,
            include_validations,
            include_crosswalk,
        )
        self._run_monthly_derived_chain(
            ctx,
            cfg.start_date,
            cfg.end_date,
            include_validations=include_validations,
            include_crosswalk=include_crosswalk,
        )
        ctx.log.info("[V8] monthly derived complete")


@dataclass
class V8DailyOpsTask(_V8OpsMixin):
    name: str = "V8DailyOpsTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        include_validations = _env_flag("V8_DAILY_INCLUDE_VALIDATIONS", True)
        include_crosswalk = _env_flag("V8_DAILY_INCLUDE_CROSSWALK_LATEST", True)
        audit_start, audit_end = self._resolve_daily_audit_window(cfg.end_date)
        ctx.log.info("[V8] daily ops start window=%s..%s", cfg.start_date, cfg.end_date)
        ctx.log.info("[V8] daily audit window=%s..%s", audit_start, audit_end)

        repair_enabled = _env_flag("V8_DAILY_AUTO_REPAIR", True)

        try:
            self._run_daily_raw_chain(ctx)
            audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
            self._log_audit_summary(ctx, audit_frames, prefix="daily-first-pass")
            self._repair_index_gaps_from_audit(ctx, audit_frames)
            audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
            self._log_audit_summary(ctx, audit_frames, prefix="daily-post-index-repair")
            self._run_daily_derived_chain(
                ctx,
                cfg.start_date,
                cfg.end_date,
                include_validations=include_validations,
                include_crosswalk=include_crosswalk,
            )
            if not self._audit_has_issues(audit_frames):
                ctx.log.info("[V8] daily ops complete without repair")
                return
            if not repair_enabled:
                raise RuntimeError("[V8] daily audit found gaps and V8_DAILY_AUTO_REPAIR is disabled")
        except Exception as exc:
            if not repair_enabled:
                raise
            ctx.log.warning("[V8] daily first pass failed; auto-repair will retry recent window: %s", exc)

        repair_start = self._resolve_repair_start(cfg.end_date, audit_frames if "audit_frames" in locals() else {})
        repair_span_days = (_parse_yyyymmdd(cfg.end_date) - _parse_yyyymmdd(repair_start)).days + 1
        ctx.log.warning(
            "[V8] daily auto-repair start=%s end=%s span_days=%s",
            repair_start,
            cfg.end_date,
            repair_span_days,
        )

        with _temporary_cfg_dates(ctx, repair_start, cfg.end_date):
            self._run_daily_raw_chain(ctx)
            audit_frames = self._run_coverage_audit_for_window(ctx, repair_start, cfg.end_date)
            self._log_audit_summary(ctx, audit_frames, prefix="daily-second-pass")
            self._repair_index_gaps_from_audit(ctx, audit_frames)
            audit_frames = self._run_coverage_audit_for_window(ctx, repair_start, cfg.end_date)
            self._log_audit_summary(ctx, audit_frames, prefix="daily-second-pass-post-index-repair")
            self._run_daily_derived_chain(
                ctx,
                repair_start,
                cfg.end_date,
                include_validations=include_validations,
                include_crosswalk=include_crosswalk,
            )

        if self._audit_has_issues(audit_frames):
            raise RuntimeError("[V8] daily auto-repair completed but coverage audit still reports gaps")
        ctx.log.info("[V8] daily ops complete after auto-repair")


@dataclass
class V8WeeklyOpsTask(_V8OpsMixin):
    name: str = "V8WeeklyOpsTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        audit_start, audit_end = self._resolve_audit_window("V8_WEEKLY", cfg.end_date, 180)
        repair_enabled = _env_flag("V8_WEEKLY_AUTO_REPAIR", True)
        ctx.log.info("[V8] weekly ops start window=%s..%s", cfg.start_date, cfg.end_date)
        ctx.log.info("[V8] weekly audit window=%s..%s", audit_start, audit_end)
        try:
            self._run_tasks(
                ctx,
                [
                    BoardMembershipRefreshTask(),
                    StockBasicWeeklyTask(),
                    EventLoaderTask(name="EventLoaderPeriodicWeekly", frequency_tag="periodic"),
                ],
            )
            audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
            self._log_audit_summary(ctx, audit_frames, prefix="weekly-first-pass")
            self._repair_index_gaps_from_audit(ctx, audit_frames)
            audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
            self._log_audit_summary(ctx, audit_frames, prefix="weekly-post-index-repair")
            if _env_flag("V8_WEEKLY_REFRESH_LATEST_SNAPSHOT", True):
                self._run_latest_snap(ctx, cfg.end_date)
            if _env_flag("V8_WEEKLY_BUILD_CROSSWALK_LATEST", True):
                self._run_crosswalk_latest(ctx)
            if not self._audit_has_issues(audit_frames):
                ctx.log.info("[V8] weekly ops complete without repair")
                return
            if not repair_enabled:
                raise RuntimeError("[V8] weekly audit found gaps and V8_WEEKLY_AUTO_REPAIR is disabled")
        except Exception as exc:
            if not repair_enabled:
                raise
            ctx.log.warning("[V8] weekly first pass failed; auto-repair will retry expanded window: %s", exc)

        repair_start = self._resolve_repair_start_for_prefix("V8_WEEKLY", cfg.end_date, audit_frames if "audit_frames" in locals() else {}, 30, 730)
        repair_span_days = (_parse_yyyymmdd(cfg.end_date) - _parse_yyyymmdd(repair_start)).days + 1
        ctx.log.warning("[V8] weekly auto-repair start=%s end=%s span_days=%s", repair_start, cfg.end_date, repair_span_days)
        with _temporary_cfg_dates(ctx, repair_start, cfg.end_date):
            self._run_tasks(
                ctx,
                [
                    BoardMembershipRefreshTask(),
                    StockBasicWeeklyTask(),
                    EventLoaderTask(name="EventLoaderPeriodicWeeklyRepair", frequency_tag="periodic"),
                ],
            )
            audit_frames = self._run_coverage_audit_for_window(ctx, repair_start, cfg.end_date)
            self._log_audit_summary(ctx, audit_frames, prefix="weekly-second-pass")
            self._repair_index_gaps_from_audit(ctx, audit_frames)
            audit_frames = self._run_coverage_audit_for_window(ctx, repair_start, cfg.end_date)
            self._log_audit_summary(ctx, audit_frames, prefix="weekly-second-pass-post-index-repair")
            if _env_flag("V8_WEEKLY_REFRESH_LATEST_SNAPSHOT", True):
                self._run_latest_snap(ctx, cfg.end_date)
            if _env_flag("V8_WEEKLY_BUILD_CROSSWALK_LATEST", True):
                self._run_crosswalk_latest(ctx)
        if self._audit_has_issues(audit_frames):
            raise RuntimeError("[V8] weekly auto-repair completed but coverage audit still reports gaps")
        ctx.log.info("[V8] weekly ops complete")


@dataclass
class V8MonthlyOpsTask(_V8OpsMixin):
    name: str = "V8MonthlyOpsTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")
        audit_start, audit_end = self._resolve_audit_window("V8_MONTHLY", cfg.end_date, 365)
        repair_enabled = _env_flag("V8_MONTHLY_AUTO_REPAIR", True)
        ctx.log.info("[V8] monthly ops start window=%s..%s", cfg.start_date, cfg.end_date)
        ctx.log.info("[V8] monthly audit window=%s..%s", audit_start, audit_end)
        try:
            with _temporary_env(
                {
                    "STOCK_FUNDAMENTAL_MONTHLY_FORCE": "1",
                    "STOCK_FUNDAMENTAL_MODE": os.getenv("V8_MONTHLY_FUNDAMENTAL_MODE", "all"),
                }
            ):
                self._run_tasks(
                    ctx,
                    [
                        StockFundamentalMonthlyTask(),
                        EventLoaderTask(name="EventLoaderPeriodicMonthly", frequency_tag="periodic"),
                    ],
                )
            audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
            self._log_audit_summary(ctx, audit_frames, prefix="monthly-first-pass")
            self._repair_index_gaps_from_audit(ctx, audit_frames)
            audit_frames = self._run_coverage_audit_for_window(ctx, audit_start, audit_end)
            self._log_audit_summary(ctx, audit_frames, prefix="monthly-post-index-repair")
            if _env_flag("V8_MONTHLY_INCLUDE_DERIVED_REFRESH", True):
                self._run_monthly_derived_chain(
                    ctx,
                    cfg.start_date,
                    cfg.end_date,
                    include_validations=_env_flag("V8_MONTHLY_INCLUDE_VALIDATIONS", True),
                    include_crosswalk=_env_flag("V8_MONTHLY_INCLUDE_CROSSWALK_LATEST", True),
                )
            if not self._audit_has_issues(audit_frames):
                ctx.log.info("[V8] monthly ops complete without repair")
                return
            if not repair_enabled:
                raise RuntimeError("[V8] monthly audit found gaps and V8_MONTHLY_AUTO_REPAIR is disabled")
        except Exception as exc:
            if not repair_enabled:
                raise
            ctx.log.warning("[V8] monthly first pass failed; auto-repair will retry expanded window: %s", exc)

        repair_start = self._resolve_repair_start_for_prefix("V8_MONTHLY", cfg.end_date, audit_frames if "audit_frames" in locals() else {}, 60, 1095)
        repair_span_days = (_parse_yyyymmdd(cfg.end_date) - _parse_yyyymmdd(repair_start)).days + 1
        ctx.log.warning("[V8] monthly auto-repair start=%s end=%s span_days=%s", repair_start, cfg.end_date, repair_span_days)
        with _temporary_cfg_dates(ctx, repair_start, cfg.end_date):
            with _temporary_env(
                {
                    "STOCK_FUNDAMENTAL_MONTHLY_FORCE": "1",
                    "STOCK_FUNDAMENTAL_MODE": os.getenv("V8_MONTHLY_FUNDAMENTAL_MODE", "all"),
                }
            ):
                self._run_tasks(
                    ctx,
                    [
                        StockFundamentalMonthlyTask(),
                        EventLoaderTask(name="EventLoaderPeriodicMonthlyRepair", frequency_tag="periodic"),
                    ],
                )
            audit_frames = self._run_coverage_audit_for_window(ctx, repair_start, cfg.end_date)
            self._log_audit_summary(ctx, audit_frames, prefix="monthly-second-pass")
            self._repair_index_gaps_from_audit(ctx, audit_frames)
            audit_frames = self._run_coverage_audit_for_window(ctx, repair_start, cfg.end_date)
            self._log_audit_summary(ctx, audit_frames, prefix="monthly-second-pass-post-index-repair")
            if _env_flag("V8_MONTHLY_INCLUDE_DERIVED_REFRESH", True):
                self._run_monthly_derived_chain(
                    ctx,
                    repair_start,
                    cfg.end_date,
                    include_validations=_env_flag("V8_MONTHLY_INCLUDE_VALIDATIONS", True),
                    include_crosswalk=_env_flag("V8_MONTHLY_INCLUDE_CROSSWALK_LATEST", True),
                )
        if self._audit_has_issues(audit_frames):
            raise RuntimeError("[V8] monthly auto-repair completed but coverage audit still reports gaps")
        ctx.log.info("[V8] monthly ops complete")


@dataclass
class V8HistoricalBackfillTask(_V8OpsMixin):
    name: str = "V8HistoricalBackfillTask"

    def run(self, ctx) -> None:
        cfg = ctx.config
        cfg.finalize_dates()
        activate_tunnel("cn")

        start_date = str(os.getenv("V8_BACKFILL_START", cfg.start_date or "20100101")).strip()
        end_date = str(os.getenv("V8_BACKFILL_END", cfg.end_date or datetime.today().strftime("%Y%m%d"))).strip()
        history_frequency = str(os.getenv("V8_BACKFILL_HIS_UNIVERSE_FREQUENCY", "monthly")).strip().lower() or "monthly"

        include_price = _env_flag("V8_BACKFILL_INCLUDE_PRICE", True)
        include_fundamental = _env_flag("V8_BACKFILL_INCLUDE_FUNDAMENTAL", True)
        include_sw_history = _env_flag("V8_BACKFILL_INCLUDE_SW_HISTORY", False)
        include_concept_history = _env_flag("V8_BACKFILL_INCLUDE_CONCEPT_HISTORY", False)
        include_daily_basic = _env_flag("V8_BACKFILL_INCLUDE_DAILY_BASIC", True)
        include_sw_daily = _env_flag("V8_BACKFILL_INCLUDE_SW_DAILY", True)
        include_board_refresh = _env_flag("V8_BACKFILL_INCLUDE_BOARD_REFRESH", True)
        include_derived = _env_flag("V8_BACKFILL_INCLUDE_DERIVED", True)
        include_validations = _env_flag("V8_BACKFILL_INCLUDE_VALIDATIONS", True)
        include_crosswalk = _env_flag("V8_BACKFILL_INCLUDE_CROSSWALK_LATEST", True)

        ctx.log.info(
            "[V8] backfill start=%s end=%s include_price=%s include_fundamental=%s include_sw_history=%s include_concept_history=%s include_daily_basic=%s include_sw_daily=%s include_board_refresh=%s include_derived=%s",
            start_date,
            end_date,
            include_price,
            include_fundamental,
            include_sw_history,
            include_concept_history,
            include_daily_basic,
            include_sw_daily,
            include_board_refresh,
            include_derived,
        )

        if include_price:
            with _temporary_cfg_history(ctx, start_date, end_date, history_frequency):
                HisStocksLoaderTask().run(ctx)

        if include_board_refresh:
            with _temporary_cfg_dates(ctx, start_date, end_date):
                BoardMembershipRefreshTask().run(ctx)

        if include_daily_basic:
            self._run_python_module(
                ctx,
                "app.tools.sync_cn_stock_daily_basic_from_tushare",
                [
                    "--provider",
                    _env_text("V8_BACKFILL_DAILY_BASIC_PROVIDER", "tushare"),
                    "--calendar-source",
                    _env_text("V8_BACKFILL_DAILY_BASIC_CALENDAR_SOURCE", "price"),
                    "--date-order",
                    _env_text("V8_BACKFILL_DAILY_BASIC_DATE_ORDER", "desc"),
                    "--batch-size",
                    str(_env_int("V8_BACKFILL_DAILY_BASIC_BATCH_SIZE", 20)),
                    "--start",
                    start_date,
                    "--end",
                    end_date,
                ],
            )

        if include_fundamental:
            with _temporary_cfg_dates(ctx, start_date, end_date):
                with _temporary_env(
                    {
                        "STOCK_FUNDAMENTAL_MONTHLY_FORCE": "1",
                        "STOCK_FUNDAMENTAL_MONTHLY_FULL_REBUILD": "1",
                        "STOCK_FUNDAMENTAL_MONTHLY_BY_SYMBOL": "1",
                        "STOCK_FUNDAMENTAL_MONTHLY_HISTORY_START": start_date,
                        "STOCK_FUNDAMENTAL_MODE": os.getenv("V8_BACKFILL_FUNDAMENTAL_MODE", "all"),
                    }
                ):
                    StockFundamentalMonthlyTask().run(ctx)

        if include_sw_daily:
            self._run_python_module(
                ctx,
                "app.tools.sync_cn_sw_industry_daily_from_tushare",
                [
                    "--start",
                    start_date,
                    "--end",
                    end_date,
                    "--src",
                    _env_text("V8_BACKFILL_SW_SRC", "SW2021"),
                    "--master-source",
                    _env_text("V8_BACKFILL_SW_MASTER_SOURCE", "TUSHARE_SW2021_L1"),
                    "--full",
                ],
            )

        if include_sw_history:
            sw_src = str(os.getenv("V8_BACKFILL_SW_SRC", "SW2021")).strip() or "SW2021"
            sw_level = str(os.getenv("V8_BACKFILL_SW_LEVEL", "L1")).strip().upper() or "L1"
            map_chunk_years = str(_env_int("V8_BACKFILL_SW_MAP_CHUNK_YEARS", 1))
            sw_args = [
                "--start",
                start_date,
                "--end",
                end_date,
                "--src",
                sw_src,
                "--level",
                sw_level,
                "--map-chunk-years",
                map_chunk_years,
            ]
            if _env_flag("V8_BACKFILL_KEEP_EXISTING_MEMBER_SOURCE", True):
                sw_args.append("--keep-existing-member-source")
            self._run_python_module(
                ctx,
                "app.tools.backfill_sw_industry_history_from_tushare",
                sw_args,
            )

        if include_concept_history:
            map_chunk_years = str(_env_int("V8_BACKFILL_CONCEPT_MAP_CHUNK_YEARS", 1))
            self._run_python_module(
                ctx,
                "app.tools.backfill_concept_history_from_tushare",
                [
                    "--start",
                    start_date,
                    "--end",
                    end_date,
                    "--source-label",
                    _env_text("V8_BACKFILL_CONCEPT_SOURCE_LABEL", "tushare_concept"),
                    "--map-chunk-years",
                    map_chunk_years,
                ],
            )

        if include_derived:
            self._run_derived_chain(
                ctx,
                start_date,
                end_date,
                include_validations=include_validations,
                include_crosswalk=include_crosswalk,
            )

        ctx.log.info("[V8] historical backfill complete start=%s end=%s", start_date, end_date)
