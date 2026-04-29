from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd
import tushare as ts
from sqlalchemy import text

from app.tasks.state_store import StateStore
from app.tools.sync_cn_stock_daily_price_from_tushare import (
    patch_pandas_fillna_method_compat,
    resolve_tushare_token,
)
from app.utils.wireguard_helper import activate_tunnel


ETF_HIST_TABLE = "CN_FUND_ETF_HIST_EM"

MYSQL_ETF_HIST_SQL = f"""
CREATE TABLE IF NOT EXISTS {ETF_HIST_TABLE} (
    CODE            VARCHAR(20)    NOT NULL,
    DATA_DATE       DATE           NOT NULL,
    OPEN_PRICE      DECIMAL(20,6),
    CLOSE_PRICE     DECIMAL(20,6),
    HIGH_PRICE      DECIMAL(20,6),
    LOW_PRICE       DECIMAL(20,6),
    VOLUME          DECIMAL(24,6),
    AMOUNT          DECIMAL(24,6),
    AMPLITUDE       DECIMAL(20,6),
    CHANGE_PCT      DECIMAL(20,6),
    CHANGE_AMOUNT   DECIMAL(20,6),
    TURNOVER_RATE   DECIMAL(20,6),
    ADJUST_TYPE     VARCHAR(10)    NOT NULL,
    SOURCE          VARCHAR(30),
    CREATED_AT      DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (CODE, DATA_DATE, ADJUST_TYPE),
    KEY IDX_FEH_CODE (CODE),
    KEY IDX_FEH_DATE (DATA_DATE),
    KEY IDX_FEH_ADJUST (ADJUST_TYPE)
)
"""


@dataclass
class EtfLoaderTask:
    name: str = "ETFLoader"

    def __init__(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.data_dir = os.path.join(root_dir, "data")

    def _table_ref(self, table_name: str) -> str:
        return table_name

    def _ensure_etf_table(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(MYSQL_ETF_HIST_SQL))

    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        self.engine = ctx.engine
        self.data_source_flag = str(getattr(cfg, "data_source_flag", "tu") or "tu").strip().lower()
        self._ts_pro = None
        self._ts_token = None

        scanned_file = getattr(cfg, "etf_scanned_file", Path(self.data_dir) / "state" / "ETF_scanned.json")
        failed_file = getattr(cfg, "etf_failed_file", Path(self.data_dir) / "state" / "ETF_failed.json")
        state = StateStore(scanned_file, failed_file, self.log)
        state_flush_every = int(getattr(cfg, "state_flush_every", 50) or 0)

        if cfg.look_back_days >= 1:
            self.load_fund_etf_hist(
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                state=state,
                is_continue_load=True,
                state_flush_every=state_flush_every,
            )
            self.log.info("[DONE] ETF historical loader finished")
        else:
            self.load_spot_as_hist_today()
            self.log.info("[DONE] ETF spot-as-hist finished")

    def get_state_files(self, cfg):
        return [
            getattr(cfg, "etf_scanned_file", Path(self.data_dir) / "state" / "ETF_scanned.json"),
            getattr(cfg, "etf_failed_file", Path(self.data_dir) / "state" / "ETF_failed.json"),
        ]

    def load_fund_etf_hist(
        self,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjustflag: str = "3",
        *,
        state: StateStore,
        is_continue_load: bool = True,
        state_flush_every: int = 50,
    ):
        self.log.info(
            f"=== Historical ETF load started: {start_date} -> {end_date} | frequency={frequency} | adjustflag={adjustflag} | source={self.data_source_flag} ==="
        )
        self._ensure_etf_table()
        activate_tunnel("cn")

        patch_pandas_fillna_method_compat()
        try:
            token, _ = resolve_tushare_token("", "")
            self._ts_token = token.strip() if token else ""
            if self._ts_token:
                self._ts_pro = ts.pro_api(self._ts_token)
            else:
                self.log.warning("[ETF] Tushare token not found; will use AkShare fallback only.")
        except Exception as e:
            self.log.warning(f"[ETF] Tushare token resolve failed: {e}")

        codes = self._resolve_etf_codes()
        if not codes:
            self.log.info("[ETF] no ETF codes resolved, skip")
            return

        scanned = state.load_scanned() if is_continue_load else set()
        failed = state.load_failed() if is_continue_load else set()
        total = 0

        for i, code in enumerate(codes, start=1):
            if state_flush_every > 0 and i % state_flush_every == 0:
                state.save_scanned(scanned)
                state.save_failed(failed)
                self.log.info(f"[ETF][STATE] flushed scanned/failed at {i}")

            if code in scanned:
                self.log.info(f"[ETF] ({i}/{len(codes)}) {code} SKIP scanned")
                continue
            if code in failed:
                self.log.info(f"[ETF] ({i}/{len(codes)}) {code} SKIP failed")
                continue

            try:
                if self.data_source_flag == "tu":
                    df = self._load_etf_hist_tushare(code, start_date, end_date)
                    source = "TUSHARE"
                else:
                    df = self._load_etf_hist_akshare(code, start_date, end_date, adjustflag)
                    source = "AKSHARE"
                if df is None or df.empty:
                    raise RuntimeError(f"no ETF history returned from source={self.data_source_flag}")

                count = self._finalize_and_upsert(df, code, source, adjustflag)
                total += count
                scanned.add(code)
                self.log.info(f"[ETF] ({i}/{len(codes)}) {code} ok rows={count} source={source}")
            except Exception as e:
                failed.add(code)
                self.log.info(f"[ETF] ({i}/{len(codes)}) {code} failed: {e}")

        state.save_scanned(scanned)
        state.save_failed(failed)
        self.log.info("[DONE][ETF] state saved")
        self.log.info(f"[DONE][ETF] total merged rows={total}")
        if failed:
            self.log.info(f"[DONE][ETF] failed symbols={len(failed)}")

    def _resolve_etf_codes(self) -> list[str]:
        codes: set[str] = set()
        try:
            spot_df = ak.fund_etf_spot_em()
            if spot_df is not None and not spot_df.empty:
                col = "代码" if "代码" in spot_df.columns else "CODE"
                for raw in spot_df[col].dropna().astype(str):
                    code = self._normalize_etf_code(raw)
                    if code:
                        codes.add(code)
        except Exception as e:
            self.log.warning(f"[ETF] resolve codes from AkShare spot failed: {e}")

        try:
            with self.engine.connect() as conn:
                hist_codes = pd.read_sql(f"SELECT DISTINCT CODE FROM {self._table_ref(ETF_HIST_TABLE)}", conn)
            if hist_codes is not None and not hist_codes.empty:
                col = "CODE" if "CODE" in hist_codes.columns else hist_codes.columns[0]
                for raw in hist_codes[col].dropna().astype(str):
                    code = self._normalize_etf_code(raw)
                    if code:
                        codes.add(code)
        except Exception as e:
            self.log.warning(f"[ETF] resolve codes from history table failed: {e}")

        return sorted(codes)

    @staticmethod
    def _normalize_etf_code(raw: str) -> str:
        s = str(raw).strip().lower()
        if s.startswith(("sh.", "sz.")):
            s = s[3:]
        elif s.startswith(("sh", "sz")) and len(s) >= 8:
            s = s[2:]
        return s if len(s) == 6 and s.isdigit() else ""

    @staticmethod
    def _symbol_to_ts_code(code: str) -> str:
        if code.startswith("5"):
            return f"{code}.SH"
        return f"{code}.SZ"

    @staticmethod
    def _adjustflag_to_ak(adjustflag: str) -> str:
        return {"1": "", "2": "qfq", "3": "hfq"}.get(adjustflag, "")

    @staticmethod
    def _adjustflag_to_label(adjustflag: str) -> str:
        return {"1": "NONE", "2": "PRE", "3": "POST"}.get(adjustflag, adjustflag)

    def _load_etf_hist_tushare(self, code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        if not self._ts_pro:
            return None
        df = self._ts_pro.fund_daily(
            ts_code=self._symbol_to_ts_code(code),
            start_date=pd.to_datetime(start_date).strftime("%Y%m%d"),
            end_date=pd.to_datetime(end_date).strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            return None

        df = df.rename(
            columns={
                "trade_date": "DATA_DATE",
                "open": "OPEN_PRICE",
                "high": "HIGH_PRICE",
                "low": "LOW_PRICE",
                "close": "CLOSE_PRICE",
                "vol": "VOLUME",
                "amount": "AMOUNT",
                "pct_chg": "CHANGE_PCT",
            }
        )
        df["DATA_DATE"] = pd.to_datetime(df["DATA_DATE"], format="%Y%m%d", errors="coerce")
        for col in ["OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "VOLUME", "AMOUNT", "CHANGE_PCT"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["DATA_DATE"]).sort_values("DATA_DATE").reset_index(drop=True)
        df["PRE_CLOSE"] = df["CLOSE_PRICE"].shift(1)
        df["CHANGE_AMOUNT"] = df["CLOSE_PRICE"] - df["PRE_CLOSE"]
        df["AMPLITUDE"] = ((df["HIGH_PRICE"] - df["LOW_PRICE"]) / df["PRE_CLOSE"] * 100).where(df["PRE_CLOSE"] != 0, np.nan)
        df["TURNOVER_RATE"] = None
        return df.drop(columns=["PRE_CLOSE"], errors="ignore").replace([np.nan, np.inf, -np.inf], None)

    def _load_etf_hist_akshare(self, code: str, start_date: str, end_date: str, adjustflag: str) -> pd.DataFrame | None:
        adjust = self._adjustflag_to_ak(adjustflag)
        kwargs = {
            "symbol": code,
            "period": "daily",
            "start_date": pd.to_datetime(start_date).strftime("%Y%m%d"),
            "end_date": pd.to_datetime(end_date).strftime("%Y%m%d"),
        }
        if adjust:
            kwargs["adjust"] = adjust
        df = ak.fund_etf_hist_em(**kwargs)
        if df is None or df.empty:
            return None

        df = df.rename(
            columns={
                "日期": "DATA_DATE",
                "开盘": "OPEN_PRICE",
                "收盘": "CLOSE_PRICE",
                "最高": "HIGH_PRICE",
                "最低": "LOW_PRICE",
                "成交量": "VOLUME",
                "成交额": "AMOUNT",
                "振幅": "AMPLITUDE",
                "涨跌幅": "CHANGE_PCT",
                "涨跌额": "CHANGE_AMOUNT",
                "换手率": "TURNOVER_RATE",
            }
        )
        df["DATA_DATE"] = pd.to_datetime(df["DATA_DATE"], errors="coerce")
        for col in [
            "OPEN_PRICE",
            "HIGH_PRICE",
            "LOW_PRICE",
            "CLOSE_PRICE",
            "VOLUME",
            "AMOUNT",
            "AMPLITUDE",
            "CHANGE_PCT",
            "CHANGE_AMOUNT",
            "TURNOVER_RATE",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["DATA_DATE"]).sort_values("DATA_DATE").reset_index(drop=True)

    def _finalize_and_upsert(self, df: pd.DataFrame, code: str, source: str, adjustflag: str) -> int:
        if df is None or df.empty:
            return 0

        df = df.copy()
        df["CODE"] = code
        df["ADJUST_TYPE"] = self._adjustflag_to_label(adjustflag)
        df["SOURCE"] = source
        df["CREATED_AT"] = pd.Timestamp.now()

        keep_cols = [
            "CODE",
            "DATA_DATE",
            "OPEN_PRICE",
            "CLOSE_PRICE",
            "HIGH_PRICE",
            "LOW_PRICE",
            "VOLUME",
            "AMOUNT",
            "AMPLITUDE",
            "CHANGE_PCT",
            "CHANGE_AMOUNT",
            "TURNOVER_RATE",
            "ADJUST_TYPE",
            "SOURCE",
            "CREATED_AT",
        ]
        for col in keep_cols:
            if col not in df.columns:
                df[col] = None
        df = df[keep_cols]
        records = df.to_dict("records")
        if not records:
            return 0

        merge_sql = f"""
            INSERT INTO {self._table_ref(ETF_HIST_TABLE)} (
                CODE, DATA_DATE, OPEN_PRICE, CLOSE_PRICE, HIGH_PRICE, LOW_PRICE,
                VOLUME, AMOUNT, AMPLITUDE, CHANGE_PCT, CHANGE_AMOUNT, TURNOVER_RATE,
                ADJUST_TYPE, SOURCE, CREATED_AT
            ) VALUES (
                :CODE, :DATA_DATE, :OPEN_PRICE, :CLOSE_PRICE, :HIGH_PRICE, :LOW_PRICE,
                :VOLUME, :AMOUNT, :AMPLITUDE, :CHANGE_PCT, :CHANGE_AMOUNT, :TURNOVER_RATE,
                :ADJUST_TYPE, :SOURCE, :CREATED_AT
            )
            ON DUPLICATE KEY UPDATE
                OPEN_PRICE = VALUES(OPEN_PRICE),
                CLOSE_PRICE = VALUES(CLOSE_PRICE),
                HIGH_PRICE = VALUES(HIGH_PRICE),
                LOW_PRICE = VALUES(LOW_PRICE),
                VOLUME = VALUES(VOLUME),
                AMOUNT = VALUES(AMOUNT),
                AMPLITUDE = VALUES(AMPLITUDE),
                CHANGE_PCT = VALUES(CHANGE_PCT),
                CHANGE_AMOUNT = VALUES(CHANGE_AMOUNT),
                TURNOVER_RATE = VALUES(TURNOVER_RATE),
                SOURCE = VALUES(SOURCE),
                CREATED_AT = VALUES(CREATED_AT)
        """
        with self.engine.connect() as conn:
            conn.execute(text(merge_sql), records)
            conn.commit()
        return len(records)

    def load_spot_as_hist_today(self):
        today_str = date.today().strftime("%Y-%m-%d")
        self.log.info(f"=== ETF spot-as-hist for {today_str} ===")
        self._ensure_etf_table()
        activate_tunnel("cn")

        try:
            df = ak.fund_etf_spot_em()
            if df is None or df.empty:
                self.log.info("ak.fund_etf_spot_em() returned empty")
                return

            rename_map = {
                "代码": "CODE",
                "最新价": "CLOSE_PRICE",
                "开盘价": "OPEN_PRICE",
                "最高价": "HIGH_PRICE",
                "最低价": "LOW_PRICE",
                "成交量": "VOLUME",
                "成交额": "AMOUNT",
                "涨跌幅": "CHANGE_PCT",
                "涨跌额": "CHANGE_AMOUNT",
                "换手率": "TURNOVER_RATE",
                "数据日期": "DATA_DATE",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            for col in [
                "OPEN_PRICE",
                "CLOSE_PRICE",
                "HIGH_PRICE",
                "LOW_PRICE",
                "VOLUME",
                "AMOUNT",
                "CHANGE_PCT",
                "CHANGE_AMOUNT",
                "TURNOVER_RATE",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", ""), errors="coerce")

            if "DATA_DATE" in df.columns:
                df["DATA_DATE"] = pd.to_datetime(df["DATA_DATE"], errors="coerce").dt.date
            else:
                df["DATA_DATE"] = date.today()

            if all(col in df.columns for col in ["HIGH_PRICE", "LOW_PRICE", "OPEN_PRICE"]):
                df["AMPLITUDE"] = ((df["HIGH_PRICE"] - df["LOW_PRICE"]) / df["OPEN_PRICE"]) * 100
            else:
                df["AMPLITUDE"] = None

            df["CODE"] = df["CODE"].astype(str).map(self._normalize_etf_code)
            df = df[df["CODE"].astype(bool)].copy()
            df = df.replace([np.nan, np.inf, -np.inf], None)
            df["ADJUST_TYPE"] = "qfq"
            df["SOURCE"] = "AKSHARE_SPOT_AS_HIST"
            df["CREATED_AT"] = pd.Timestamp.now()

            keep_cols = [
                "CODE",
                "DATA_DATE",
                "OPEN_PRICE",
                "CLOSE_PRICE",
                "HIGH_PRICE",
                "LOW_PRICE",
                "VOLUME",
                "AMOUNT",
                "AMPLITUDE",
                "CHANGE_PCT",
                "CHANGE_AMOUNT",
                "TURNOVER_RATE",
                "ADJUST_TYPE",
                "SOURCE",
                "CREATED_AT",
            ]
            for col in keep_cols:
                if col not in df.columns:
                    df[col] = None
            records = df[keep_cols].to_dict("records")
            if not records:
                self.log.info("no ETF spot records to insert")
                return

            with self.engine.connect() as conn:
                existing_sql = text(
                    f"""
                    SELECT CODE, DATA_DATE
                    FROM {self._table_ref(ETF_HIST_TABLE)}
                    WHERE DATA_DATE = :dt AND ADJUST_TYPE = 'qfq'
                    """
                )
                existing = pd.read_sql(existing_sql, conn, params={"dt": records[0]["DATA_DATE"]})
                existing.columns = [col.upper() for col in existing.columns]
                existing_set = {(row["CODE"], row["DATA_DATE"]) for _, row in existing.iterrows()}
                missing_records = [
                    rec for rec in records if (rec["CODE"], rec["DATA_DATE"]) not in existing_set
                ]
                if not missing_records:
                    self.log.info("all ETF spot-as-hist rows already exist")
                    return

                conn.execute(
                    text(
                        f"""
                        INSERT INTO {self._table_ref(ETF_HIST_TABLE)}
                        (CODE, DATA_DATE, OPEN_PRICE, CLOSE_PRICE, HIGH_PRICE, LOW_PRICE,
                         VOLUME, AMOUNT, AMPLITUDE, CHANGE_PCT, CHANGE_AMOUNT, TURNOVER_RATE,
                         ADJUST_TYPE, SOURCE, CREATED_AT)
                        VALUES
                        (:CODE, :DATA_DATE, :OPEN_PRICE, :CLOSE_PRICE, :HIGH_PRICE, :LOW_PRICE,
                         :VOLUME, :AMOUNT, :AMPLITUDE, :CHANGE_PCT, :CHANGE_AMOUNT, :TURNOVER_RATE,
                         :ADJUST_TYPE, :SOURCE, :CREATED_AT)
                        """
                    ),
                    missing_records,
                )
                conn.commit()

            self.log.info(f"inserted ETF spot-as-hist rows={len(missing_records)} for {today_str}")
        except Exception as e:
            self.log.info(f"ETF spot-as-hist failed: {e}")
            traceback.print_exc()
