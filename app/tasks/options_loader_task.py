from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import date
from sqlalchemy import text
import akshare as ak
import time
import random

from app.utils.wireguard_helper import switch_wire_guard
from app.tasks.state_store import StateStore
# ==============================
#     Option Daily Table
# ==============================


OPTION_DAILY_TABLE = "CN_OPTION_SSE_DAILY"

MYSQL_OPTION_DAILY_SQL = f"""
CREATE TABLE IF NOT EXISTS {OPTION_DAILY_TABLE} (
    CONTRACT_CODE     VARCHAR(20)    NOT NULL,
    UNDERLYING_CODE   VARCHAR(20),
    EXPIRY_MONTH      VARCHAR(6),
    DATA_DATE         DATE           NOT NULL,
    OPEN_PRICE        DECIMAL(20,6),
    HIGH_PRICE        DECIMAL(20,6),
    LOW_PRICE         DECIMAL(20,6),
    CLOSE_PRICE       DECIMAL(20,6),
    VOLUME            DECIMAL(24,6),
    SOURCE            VARCHAR(30),
    CREATED_AT        DATETIME       DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (CONTRACT_CODE, DATA_DATE),
    KEY IDX_OPTION_CONTRACT (CONTRACT_CODE),
    KEY IDX_OPTION_DATE (DATA_DATE),
    KEY IDX_OPTION_UNDERLYING (UNDERLYING_CODE)
)
"""


@dataclass
class OptionsLoaderTask:
    name: str = "OptionsSSELoader"

    def __init__(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.data_dir = os.path.join(root_dir, "data")

    def _table_ref(self) -> str:
        return OPTION_DAILY_TABLE

    def _ensure_table(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(MYSQL_OPTION_DAILY_SQL))

    def run(self, ctx) -> None:
        cfg = ctx.config
        self.log = ctx.log
        self.engine = ctx.engine

        scanned_file = getattr(cfg, "opt_scanned_file", Path(self.data_dir) / "state" / "OPT_scanned.json")
        failed_file = getattr(cfg, "opt_failed_file", Path(self.data_dir) / "state" / "OPT_failed.json")
        self._state = StateStore(scanned_file, failed_file, self.log)
        self._state_flush_every = int(getattr(cfg, "state_flush_every", 10) or 0)

        self.load_option_sse_daily()

    # RunnerApp refresh support
    def get_state_files(self, cfg):
        return [
            getattr(cfg, "opt_scanned_file", Path(self.data_dir) / "state" / "OPT_scanned.json"),
            getattr(cfg, "opt_failed_file", Path(self.data_dir) / "state" / "OPT_failed.json"),
        ]

    def load_option_sse_daily(self):
        today = date.today()
        state = getattr(self, '_state', None)
        state_flush_every = int(getattr(self, '_state_flush_every', 0) or 0)
        scanned = state.load_scanned() if state else set()
        failed = state.load_failed() if state else set()
        processed_underlying = 0

        self.log.info("=== Creating/Checking Option SSE Daily table ===")
        self._ensure_table()
    
        self.log.info(f"\n=== Loading SSE/SZSE ETF Option Daily Data for {today} ===")
        
        etf_codes = [
            "510050", "510300", "510500", "588000", "588080",
            "159919", "159922", "159915", "159901"
        ]
    
        total_processed = 0
        max_retries = 5
               
        for attempt in range(1, max_retries + 1):
            try:
                for etf_idx, underlying in enumerate(etf_codes, 1):
                    processed_underlying += 1
                    if state and state_flush_every > 0 and processed_underlying % state_flush_every == 0:
                        state.save_scanned(scanned)
                        state.save_failed(failed)
                        self.log.info(f"[STATE] flushed scanned/failed at {processed_underlying}")

                    if underlying in scanned:
                        self.log.info(f"\n[{etf_idx:2d}/{len(etf_codes)}] Processing underlying: {underlying} ... SKIP scanned")
                        continue
                    if underlying in failed:
                        self.log.info(f"\n[{etf_idx:2d}/{len(etf_codes)}] Processing underlying: {underlying} ... SKIP failed")
                        continue

                    self.log.info(f"\n[{etf_idx:2d}/{len(etf_codes)}] Processing underlying: {underlying}")
            
                    # 检查该 underlying 当天是否已有数据（若有则跳过整个 underlying）
                    with self.engine.connect() as conn:
                        count_sql = text(f"""
                            SELECT COUNT(*) FROM {self._table_ref()}
                            WHERE UNDERLYING_CODE = :underlying AND DATA_DATE = :dt
                        """)
                        count = conn.scalar(count_sql, {"underlying": underlying, "dt": today})
            
                    if count > 0:
                        self.log.info(f"    Skip: Already has {count} records for today")
                        scanned.add(underlying)
                        continue
            
                    records = []  # 每个 underlying 单独收集
            
                    try:
                        expire_list = ak.option_sse_list_sina(symbol=underlying)
                        if not expire_list:
                            self.log.info("    No expiry months available")
                            continue
            
                        self.log.info(f"    Found {len(expire_list)} expiry months: {', '.join(expire_list)}")
            
                        for expiry_idx, expiry in enumerate(expire_list, 1):
                            self.log.info(f"  [{expiry_idx:2d}/{len(expire_list)}] Expiry: {expiry}")
            
                            try:
                                df_codes = ak.option_sse_codes_sina(trade_date=expiry, underlying=underlying)
                                if df_codes.empty:
                                    self.log.info("      No contracts found")
                                    continue
            
                                contracts = df_codes['期权代码'].tolist()
                                self.log.info(f"      Found {len(contracts)} contracts")
            
                                for contract_idx, contract in enumerate(contracts, 1):
                                    self.log.info(f"        [{contract_idx:3d}/{len(contracts)}] {contract} ... " )
            
                                    time.sleep(random.uniform(1.5, 4.0))
            
                                    try:
                                        df_hist = ak.option_sse_daily_sina(symbol=contract)
                                        if df_hist.empty:
                                            self.log.info("empty")
                                            continue
            
                                        df_latest = df_hist.iloc[[0]].copy()
                                        latest_date = pd.to_datetime(df_latest['日期'].iloc[0]).date()
            
                                        df_latest['CONTRACT_CODE'] = contract
                                        df_latest['UNDERLYING_CODE'] = underlying
                                        df_latest['EXPIRY_MONTH'] = expiry
                                        df_latest['SOURCE'] = 'AKSHARE_OPTION_SSE_SINA'
                                        df_latest['CREATED_AT'] = pd.Timestamp.now()
            
                                        records.extend(df_latest.to_dict('records'))
                                        total_processed += 1
                                        self.log.info("ok")
                                        
                                    except Exception as inner_e:
                                        self.log.info(f"error: {str(inner_e)}")
            
                            except Exception as expiry_e:
                                self.log.info(f"    Expiry {expiry} failed: {str(expiry_e)}")
            
                    except Exception as etf_e:
                        self.log.info(f"  Underlying {underlying} failed: {str(etf_e)}")
                        failed.add(underlying)
            
                    # 每个 underlying 处理完后 → 只插入缺失的数据
                    if records:
                        df_etf = pd.DataFrame(records)
            
                        # 调试：显示实际列
                        self.log.info(f"    Columns collected for {underlying}: {list(df_etf.columns)}")
            
                        # 重命名 & 类型转换（只处理实际存在的字段）
                        rename_map = {
                            '日期': 'DATA_DATE',
                            '开盘': 'OPEN_PRICE',
                            '最高': 'HIGH_PRICE',
                            '最低': 'LOW_PRICE',
                            '收盘': 'CLOSE_PRICE',
                            '成交量': 'VOLUME',
                        }
                        df_etf.rename(columns=rename_map, inplace=True)
            
                        numeric_cols = ['OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'VOLUME']
                        for col in numeric_cols:
                            if col in df_etf.columns:
                                df_etf[col] = pd.to_numeric(df_etf[col], errors='coerce')
            
                        df_etf['DATA_DATE'] = pd.to_datetime(df_etf['DATA_DATE'], errors='coerce').dt.date
                        df_etf = df_etf.replace([np.nan, np.inf, -np.inf], None)
            
                        # 确保关键字段存在
                        for required in ['CONTRACT_CODE', 'UNDERLYING_CODE', 'EXPIRY_MONTH', 'SOURCE', 'CREATED_AT']:
                            if required not in df_etf.columns:
                                df_etf[required] = None
            
                        etf_records = df_etf.to_dict('records')
                        ###
                        if etf_records:
                            with self.engine.connect() as conn:
                                # 预过滤缺失记录（可选，但保留以减少 MERGE 负载）
                                existing_sql = text(f"""
                                    SELECT CONTRACT_CODE 
                                    FROM {self._table_ref()}
                                    WHERE UNDERLYING_CODE = :underlying AND DATA_DATE = :dt
                                """)
                                existing_df = pd.read_sql(
                                    existing_sql,
                                    conn,
                                    params={"underlying": underlying, "dt": today}
                                )
            
                                if not existing_df.empty:
                                    existing_contracts = set(existing_df['CONTRACT_CODE'])
                                else:
                                    existing_contracts = set()
            
                                missing_records = [
                                    rec for rec in etf_records
                                    if rec['CONTRACT_CODE'] not in existing_contracts
                                ]
            
                                if not missing_records:
                                    self.log.info(f"    All {len(etf_records)} records already exist for {underlying}, skip")
                                    continue
            
                                # 固定字段列表（根据实际数据）
                                core_cols = [
                                    'CONTRACT_CODE', 'DATA_DATE', 'UNDERLYING_CODE', 'EXPIRY_MONTH',
                                    'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'VOLUME', 'SOURCE'
                                ]
            
                                # 过滤掉实际不存在的列（防止接口返回少字段）
                                available_cols = [c for c in core_cols if c in missing_records[0]]
            
                                # UPDATE SET：只更新数值字段 + SOURCE
                                updatable_cols = [c for c in available_cols 
                                                  if c in ['OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'VOLUME', 'SOURCE']]
                                update_set = ", ".join([f"t.{c} = s.{c}" for c in updatable_cols])
            
                                # INSERT 列：核心字段 + CREATED_AT（不重复）
                                insert_cols = ", ".join(available_cols + ['CREATED_AT'])
                                insert_vals = ", ".join([f":{c}" for c in available_cols] + ["SYSDATE"])
            
                                merge_sql = f"""
                                INSERT INTO {self._table_ref()} (
                                    CONTRACT_CODE, DATA_DATE, UNDERLYING_CODE, EXPIRY_MONTH,
                                    OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, VOLUME, SOURCE, CREATED_AT
                                ) VALUES (
                                    :CONTRACT_CODE, :DATA_DATE, :UNDERLYING_CODE, :EXPIRY_MONTH,
                                    :OPEN_PRICE, :HIGH_PRICE, :LOW_PRICE, :CLOSE_PRICE, :VOLUME, :SOURCE, NOW()
                                )
                                ON DUPLICATE KEY UPDATE
                                    OPEN_PRICE = VALUES(OPEN_PRICE),
                                    HIGH_PRICE = VALUES(HIGH_PRICE),
                                    LOW_PRICE = VALUES(LOW_PRICE),
                                    CLOSE_PRICE = VALUES(CLOSE_PRICE),
                                    VOLUME = VALUES(VOLUME),
                                    SOURCE = VALUES(SOURCE),
                                    CREATED_AT = NOW()
                                """
            
                                # 调试输出（运行一次后可注释掉）
                                self.log.info(f"Generated MERGE for {underlying}:\n{merge_sql}")
            
                                conn.execute(text(merge_sql), missing_records)
                                conn.commit()
                                scanned.add(underlying)
            
                                self.log.info(f"    Merged {len(missing_records)} records for {underlying} (total collected: {len(etf_records)})")            
            
                        ###
            #try
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"获取etf 数据失败（已重试 {max_retries} 次）：{str(e)}"
                    ) from e
                #switch_wire_guard("cn")
                time.sleep(60)
         
        #for 
        if state:
            state.save_scanned(scanned)
            state.save_failed(failed)
            self.log.info("[DONE][OPT] state saved")

        self.log.info(f"\n=== 完成 ===\n总处理合约数量: {total_processed}")
    
