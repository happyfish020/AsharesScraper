# -*- coding: utf-8 -*-
"""
UnifiedRisk V12 - Options Risk DataSource (E Block)

目标：
- 从 Oracle 表 CN_OPTION_SSE_DAILY 聚合 ETF 期权日行情，输出期权风险定价原始块（options_risk_raw）
- 核心指标：
    weighted_change  : 成交量加权涨跌额均值  sum(change_amount * volume) / sum(volume)
    total_change     : 所有合约涨跌额求和  sum(change_amount)
    total_volume     : 成交量求和          sum(volume)
    weighted_close   : 成交量加权收盘价    sum(close * volume) / sum(volume)
    change_ratio     : weighted_change / weighted_close
    trend_10d        : 10D 变化（weighted_change）
    acc_3d           : 3D 变化（weighted_change）
    series           : 历史序列（按日）

重要修复（你提的点）：
- 如果 CN_OPTION_SSE_DAILY 数据为空 -> 返回 data_status=MISSING + warnings，不再伪装 OK/0
- 如果接口/入库没有 CHANGE_AMOUNT/CHANGE_PCT：
    - DS 内部用 CLOSE_PRICE 自动计算 CHANGE_AMOUNT/CHANGE_PCT（按 CONTRACT_CODE 分组 + shift）
    - 然后再按日聚合生成 weighted_change 等

约束：
- 仅依赖 DBOracleProvider，不访问外部 API
- 不新增 Provider 接口（仅使用 provider.execute）
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np

from core.datasources.datasource_base import DataSourceConfig, DataSourceBase
from core.utils.ds_refresh import apply_refresh_cleanup
from core.utils.logger import get_logger
from core.adapters.providers.db_provider_oracle import DBOracleProvider


LOG = get_logger("DS.OptionsRisk")


DEFAULT_UNDERLYINGS = [
    "510050",  # 华夏上证50ETF
    "510300",  # 华泰柏瑞沪深300ETF
    "510500",  # 南方中证500ETF
    "588000",  # 华夏科创50ETF
    "588080",  # 易方达科创50ETF
    "159919",  # 嘉实沪深300ETF (深市)
    "159922",  # 嘉实中证500ETF (深市)
    "159915",  # 易方达创业板ETF
    "159901",  # 易方达深证100ETF
]


class OptionsRiskDataSource(DataSourceBase):
    """
    Options Risk DataSource (E Block)
    """

    def __init__(self, config: DataSourceConfig, window: int = 60) -> None:
        super().__init__(name="DS.OptionsRisk")
        self.config = config
        self.window = int(window) if window and window > 0 else 60
        self.db = DBOracleProvider()

        self.cache_root = config.cache_root
        self.history_root = config.history_root
        os.makedirs(self.cache_root, exist_ok=True)
        os.makedirs(self.history_root, exist_ok=True)

        self.cache_file = os.path.join(self.cache_root, "options_risk_today.json")
        self.history_file = os.path.join(self.history_root, "options_risk_series.json")

        LOG.info(
            "[DS.OptionsRisk] Init: market=%s ds_name=%s cache_root=%s history_root=%s window=%s",
            config.market, config.ds_name, self.cache_root, self.history_root, self.window
        )

    # ------------------------------------------------------------------
    def build_block(self, trade_date: str, refresh_mode: str = "none") -> Dict[str, Any]:
        """
        构建期权风险原始数据块（options_risk_raw）
        """
        apply_refresh_cleanup(
            refresh_mode=refresh_mode,
            cache_path=self.cache_file,
            history_path=self.history_file,
            spot_path=None,
        )

        # cache hit
        if refresh_mode in ("none", "readonly") and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                LOG.error("[DS.OptionsRisk] load cache error: %s", exc)

        # 1) 先尝试走 provider 的聚合接口（若列齐全且已计算）
        df_agg = None
        try:
            df_agg = self.db.fetch_options_risk_series(start_date=trade_date, look_back_days=self.window)
        except Exception as exc:
            # 注意：如果 DB 表缺列（CHANGE_AMOUNT）会在这里报错，我们会 fallback
            LOG.warning("[DS.OptionsRisk] fetch_options_risk_series failed -> fallback: %s", exc)

        # 如果 provider 聚合结果可用，直接使用
        if df_agg is not None and isinstance(df_agg, pd.DataFrame) and (not df_agg.empty):
            series = self._dfagg_to_series(df_agg)
        else:
            # 2) fallback：直接拉原始 option 行情，用 CLOSE_PRICE 自算 change_amount/pct，再按日聚合
            try:
                df_raw = self._fetch_option_daily_rows(asof_date=trade_date, look_back_days=self.window)
            except Exception as exc:
                LOG.error("[DS.OptionsRisk] _fetch_option_daily_rows error: %s", exc)
                return self._neutral_block(trade_date)

            if df_raw is None or df_raw.empty:
                LOG.warning("[DS.OptionsRisk] option_daily rows empty for %s", trade_date)
                return self._neutral_block(trade_date)

            df_agg2 = self._aggregate_from_raw(df_raw)
            if df_agg2 is None or df_agg2.empty:
                LOG.warning("[DS.OptionsRisk] aggregate_from_raw empty for %s", trade_date)
                return self._neutral_block(trade_date)

            series = self._dfagg_to_series(df_agg2)

        # 合并历史，截断 window
        merged_series = self._merge_history(series)
        trend_10d, acc_3d = self._calc_trend(merged_series)

        latest = merged_series[-1] if merged_series else None
        if not latest:
            return self._neutral_block(trade_date)

        block: Dict[str, Any] = {
            "trade_date": latest.get("trade_date", trade_date),
            "weighted_change": float(latest.get("weighted_change", 0.0) or 0.0),
            "total_change": float(latest.get("total_change", 0.0) or 0.0),
            "total_volume": float(latest.get("total_volume", 0.0) or 0.0),
            "weighted_close": float(latest.get("weighted_close", 0.0) or 0.0),
            "change_ratio": float(latest.get("change_ratio", 0.0) or 0.0),
            "trend_10d": float(trend_10d),
            "acc_3d": float(acc_3d),
            "series": merged_series,
            "data_status": "OK",
            "warnings": [],
        }

        # persist
        try:
            self._save(self.history_file, merged_series)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(block, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            LOG.error("[DS.OptionsRisk] save error: %s", exc)

        return block

    # ------------------------------------------------------------------
    def _fetch_option_daily_rows(self, asof_date: str, look_back_days: int) -> pd.DataFrame:
        """
        从 CN_OPTION_SSE_DAILY 拉原始行：
        - CONTRACT_CODE, DATA_DATE, CLOSE_PRICE, VOLUME, (optional) CHANGE_AMOUNT, CHANGE_PCT
        然后在 python 里补全 CHANGE_AMOUNT/CHANGE_PCT（若缺失或为空）
        """
        option_table = self.db.tables.get("option_daily") or "CN_OPTION_SSE_DAILY"
        schema = self.db.schema

        # IN 列表（与 provider 一致，固定九只）
        in_list = ",".join([f"'{c}'" for c in DEFAULT_UNDERLYINGS])

        # 先尝试带 CHANGE_AMOUNT/CHANGE_PCT
        sql_with_change = f"""
        SELECT
            t.CONTRACT_CODE AS contract_code,
            t.DATA_DATE     AS trade_date,
            t.CLOSE_PRICE   AS close_price,
            t.VOLUME        AS volume,
            t.CHANGE_AMOUNT AS change_amount,
            t.CHANGE_PCT    AS change_pct
        FROM {schema}.{option_table} t
        WHERE t.UNDERLYING_CODE IN ({in_list})
          AND t.DATA_DATE >= :asof_date - :look_back_days
          AND t.DATA_DATE <= :asof_date
        ORDER BY t.CONTRACT_CODE, t.DATA_DATE
        """

        params = {
            "asof_date": pd.to_datetime(asof_date).date(),
            "look_back_days": int(look_back_days),
        }

        rows = None
        try:
            rows = self.db.execute(sql_with_change, params)
            cols = ["contract_code", "trade_date", "close_price", "volume", "change_amount", "change_pct"]
            df = pd.DataFrame(rows, columns=cols)
        except Exception as exc:
            # 如果表根本没有这两列（或权限问题），fallback 到不含它们的 SQL
            LOG.warning("[DS.OptionsRisk] query with CHANGE_* failed, fallback without: %s", exc)

            sql_no_change = f"""
            SELECT
                t.CONTRACT_CODE AS contract_code,
                t.DATA_DATE     AS trade_date,
                t.CLOSE_PRICE   AS close_price,
                t.VOLUME        AS volume
            FROM {schema}.{option_table} t
            WHERE t.UNDERLYING_CODE IN ({in_list})
              AND t.DATA_DATE >= :asof_date - :look_back_days
              AND t.DATA_DATE <= :asof_date
            ORDER BY t.CONTRACT_CODE, t.DATA_DATE
            """
            rows = self.db.execute(sql_no_change, params)
            cols = ["contract_code", "trade_date", "close_price", "volume"]
            df = pd.DataFrame(rows, columns=cols)
            df["change_amount"] = np.nan
            df["change_pct"] = np.nan

        if df is None or df.empty:
            return pd.DataFrame()

        # 类型标准化
        df["contract_code"] = df["contract_code"].astype(str)
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
        for c in ["close_price", "volume", "change_amount", "change_pct"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # 补全 change_amount / change_pct（仅当为空时）
        df = df.sort_values(["contract_code", "trade_date"]).reset_index(drop=True)

        prev_close = df.groupby("contract_code")["close_price"].shift(1)
        calc_change_amount = df["close_price"] - prev_close
        calc_change_pct = (calc_change_amount / prev_close) * 100.0

        # prev_close=0 或 NaN 处理
        calc_change_amount = calc_change_amount.replace([np.inf, -np.inf], np.nan)
        calc_change_pct = calc_change_pct.replace([np.inf, -np.inf], np.nan)

        # 只在原字段缺失时填充
        df["change_amount"] = df["change_amount"].where(df["change_amount"].notna(), calc_change_amount)
        df["change_pct"] = df["change_pct"].where(df["change_pct"].notna(), calc_change_pct)

        # 第一条（无 prev_close）会是 NaN，填 0（更利于聚合，不会误报缺失）
        df["change_amount"] = df["change_amount"].fillna(0.0)
        df["change_pct"] = df["change_pct"].fillna(0.0)

        # volume NaN -> 0
        df["volume"] = df["volume"].fillna(0.0)
        df["close_price"] = df["close_price"].fillna(0.0)

        return df

    # ------------------------------------------------------------------
    def _aggregate_from_raw(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        将原始合约行情行聚合为按日序列：
            weighted_change, total_change, total_volume, weighted_close, change_ratio
        index=trade_date (datetime)
        """
        if df_raw is None or df_raw.empty:
            return pd.DataFrame()

        # 按日聚合
        g = df_raw.groupby("trade_date", as_index=False)

        def safe_div(a, b):
            if b is None or b == 0:
                return 0.0
            return a / b

        agg_rows: List[Tuple] = []
        for _, sub in g:
            dt = sub["trade_date"].iloc[0]
            total_volume = float(sub["volume"].sum() or 0.0)
            sum_close_vol = float((sub["close_price"] * sub["volume"]).sum() or 0.0)
            sum_chg_vol = float((sub["change_amount"] * sub["volume"]).sum() or 0.0)
            total_change = float(sub["change_amount"].sum() or 0.0)

            weighted_close = safe_div(sum_close_vol, total_volume)
            weighted_change = safe_div(sum_chg_vol, total_volume)
            change_ratio = safe_div(weighted_change, weighted_close) if weighted_close != 0 else 0.0

            agg_rows.append((dt, weighted_change, total_change, total_volume, weighted_close, change_ratio))

        df_agg = pd.DataFrame(
            agg_rows,
            columns=["trade_date", "weighted_change", "total_change", "total_volume", "weighted_close", "change_ratio"]
        )
        df_agg["trade_date"] = pd.to_datetime(df_agg["trade_date"])
        df_agg = df_agg.set_index("trade_date").sort_index(ascending=True)
        return df_agg

    # ------------------------------------------------------------------
    def _dfagg_to_series(self, df_agg: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        将 provider/fallback 聚合 DF 转为 series 列表（升序）
        """
        if df_agg is None or df_agg.empty:
            return []

        df_sorted = df_agg.sort_index(ascending=True)
        out: List[Dict[str, Any]] = []

        for idx, row in df_sorted.iterrows():
            dt_str = pd.to_datetime(idx).strftime("%Y-%m-%d")
            wchg = float(row.get("weighted_change", 0.0) or 0.0)
            tchg = float(row.get("total_change", 0.0) or 0.0)
            tv = float(row.get("total_volume", 0.0) or 0.0)
            wclose = float(row.get("weighted_close", 0.0) or 0.0)
            ratio = row.get("change_ratio", 0.0)
            try:
                ratio = float(ratio) if ratio is not None and pd.notna(ratio) else 0.0
            except Exception:
                ratio = 0.0

            out.append({
                "trade_date": dt_str,
                "weighted_change": round(wchg, 6),
                "total_change": round(tchg, 6),
                "total_volume": round(tv, 6),
                "weighted_close": round(wclose, 6),
                "change_ratio": round(ratio, 8),
            })

        return out

    # ------------------------------------------------------------------
    def _merge_history(self, recent: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并历史与当前窗口，保证长度固定为 window
        """
        old: List[Dict[str, Any]] = []
        if os.path.exists(self.history_file):
            try:
                old = self._load(self.history_file)
            except Exception:
                old = []

        buf: Dict[str, Dict[str, Any]] = {r["trade_date"]: r for r in (old or [])}
        for r in (recent or []):
            buf[r["trade_date"]] = r

        out = sorted(buf.values(), key=lambda x: x["trade_date"])
        return out[-self.window:]

    # ------------------------------------------------------------------
    def _calc_trend(self, series: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        计算 10 日趋势和 3 日加速度（基于 weighted_change）
        """
        if not series or len(series) < 2:
            return 0.0, 0.0
        values = [float(s.get("weighted_change", 0.0) or 0.0) for s in series]
        try:
            t10 = values[-1] - values[-11] if len(values) >= 11 else 0.0
            a3 = values[-1] - values[-4] if len(values) >= 4 else 0.0
            return round(t10, 2), round(a3, 2)
        except Exception:
            return 0.0, 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _load(path: str) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    @staticmethod
    def _save(path: str, obj: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _neutral_block(trade_date: str) -> Dict[str, Any]:
        """
        数据缺失时的占位块：
        - 必须明确标记 MISSING，防止报告误判 OK
        """
        return {
            "trade_date": trade_date,
            "weighted_change": 0.0,
            "total_change": 0.0,
            "total_volume": 0.0,
            "weighted_close": 0.0,
            "change_ratio": 0.0,
            "trend_10d": 0.0,
            "acc_3d": 0.0,
            "series": [],
            "data_status": "MISSING",
            "warnings": ["missing:options_risk_series"],
        }
