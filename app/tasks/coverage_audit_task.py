from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Optional

import pandas as pd
from sqlalchemy import Tuple, text
import akshare as ak

from app.utils.wireguard_helper import activate_tunnel, switch_wire_guard
import os 


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(str(raw).strip())
    except Exception:
        return default


@dataclass
class CoverageAuditTask:
    name: str = "CoverageAudit"
    
    
    def __init__(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".." ,".."))
        self.audit_reports_dir = os.path.join(root_dir, "audit_reports")

    @staticmethod
    def _resolve_col(df: pd.DataFrame, *candidates: str) -> str:
        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}
        for c in candidates:
            hit = lower_map.get(c.lower())
            if hit is not None:
                return hit
        raise KeyError(candidates[0])

    def _relation_exists(self, name: str) -> bool:
        q = text(
            """
            SELECT COUNT(*)
            FROM (
                SELECT table_name AS rel_name
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
                UNION ALL
                SELECT table_name AS rel_name
                FROM information_schema.views
                WHERE table_schema = DATABASE()
            ) x
            WHERE x.rel_name = :t
            """
        )
        with self.ctx.engine.connect() as conn:
            n = conn.execute(q, {"t": name}).scalar()
        return int(n or 0) > 0

    def run(self, ctx) -> None:
        self.ctx = ctx
        cfg = ctx.config
        self.log = ctx.log
        self.engine = ctx.engine
        activate_tunnel("cn")
         
        self.run_full_coverage_audit(
             
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            stock_symbols=None,
            index_codes=cfg.index_symbols,
        )

    def run_full_coverage_audit(self,
        
        start_date: str,
        end_date: str,
        stock_symbols: Optional[List[Tuple[str, str]]] = None,
        index_codes: Optional[List[str]] = None,
    ):
        """
        涓€娆℃€ц窇瀹岋細
          - 鑲＄エ缂哄け瀹¤
          - 鎸囨暟 GAP + health
        """
        cfg = self.ctx.config
        self.audit_reports_dir = os.path.abspath(getattr(cfg, "audit_reports_dir", self.audit_reports_dir))
        os.makedirs(self.audit_reports_dir, exist_ok=True)
        base_index = "sh000300"  # 鎴?"sh000001"
    
        max_retries = 8
        for attempt in range(1, max_retries + 1):
            try:
                # Direct replacement: use Sina index daily endpoint (EM endpoint is unstable here)
                base_index_df = ak.stock_zh_index_daily(symbol=base_index)
                if base_index_df is None or base_index_df.empty:
                    raise RuntimeError("stock_zh_index_daily returned empty dataframe")

                # Keep the same window semantics as before.
                base_index_df["date"] = pd.to_datetime(base_index_df["date"], errors="coerce")
                d1 = pd.to_datetime(start_date)
                d2 = pd.to_datetime(end_date)
                base_index_df = base_index_df[
                    (base_index_df["date"] >= d1) & (base_index_df["date"] <= d2)
                ].copy()
                break
            except Exception as e:
                if attempt == max_retries:
                    self.log.info(e)
                    raise RuntimeError(
                        f"获取指数 {base_index} 日线数据失败（已重试 {max_retries} 次）：{str(e)}"
                    ) from e
                self.log.warn("鑾峰彇澶辫触")
        
        if base_index_df.empty:
            self.log.info("鑾峰彇鍩哄噯浜ゆ槗鏃ュ巻澶辫触")
        else:
            # Step 2: 鑲＄エ缂哄け瀹¤
            gap_df, window_start_df = self.audit_index_missing_days(
                 
                base_index_df=base_index_df,
                index_codes=index_codes,
                start_date=start_date,
                end_date=end_date,
            )
            
            index_gap_path = os.path.join(self.audit_reports_dir, "audit_index_gap.csv")
            index_window_path = os.path.join(self.audit_reports_dir, "audit_index_window_start.csv")
            gap_df.to_csv(index_gap_path, index=False)
            window_start_df.to_csv(index_window_path, index=False)
            self.log.info(f"指数 GAP 明细报告: {index_gap_path}")
            self.log.info(f"指数 WINDOW_START 明细报告: {index_window_path}")
            
            ##############################################################
            stock_missing_df = self.audit_stock_missing_days(
                
                base_index_df=base_index_df,
                start_date=start_date,
                end_date=end_date,
            )
            stock_missing_path = os.path.join(self.audit_reports_dir, "audit_stock_missing.csv")
            stock_missing_df.to_csv(stock_missing_path, index=False)
            self.log.info(f"股票缺失明细报告: {stock_missing_path}")
            
            if len(stock_missing_df) >0:
                return stock_missing_df['symbol'].tolist()
                  
        
            # Step 3: 鎸囨暟鍋ュ悍瀹¤
            #index_codes = ["sh000001", "sh000300", "sz399001", "sz399006", "sh000688"]
    def audit_stock_missing_days(
        self,
        base_index_df: pd.DataFrame,
        start_date: str,
        end_date: str,
        stock_symbols: list | None = None,
    ) -> pd.DataFrame:
        """
        Audit stock missing trading days in the window and classify as WINDOW_START / GAP.

        stock_symbols supports three input shapes:
        - list[str]
        - list[tuple[str, first_trade_date]]
        - DataFrame with symbol + optional first_trade_date columns
        """
        self.log.info(f"开始审计股票缺失数据 [{start_date} ~ {end_date}]，使用外部传入基准交易日历")

        try:
            base_dates = set(pd.to_datetime(base_index_df["date"]).dt.date)
            sorted_base_dates = sorted(base_dates)
            self.log.info(f"基准交易日总数: {len(sorted_base_dates)}")
        except Exception as e:
            self.log.info(f"解析基准交易日历失败: {e}")
            raise

        if not base_dates:
            self.log.info("基准交易日为空，无法审计")
            return pd.DataFrame(columns=["symbol", "missing_type", "missing_date", "missing_count"])

        symbol_start_dates: dict[str, object] = {}
        symbol_last_trade_dates: dict[str, object] = {}

        if stock_symbols is None:
            if self._relation_exists("cn_stock_active_universe_v"):
                symbol_query = """
                    SELECT symbol, first_trade_date, last_trade_date, recent_trade_days
                    FROM cn_stock_active_universe_v
                """
                self.log.info("使用活跃股票视图 cn_stock_active_universe_v")
            elif self._relation_exists("cn_stock_universe_status_t"):
                symbol_query = """
                    SELECT symbol, first_trade_date, last_trade_date, recent_trade_days
                    FROM cn_stock_universe_status_t
                    WHERE IFNULL(is_active, 1) = 1
                """
                self.log.info("使用活跃股票池 cn_stock_universe_status_t(is_active=1)")
            else:
                symbol_query = """
                    SELECT symbol, MIN(TRADE_DATE) AS first_trade_date
                    FROM cn_stock_daily_price
                    GROUP BY symbol
                """
                self.log.info("未检测到状态表，回退到历史价格表推断首个交易日")

            all_symbols_df = pd.read_sql(symbol_query, self.ctx.engine)
            sym_col = self._resolve_col(all_symbols_df, "symbol", "SYMBOL")
            all_symbols_df = all_symbols_df.rename(columns={sym_col: "symbol"})

            ftd_col = None
            for c in all_symbols_df.columns:
                if str(c).lower() == "first_trade_date":
                    ftd_col = c
                    break
            if ftd_col is not None:
                all_symbols_df[ftd_col] = pd.to_datetime(all_symbols_df[ftd_col], errors="coerce").dt.date
                symbol_start_dates = dict(zip(all_symbols_df["symbol"].astype(str), all_symbols_df[ftd_col]))
            ltd_col = None
            for c in all_symbols_df.columns:
                if str(c).lower() == "last_trade_date":
                    ltd_col = c
                    break
            if ltd_col is not None:
                all_symbols_df[ltd_col] = pd.to_datetime(all_symbols_df[ltd_col], errors="coerce").dt.date
                symbol_last_trade_dates = dict(zip(all_symbols_df["symbol"].astype(str), all_symbols_df[ltd_col]))

            # Prefer auditing symbols that are actually "recently tradable" to avoid
            # counting long-halted or stale active-pool members as daily missing.
            rtd_col = None
            for c in all_symbols_df.columns:
                if str(c).lower() == "recent_trade_days":
                    rtd_col = c
                    break
            if rtd_col is not None:
                before_n = len(all_symbols_df)
                rtd = pd.to_numeric(all_symbols_df[rtd_col], errors="coerce").fillna(0)
                all_symbols_df = all_symbols_df.loc[rtd > 0].copy()
                self.log.info(
                    f"按 recent_trade_days>0 过滤股票池: {before_n} -> {len(all_symbols_df)}"
                )
                # keep maps aligned with filtered universe
                kept = set(all_symbols_df["symbol"].astype(str).tolist())
                symbol_start_dates = {k: v for k, v in symbol_start_dates.items() if k in kept}
                symbol_last_trade_dates = {k: v for k, v in symbol_last_trade_dates.items() if k in kept}

            stock_symbols = all_symbols_df["symbol"].astype(str).tolist()
            self.log.info(f"自动获取数据库中股票总数: {len(stock_symbols)} 只")
        else:
            if isinstance(stock_symbols, pd.DataFrame):
                sym_col = self._resolve_col(stock_symbols, "symbol", "SYMBOL")
                tmp = stock_symbols.rename(columns={sym_col: "symbol"}).copy()

                ftd_col = None
                for c in tmp.columns:
                    if str(c).lower() == "first_trade_date":
                        ftd_col = c
                        break
                if ftd_col is not None:
                    tmp[ftd_col] = pd.to_datetime(tmp[ftd_col], errors="coerce").dt.date
                    symbol_start_dates = dict(zip(tmp["symbol"].astype(str), tmp[ftd_col]))

                stock_symbols = tmp["symbol"].astype(str).tolist()
            elif stock_symbols and isinstance(stock_symbols[0], (tuple, list)):
                normalized_symbols = []
                for item in stock_symbols:
                    if not item:
                        continue
                    sym = str(item[0])
                    normalized_symbols.append(sym)
                    ftd = item[1] if len(item) > 1 else None
                    if ftd is not None:
                        symbol_start_dates[sym] = pd.to_datetime(ftd, errors="coerce").date()
                stock_symbols = normalized_symbols
            else:
                stock_symbols = [str(s) for s in stock_symbols]

            self.log.info(f"指定审计股票数: {len(stock_symbols)} 只")

        if not stock_symbols:
            self.log.info("无股票可审计")
            return pd.DataFrame(columns=["symbol", "missing_type", "missing_date", "missing_count"])

        # Fallback for symbols with missing first_trade_date in status tables.
        missing_start_symbols = [
            s for s in stock_symbols if (s not in symbol_start_dates) or pd.isna(symbol_start_dates.get(s))
        ]
        if missing_start_symbols:
            infer_batch_size = 900
            infer_records = []
            for i in range(0, len(missing_start_symbols), infer_batch_size):
                batch_symbols = missing_start_symbols[i : i + infer_batch_size]
                symbols_str = "','".join(batch_symbols)
                infer_query = f"""
                    SELECT symbol, MIN(TRADE_DATE) AS first_trade_date
                    FROM cn_stock_daily_price
                    WHERE symbol IN ('{symbols_str}')
                    GROUP BY symbol
                """
                infer_df = pd.read_sql(infer_query, self.ctx.engine)
                if infer_df is not None and not infer_df.empty:
                    sym_col = self._resolve_col(infer_df, "symbol", "SYMBOL")
                    ftd_col = self._resolve_col(infer_df, "first_trade_date", "FIRST_TRADE_DATE")
                    infer_df = infer_df.rename(columns={sym_col: "symbol", ftd_col: "first_trade_date"})
                    infer_df["first_trade_date"] = pd.to_datetime(infer_df["first_trade_date"], errors="coerce").dt.date
                    infer_records.append(infer_df[["symbol", "first_trade_date"]])
            if infer_records:
                infer_all = pd.concat(infer_records, ignore_index=True).drop_duplicates(subset=["symbol"])
                inferred_map = dict(zip(infer_all["symbol"].astype(str), infer_all["first_trade_date"]))
                symbol_start_dates.update(inferred_map)
                self.log.info(
                    f"已为 {len(inferred_map)} 只股票从历史价格表回填 first_trade_date（原状态表为空）"
                )

        # If first_trade_date is still unknown and the symbol has never appeared in history,
        # skip it from strict missing audit to avoid systematic false positives.
        unknown_start_symbols = [
            s for s in stock_symbols if (s not in symbol_start_dates) or pd.isna(symbol_start_dates.get(s))
        ]
        never_seen_symbols: set[str] = set()
        if unknown_start_symbols:
            seen_batch_size = 900
            seen_symbols: set[str] = set()
            for i in range(0, len(unknown_start_symbols), seen_batch_size):
                batch_symbols = unknown_start_symbols[i : i + seen_batch_size]
                symbols_str = "','".join(batch_symbols)
                seen_query = f"""
                    SELECT DISTINCT symbol
                    FROM cn_stock_daily_price
                    WHERE symbol IN ('{symbols_str}')
                      AND TRADE_DATE <= '{end_date}'
                """
                seen_df = pd.read_sql(seen_query, self.ctx.engine)
                if seen_df is not None and not seen_df.empty:
                    sym_col = self._resolve_col(seen_df, "symbol", "SYMBOL")
                    seen_symbols.update(seen_df[sym_col].astype(str).tolist())
            never_seen_symbols = set(unknown_start_symbols) - seen_symbols
            if never_seen_symbols:
                self.log.info(
                    f"跳过 {len(never_seen_symbols)} 只从未入库且首日未知的股票，避免 WINDOW_START 误报"
                )

        batch_size = 900
        batches = math.ceil(len(stock_symbols) / batch_size)
        self.log.info(f"分 {batches} 批查询数据库现有交易日记录")

        existing_records = []
        for i in range(0, len(stock_symbols), batch_size):
            batch_symbols = stock_symbols[i : i + batch_size]
            symbols_str = "','".join(batch_symbols)
            query = f"""
                SELECT symbol, TRADE_DATE
                FROM cn_stock_daily_price
                WHERE symbol IN ('{symbols_str}')
                  AND TRADE_DATE BETWEEN '{start_date}' AND '{end_date}'
            """
            try:
                batch_df = pd.read_sql(query, self.ctx.engine)
                if not batch_df.empty:
                    td_col = self._resolve_col(batch_df, "trade_date", "TRADE_DATE")
                    sym_col = self._resolve_col(batch_df, "symbol", "SYMBOL")
                    batch_df = batch_df.rename(columns={td_col: "trade_date", sym_col: "symbol"})
                    batch_df["trade_date"] = pd.to_datetime(batch_df["trade_date"]).dt.date
                    existing_records.append(batch_df)
                self.log.info(f"批次 {i // batch_size + 1}/{batches} 查询完成，记录数: {len(batch_df)}")
            except Exception as e:
                self.log.info(f"批次查询失败: {e}")
                raise

        existing_df = (
            pd.concat(existing_records, ignore_index=True)
            if existing_records
            else pd.DataFrame(columns=["symbol", "trade_date"])
        )

        missing_records = []
        start_dt = pd.to_datetime(start_date, errors="coerce").date()
        self.log.info("股票缺失审计口径版本: v3(last_trade_date_guard)")
        skipped_by_last_trade = 0

        for symbol in stock_symbols:
            if symbol in never_seen_symbols:
                continue
            last_trade_date = symbol_last_trade_dates.get(symbol)
            if last_trade_date is not None and not pd.isna(last_trade_date) and last_trade_date < start_dt:
                skipped_by_last_trade += 1
                continue
            stock_dates = set(existing_df[existing_df["symbol"] == symbol]["trade_date"])

            first_trade_date = symbol_start_dates.get(symbol)
            if first_trade_date is not None and not pd.isna(first_trade_date):
                effective_start = max(start_dt, first_trade_date)
                symbol_base_dates = [d for d in sorted_base_dates if d >= effective_start]
            else:
                symbol_base_dates = sorted_base_dates

            if not symbol_base_dates:
                continue

            symbol_base_dates_set = set(symbol_base_dates)
            missing_dates = symbol_base_dates_set - stock_dates
            if not missing_dates:
                continue

            sorted_missing = sorted(missing_dates)
            window_start_date = symbol_base_dates[0]

            continuous_from_start = []
            for d in symbol_base_dates:
                if d in missing_dates:
                    continuous_from_start.append(d)
                else:
                    break

            if continuous_from_start and continuous_from_start[0] == window_start_date:
                missing_records.append(
                    {
                        "symbol": symbol,
                        "missing_type": "WINDOW_START",
                        "missing_reason": "WINDOW_START",
                        "missing_date": window_start_date.strftime("%Y-%m-%d"),
                        "expected_trade_date": window_start_date.strftime("%Y-%m-%d"),
                        "missing_count": len(continuous_from_start),
                        "first_trade_date": first_trade_date.strftime("%Y-%m-%d") if first_trade_date is not None and not pd.isna(first_trade_date) else "",
                        "last_trade_date": last_trade_date.strftime("%Y-%m-%d") if last_trade_date is not None and not pd.isna(last_trade_date) else "",
                        "skipped_by_last_trade": False,
                        "source_table": "cn_stock_daily_price",
                        "audit_version": "v3(last_trade_date_guard)",
                    }
                )

            gap_dates = [d for d in sorted_missing if d not in continuous_from_start]
            for gap_date in gap_dates:
                missing_records.append(
                    {
                        "symbol": symbol,
                        "missing_type": "GAP",
                        "missing_reason": "GAP",
                        "missing_date": gap_date.strftime("%Y-%m-%d"),
                        "expected_trade_date": gap_date.strftime("%Y-%m-%d"),
                        "missing_count": 1,
                        "first_trade_date": first_trade_date.strftime("%Y-%m-%d") if first_trade_date is not None and not pd.isna(first_trade_date) else "",
                        "last_trade_date": last_trade_date.strftime("%Y-%m-%d") if last_trade_date is not None and not pd.isna(last_trade_date) else "",
                        "skipped_by_last_trade": False,
                        "source_table": "cn_stock_daily_price",
                        "audit_version": "v3(last_trade_date_guard)",
                    }
                )

        result_df = pd.DataFrame(missing_records)

        # Tushare/AK daily stock bars are sparse by design: a listed stock has no
        # daily bar on suspension / no-trade days.  The old audit compared every
        # active symbol against the market index calendar, so normal single-stock
        # suspensions were reported as fatal cn_stock_daily_price GAPs even after
        # a successful --refresh.
        #
        # v4 sparse-classification rule:
        #   - Compute missing symbol breadth per missing_date across ALL stock missing rows,
        #     not just GAP rows.
        #   - Treat both GAP and WINDOW_START as non-fatal when the breadth ratio is below
        #     V8_STOCK_DAILY_FATAL_GAP_MIN_RATIO (default 10%).  This covers:
        #       * ordinary single-stock suspensions/no-trade days
        #       * stocks missing at the start of an audit window because they were suspended
        #         on the first few benchmark trading days
        #   - Keep truly broad source/table outages fatal when the missing breadth ratio is high.
        #
        # The old v3.1 rule also required missing_symbols_on_date < 50.  That was too strict
        # for 2010/2011 A-share history: 50-70 names missing out of ~2,500 symbols is only
        # ~2-3% breadth and is still normal sparse/suspension behavior, not a table-wide gap.
        nonfatal_df = pd.DataFrame()
        if (
            not result_df.empty
            and _env_flag("V8_STOCK_DAILY_SPARSE_GAP_AS_NONFATAL", True)
            and "missing_type" in result_df.columns
        ):
            missing_by_date = (
                result_df.groupby("missing_date")["symbol"]
                .nunique()
                .rename("missing_symbols_on_date")
            )
            total_symbols = max(len(stock_symbols), 1)
            min_ratio = _env_float("V8_STOCK_DAILY_FATAL_GAP_MIN_RATIO", 0.10)
            # Kept only for logging/backward compatibility.  The default 0 disables
            # count-based fatal classification; ratio is the authoritative signal.
            min_symbols = _env_int("V8_STOCK_DAILY_FATAL_GAP_MIN_SYMBOLS", 0)

            result_df = result_df.merge(
                missing_by_date,
                left_on="missing_date",
                right_index=True,
                how="left",
            )
            result_df["missing_symbols_on_date"] = (
                pd.to_numeric(result_df["missing_symbols_on_date"], errors="coerce").fillna(0).astype(int)
            )
            result_df["audited_symbol_count"] = int(total_symbols)
            result_df["missing_symbol_ratio_on_date"] = result_df["missing_symbols_on_date"] / float(total_symbols)

            sparse_mask = result_df["missing_symbol_ratio_on_date"] < min_ratio

            if sparse_mask.any():
                nonfatal_df = result_df.loc[sparse_mask].copy()
                nonfatal_df["original_missing_type"] = nonfatal_df["missing_type"].astype(str)
                nonfatal_df["missing_type"] = "SUSPENSION_OR_NO_TRADE"
                nonfatal_df["missing_reason"] = "SUSPENSION_OR_NO_TRADE"
                nonfatal_df["audit_action"] = (
                    "non_fatal: sparse per-symbol absence on a market trading date; "
                    "after --refresh this is normally suspension/no-trade or window-start suspension, "
                    "not a source-table gap"
                )
                nonfatal_path = os.path.join(self.audit_reports_dir, "audit_stock_non_trading_absence.csv")
                nonfatal_df.sort_values(["symbol", "missing_date"]).to_csv(
                    nonfatal_path, index=False, encoding="utf-8-sig"
                )
                self.log.info(
                    "股票稀疏日线诊断报告: %s rows=%s fatal_threshold_symbols=%s fatal_threshold_ratio=%.4f",
                    nonfatal_path,
                    len(nonfatal_df),
                    min_symbols,
                    min_ratio,
                )
                result_df = result_df.loc[~sparse_mask].copy()

        if not result_df.empty:
            result_df = result_df.sort_values(["missing_type", "symbol", "missing_date"])
            window_count = len(result_df[result_df["missing_type"] == "WINDOW_START"])
            gap_count = len(result_df[result_df["missing_type"] == "GAP"])
            self.log.info(
                f"股票审计完成，发现致命缺失 {len(result_df)} 条（WINDOW_START: {window_count}，GAP: {gap_count}）"
            )
            sample_unique = ",".join(result_df["symbol"].astype(str).drop_duplicates().head(20).tolist())
            sample_rows = ",".join(
                (
                    result_df["symbol"].astype(str)
                    + ":"
                    + result_df["missing_date"].astype(str)
                    + ":"
                    + result_df["missing_type"].astype(str)
                ).head(20).tolist()
            )
            self.log.info(
                f"审计附加统计: skipped_by_last_trade={skipped_by_last_trade}, "
                f"missing_sample_symbols_unique={sample_unique}, missing_sample_rows={sample_rows}"
            )
        else:
            if not nonfatal_df.empty:
                self.log.info(
                    "股票审计完成，未发现致命缺失；非致命停牌/无成交日诊断 %s 条",
                    len(nonfatal_df),
                )
            else:
                self.log.info("股票审计完成，未发现缺失数据")

        return result_df
    def audit_index_missing_days(
        self,
        base_index_df: pd.DataFrame,
        index_codes: list,
        start_date: str,
        end_date: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Audit index missing trading days in the window and classify as WINDOW_START / GAP.

        Returns:
        - gap_df: GAP rows
        - window_start_df: WINDOW_START rows
        """
        self.log.info(f"开始审计指数日历健康 [{start_date} ~ {end_date}]，审计指数数: {len(index_codes)}")

        try:
            base_dates = set(pd.to_datetime(base_index_df["date"]).dt.date)
            sorted_base_dates = sorted(base_dates)
            self.log.info(f"基准交易日总数: {len(sorted_base_dates)}")
        except Exception as e:
            self.log.info(f"解析基准交易日历失败: {e}")
            raise

        if not base_dates:
            self.log.info("基准交易日为空，无法审计指数")
            empty_df = pd.DataFrame(columns=["index_code", "missing_type", "missing_date", "missing_count"])
            return empty_df.copy(), empty_df.copy()

        if not index_codes:
            self.log.info("无指数可审计")
            empty_df = pd.DataFrame(columns=["index_code", "missing_type", "missing_date", "missing_count"])
            return empty_df.copy(), empty_df.copy()

        batch_size = 900
        batches = math.ceil(len(index_codes) / batch_size)
        self.log.info(f"分 {batches} 批查询数据库中指数现有记录")

        existing_records = []
        for i in range(0, len(index_codes), batch_size):
            batch_codes = index_codes[i : i + batch_size]
            codes_str = "','".join(batch_codes)
            query = f"""
                SELECT index_code, TRADE_DATE
                FROM cn_index_daily_price
                WHERE index_code IN ('{codes_str}')
                  AND TRADE_DATE BETWEEN '{start_date}' AND '{end_date}'
            """
            try:
                batch_df = pd.read_sql(query, self.ctx.engine)
                if not batch_df.empty:
                    td_col = self._resolve_col(batch_df, "trade_date", "TRADE_DATE")
                    idx_col = self._resolve_col(batch_df, "index_code", "INDEX_CODE")
                    batch_df = batch_df.rename(columns={td_col: "trade_date", idx_col: "index_code"})
                    batch_df["trade_date"] = pd.to_datetime(batch_df["trade_date"]).dt.date
                    existing_records.append(batch_df)
                self.log.info(f"指数批次 {i // batch_size + 1}/{batches} 查询完成，记录数: {len(batch_df)}")
            except Exception as e:
                self.log.info(f"指数批次查询失败: {e}")
                raise

        existing_df = (
            pd.concat(existing_records, ignore_index=True)
            if existing_records
            else pd.DataFrame(columns=["index_code", "trade_date"])
        )

        gap_records = []
        window_start_records = []

        window_start_date = sorted_base_dates[0]
        for index_code in index_codes:
            index_dates = set(existing_df[existing_df["index_code"] == index_code]["trade_date"])
            missing_dates = base_dates - index_dates
            if not missing_dates:
                continue

            sorted_missing = sorted(missing_dates)

            continuous_from_start = []
            for d in sorted_base_dates:
                if d in missing_dates:
                    continuous_from_start.append(d)
                else:
                    break

            if continuous_from_start and continuous_from_start[0] == window_start_date:
                window_start_records.append(
                    {
                        "index_code": index_code,
                        "missing_type": "WINDOW_START",
                        "missing_reason": "WINDOW_START",
                        "missing_date": window_start_date.strftime("%Y-%m-%d"),
                        "expected_trade_date": window_start_date.strftime("%Y-%m-%d"),
                        "missing_count": len(continuous_from_start),
                        "source_table": "cn_index_daily_price",
                        "audit_version": "v3(index_window_start_separated)",
                    }
                )

            gap_dates = [d for d in sorted_missing if d not in continuous_from_start]
            for gap_date in gap_dates:
                gap_records.append(
                    {
                        "index_code": index_code,
                        "missing_type": "GAP",
                        "missing_reason": "GAP",
                        "missing_date": gap_date.strftime("%Y-%m-%d"),
                        "expected_trade_date": gap_date.strftime("%Y-%m-%d"),
                        "missing_count": 1,
                        "source_table": "cn_index_daily_price",
                        "audit_version": "v3(index_window_start_separated)",
                    }
                )

        gap_df = pd.DataFrame(gap_records)
        window_start_df = pd.DataFrame(window_start_records)

        if not gap_df.empty:
            gap_df = gap_df.sort_values(["index_code", "missing_date"])
        if not window_start_df.empty:
            window_start_df = window_start_df.sort_values(["index_code"])

        total_gaps = len(gap_df)
        total_windows = len(window_start_df)
        self.log.info(f"指数日历审计完成，GAP 缺失 {total_gaps} 条，WINDOW_START 缺失 {total_windows} 条")

        return gap_df, window_start_df
