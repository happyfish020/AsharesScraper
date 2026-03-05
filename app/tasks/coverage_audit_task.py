from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Optional

import pandas as pd
from sqlalchemy import Tuple, text
import akshare as ak

from app.utils.wireguard_helper import activate_tunnel, switch_wire_guard
import os 
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
        一次性跑完：
          - 股票缺失审计
          - 指数 GAP + health
        """
        cfg = self.ctx.config
        base_index = "sh000300"  # 或 "sh000001"
    
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
                self.log.warn("获取失败")
        
        if base_index_df.empty:
            self.log.info("获取基准交易日历失败")
        else:
            # Step 2: 股票缺失审计
            gap_df, window_start_df = self.audit_index_missing_days(
                 
                base_index_df=base_index_df,
                index_codes=index_codes,
                start_date=start_date,
                end_date=end_date,
            )
            
            gap_df.to_csv(os.path.join(cfg.audit_reports_dir, "audit_index_gap.csv"), index=False)
            window_start_df.to_csv(os.path.join(cfg.audit_reports_dir, "audit_index_window_start.csv"), index=False)
            
            ##############################################################
            stock_missing_df = self.audit_stock_missing_days(
                
                base_index_df=base_index_df,
                start_date=start_date,
                end_date=end_date,
            )
            stock_missing_df.to_csv(os.path.join(self.audit_reports_dir, "audit_stock_missing.csv"), index=False)
            
            if len(stock_missing_df) >0:
                return stock_missing_df['symbol'].tolist()
                  
        
            # Step 3: 指数健康审计
            #index_codes = ["sh000001", "sh000300", "sz399001", "sz399006", "sh000688"]


    def audit_stock_missing_days(
        self, 
        base_index_df: pd.DataFrame,
        start_date: str,
        end_date: str,
        stock_symbols: list | None = None,
    ) -> pd.DataFrame:
        """
        审计股票在指定区间内的数据缺失情况（WINDOW_START 和 GAP 类型）
        
        不内部获取交易日历，直接使用外部传入的基准指数 DataFrame 作为标准
        
        Parameters:
        - engine: SQLAlchemy engine
        - base_index_df: pd.DataFrame - 基准指数日线数据（必须包含 'date' 列）
        - start_date: str - 'YYYY-MM-DD'
        - end_date: str - 'YYYY-MM-DD'
        - stock_symbols: list[str] | None - 指定股票列表；None 表示数据库中所有股票
        
        Returns:
        - pd.DataFrame: 包含 symbol, missing_type, missing_date, missing_count 等
        """
        self.log.info(f"开始审计股票缺失数据 [{start_date} ~ {end_date}]，使用外部传入基准交易日历")
    
        # Step 1: 从传入的 base_index_df 提取预期交易日集合
        try:
            base_dates = set(pd.to_datetime(base_index_df['date']).dt.date)
            sorted_base_dates = sorted(base_dates)
            self.log.info(f"基准交易日总数: {len(sorted_base_dates)}")
        except Exception as e:
            self.log.info(f"解析基准交易日历失败: {e}")
            raise
    
        if not base_dates:
            self.log.info("基准交易日为空，无法审计")
            return pd.DataFrame(columns=['symbol', 'missing_type', 'missing_date', 'missing_count'])
    
        # Step 2: 获取待审计股票列表
        if stock_symbols is None:
            if self._relation_exists("cn_stock_active_universe_v"):
                symbol_query = "SELECT symbol FROM cn_stock_active_universe_v"
                self.log.info("使用活跃股票视图 cn_stock_active_universe_v")
            elif self._relation_exists("cn_stock_universe_status_t"):
                symbol_query = """
                    SELECT symbol
                    FROM cn_stock_universe_status_t
                    WHERE IFNULL(is_active, 1) = 1
                """
                self.log.info("使用活跃股票池 cn_stock_universe_status_t(is_active=1)")
            else:
                symbol_query = "SELECT DISTINCT symbol FROM cn_stock_daily_price"
                self.log.info("未检测到状态表，回退到全历史股票池")
            all_symbols_df = pd.read_sql(symbol_query, self.ctx.engine)
            stock_symbols = all_symbols_df['symbol'].tolist()
            self.log.info(f"自动获取数据库中股票总数: {len(stock_symbols)} 只")
        else:
            self.log.info(f"指定审计股票数: {len(stock_symbols)} 只")
    
        if not stock_symbols:
            self.log.info("无股票可审计")
            return pd.DataFrame(columns=['symbol', 'missing_type', 'missing_date', 'missing_count'])
    
        # Step 3: 分批查询数据库现有记录（解决 ORA-01795）
        batch_size = 900
        batches = math.ceil(len(stock_symbols) / batch_size)
        self.log.info(f"分 {batches} 批查询数据库现有交易日记录")
    
        existing_records = []
        for i in range(0, len(stock_symbols), batch_size):
            batch_symbols = stock_symbols[i:i + batch_size]
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
                    batch_df['trade_date'] = pd.to_datetime(batch_df['trade_date']).dt.date
                    existing_records.append(batch_df)
                self.log.info(f"批次 {i//batch_size + 1}/{batches} 查询完成，记录数: {len(batch_df)}")
            except Exception as e:
                self.log.info(f"批次查询失败: {e}")
                raise
    
        # 合并所有批次
        existing_df = pd.concat(existing_records, ignore_index=True) if existing_records else pd.DataFrame(columns=['symbol', 'trade_date'])
    
        # Step 4: 计算每只股票的缺失
        missing_records = []
    
        for symbol in stock_symbols:
            stock_dates = set(existing_df[existing_df['symbol'] == symbol]['trade_date'])
            missing_dates = base_dates - stock_dates
    
            if not missing_dates:
                continue
    
            sorted_missing = sorted(missing_dates)
    
            # WINDOW_START 判断
            window_start_date = sorted_base_dates[0]
            continuous_from_start = []
            for d in sorted_base_dates:
                if d in missing_dates:
                    continuous_from_start.append(d)
                else:
                    break
    
            if continuous_from_start and continuous_from_start[0] == window_start_date:
                missing_records.append({
                    'symbol': symbol,
                    'missing_type': 'WINDOW_START',
                    'missing_date': window_start_date.strftime('%Y-%m-%d'),
                    'missing_count': len(continuous_from_start)
                })
    
            # GAP 类型
            gap_dates = [d for d in sorted_missing if d not in continuous_from_start]
            for gap_date in gap_dates:
                missing_records.append({
                    'symbol': symbol,
                    'missing_type': 'GAP',
                    'missing_date': gap_date.strftime('%Y-%m-%d'),
                    'missing_count': 1
                })
    
        # Step 5: 输出结果
        result_df = pd.DataFrame(missing_records)
        if not result_df.empty:
            result_df = result_df.sort_values(['missing_type', 'symbol', 'missing_date'])
            window_count = len(result_df[result_df['missing_type'] == 'WINDOW_START'])
            gap_count = len(result_df[result_df['missing_type'] == 'GAP'])
            self.log.info(f"股票审计完成，发现缺失 {len(result_df)} 条（WINDOW_START: {window_count}，GAP: {gap_count}）")
        else:
            self.log.info("股票审计完成，未发现缺失数据")
    
        return result_df
    # =====================================================
    # 核心函数 2：指数 GAP 交叉审计 + Health
    # =====================================================
     
    



    def audit_index_missing_days(self,
        
        base_index_df: pd.DataFrame,
        index_codes: list,
        start_date: str,
        end_date: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        审计多个指数在指定区间的日历健康情况（相对于实时基准交易日历）
        
        参照 audit_stock_missing_days 风格：从数据库读取已有数据，分批查询
        
        Parameters:
        - base_index_df: pd.DataFrame - 实时获取的基准指数日线（包含 'date' 列）
        - index_codes: list[str] - 要审计的指数代码列表
        - start_date: str - 'YYYY-MM-DD'
        - end_date: str - 'YYYY-MM-DD'
        
        Returns:
        - gap_df: pd.DataFrame - GAP 类型缺失
        - window_start_df: pd.DataFrame - WINDOW_START 类型缺失
        """
        self.log.info(f"开始审计指数日历健康 [{start_date} ~ {end_date}]，审计指数数: {len(index_codes)}")
    
        # Step 1: 从传入的 base_index_df 提取标准交易日
        try:
            base_dates = set(pd.to_datetime(base_index_df['date']).dt.date)
            sorted_base_dates = sorted(base_dates)
            self.log.info(f"基准交易日总数: {len(sorted_base_dates)}")
        except Exception as e:
            self.log.info(f"解析基准交易日历失败: {e}")
            raise
    
        if not base_dates:
            self.log.info("基准交易日为空，无法审计指数")
            empty_df = pd.DataFrame(columns=['index_code', 'missing_type', 'missing_date', 'missing_count'])
            return empty_df.copy(), empty_df.copy()
    
        if not index_codes:
            self.log.info("无指数可审计")
            empty_df = pd.DataFrame(columns=['index_code', 'missing_type', 'missing_date', 'missing_count'])
            return empty_df.copy(), empty_df.copy()
    
        # Step 2: 分批从数据库查询指数已有交易日（解决 ORA-01795）
        batch_size = 900
        batches = math.ceil(len(index_codes) / batch_size)
        self.log.info(f"分 {batches} 批查询数据库中指数现有记录")
    
        existing_records = []
        for i in range(0, len(index_codes), batch_size):
            batch_codes = index_codes[i:i + batch_size]
            codes_str = "','" .join(batch_codes)
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
                    batch_df['trade_date'] = pd.to_datetime(batch_df['trade_date']).dt.date
                    existing_records.append(batch_df)
                self.log.info(f"指数批次 {i//batch_size + 1}/{batches} 查询完成，记录数: {len(batch_df)}")
            except Exception as e:
                self.log.info(f"指数批次查询失败: {e}")
                raise
    
        # 合并结果
        existing_df = pd.concat(existing_records, ignore_index=True) if existing_records else pd.DataFrame(columns=['index_code', 'trade_date'])
    
        # Step 3: 计算每个指数的缺失
        gap_records = []
        window_start_records = []
    
        window_start_date = sorted_base_dates[0]  # 区间起始日
    
        for index_code in index_codes:
            index_dates = set(existing_df[existing_df['index_code'] == index_code]['trade_date'])
            missing_dates = base_dates - index_dates
    
            if not missing_dates:
                continue  # 该指数完整，无缺失
    
            sorted_missing = sorted(missing_dates)
    
            # WINDOW_START 判断：从起始日期连续缺失
            continuous_from_start = []
            for d in sorted_base_dates:
                if d in missing_dates:
                    continuous_from_start.append(d)
                else:
                    break
    
            if continuous_from_start and continuous_from_start[0] == window_start_date:
                window_start_records.append({
                    'index_code': index_code,
                    'missing_type': 'WINDOW_START',
                    'missing_date': window_start_date.strftime('%Y-%m-%d'),
                    'missing_count': len(continuous_from_start)
                })
    
            # GAP 类型：非起始部分的缺失
            gap_dates = [d for d in sorted_missing if d not in continuous_from_start]
            for gap_date in gap_dates:
                gap_records.append({
                    'index_code': index_code,
                    'missing_type': 'GAP',
                    'missing_date': gap_date.strftime('%Y-%m-%d'),
                    'missing_count': 1
                })
    
        # Step 4: 转为 DataFrame 并排序
        gap_df = pd.DataFrame(gap_records)
        window_start_df = pd.DataFrame(window_start_records)
    
        if not gap_df.empty:
            gap_df = gap_df.sort_values(['index_code', 'missing_date'])
        if not window_start_df.empty:
            window_start_df = window_start_df.sort_values(['index_code'])
    
        total_gaps = len(gap_df)
        total_windows = len(window_start_df)
        self.log.info(f"指数日历审计完成，GAP 缺失 {total_gaps} 条，WINDOW_START 缺失 {total_windows} 条")
    
        return gap_df, window_start_df 
 
