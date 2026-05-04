from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
import time
from typing import Iterable, List

import akshare as ak
import pandas as pd
import tushare as ts
from sqlalchemy import text

from app.settings import build_engine
from app.utils.progress import ProgressLogger
from app.tools.sync_cn_stock_daily_basic_from_tushare import (
    UPSERT_COLS as DAILY_BASIC_COLS,
    fetch_daily_basic_akshare_snapshot,
    normalize_daily_basic,
)
from app.tools.sync_cn_stock_daily_price_from_tushare import (
    _parse_ymd,
    patch_pandas_fillna_method_compat,
    resolve_tushare_token,
)


MONTHLY_BASIC_COLS = ["symbol", "trade_date", "month_key", *[c for c in DAILY_BASIC_COLS if c not in {"symbol", "trade_date"}], "raw_payload"]

INCOME_COLS = [
    "symbol",
    "end_date",
    "ann_date",
    "f_ann_date",
    "report_type",
    "comp_type",
    "end_type",
    "basic_eps",
    "diluted_eps",
    "total_revenue",
    "revenue",
    "operate_profit",
    "total_profit",
    "income_tax",
    "n_income",
    "n_income_attr_p",
    "minority_gain",
    "oth_compr_income",
    "t_compr_income",
    "compr_inc_attr_p",
    "compr_inc_attr_m_s",
    "ebit",
    "ebitda",
    "undist_profit",
    "source",
    "raw_payload",
]

BALANCE_COLS = [
    "symbol",
    "end_date",
    "ann_date",
    "f_ann_date",
    "report_type",
    "comp_type",
    "end_type",
    "money_cap",
    "trad_asset",
    "notes_receiv",
    "accounts_receiv",
    "oth_receiv",
    "prepayment",
    "inventories",
    "total_cur_assets",
    "fix_assets",
    "cip",
    "intan_assets",
    "goodwill",
    "total_nca",
    "total_assets",
    "st_borr",
    "notes_payable",
    "acct_payable",
    "adv_receipts",
    "total_cur_liab",
    "lt_borr",
    "bond_payable",
    "total_ncl",
    "total_liab",
    "total_hldr_eqy_exc_min_int",
    "total_hldr_eqy_inc_min_int",
    "source",
    "raw_payload",
]

FINA_COLS = [
    "symbol",
    "end_date",
    "ann_date",
    "report_type",
    "eps",
    "dt_eps",
    "bps",
    "ocfps",
    "roe",
    "roe_dt",
    "roa",
    "roic",
    "grossprofit_margin",
    "netprofit_margin",
    "profit_to_gr",
    "ocf_to_or",
    "debt_to_eqt",
    "debt_to_assets",
    "current_ratio",
    "quick_ratio",
    "ar_turn",
    "arturn_days",
    "inv_turn",
    "invturn_days",
    "or_yoy",
    "netprofit_yoy",
    "tr_yoy",
    "q_sales_yoy",
    "q_profit_yoy",
    "q_ocf_yoy",
    "source",
    "raw_payload",
]

CASHFLOW_COLS = [
    "symbol",
    "end_date",
    "ann_date",
    "f_ann_date",
    "report_type",
    "comp_type",
    "end_type",
    "n_cashflow_act",
    "n_cash_flows_inv_act",
    "n_cash_flows_fnc_act",
    "source",
    "raw_payload",
]


def ensure_tables(engine) -> None:
    for ddl_path in [
        "docs/DDL/cn_market.cn_stock_monthly_basic.sql",
        "docs/DDL/cn_market.cn_stock_income.sql",
        "docs/DDL/cn_market.cn_stock_balancesheet.sql",
        "docs/DDL/cn_market.cn_stock_fina_indicator.sql",
        "docs/DDL/cn_market.cn_stock_cashflow.sql",
        "docs/DDL/cn_market.cn_fundamental_quality_param_t.sql",
        "docs/DDL/cn_market.cn_stock_fundamental_quality_snap.sql",
    ]:
        apply_ddl(engine, ddl_path)
    _ensure_mysql_column(engine, "cn_stock_fina_indicator", "ar_turn", "DECIMAL(18,6) NULL AFTER quick_ratio")
    _ensure_mysql_column(engine, "cn_stock_fina_indicator", "arturn_days", "DECIMAL(18,6) NULL AFTER ar_turn")
    _ensure_mysql_column(engine, "cn_stock_fina_indicator", "inv_turn", "DECIMAL(18,6) NULL AFTER arturn_days")
    _ensure_mysql_column(engine, "cn_stock_fina_indicator", "invturn_days", "DECIMAL(18,6) NULL AFTER inv_turn")
    _ensure_mysql_column(engine, "cn_stock_fundamental_quality_snap", "roe", "DECIMAL(18,6) NULL AFTER netprofit_margin")
    _ensure_mysql_column(engine, "cn_stock_fundamental_quality_snap", "netprofit_yoy", "DOUBLE NULL AFTER roe")
    _ensure_mysql_column(engine, "cn_stock_fundamental_quality_snap", "ocf_to_np", "DECIMAL(18,6) NULL AFTER netprofit_yoy")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE cn_stock_fina_indicator MODIFY COLUMN or_yoy DOUBLE NULL"))
        conn.execute(text("ALTER TABLE cn_stock_fina_indicator MODIFY COLUMN netprofit_yoy DOUBLE NULL"))
        conn.execute(text("ALTER TABLE cn_stock_fina_indicator MODIFY COLUMN tr_yoy DOUBLE NULL"))
        conn.execute(text("ALTER TABLE cn_stock_fina_indicator MODIFY COLUMN q_sales_yoy DOUBLE NULL"))
        conn.execute(text("ALTER TABLE cn_stock_fina_indicator MODIFY COLUMN q_profit_yoy DOUBLE NULL"))
        conn.execute(text("ALTER TABLE cn_stock_fina_indicator MODIFY COLUMN q_ocf_yoy DOUBLE NULL"))


def apply_ddl(engine, ddl_path: str) -> None:
    sql = Path(ddl_path).read_text(encoding="utf-8")
    with engine.begin() as conn:
        statements = [part.strip() for part in sql.split(";") if part.strip()]
        for stmt in statements:
            conn.execute(text(stmt))


def _ensure_mysql_column(engine, table_name: str, column_name: str, definition_sql: str) -> None:
    exists_sql = """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = :table_name
          AND column_name = :column_name
        LIMIT 1
    """
    with engine.begin() as conn:
        exists = conn.execute(
            text(exists_sql),
            {"table_name": table_name, "column_name": column_name},
        ).scalar()
        if not exists:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition_sql}"))


def rebuild_quality_snapshot(engine, *, log=None, include_payload: bool = False) -> int:
    sql_months = """
        SELECT DISTINCT month_key
        FROM cn_stock_monthly_basic
        WHERE month_key IS NOT NULL
        ORDER BY month_key
    """
    sql_truncate = "TRUNCATE TABLE cn_stock_fundamental_quality_snap"
    sql_insert = """
        INSERT INTO cn_stock_fundamental_quality_snap (
            symbol, basic_trade_date, month_key, fina_end_date, ann_date,
            total_mv, circ_mv, pe, pe_ttm, pb, ps, ps_ttm,
            eps, revenue_growth_pct, or_yoy, tr_yoy, q_sales_yoy,
            debt_to_eqt, grossprofit_margin, netprofit_margin,
            roe, netprofit_yoy, ocf_to_np,
            pass_eps_positive, pass_revenue_growth_5, pass_revenue_growth_10,
            pass_debt_to_eqt_lt_2, pass_gross_margin_positive,
            quality_core_score, quality_total_score, quality_pass_core, quality_pass_with_margin,
            basic_source, fina_source, basic_raw_payload, fina_raw_payload
        )
        WITH params AS (
            SELECT
                p.parameter_set,
                p.eps_min,
                p.revenue_growth_min,
                p.revenue_growth_strict_min,
                p.debt_to_eqt_max,
                p.grossprofit_margin_min
            FROM cn_fundamental_quality_param_t p
            WHERE p.is_active = 1
            ORDER BY p.updated_at DESC, p.parameter_set
            LIMIT 1
        ),
        basic_base AS (
            SELECT
                b.symbol,
                b.trade_date,
                b.month_key,
                b.total_mv,
                b.circ_mv,
                b.pe,
                b.pe_ttm,
                  b.pb,
                  b.ps,
                  b.ps_ttm,
                  b.source AS basic_source
              FROM cn_stock_monthly_basic b
              WHERE b.month_key = :month_key
          ),
          joined AS (
              SELECT
                b.*,
                f.end_date AS fina_end_date,
                f.ann_date,
                f.eps,
                f.or_yoy,
                f.tr_yoy,
                  f.q_sales_yoy,
                  f.debt_to_eqt,
                  f.grossprofit_margin,
                  f.netprofit_margin,
                  f.roe,
                  f.netprofit_yoy,
                  cf.n_cashflow_act,
                  ic.n_income_attr_p,
                  f.source AS fina_source,
                  ROW_NUMBER() OVER (
                      PARTITION BY b.symbol, b.trade_date
                      ORDER BY
                          CASE
                              WHEN COALESCE(f.ann_date, f.end_date) <= b.trade_date THEN 0
                            ELSE 1
                        END,
                        COALESCE(f.ann_date, f.end_date) DESC,
                        f.end_date DESC
                ) AS rn
            FROM basic_base b
            LEFT JOIN cn_stock_fina_indicator f
              ON f.symbol = b.symbol
            LEFT JOIN cn_stock_cashflow cf
              ON cf.symbol = b.symbol AND cf.end_date = f.end_date
            LEFT JOIN cn_stock_income ic
              ON ic.symbol = b.symbol AND ic.end_date = f.end_date
        )
        SELECT
            j.symbol,
            j.trade_date AS basic_trade_date,
            j.month_key,
            j.fina_end_date,
            j.ann_date,
            j.total_mv,
            j.circ_mv,
            j.pe,
            j.pe_ttm,
            j.pb,
            j.ps,
            j.ps_ttm,
            j.eps,
            COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) AS revenue_growth_pct,
            j.or_yoy,
            j.tr_yoy,
            j.q_sales_yoy,
            j.debt_to_eqt,
            j.grossprofit_margin,
            j.netprofit_margin,
            j.roe,
            j.netprofit_yoy,
            CASE
                WHEN j.n_income_attr_p IS NOT NULL AND j.n_income_attr_p <> 0
                THEN j.n_cashflow_act / j.n_income_attr_p
                ELSE NULL
            END AS ocf_to_np,
            CASE WHEN j.eps > prm.eps_min THEN 1 ELSE 0 END AS pass_eps_positive,
            CASE WHEN COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) >= prm.revenue_growth_min THEN 1 ELSE 0 END AS pass_revenue_growth_5,
            CASE WHEN COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) >= prm.revenue_growth_strict_min THEN 1 ELSE 0 END AS pass_revenue_growth_10,
            CASE WHEN j.debt_to_eqt IS NOT NULL AND j.debt_to_eqt < prm.debt_to_eqt_max THEN 1 ELSE 0 END AS pass_debt_to_eqt_lt_2,
            CASE WHEN j.grossprofit_margin IS NOT NULL AND j.grossprofit_margin > prm.grossprofit_margin_min THEN 1 ELSE 0 END AS pass_gross_margin_positive,
            (
                CASE WHEN j.eps > prm.eps_min THEN 1 ELSE 0 END
                + CASE WHEN COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) >= prm.revenue_growth_min THEN 1 ELSE 0 END
                + CASE WHEN j.debt_to_eqt IS NOT NULL AND j.debt_to_eqt < prm.debt_to_eqt_max THEN 1 ELSE 0 END
            ) AS quality_core_score,
            (
                CASE WHEN j.eps > prm.eps_min THEN 1 ELSE 0 END
                + CASE WHEN COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) >= prm.revenue_growth_min THEN 1 ELSE 0 END
                + CASE WHEN j.debt_to_eqt IS NOT NULL AND j.debt_to_eqt < prm.debt_to_eqt_max THEN 1 ELSE 0 END
                + CASE WHEN j.grossprofit_margin IS NOT NULL AND j.grossprofit_margin > prm.grossprofit_margin_min THEN 1 ELSE 0 END
            ) AS quality_total_score,
            CASE
                WHEN j.eps > prm.eps_min
                 AND COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) >= prm.revenue_growth_min
                 AND j.debt_to_eqt IS NOT NULL
                 AND j.debt_to_eqt < prm.debt_to_eqt_max
                THEN 1 ELSE 0
            END AS quality_pass_core,
            CASE
                WHEN j.eps > prm.eps_min
                 AND COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) >= prm.revenue_growth_min
                 AND j.debt_to_eqt IS NOT NULL
                 AND j.debt_to_eqt < prm.debt_to_eqt_max
                 AND j.grossprofit_margin IS NOT NULL
                 AND j.grossprofit_margin > prm.grossprofit_margin_min
                THEN 1 ELSE 0
              END AS quality_pass_with_margin,
              j.basic_source,
              j.fina_source,
              NULL AS basic_raw_payload,
              NULL AS fina_raw_payload
          FROM joined j
          JOIN params prm
          WHERE j.rn = 1
            AND (j.ann_date IS NULL OR j.ann_date <= j.trade_date OR j.fina_end_date <= j.trade_date)
    """
    with engine.begin() as conn:
        month_rows = conn.execute(text(sql_months)).fetchall()
        conn.execute(text(sql_truncate))
        total = 0
        progress = ProgressLogger(
            name="stock_quality_snapshot.rebuild",
            total=len(month_rows),
            unit="months",
            log=log,
            every=1,
            min_interval_seconds=5.0,
        )
        for row in month_rows:
            month_key = str(row[0]).strip()
            if not month_key:
                progress.update(current_item="blank", rows=0, affected=0, extra="skip=blank_month_key")
                continue
            ret = conn.execute(text(sql_insert), {"month_key": month_key})
            affected = int(ret.rowcount or 0)
            total += affected
            extra = "payload=on" if include_payload else "payload=off"
            progress.update(current_item=month_key, rows=affected, affected=affected, extra=extra)
        progress.finish(extra="payload=on" if include_payload else "payload=off")
        return total


def chunked(records: List[dict], chunk_size: int) -> Iterable[List[dict]]:
    for i in range(0, len(records), chunk_size):
        yield records[i : i + chunk_size]


def _coerce_date(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    dt = pd.to_datetime(v, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.date()


def _normalize_numeric_series(series: pd.Series) -> pd.Series:
    work = series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
    work = work.replace({"None": None, "nan": None, "NaN": None, "--": None, "": None})
    return pd.to_numeric(work, errors="coerce")


def _symbol_to_ts_code(symbol: str) -> str:
    code = str(symbol).strip()[-6:]
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "3")):
        return f"{code}.SZ"
    if code.startswith("8"):
        return f"{code}.BJ"
    return code


def _frame_row_payloads(raw: pd.DataFrame) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype="object")
    work = raw.copy().astype(object)
    work = work.where(pd.notna(work), None)
    payloads = []
    for record in work.to_dict(orient="records"):
        payloads.append(json.dumps(record, ensure_ascii=False, default=str))
    return pd.Series(payloads, index=raw.index, dtype="object")


def get_trade_month_dates(engine, start_date: date, end_date: date, calendar_source: str) -> List[date]:
    if calendar_source == "price":
        source_table = "cn_stock_daily_price"
        extra_where = ""
    elif calendar_source == "board-map":
        source_table = "cn_board_member_map_d"
        extra_where = "AND sector_type='INDUSTRY' AND sector_id LIKE 'BK%%'"
    else:
        raise ValueError(f"unsupported calendar_source: {calendar_source}")

    sql = f"""
        SELECT MAX(trade_date) AS trade_date
        FROM {source_table}
        WHERE trade_date >= :start_date
          AND trade_date <= :end_date
          {extra_where}
        GROUP BY DATE_FORMAT(trade_date, '%%Y%%m')
        ORDER BY trade_date
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"start_date": start_date, "end_date": end_date}).fetchall()
    out: List[date] = []
    for row in rows:
        dt = _coerce_date(row[0])
        if dt is not None:
            out.append(dt)
    return out


def get_trade_month_dates_tushare(token: str, start_date: date, end_date: date) -> List[date]:
    pro = ts.pro_api(token)
    raw = pro.trade_cal(
        exchange="SSE",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        fields="cal_date,is_open",
    )
    if raw is None or raw.empty:
        return []
    work = raw.copy()
    work = work[work["is_open"].astype(str) == "1"].copy()
    if work.empty:
        return []
    work["trade_date"] = pd.to_datetime(work["cal_date"], format="%Y%m%d", errors="coerce").dt.date
    work = work.dropna(subset=["trade_date"])
    month_dates = (
        work.assign(month_key=pd.to_datetime(work["trade_date"]).dt.strftime("%Y%m"))
        .groupby("month_key", as_index=False)["trade_date"]
        .max()
        .sort_values("trade_date")
    )
    return month_dates["trade_date"].tolist()


def iter_report_periods(start_date: date, end_date: date) -> List[str]:
    periods: List[str] = []
    year = start_date.year
    while year <= end_date.year:
        for month, day in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            period = date(year, month, day)
            if period < start_date or period > end_date:
                continue
            periods.append(period.strftime("%Y%m%d"))
        year += 1
    return periods


def upsert_monthly_basic(engine, df: pd.DataFrame, chunk_size: int = 4000) -> int:
    if df is None or df.empty:
        return 0
    work = df.copy().astype(object).where(pd.notna(df), None)
    insert_sql = """
        INSERT INTO cn_stock_monthly_basic (
            symbol, trade_date, month_key, total_share, float_share, free_share,
            total_mv, circ_mv, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
            turnover_rate_f, volume_ratio, source, raw_payload
        ) VALUES (
            :symbol, :trade_date, :month_key, :total_share, :float_share, :free_share,
            :total_mv, :circ_mv, :pe, :pe_ttm, :pb, :ps, :ps_ttm, :dv_ratio, :dv_ttm,
            :turnover_rate_f, :volume_ratio, :source, :raw_payload
        )
        ON DUPLICATE KEY UPDATE
            month_key = VALUES(month_key),
            total_share = VALUES(total_share),
            float_share = VALUES(float_share),
            free_share = VALUES(free_share),
            total_mv = VALUES(total_mv),
            circ_mv = VALUES(circ_mv),
            pe = VALUES(pe),
            pe_ttm = VALUES(pe_ttm),
            pb = VALUES(pb),
            ps = VALUES(ps),
            ps_ttm = VALUES(ps_ttm),
            dv_ratio = VALUES(dv_ratio),
            dv_ttm = VALUES(dv_ttm),
            turnover_rate_f = VALUES(turnover_rate_f),
            volume_ratio = VALUES(volume_ratio),
            source = VALUES(source),
            raw_payload = VALUES(raw_payload)
    """
    affected = 0
    with engine.begin() as conn:
        for batch in chunked(work[MONTHLY_BASIC_COLS].to_dict(orient="records"), chunk_size):
            ret = conn.execute(text(insert_sql), batch)
            affected += int(ret.rowcount or 0)
    return affected


def upsert_income(engine, df: pd.DataFrame, chunk_size: int = 4000) -> int:
    if df is None or df.empty:
        return 0
    work = df.copy().astype(object).where(pd.notna(df), None)
    insert_sql = """
        INSERT INTO cn_stock_income (
            symbol, end_date, ann_date, f_ann_date, report_type, comp_type, end_type,
            basic_eps, diluted_eps, total_revenue, revenue, operate_profit, total_profit,
            income_tax, n_income, n_income_attr_p, minority_gain, oth_compr_income,
            t_compr_income, compr_inc_attr_p, compr_inc_attr_m_s, ebit, ebitda,
            undist_profit, source, raw_payload
        ) VALUES (
            :symbol, :end_date, :ann_date, :f_ann_date, :report_type, :comp_type, :end_type,
            :basic_eps, :diluted_eps, :total_revenue, :revenue, :operate_profit, :total_profit,
            :income_tax, :n_income, :n_income_attr_p, :minority_gain, :oth_compr_income,
            :t_compr_income, :compr_inc_attr_p, :compr_inc_attr_m_s, :ebit, :ebitda,
            :undist_profit, :source, :raw_payload
        )
        ON DUPLICATE KEY UPDATE
            ann_date = VALUES(ann_date),
            f_ann_date = VALUES(f_ann_date),
            report_type = VALUES(report_type),
            comp_type = VALUES(comp_type),
            end_type = VALUES(end_type),
            basic_eps = VALUES(basic_eps),
            diluted_eps = VALUES(diluted_eps),
            total_revenue = VALUES(total_revenue),
            revenue = VALUES(revenue),
            operate_profit = VALUES(operate_profit),
            total_profit = VALUES(total_profit),
            income_tax = VALUES(income_tax),
            n_income = VALUES(n_income),
            n_income_attr_p = VALUES(n_income_attr_p),
            minority_gain = VALUES(minority_gain),
            oth_compr_income = VALUES(oth_compr_income),
            t_compr_income = VALUES(t_compr_income),
            compr_inc_attr_p = VALUES(compr_inc_attr_p),
            compr_inc_attr_m_s = VALUES(compr_inc_attr_m_s),
            ebit = VALUES(ebit),
            ebitda = VALUES(ebitda),
            undist_profit = VALUES(undist_profit),
            source = VALUES(source),
            raw_payload = VALUES(raw_payload)
    """
    affected = 0
    with engine.begin() as conn:
        for batch in chunked(work[INCOME_COLS].to_dict(orient="records"), chunk_size):
            ret = conn.execute(text(insert_sql), batch)
            affected += int(ret.rowcount or 0)
    return affected


def upsert_balancesheet(engine, df: pd.DataFrame, chunk_size: int = 4000) -> int:
    if df is None or df.empty:
        return 0
    work = df.copy().astype(object).where(pd.notna(df), None)
    insert_sql = """
        INSERT INTO cn_stock_balancesheet (
            symbol, end_date, ann_date, f_ann_date, report_type, comp_type, end_type,
            money_cap, trad_asset, notes_receiv, accounts_receiv, oth_receiv, prepayment,
            inventories, total_cur_assets, fix_assets, cip, intan_assets, goodwill,
            total_nca, total_assets, st_borr, notes_payable, acct_payable, adv_receipts,
            total_cur_liab, lt_borr, bond_payable, total_ncl, total_liab,
            total_hldr_eqy_exc_min_int, total_hldr_eqy_inc_min_int, source, raw_payload
        ) VALUES (
            :symbol, :end_date, :ann_date, :f_ann_date, :report_type, :comp_type, :end_type,
            :money_cap, :trad_asset, :notes_receiv, :accounts_receiv, :oth_receiv, :prepayment,
            :inventories, :total_cur_assets, :fix_assets, :cip, :intan_assets, :goodwill,
            :total_nca, :total_assets, :st_borr, :notes_payable, :acct_payable, :adv_receipts,
            :total_cur_liab, :lt_borr, :bond_payable, :total_ncl, :total_liab,
            :total_hldr_eqy_exc_min_int, :total_hldr_eqy_inc_min_int, :source, :raw_payload
        )
        ON DUPLICATE KEY UPDATE
            ann_date = VALUES(ann_date),
            f_ann_date = VALUES(f_ann_date),
            report_type = VALUES(report_type),
            comp_type = VALUES(comp_type),
            end_type = VALUES(end_type),
            money_cap = VALUES(money_cap),
            trad_asset = VALUES(trad_asset),
            notes_receiv = VALUES(notes_receiv),
            accounts_receiv = VALUES(accounts_receiv),
            oth_receiv = VALUES(oth_receiv),
            prepayment = VALUES(prepayment),
            inventories = VALUES(inventories),
            total_cur_assets = VALUES(total_cur_assets),
            fix_assets = VALUES(fix_assets),
            cip = VALUES(cip),
            intan_assets = VALUES(intan_assets),
            goodwill = VALUES(goodwill),
            total_nca = VALUES(total_nca),
            total_assets = VALUES(total_assets),
            st_borr = VALUES(st_borr),
            notes_payable = VALUES(notes_payable),
            acct_payable = VALUES(acct_payable),
            adv_receipts = VALUES(adv_receipts),
            total_cur_liab = VALUES(total_cur_liab),
            lt_borr = VALUES(lt_borr),
            bond_payable = VALUES(bond_payable),
            total_ncl = VALUES(total_ncl),
            total_liab = VALUES(total_liab),
            total_hldr_eqy_exc_min_int = VALUES(total_hldr_eqy_exc_min_int),
            total_hldr_eqy_inc_min_int = VALUES(total_hldr_eqy_inc_min_int),
            source = VALUES(source),
            raw_payload = VALUES(raw_payload)
    """
    affected = 0
    with engine.begin() as conn:
        for batch in chunked(work[BALANCE_COLS].to_dict(orient="records"), chunk_size):
            ret = conn.execute(text(insert_sql), batch)
            affected += int(ret.rowcount or 0)
    return affected


def upsert_fina_indicator(engine, df: pd.DataFrame, chunk_size: int = 4000) -> int:
    if df is None or df.empty:
        return 0
    work = df.copy().astype(object).where(pd.notna(df), None)
    insert_sql = """
        INSERT INTO cn_stock_fina_indicator (
            symbol, end_date, ann_date, report_type, eps, dt_eps, bps, ocfps,
            roe, roe_dt, roa, roic, grossprofit_margin, netprofit_margin,
            profit_to_gr, ocf_to_or, debt_to_eqt, debt_to_assets, current_ratio, quick_ratio,
            ar_turn, arturn_days, inv_turn, invturn_days,
            or_yoy, netprofit_yoy, tr_yoy, q_sales_yoy, q_profit_yoy, q_ocf_yoy, source, raw_payload
        ) VALUES (
            :symbol, :end_date, :ann_date, :report_type, :eps, :dt_eps, :bps, :ocfps,
            :roe, :roe_dt, :roa, :roic, :grossprofit_margin, :netprofit_margin,
            :profit_to_gr, :ocf_to_or, :debt_to_eqt, :debt_to_assets, :current_ratio, :quick_ratio,
            :ar_turn, :arturn_days, :inv_turn, :invturn_days,
            :or_yoy, :netprofit_yoy, :tr_yoy, :q_sales_yoy, :q_profit_yoy, :q_ocf_yoy, :source, :raw_payload
        )
        ON DUPLICATE KEY UPDATE
            ann_date = VALUES(ann_date),
            report_type = VALUES(report_type),
            eps = VALUES(eps),
            dt_eps = VALUES(dt_eps),
            bps = VALUES(bps),
            ocfps = VALUES(ocfps),
            roe = VALUES(roe),
            roe_dt = VALUES(roe_dt),
            roa = VALUES(roa),
            roic = VALUES(roic),
            grossprofit_margin = VALUES(grossprofit_margin),
            netprofit_margin = VALUES(netprofit_margin),
            profit_to_gr = VALUES(profit_to_gr),
            ocf_to_or = VALUES(ocf_to_or),
            debt_to_eqt = VALUES(debt_to_eqt),
            debt_to_assets = VALUES(debt_to_assets),
            current_ratio = VALUES(current_ratio),
            quick_ratio = VALUES(quick_ratio),
            ar_turn = VALUES(ar_turn),
            arturn_days = VALUES(arturn_days),
            inv_turn = VALUES(inv_turn),
            invturn_days = VALUES(invturn_days),
            or_yoy = VALUES(or_yoy),
            netprofit_yoy = VALUES(netprofit_yoy),
            tr_yoy = VALUES(tr_yoy),
            q_sales_yoy = VALUES(q_sales_yoy),
            q_profit_yoy = VALUES(q_profit_yoy),
            q_ocf_yoy = VALUES(q_ocf_yoy),
            source = VALUES(source),
            raw_payload = VALUES(raw_payload)
    """
    affected = 0
    with engine.begin() as conn:
        for batch in chunked(work[FINA_COLS].to_dict(orient="records"), chunk_size):
            ret = conn.execute(text(insert_sql), batch)
            affected += int(ret.rowcount or 0)
    return affected


def normalize_monthly_basic(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    out = normalize_daily_basic(raw, source_label)
    if out.empty:
        return pd.DataFrame(columns=MONTHLY_BASIC_COLS)
    out["month_key"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.strftime("%Y%m")
    out["raw_payload"] = _frame_row_payloads(raw).reindex(out.index)
    return out[MONTHLY_BASIC_COLS].copy()


def _fetch_fina_indicator_tushare(pro, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame:
    fields = (
        "ts_code,ann_date,end_date,report_type,eps,dt_eps,bps,ocfps,roe,roe_dt,roa,roic,"
        "grossprofit_margin,netprofit_margin,profit_to_gr,ocf_to_or,debt_to_eqt,debt_to_assets,"
        "current_ratio,quick_ratio,ar_turn,arturn_days,inv_turn,invturn_days,"
        "or_yoy,netprofit_yoy,tr_yoy,q_sales_yoy,q_profit_yoy,q_ocf_yoy"
    )
    return pro.fina_indicator(
        ts_code=ts_code,
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        fields=fields,
    )


def _fetch_balancesheet_tushare(pro, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame:
    fields = (
        "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,end_type,money_cap,trad_asset,"
        "notes_receiv,accounts_receiv,oth_receiv,prepayment,inventories,total_cur_assets,fix_assets,"
        "cip,intan_assets,goodwill,total_nca,total_assets,st_borr,notes_payable,acct_payable,"
        "adv_receipts,total_cur_liab,lt_borr,bond_payable,total_ncl,total_liab,"
        "total_hldr_eqy_exc_min_int,total_hldr_eqy_inc_min_int"
    )
    return pro.balancesheet(
        ts_code=ts_code,
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        fields=fields,
    )


def _fetch_income_tushare(pro, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame:
    fields = (
        "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,end_type,basic_eps,diluted_eps,"
        "total_revenue,revenue,operate_profit,total_profit,income_tax,n_income,n_income_attr_p,"
        "minority_gain,oth_compr_income,t_compr_income,compr_inc_attr_p,compr_inc_attr_m_s,"
        "ebit,ebitda,undist_profit"
    )
    return pro.income(
        ts_code=ts_code,
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        fields=fields,
    )


def _is_tushare_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return "每分钟最多访问该接口200次" in msg or "每分钟最多访问" in msg


def _fetch_with_retry(fetch_fn, label: str) -> pd.DataFrame:
    for attempt in range(1, 5):
        try:
            return fetch_fn()
        except Exception as exc:
            if _is_tushare_rate_limit_error(exc):
                wait_seconds = 60
            else:
                wait_seconds = min(12, attempt * 2)
            if attempt == 4:
                raise RuntimeError(f"{label} failed after retries: {exc}") from exc
            print(f"{label} retry attempt={attempt} wait={wait_seconds}s err={exc}")
            time.sleep(wait_seconds)
    return pd.DataFrame()




def _fetch_financial_by_ann_date(pro, api_name: str, fields: str, start_date: date, end_date: date, log=None) -> pd.DataFrame:
    frames = []
    api = getattr(pro, api_name)
    cur = start_date
    total_days = (end_date - start_date).days + 1
    progress = ProgressLogger(name=f"stock_fundamental.{api_name}", total=total_days, unit="days", log=log, every=10, min_interval_seconds=15.0)
    while cur <= end_date:
        ymd = cur.strftime("%Y%m%d")
        raw = pd.DataFrame()
        try:
            raw = _fetch_with_retry(lambda d=ymd: api(ann_date=d, fields=fields), f"{api_name} ann_date={ymd}")
        except Exception as exc:
            progress.update(current_item=ymd, extra=f"failed={exc}")
            cur += timedelta(days=1)
            continue
        if raw is not None and not raw.empty:
            frames.append(raw)
            progress.update(current_item=ymd, rows=int(len(raw)))
        else:
            progress.update(current_item=ymd)
        cur += timedelta(days=1)
    progress.finish()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)



def _get_disclosure_symbols_by_date(engine, start_date: date, end_date: date) -> dict[date, List[str]]:
    """Return disclosure-driven symbols sliced by actual disclosure date only.

    income/balancesheet/fina_indicator TuShare APIs require ts_code in this
    environment. Use actual_date only; pre_date is scheduled/expected and can
    expand one daily catch-up into thousands of unnecessary requests.
    """
    sql = text(
        """
        SELECT actual_date AS disclosure_date, symbol
        FROM cn_event_disclosure_date
        WHERE actual_date IS NOT NULL
          AND actual_date BETWEEN :start_date AND :end_date
          AND symbol IS NOT NULL AND symbol <> ''
        ORDER BY actual_date, symbol
        """
    )
    by_date: dict[date, set[str]] = {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"start_date": start_date, "end_date": end_date}).fetchall()
    except Exception:
        return {}

    for raw_dt, raw_symbol in rows:
        dt = pd.to_datetime(raw_dt, errors="coerce")
        if pd.isna(dt):
            continue
        symbol = str(raw_symbol).strip()
        if not symbol:
            continue
        by_date.setdefault(dt.date(), set()).add(symbol)

    return {dt: sorted(symbols) for dt, symbols in sorted(by_date.items())}

def _fetch_financial_by_disclosure_dates(
    engine,
    pro,
    api_name: str,
    fields: str,
    start_date: date,
    end_date: date,
    log=None,
) -> pd.DataFrame:
    symbols_by_date = _get_disclosure_symbols_by_date(engine, start_date, end_date)
    total = sum(len(symbols) for symbols in symbols_by_date.values())
    if total <= 0:
        msg = f"[stock_fundamental.{api_name}] no disclosure symbols in {start_date}..{end_date}; skip ts_code fetch"
        if log:
            log.info(msg)
        else:
            print(msg)
        return pd.DataFrame()

    day_summary = ", ".join(f"{dt.strftime('%Y%m%d')}={len(symbols)}" for dt, symbols in symbols_by_date.items())
    msg = f"[stock_fundamental.{api_name}] disclosure_date_slices={len(symbols_by_date)} total_symbols={total} {day_summary}"
    if log:
        log.info(msg)
    else:
        print(msg)

    api = getattr(pro, api_name)
    frames = []
    progress = ProgressLogger(
        name=f"stock_fundamental.{api_name}",
        total=total,
        unit="disclosure_date_symbols",
        log=log,
        every=10,
        min_interval_seconds=15.0,
    )
    for disclosure_dt, symbols in symbols_by_date.items():
        ymd = disclosure_dt.strftime("%Y%m%d")
        for symbol in symbols:
            ts_code = _symbol_to_ts_code(symbol)
            try:
                raw = _fetch_with_retry(
                    lambda t=ts_code, d=ymd: api(ts_code=t, start_date=d, end_date=d, fields=fields),
                    f"{api_name} ts_code={ts_code} disclosure_date={ymd}",
                )
            except Exception as exc:
                progress.update(current_item=f"{ymd}:{symbol}", extra=f"failed={exc}")
                continue
            if raw is not None and not raw.empty:
                frames.append(raw)
                progress.update(current_item=f"{ymd}:{symbol}", rows=int(len(raw)))
            else:
                progress.update(current_item=f"{ymd}:{symbol}")
    progress.finish()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()

def normalize_income(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=INCOME_COLS)
    out = raw.copy()
    out["symbol"] = out.get("ts_code", pd.Series(dtype="object")).astype(str).str.split(".").str[0]
    for date_col in ["end_date", "ann_date", "f_ann_date"]:
        out[date_col] = pd.to_datetime(out.get(date_col), format="%Y%m%d", errors="coerce").dt.date
    for col in INCOME_COLS:
        if col in {"symbol", "end_date", "ann_date", "f_ann_date", "report_type", "comp_type", "end_type", "source", "raw_payload"}:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce") if col in out.columns else None
    out["source"] = source_label
    out["raw_payload"] = _frame_row_payloads(raw)
    out = out.dropna(subset=["symbol", "end_date"])
    return out[INCOME_COLS].copy()


def normalize_balancesheet(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=BALANCE_COLS)
    out = raw.copy()
    out["symbol"] = out.get("ts_code", pd.Series(dtype="object")).astype(str).str.split(".").str[0]
    for date_col in ["end_date", "ann_date", "f_ann_date"]:
        out[date_col] = pd.to_datetime(out.get(date_col), format="%Y%m%d", errors="coerce").dt.date
    for col in BALANCE_COLS:
        if col in {"symbol", "end_date", "ann_date", "f_ann_date", "report_type", "comp_type", "end_type", "source", "raw_payload"}:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce") if col in out.columns else None
    out["source"] = source_label
    out["raw_payload"] = _frame_row_payloads(raw)
    out = out.dropna(subset=["symbol", "end_date"])
    return out[BALANCE_COLS].copy()


def normalize_fina_indicator(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=FINA_COLS)
    out = raw.copy()
    out["symbol"] = out.get("ts_code", pd.Series(dtype="object")).astype(str).str.split(".").str[0]
    out["end_date"] = pd.to_datetime(out.get("end_date"), format="%Y%m%d", errors="coerce").dt.date
    out["ann_date"] = pd.to_datetime(out.get("ann_date"), format="%Y%m%d", errors="coerce").dt.date
    if "report_type" not in out.columns:
        out["report_type"] = None
    for col in FINA_COLS:
        if col in {"symbol", "end_date", "ann_date", "report_type", "source", "raw_payload"}:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce") if col in out.columns else None
    out["source"] = source_label
    out["raw_payload"] = _frame_row_payloads(raw)
    out = out.dropna(subset=["symbol", "end_date"])
    return out[FINA_COLS].copy()


def load_monthly_basic_tushare(engine, start_date: date, end_date: date, calendar_source: str, source_label: str, token: str, log=None):
    trade_dates = get_trade_month_dates_tushare(token, start_date, end_date)
    if not trade_dates:
        trade_dates = get_trade_month_dates(engine, start_date, end_date, calendar_source)
    if not trade_dates:
        return 0, 0, [], "tushare"
    pro = ts.pro_api(token)
    total_rows = 0
    total_affected = 0
    progress = ProgressLogger(name="stock_fundamental.monthly_basic", total=len(trade_dates), unit="trade_dates", log=log, every=12, min_interval_seconds=20.0)
    for trade_dt in trade_dates:
        raw = pro.daily_basic(
            trade_date=trade_dt.strftime("%Y%m%d"),
            fields=(
                "ts_code,trade_date,total_share,float_share,free_share,total_mv,circ_mv,"
                "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,turnover_rate_f,volume_ratio"
            ),
        )
        df = normalize_monthly_basic(raw, source_label)
        affected = upsert_monthly_basic(engine, df)
        total_rows += int(len(df))
        total_affected += affected
        progress.update(current_item=str(trade_dt), rows=int(len(df)), affected=affected)
    progress.finish()
    return total_rows, total_affected, trade_dates, "tushare"


def load_income_tushare(engine, start_date: date, end_date: date, source_label: str, token: str, log=None):
    pro = ts.pro_api(token)
    fields = (
        "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,end_type,basic_eps,diluted_eps,"
        "total_revenue,revenue,operate_profit,total_profit,income_tax,n_income,n_income_attr_p,"
        "minority_gain,oth_compr_income,t_compr_income,compr_inc_attr_p,compr_inc_attr_m_s,"
        "ebit,ebitda,undist_profit"
    )
    raw = _fetch_financial_by_disclosure_dates(engine, pro, "income", fields, start_date, end_date, log=log)
    df = normalize_income(raw, source_label)
    affected = upsert_income(engine, df)
    loaded_periods = sorted({dt.strftime("%Y%m%d") for dt in df["end_date"].dropna().tolist()}) if not df.empty else []
    return int(len(df)), int(affected), loaded_periods, "tushare"


def load_balancesheet_tushare(engine, start_date: date, end_date: date, source_label: str, token: str, log=None):
    pro = ts.pro_api(token)
    fields = (
        "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,end_type,money_cap,trad_asset,"
        "notes_receiv,accounts_receiv,oth_receiv,prepayment,inventories,total_cur_assets,fix_assets,"
        "cip,intan_assets,goodwill,total_nca,total_assets,st_borr,notes_payable,acct_payable,"
        "adv_receipts,total_cur_liab,lt_borr,bond_payable,total_ncl,total_liab,"
        "total_hldr_eqy_exc_min_int,total_hldr_eqy_inc_min_int"
    )
    raw = _fetch_financial_by_disclosure_dates(engine, pro, "balancesheet", fields, start_date, end_date, log=log)
    df = normalize_balancesheet(raw, source_label)
    affected = upsert_balancesheet(engine, df)
    loaded_periods = sorted({dt.strftime("%Y%m%d") for dt in df["end_date"].dropna().tolist()}) if not df.empty else []
    return int(len(df)), int(affected), loaded_periods, "tushare"


def load_fina_indicator_tushare(engine, start_date: date, end_date: date, source_label: str, token: str, log=None):
    pro = ts.pro_api(token)
    fields = (
        "ts_code,ann_date,end_date,report_type,eps,dt_eps,bps,ocfps,roe,roe_dt,roa,roic,"
        "grossprofit_margin,netprofit_margin,profit_to_gr,ocf_to_or,debt_to_eqt,debt_to_assets,"
        "current_ratio,quick_ratio,ar_turn,arturn_days,inv_turn,invturn_days,"
        "or_yoy,netprofit_yoy,tr_yoy,q_sales_yoy,q_profit_yoy,q_ocf_yoy"
    )
    raw = _fetch_financial_by_disclosure_dates(engine, pro, "fina_indicator", fields, start_date, end_date, log=log)
    df = normalize_fina_indicator(raw, source_label)
    affected = upsert_fina_indicator(engine, df)
    loaded_periods = sorted({dt.strftime("%Y%m%d") for dt in df["end_date"].dropna().tolist()}) if not df.empty else []
    return int(len(df)), int(affected), loaded_periods, "tushare"

def upsert_cashflow(engine, df: pd.DataFrame, chunk_size: int = 4000) -> int:
    if df is None or df.empty:
        return 0
    work = df.copy().astype(object).where(pd.notna(df), None)
    insert_sql = """
        INSERT INTO cn_stock_cashflow (
            symbol, end_date, ann_date, f_ann_date, report_type, comp_type, end_type,
            n_cashflow_act, n_cash_flows_inv_act, n_cash_flows_fnc_act, source, raw_payload
        ) VALUES (
            :symbol, :end_date, :ann_date, :f_ann_date, :report_type, :comp_type, :end_type,
            :n_cashflow_act, :n_cash_flows_inv_act, :n_cash_flows_fnc_act, :source, :raw_payload
        )
        ON DUPLICATE KEY UPDATE
            ann_date = VALUES(ann_date),
            f_ann_date = VALUES(f_ann_date),
            report_type = VALUES(report_type),
            comp_type = VALUES(comp_type),
            end_type = VALUES(end_type),
            n_cashflow_act = VALUES(n_cashflow_act),
            n_cash_flows_inv_act = VALUES(n_cash_flows_inv_act),
            n_cash_flows_fnc_act = VALUES(n_cash_flows_fnc_act),
            source = VALUES(source),
            raw_payload = VALUES(raw_payload)
    """
    affected = 0
    with engine.begin() as conn:
        for batch in chunked(work[CASHFLOW_COLS].to_dict(orient="records"), chunk_size):
            ret = conn.execute(text(insert_sql), batch)
            affected += int(ret.rowcount or 0)
    return affected


def normalize_cashflow(raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=CASHFLOW_COLS)
    out = raw.copy()
    out["symbol"] = out.get("ts_code", pd.Series(dtype="object")).astype(str).str.split(".").str[0]
    for date_col in ["end_date", "ann_date", "f_ann_date"]:
        out[date_col] = pd.to_datetime(out.get(date_col), format="%Y%m%d", errors="coerce").dt.date
    for col in ["n_cashflow_act", "n_cash_flows_inv_act", "n_cash_flows_fnc_act"]:
        out[col] = pd.to_numeric(out[col], errors="coerce") if col in out.columns else None
    out["source"] = source_label
    out["raw_payload"] = _frame_row_payloads(raw)
    out = out.dropna(subset=["symbol", "end_date"])
    for col in CASHFLOW_COLS:
        if col not in out.columns:
            out[col] = None
    return out[CASHFLOW_COLS].copy()


def load_cashflow_tushare(engine, start_date: date, end_date: date, source_label: str, token: str, log=None, by_ann_date: bool = False):
    pro = ts.pro_api(token)
    fields = (
        "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,end_type,"
        "n_cashflow_act,n_cash_flows_inv_act,n_cash_flows_fnc_act"
    )
    if by_ann_date:
        raw = _fetch_financial_by_ann_date(pro, "cashflow", fields, start_date, end_date, log=log)
    else:
        raw = _fetch_financial_by_disclosure_dates(engine, pro, "cashflow", fields, start_date, end_date, log=log)
    df = normalize_cashflow(raw, source_label)
    affected = upsert_cashflow(engine, df)
    loaded_periods = sorted({dt.strftime("%Y%m%d") for dt in df["end_date"].dropna().tolist()}) if not df.empty else []
    return int(len(df)), int(affected), loaded_periods, "tushare"


def load_monthly_basic_akshare(engine, end_date: date, source_label: str, max_workers: int = 12, timeout: float = 15.0):
    df, failures = fetch_daily_basic_akshare_snapshot(
        trade_date=end_date,
        source_label=source_label,
        max_workers=max_workers,
        timeout=timeout,
    )
    if not df.empty:
        df["month_key"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y%m")
        df["raw_payload"] = _frame_row_payloads(df)
        df = df[MONTHLY_BASIC_COLS].copy()
    affected = upsert_monthly_basic(engine, df)
    return int(len(df)), int(affected), [end_date], "akshare", int(failures)


def _get_symbol_universe(engine) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT symbol FROM cn_stock_daily_price WHERE symbol IS NOT NULL AND symbol <> '' ORDER BY symbol")).fetchall()
    return [str(row[0]).strip() for row in rows if str(row[0]).strip()]


def _extract_akshare_financial_column(df: pd.DataFrame, keywords: List[str]) -> pd.Series | None:
    for col in df.columns:
        label = str(col).lower()
        if all(k.lower() in label for k in keywords):
            return _normalize_numeric_series(df[col])
    return None


def _normalize_akshare_fina(symbol: str, raw: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=FINA_COLS)
    report_col = raw.columns[0]
    out = pd.DataFrame(index=raw.index)
    out["end_date"] = pd.to_datetime(raw[report_col], errors="coerce").dt.date
    out["symbol"] = symbol
    out["ann_date"] = None
    out["report_type"] = "akshare"
    mappings = {
        "eps": [["eps"]],
        "dt_eps": [["dt", "eps"]],
        "bps": [["bps"]],
        "ocfps": [["ocfps"]],
        "roe": [["roe"]],
        "roe_dt": [["roe", "dt"]],
        "roa": [["roa"]],
        "roic": [["roic"]],
        "grossprofit_margin": [["gross", "margin"]],
        "netprofit_margin": [["net", "margin"]],
        "profit_to_gr": [["profit", "revenue"]],
        "ocf_to_or": [["ocf", "revenue"]],
        "debt_to_eqt": [["debt", "equity"]],
        "debt_to_assets": [["debt", "asset"]],
        "current_ratio": [["current", "ratio"]],
        "quick_ratio": [["quick", "ratio"]],
        "or_yoy": [["revenue", "yoy"]],
        "netprofit_yoy": [["profit", "yoy"]],
        "tr_yoy": [["total", "revenue", "yoy"]],
        "q_sales_yoy": [["quarter", "sales", "yoy"]],
        "q_profit_yoy": [["quarter", "profit", "yoy"]],
        "q_ocf_yoy": [["quarter", "ocf", "yoy"]],
    }
    for target, candidates in mappings.items():
        series = None
        for keywords in candidates:
            series = _extract_akshare_financial_column(raw, keywords)
            if series is not None:
                break
        out[target] = series if series is not None else None
    out["source"] = source_label
    out["raw_payload"] = _frame_row_payloads(raw).reindex(out.index)
    out = out.dropna(subset=["end_date"])
    for col in FINA_COLS:
        if col not in out.columns:
            out[col] = None
    return out[FINA_COLS].copy()


def _fetch_akshare_fina_for_symbol(symbol: str, source_label: str) -> pd.DataFrame:
    last_err = None
    candidates = [
        ("stock_financial_abstract_ths", {"symbol": symbol}),
        ("stock_financial_abstract", {"symbol": symbol}),
        ("stock_financial_analysis_indicator", {"symbol": symbol}),
    ]
    for func_name, kwargs in candidates:
        fn = getattr(ak, func_name, None)
        if not callable(fn):
            continue
        try:
            return _normalize_akshare_fina(symbol, fn(**kwargs), source_label)
        except TypeError as e:
            last_err = e
            if func_name == "stock_financial_abstract_ths":
                try:
                    return _normalize_akshare_fina(symbol, fn(symbol=symbol, indicator="report"), source_label)
                except Exception as inner:
                    last_err = inner
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return pd.DataFrame(columns=FINA_COLS)


def load_fina_indicator_akshare(engine, start_date: date, end_date: date, source_label: str, max_workers: int = 8):
    symbols = _get_symbol_universe(engine)
    if not symbols:
        return 0, 0, [], "akshare", 0
    rows: List[pd.DataFrame] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = {pool.submit(_fetch_akshare_fina_for_symbol, sym, source_label): sym for sym in symbols}
        for fut in as_completed(futures):
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    rows.append(df)
            except Exception:
                failures += 1
    if not rows:
        return 0, 0, [], "akshare", failures
    merged = pd.concat(rows, ignore_index=True)
    merged["end_date"] = pd.to_datetime(merged["end_date"], errors="coerce").dt.date
    merged = merged[(merged["end_date"] >= start_date) & (merged["end_date"] <= end_date)].copy()
    merged = merged.sort_values(["symbol", "end_date"]).drop_duplicates(subset=["symbol", "end_date"], keep="last")
    if merged.empty:
        return 0, 0, [], "akshare", failures
    affected = upsert_fina_indicator(engine, merged)
    periods = sorted({dt.strftime("%Y%m%d") for dt in merged["end_date"].dropna().tolist()})
    return int(len(merged)), int(affected), periods, "akshare", int(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load monthly valuation snapshots and quarterly financial indicators.")
    parser.add_argument("--token", default="", help="Tushare token")
    parser.add_argument("--config", default="", help="Config file path for Tushare token")
    parser.add_argument("--start", default="2008-01-01", help="Start date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end", default="", help="End date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--provider", choices=["auto", "tushare", "akshare"], default="auto")
    parser.add_argument("--calendar-source", choices=["price", "board-map"], default="price")
    parser.add_argument("--basic-source-label", default="tushare_monthly_basic")
    parser.add_argument("--income-source-label", default="tushare_income")
    parser.add_argument("--balance-source-label", default="tushare_balancesheet")
    parser.add_argument("--fina-source-label", default="tushare_fina_indicator")
    parser.add_argument("--cashflow-source-label", default="tushare_cashflow")
    parser.add_argument("--cashflow-by-ann-date", action="store_true",
                        help="Backfill cashflow by scanning each ann_date instead of using cn_event_disclosure_date (use for historical backfill)")
    parser.add_argument("--akshare-workers", type=int, default=8)
    args = parser.parse_args()

    patch_pandas_fillna_method_compat()
    end_date = _parse_ymd(args.end) if str(args.end).strip() else date.today()
    start_date = _parse_ymd(args.start)
    if start_date > end_date:
        raise SystemExit(f"invalid date range: {start_date} > {end_date}")

    token = ""
    tried_files = []
    if args.provider in {"auto", "tushare"}:
        token, tried_files = resolve_tushare_token(args.token, args.config)
        if args.provider == "tushare" and not token:
            msg = "Tushare token is required. Use --token, env TUSHARE_TOKEN/TS_TOKEN, or --config."
            if tried_files:
                msg += f" Tried files: {', '.join(str(p) for p in tried_files)}"
            raise SystemExit(msg)

    engine = build_engine()
    ensure_tables(engine)
    basic_rows = basic_affected = income_rows = income_affected = balance_rows = balance_affected = 0
    fina_rows = fina_affected = cashflow_rows = cashflow_affected = 0
    basic_dates: List[date] = []
    income_periods: List[str] = []
    balance_periods: List[str] = []
    fina_periods: List[str] = []
    cashflow_periods: List[str] = []
    used_provider = args.provider
    ak_failures = 0

    try:
        if args.provider == "akshare":
            basic_rows, basic_affected, basic_dates, used_provider, ak_failures = load_monthly_basic_akshare(
                engine=engine,
                end_date=end_date,
                source_label="akshare_monthly_basic",
                max_workers=args.akshare_workers,
            )
            fina_rows, fina_affected, fina_periods, _, fina_failures = load_fina_indicator_akshare(
                engine=engine,
                start_date=start_date,
                end_date=end_date,
                source_label="akshare_fina_indicator",
                max_workers=args.akshare_workers,
            )
            ak_failures += fina_failures
        else:
            if not token:
                raise RuntimeError("tushare token missing")
            basic_rows, basic_affected, basic_dates, used_provider = load_monthly_basic_tushare(
                engine=engine,
                start_date=start_date,
                end_date=end_date,
                calendar_source=args.calendar_source,
                source_label=args.basic_source_label,
                token=token,
            )
            income_rows, income_affected, income_periods, _ = load_income_tushare(
                engine=engine,
                start_date=start_date,
                end_date=end_date,
                source_label=args.income_source_label,
                token=token,
            )
            balance_rows, balance_affected, balance_periods, _ = load_balancesheet_tushare(
                engine=engine,
                start_date=start_date,
                end_date=end_date,
                source_label=args.balance_source_label,
                token=token,
            )
            fina_rows, fina_affected, fina_periods, _ = load_fina_indicator_tushare(
                engine=engine,
                start_date=start_date,
                end_date=end_date,
                source_label=args.fina_source_label,
                token=token,
            )
            cashflow_rows, cashflow_affected, cashflow_periods, _ = load_cashflow_tushare(
                engine=engine,
                start_date=start_date,
                end_date=end_date,
                source_label=args.cashflow_source_label,
                token=token,
            )
    except Exception as e:
        if args.provider != "auto":
            raise
        print(f"provider_fallback triggered=tushare_to_akshare err={e}")
        basic_rows, basic_affected, basic_dates, used_provider, ak_failures = load_monthly_basic_akshare(
            engine=engine,
            end_date=end_date,
            source_label="akshare_monthly_basic",
            max_workers=args.akshare_workers,
        )
        if not token:
            raise
        income_rows, income_affected, income_periods, _ = load_income_tushare(
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            source_label=args.income_source_label,
            token=token,
        )
        balance_rows, balance_affected, balance_periods, _ = load_balancesheet_tushare(
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            source_label=args.balance_source_label,
            token=token,
        )
        fina_rows, fina_affected, fina_periods, _, fina_failures = load_fina_indicator_akshare(
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            source_label="akshare_fina_indicator",
            max_workers=args.akshare_workers,
        )
        ak_failures += fina_failures
        cashflow_rows, cashflow_affected, cashflow_periods, _ = load_cashflow_tushare(
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            source_label=args.cashflow_source_label,
            token=token,
        )

    basic_range = f"{basic_dates[0]}..{basic_dates[-1]}" if basic_dates else "NA"
    income_range = f"{income_periods[0]}..{income_periods[-1]}" if income_periods else "NA"
    balance_range = f"{balance_periods[0]}..{balance_periods[-1]}" if balance_periods else "NA"
    fina_range = f"{fina_periods[0]}..{fina_periods[-1]}" if fina_periods else "NA"
    cashflow_range = f"{cashflow_periods[0]}..{cashflow_periods[-1]}" if cashflow_periods else "NA"
    apply_ddl(engine, "docs/DDL/cn_market.cn_stock_fundamental_quality_v1.sql")
    apply_ddl(engine, "docs/DDL/cn_market.cn_stock_fundamental_quality_hist_v1.sql")
    apply_ddl(engine, "docs/DDL/cn_market.cn_stock_working_capital_alert_v1.sql")
    snap_rows = rebuild_quality_snapshot(engine)
    print(
        f"stock_fundamental_monthly provider={used_provider} "
        f"basic_dates={len(basic_dates)} basic_range={basic_range} basic_rows={basic_rows} basic_affected={basic_affected} "
        f"income_periods={len(income_periods)} income_range={income_range} income_rows={income_rows} income_affected={income_affected} "
        f"balance_periods={len(balance_periods)} balance_range={balance_range} balance_rows={balance_rows} balance_affected={balance_affected} "
        f"fina_periods={len(fina_periods)} fina_range={fina_range} fina_rows={fina_rows} fina_affected={fina_affected} "
        f"cashflow_periods={len(cashflow_periods)} cashflow_range={cashflow_range} cashflow_rows={cashflow_rows} cashflow_affected={cashflow_affected} "
        f"ak_failures={ak_failures} views_applied=cn_stock_fundamental_quality_v1,cn_stock_fundamental_quality_hist_v1,cn_stock_working_capital_alert_v1 "
        f"snap_rows={snap_rows}"
    )


if __name__ == "__main__":
    main()
