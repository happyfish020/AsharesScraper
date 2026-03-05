from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from sqlalchemy import text

from app.settings import build_engine


INSERT_SQL = text(
    """
    INSERT IGNORE INTO cn_stock_daily_price
    (symbol, trade_date, close, pre_close, chg_pct, `change`, source, window_start, exchange)
    VALUES
    (:symbol, :trade_date, :close, :pre_close, :chg_pct, :change, :source, :window_start, :exchange)
    """
)


def _batched(rows: List[dict], batch_size: int) -> Iterable[List[dict]]:
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def main() -> None:
    p = argparse.ArgumentParser(description="Import wide close-price CSV into cn_stock_daily_price")
    p.add_argument("--csv", required=True, help="Path to wide matrix CSV")
    p.add_argument("--start", default="20050101", help="YYYYMMDD")
    p.add_argument("--end", default="20191231", help="YYYYMMDD")
    p.add_argument("--source", default="csv_2005_2019_qfq_close", help="source tag")
    p.add_argument("--chunksize", type=int, default=120, help="rows per CSV chunk")
    p.add_argument("--insert-batch", type=int, default=5000, help="insert batch size")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    start_dt = datetime.strptime(args.start, "%Y%m%d").date()
    end_dt = datetime.strptime(args.end, "%Y%m%d").date()
    if start_dt > end_dt:
        raise ValueError(f"invalid date range: {args.start}>{args.end}")

    engine = build_engine()
    window_start = args.start
    total_inserted = 0
    total_seen = 0

    with engine.begin() as conn:
        for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, dtype=str, chunksize=args.chunksize), start=1):
            date_col = chunk.columns[0]
            chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce").dt.date
            chunk = chunk[(chunk[date_col] >= start_dt) & (chunk[date_col] <= end_dt)]
            if chunk.empty:
                continue

            long_df = chunk.melt(id_vars=[date_col], var_name="symbol_raw", value_name="close_raw")
            long_df["close_raw"] = long_df["close_raw"].astype(str).str.strip()
            long_df = long_df[(long_df["close_raw"] != "") & (long_df["close_raw"].str.lower() != "nan")]
            if long_df.empty:
                continue

            sym_ex = long_df["symbol_raw"].str.extract(r"(?P<symbol>\d{6})\.(?P<ex>[A-Za-z]{2})")
            long_df["symbol"] = sym_ex["symbol"]
            long_df["ex"] = sym_ex["ex"].str.upper()
            long_df = long_df.dropna(subset=["symbol", "ex"])
            if long_df.empty:
                continue

            long_df["close"] = pd.to_numeric(long_df["close_raw"], errors="coerce")
            long_df = long_df.dropna(subset=["close"])
            if long_df.empty:
                continue

            long_df["exchange"] = long_df["ex"].map({"SH": "SSE", "SZ": "SZSE"}).fillna("SZSE")
            long_df = long_df.rename(columns={date_col: "trade_date"})

            records = []
            for r in long_df[["symbol", "trade_date", "close", "exchange"]].itertuples(index=False):
                records.append(
                    {
                        "symbol": r.symbol,
                        "trade_date": r.trade_date,
                        "close": float(r.close),
                        "pre_close": None,
                        "chg_pct": None,
                        "change": None,
                        "source": args.source,
                        "window_start": window_start,
                        "exchange": r.exchange,
                    }
                )

            total_seen += len(records)
            inserted_this_chunk = 0
            for batch in _batched(records, args.insert_batch):
                ret = conn.execute(INSERT_SQL, batch)
                inserted_this_chunk += int(getattr(ret, "rowcount", 0) or 0)
            total_inserted += inserted_this_chunk

            print(
                f"[chunk {chunk_idx}] rows={len(records)} inserted={inserted_this_chunk} "
                f"cum_seen={total_seen} cum_inserted={total_inserted}"
            )

        # Fill derived fields for imported rows where pre_close/chg_pct/change are empty.
        conn.execute(
            text(
                """
                UPDATE cn_stock_daily_price t
                JOIN (
                    SELECT
                        symbol,
                        trade_date,
                        LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date) AS pre_close_calc
                    FROM cn_stock_daily_price
                    WHERE source = :src
                ) x
                  ON t.symbol = x.symbol AND t.trade_date = x.trade_date
                SET
                  t.pre_close = COALESCE(t.pre_close, x.pre_close_calc),
                  t.`change` = COALESCE(t.`change`, t.close - x.pre_close_calc),
                  t.chg_pct = COALESCE(
                      t.chg_pct,
                      CASE WHEN x.pre_close_calc IS NULL OR x.pre_close_calc = 0
                           THEN NULL
                           ELSE (t.close / x.pre_close_calc - 1) * 100
                      END
                  )
                WHERE t.source = :src
                """
            ),
            {"src": args.source},
        )

    print(f"[done] total_seen={total_seen} total_inserted={total_inserted} source={args.source}")


if __name__ == "__main__":
    main()

