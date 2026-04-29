from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import akshare as ak

from app.utils.wireguard_helper import activate_tunnel


def get_latest_trade_date(today: Optional[date] = None, lookback_days: int = 45) -> str:
    """Return latest China A-share trading date as YYYYMMDD using akshare trade calendar.

    Uses ak.tool_trade_date_hist_sina() to fetch the full trading-day list,
    then returns the most recent date that is <= today.
    Falls back to local today if akshare fails.
    """
    d0 = today or date.today()
    end = d0.strftime("%Y-%m-%d")

    try:
        activate_tunnel("cn")

        # 返回 DataFrame，列名为 "trade_date"，类型为 datetime.date 或 str
        df = ak.tool_trade_date_hist_sina()

        if df is None or df.empty:
            return d0.strftime("%Y%m%d")

        # 统一转为 date 对象方便比较
        col = df.columns[0]  # 通常为 "trade_date"
        dates = df[col].tolist()

        # 兼容 str / datetime / date 类型
        def to_date(v):
            if isinstance(v, date):
                return v
            # pandas Timestamp 或 datetime
            if hasattr(v, "date"):
                return v.date()
            # str: YYYY-MM-DD 或 YYYYMMDD
            s = str(v).strip()
            if len(s) == 10:
                return date.fromisoformat(s)
            return date(int(s[:4]), int(s[4:6]), int(s[6:8]))

        trade_dates = sorted([to_date(v) for v in dates])

        # 找到 <= d0 的最大交易日
        latest = None
        for td in trade_dates:
            if td <= d0:
                latest = td

        if latest:
            return latest.strftime("%Y%m%d")
        return d0.strftime("%Y%m%d")

    except Exception:
        return d0.strftime("%Y%m%d")
