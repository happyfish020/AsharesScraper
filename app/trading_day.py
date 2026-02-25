from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import baostock as bs

from app.utils.wireguard_helper import activate_tunnel

def get_latest_trade_date(today: Optional[date] = None, lookback_days: int = 45) -> str:
    """Return latest China A-share trading date as YYYYMMDD using baostock trade calendar.

    Falls back to local today if baostock fails.
    """
    d0 = today or date.today()
    start = (d0 - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = d0.strftime("%Y-%m-%d")

    try:
        activate_tunnel("cn")
        lg = bs.login()
        # even if login fails, proceed to fallback
        if lg.error_code != "0":
            return d0.strftime("%Y%m%d")

        rs = bs.query_trade_dates(start_date=start, end_date=end)
        if rs.error_code != "0":
            return d0.strftime("%Y%m%d")

        latest = None
        while (rs.error_code == "0") and rs.next():
            row = rs.get_row_data()
            # row: [calendar_date, is_trading_day]
            cal_date, is_td = row[0], row[1]
            if is_td == "1":
                latest = cal_date  # YYYY-MM-DD
        bs.logout()

        if latest:
            return latest.replace("-", "")
        return d0.strftime("%Y%m%d")
    except Exception:
        try:
            bs.logout()
        except Exception:
            pass
        return d0.strftime("%Y%m%d")
