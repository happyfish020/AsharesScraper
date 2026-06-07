from datetime import date

import pandas as pd

from app.us_scraper.datasources.base import BaseDataSource


class FinraMarginDebtSource(BaseDataSource):
    """
    FINRA monthly margin statistics downloader.

    Source file:
      https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx
    """

    XLSX_URL = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
    DATE_COL = "Year-Month"
    VALUE_COL = "Debit Balances in Customers' Securities Margin Accounts"

    def __init__(self, name: str = "margin_debt") -> None:
        super().__init__(name=name)

    def _fetch(self, start_date: date, end_date: date) -> pd.DataFrame:
        df = pd.read_excel(self.XLSX_URL)
        if df.empty:
            return df

        # Convert FINRA month label (YYYY-MM) to month-start timestamp.
        df[self.DATE_COL] = pd.to_datetime(df[self.DATE_COL], format="%Y-%m", errors="coerce")
        df = df.dropna(subset=[self.DATE_COL])
        df = df.rename(columns={self.DATE_COL: "date", self.VALUE_COL: "value"})
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
        df = df.loc[mask, ["date", "value"]]
        return df.set_index("date")
