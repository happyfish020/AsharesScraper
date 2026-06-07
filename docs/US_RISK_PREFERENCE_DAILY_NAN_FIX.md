# US Risk Preference Daily NaN Fix

## Fix
`us_risk_preference_daily` now sanitizes every record before MySQL insert:

- `NaN` -> `NULL`
- `pd.NA` -> `NULL`
- `inf/-inf` -> `NULL`

This is required because early rolling-window rows can have empty 20D fields, and PyMySQL rejects float NaN values.

## Scope
Only `app/us_scraper/runner.py` is changed.

## Test
```bat
python runner.py --tasks us_global --us-history-backfill --us-years 1
```

Then:

```sql
SELECT *
FROM us_risk_preference_daily
ORDER BY trade_date DESC
LIMIT 10;
```
