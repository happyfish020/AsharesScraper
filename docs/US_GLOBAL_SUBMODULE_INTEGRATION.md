# US/global data submodule integration

## Boundary

- `AshareScraper` remains the single collection project.
- Existing A-share collection tasks are not changed.
- `app/us_scraper/` is a new submodule for non-A-share/global risk data.
- `GrowthAlpha` must not call external data APIs; it consumes MySQL/SQLite outputs only.

## New task

Run only US/global data:

```bat
python -m app.cli --tasks us_global --no-vpn --us-sources spy,qqq,hyg,vix,ust10,hy_oas --us-start-date 2026-01-01 --us-end-date latest
```

Run built-in daily US/global set:

```bat
python -m app.cli --tasks us_global --no-vpn
```

Download CSV/state only, without MySQL upsert:

```bat
python -m app.cli --tasks us_global --no-vpn --us-no-db
```

Load existing CSVs only:

```bat
python -m app.cli --tasks us_global --no-vpn --us-load-only
```

## Config

Default token/config file:

```text
config/config.json
```

Fields:

```json
{
  "tushare_token": "",
  "fred_api_key": "",
  "alpha_vantage_key": "",
  "fmp_api_key": "",
  "polygon_api_key": "",
  "tiingo_api_key": ""
}
```

The US/global Tushare data source uses the existing shared client:

```text
app/utils/tushare_pro_client.py
```

## Output

CSV/state:

```text
data/us_scraper/*.csv
state/us_scraper/*.json
```

MySQL tables:

```text
us_<source>
```

Examples:

```text
us_spy
us_qqq
us_hyg
us_vix
us_ust10
us_hy_oas
```

## Disabled

`aaii_sentiment` is intentionally not part of the built-in daily US/global set.


## 2026-06-06 Update: DB Placement / Table Prefix

- US/global data uses the same AshareScraper MySQL engine/context, therefore it writes to the same `cn_market_red` database configured by the existing AshareScraper runtime.
- No independent US database connection is introduced.
- All US/global physical tables use the `us_` prefix, for example:
  - `us_spy`
  - `us_qqq`
  - `us_vix`
  - `us_hy_oas`
  - `us_breadth_proxy_a`
- Tables are created automatically with `CREATE TABLE IF NOT EXISTS` before upsert.
- Legacy `us_global_` table prefix is no longer used.
- A-share tasks and existing A-share table names remain untouched.


## 2026-06-06 Default Source Adjustment

The default `us_global` source list is now optimized for short-/medium-term risk reduction decisions.

Default daily set:

```text
spy, qqq, xlu, xlp, hyg, lqd, vix, ust10, ust30, soxx
```

These support the core ratios used later by GrowthAlpha:

```text
QQQ/SPY     growth risk appetite
XLU/SPY     defensive rotation
XLP/SPY     consumer-staple defensive rotation
HYG/LQD     credit risk appetite
VIX         volatility pressure
UST10/UST30 rate pressure
SOXX        AI / semiconductor beta pressure
```

Slow or lagging research sources are no longer included in the daily default run:

```text
tips10, fed_balance_sheet, margin_debt, fear_greed, gdelt_risk, wikipedia_pageviews_risk
```

They are still callable manually, for example:

```bat
python runner.py --tasks us_global --us-sources tips10,fed_balance_sheet --us-years 5
```

Recommended smoke test without DB write:

```bat
python runner.py --tasks us_global --us-sources spy,qqq,vix --us-years 1 --us-no-db
```

Recommended daily run:

```bat
python runner.py --tasks us_global
```

## 2026-06-06 Practical Source Adjustment: FRED Disabled for Daily Risk

Runtime testing showed `fred.stlouisfed.org` is not reliably reachable in the current environment. This is treated as a source availability problem, not a normal slow source. Therefore FRED is no longer a daily/default dependency.

Daily default set is now fully practical for 1-4 week de-risking / cash-raising decisions:

```text
spy, qqq, xlu, xlp, hyg, lqd, vix, soxx, ief, tlt
```

Interpretation:

```text
QQQ/SPY     growth / technology risk appetite
XLU/SPY     defensive utility rotation
XLP/SPY     defensive consumer-staple rotation
HYG/LQD     credit risk appetite
VIX         volatility pressure
SOXX        AI / semiconductor beta pressure
IEF         7-10Y Treasury ETF rate-pressure proxy
TLT         20Y+ Treasury ETF rate-pressure proxy
```

Exact FRED yield sources are disabled from the daily path:

```text
ust10, ust30, tips10, hy_oas, ig_oas, fed_balance_sheet
```

For compatibility, explicit `ust10` and `ust30` requests are redirected to ETF proxies:

```text
ust10 -> ust10_proxy_ief
ust30 -> ust30_proxy_tlt
```

Recommended fast test:

```bat
python runner.py --tasks us_global --us-no-db
```

Recommended explicit test:

```bat
python runner.py --tasks us_global --us-sources spy,qqq,xlu,xlp,hyg,lqd,vix,soxx,ief,tlt --us-years 1 --us-no-db
```


## 2026-06-06 Historical Backfill Mode

US/global raw data now supports a dedicated history-backfill mode for trend calculation.
Daily mode remains one-day / normal incremental operation and does not disturb existing A-share tasks.

Recommended one-year backfill without DB write:

```bat
python runner.py --tasks us_global --us-history-backfill --us-years 1 --us-no-db
```

Recommended one-year backfill into `cn_market_red`:

```bat
python runner.py --tasks us_global --us-history-backfill --us-years 1
```

Explicit window example:

```bat
python runner.py --tasks us_global --us-history-backfill --us-start-date 2025-06-01 --us-end-date latest
```

The backfill mode uses the current practical daily source set unless `--us-sources` is provided:

```text
spy, qqq, xlu, xlp, hyg, lqd, vix, soxx, ief, tlt
```

Purpose: provide enough historical depth for later `us_risk_preference_daily` trend fields such as QQQ/SPY, HYG/LQD, XLU/SPY, VIX trend, and SOXX/SPY.

## US Risk Preference Derived Table

After `us_global` finishes loading raw daily US/global tables into `cn_market_red`, it now automatically refreshes:

```sql
us_risk_preference_daily
```

This is a derived feature table owned by AshareScraper. GrowthAlpha should consume this table directly and should not re-pull or re-calculate raw US/global market data.

Core raw inputs:

- `us_spy`
- `us_qqq`
- `us_soxx`
- `us_xlu`
- `us_xlp`
- `us_hyg`
- `us_lqd`
- `us_vix`
- optional: `us_fear_greed`

Main derived fields:

- `qqq_spy_ratio`, `qqq_spy_ratio_20d`
- `soxx_spy_ratio`, `soxx_spy_ratio_20d`
- `xlu_spy_ratio`, `xlu_spy_ratio_20d`
- `xlp_spy_ratio`, `xlp_spy_ratio_20d`
- `hyg_lqd_ratio`, `hyg_lqd_ratio_20d`
- `vix_close`, `vix_ma20`
- `fear_greed`
- `risk_off_score`
- `risk_regime`

Run one-year backfill first so 20-day trend fields are meaningful:

```bat
python runner.py --tasks us_global --us-history-backfill --us-years 1
```

Then run daily as usual:

```bat
python runner.py --tasks us_global
```
