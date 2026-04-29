-- cn_market.cn_sw_industry_daily_latest_v
-- Latest snapshot view for Shenwan industry daily行情.

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `cn_sw_industry_daily_latest_v` AS
SELECT
    d.ts_code,
    d.name,
    d.trade_date,
    d.open,
    d.high,
    d.low,
    d.close,
    d.`change`,
    d.pct_change,
    d.vol,
    d.amount,
    d.pe,
    d.pb,
    d.float_mv,
    d.source
FROM cn_sw_industry_daily d
WHERE d.trade_date = (
    SELECT MAX(x.trade_date)
    FROM cn_sw_industry_daily x
);
