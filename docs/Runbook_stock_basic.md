# Stock Basic Runbook

目标：维护 `cn_stock_daily_basic`，为日常估值/市值查询和 `cn_stock_leader_score_v2` 提供稳定输入。

## Current Object Note

- checked in local `cn_market` on `2026-05-05`
- `cn_stock_leader_score_v2` is currently a `VIEW`, not a physical table
- `stock_basic` refresh rebuilds the leader-score views after loading `cn_stock_daily_basic`
- in the current DB, `cn_stock_leader_score_v2` returns `0` rows on `2026-04-29` and `2026-04-30`

## API Strategy

- `daily_basic` 使用 `trade_date` 作为主抓取维度
- 一次 `pro.daily_basic(trade_date=YYYYMMDD)` 拉取一个交易日的全市场快照
- 日更优先按最近交易日刷新
- 历史回填建议按 `price` 交易日历、从近到远执行

## Progress Output

- loader prints a launch banner with provider, date range, calendar source, order, and batch size
- rolling progress includes percent, current trade date, cumulative rows, affected rows, elapsed time, and ETA
- when `--batch-size > 0`, console also prints `batch` and `batch_done` markers

## Runner Task

主程序任务名：

```powershell
python runner.py --tasks stock_basic --asof latest
```

默认行为：

- 启用 `stock_basic` 日更任务
- 默认用 `price` 交易日历
- 默认回刷最近 `7` 天，修复近期漏日也不怕
- 完成后重建视图：
  - `cn_stock_leader_score_v1`
  - `cn_stock_leader_score_v2`

## 常用环境变量

- `STOCK_BASIC_ENABLED=1`
- `STOCK_BASIC_FORCE=0`
- `STOCK_BASIC_PROVIDER=tushare`
- `STOCK_BASIC_CALENDAR_SOURCE=price`
- `STOCK_BASIC_LOOKBACK_DAYS=7`
- `STOCK_BASIC_DATE_ORDER=asc`
- `STOCK_BASIC_BATCH_SIZE=0`
- `STOCK_BASIC_SOURCE_LABEL=tushare_daily_basic`
- `STOCK_BASIC_AKSHARE_WORKERS=12`
- `STOCK_BASIC_AKSHARE_TIMEOUT=15`

兼容旧变量：

- `STOCK_BASIC_WEEKLY_*` 仍可用，但建议逐步迁移到 `STOCK_BASIC_*`

## 日更建议

生产日更推荐：

```powershell
set STOCK_BASIC_ENABLED=1
set STOCK_BASIC_PROVIDER=tushare
set STOCK_BASIC_CALENDAR_SOURCE=price
set STOCK_BASIC_LOOKBACK_DAYS=7
python runner.py --tasks stock,board,stock_basic --asof latest
```

说明：

- `stock` 先补日线价格
- `board` 再补行业映射
- `stock_basic` 最后补估值/市值并重建 leader 视图

## 历史回填

近到远回填：

```powershell
C:\Users\nling\AppData\Local\Python\bin\python.exe -m app.tools.sync_cn_stock_daily_basic_from_tushare --provider tushare --calendar-source price --date-order desc --batch-size 20 --start 2000-01-04 --end 2026-04-29
```

更稳妥的分段方式：

1. 先补最近 1-2 年
2. 再逐年向前补
3. 每段补完后检查日期覆盖

## 校验 SQL

全表覆盖：

```sql
SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT trade_date)
FROM cn_stock_daily_basic;
```

单票覆盖：

```sql
SELECT MIN(trade_date), MAX(trade_date), COUNT(*)
FROM cn_stock_daily_basic
WHERE symbol='603986';
```

指定日期行数：

```sql
SELECT trade_date, COUNT(*) AS cnt
FROM cn_stock_daily_basic
WHERE trade_date BETWEEN '2026-04-28' AND '2026-04-29'
GROUP BY trade_date
ORDER BY trade_date DESC;
```
