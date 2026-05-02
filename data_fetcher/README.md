# Data Fetcher Tables Guide

这个目录的任务现在以 MySQL 表为主存储。消费端应直接读取数据库表，不再依赖板块 CSV 文件。

## 安装依赖

```bash
pip install -r requirements.txt
pip install akshare --upgrade
```

## 一、板块日线入库

运行命令：

```bash
python data_fetcher/fetch_akshare_boards.py --type industry
python data_fetcher/fetch_akshare_boards.py --type concept
python data_fetcher/fetch_akshare_boards.py --type all
python data_fetcher/fetch_akshare_boards.py --type industry --board-name 光学光电子
python data_fetcher/fetch_akshare_boards.py --type industry --resume-failed
python data_fetcher/fetch_akshare_boards.py --type all --load-csv
```

这些命令直接更新 MySQL 表：

- `cn_board_industry_daily_em`
- `cn_board_concept_daily_em`
- `cn_board_fetch_failures`

### `cn_board_industry_daily_em`

用途：

- 东方财富行业板块日线标准化结果
- 每次运行按 `(board_name, trade_date)` 增量 upsert

主键：

- `(board_name, trade_date)`

字段：

- `board_code`
- `board_name`
- `trade_date`
- `open`
- `close`
- `high`
- `low`
- `volume`
- `amount`
- `pct_change`
- `source`
- `update_time`

消费示例：

```sql
SELECT trade_date, board_name, close, pct_change, amount
FROM cn_board_industry_daily_em
WHERE board_name = '半导体'
ORDER BY trade_date;
```

### `cn_board_concept_daily_em`

用途：

- 东方财富概念板块日线标准化结果

主键：

- `(board_name, trade_date)`

字段与行业表一致：

- `board_code`
- `board_name`
- `trade_date`
- `open`
- `close`
- `high`
- `low`
- `volume`
- `amount`
- `pct_change`
- `source`
- `update_time`

消费示例：

```sql
SELECT board_name, trade_date, close, pct_change
FROM cn_board_concept_daily_em
WHERE trade_date >= '2026-04-01'
ORDER BY trade_date, board_name;
```

### `cn_board_fetch_failures`

用途：

- 记录单个行业/概念抓取失败项
- `--resume-failed` 会优先重跑这张表里的失败板块

主键：

- `(board_type, board_name)`

字段：

- `board_type`
- `board_name`
- `start_date`
- `end_date`
- `error_type`
- `error_message`
- `retry_count`
- `update_time`

消费示例：

```sql
SELECT *
FROM cn_board_fetch_failures
ORDER BY update_time DESC;
```

### 一次性 CSV 导库

如果历史上已经有标准化 CSV，可以执行：

```bash
python data_fetcher/fetch_akshare_boards.py --type all --load-csv
```

说明：

- 导入 `data/normalized/industry_daily/*.csv`
- 导入 `data/normalized/concept_daily/*.csv`
- 按主键自动 upsert
- 我已检查当前仓库，这两个目录暂时没有现成 CSV 文件

## 二、本地行业映射表 + 自建行业代理指数

运行命令：

```bash
python data_fetcher/build_local_industry_proxy.py --mode refresh-map
python data_fetcher/build_local_industry_proxy.py --mode validate-map
python data_fetcher/build_local_industry_proxy.py --mode daily
python data_fetcher/build_local_industry_proxy.py --mode weekly
python data_fetcher/build_local_industry_proxy.py --mode backfill --start 20260401 --end 20260429
python data_fetcher/build_local_industry_proxy.py --mode audit
```

这些命令直接更新 MySQL 表：

- `cn_local_industry_map_hist`
- `cn_local_industry_proxy_daily`
- `cn_local_task_status`

### `cn_local_industry_map_hist`

用途：

- 股票到 `SW L1` 的本地历史映射表
- 来源是 `cn_board_member_map_d` 过滤 `TUSHARE_SW2021_L1`

主键：

- `(symbol, industry_id, valid_from)`

字段：

- `symbol`
- `industry_id`
- `industry_name`
- `industry_level`
- `valid_from`
- `valid_to`
- `source`
- `is_manual_override`
- `updated_at`

约束规则：

- 任一日期同一 `symbol` 只能归属一个 `SW L1`
- `valid_from <= valid_to`
- 最后一条 `valid_to = 9999-12-31`

消费示例：

```sql
SELECT *
FROM cn_local_industry_map_hist
WHERE symbol = '000001'
ORDER BY valid_from;
```

### `cn_local_industry_proxy_daily`

用途：

- 基于股票日线和本地行业映射生成的行业代理指数表

主键：

- `(trade_date, industry_id)`

字段：

- `trade_date`
- `industry_id`
- `industry_name`
- `industry_level`
- `member_count`
- `total_member_count`
- `coverage_ratio`
- `ret_eqw`
- `ret_amt_w`
- `amount_sum`
- `up_count`
- `down_count`
- `up_ratio`
- `rs20_eqw`
- `rs20_amt_w`
- `rs60_eqw`
- `rs60_amt_w`
- `proxy_close_eqw`
- `proxy_close_amt_w`
- `source`
- `updated_at`

字段含义：

- `ret_eqw`
  - 行业内等权收益
- `ret_amt_w`
  - 行业内成交额加权收益
- `up_ratio`
  - 当日上涨股票数 / 实际参与计算股票数
- `coverage_ratio`
  - 实际参与计算股票数 / 当天行业总成员数
- `rs20_*`
  - 20 日相对强度
- `rs60_*`
  - 60 日相对强度

消费建议：

- 主判断优先看 `ret_amt_w`
- 行业扩散/广度优先看 `ret_eqw + up_ratio + coverage_ratio`
- 若 `coverage_ratio < 0.6`，建议消费端降权或标记低质量

消费示例：

```sql
SELECT trade_date, industry_name, ret_amt_w, up_ratio, rs20_amt_w, rs60_amt_w
FROM cn_local_industry_proxy_daily
WHERE trade_date = '2026-04-29'
ORDER BY rs20_amt_w DESC, ret_amt_w DESC;
```

### `cn_local_task_status`

用途：

- 记录本地映射/代理指数任务最近运行状态

主键：

- `task_name`

字段：

- `task_name`
- `last_success_date`
- `last_checked_date`
- `last_source_max_date`
- `status`
- `message`
- `updated_at`

消费示例：

```sql
SELECT *
FROM cn_local_task_status
ORDER BY updated_at DESC;
```

## 三、消费端推荐读取方式

板块日线：

- 行业板块读 `cn_board_industry_daily_em`
- 概念板块读 `cn_board_concept_daily_em`

本地行业代理：

- 股票映射读 `cn_local_industry_map_hist`
- 行业强弱/广度/代理指数读 `cn_local_industry_proxy_daily`

推荐 Join 示例：

```sql
SELECT
    p.trade_date,
    p.symbol,
    m.industry_id,
    m.industry_name,
    x.ret_amt_w,
    x.up_ratio,
    x.rs20_amt_w,
    x.rs60_amt_w
FROM cn_stock_daily_price p
JOIN cn_local_industry_map_hist m
  ON m.symbol = p.symbol
 AND p.trade_date BETWEEN m.valid_from AND m.valid_to
LEFT JOIN cn_local_industry_proxy_daily x
  ON x.trade_date = p.trade_date
 AND x.industry_id = m.industry_id
WHERE p.trade_date = '2026-04-29';
```
