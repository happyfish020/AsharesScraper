# Data Fetcher / Local Industry Proxy

这个目录现在包含两类独立数据任务：

- `fetch_akshare_boards.py`
- `build_local_industry_proxy.py`

它们都只服务于数据采集、标准化、落盘、数据库刷新，不包含任何策略、选股、回测或交易逻辑。

## 安装依赖

```bash
pip install -r requirements.txt
pip install akshare --upgrade
```

## 一、AkShare 板块采集器

常用命令：

```bash
python data_fetcher/fetch_akshare_boards.py --type industry
python data_fetcher/fetch_akshare_boards.py --type industry --board-name 光学光电子
python data_fetcher/fetch_akshare_boards.py --type industry --resume-failed
```

输出目录：

- `data/raw/industry_daily/*.csv`
- `data/raw/concept_daily/*.csv`
- `data/normalized/industry_daily/*.csv`
- `data/normalized/concept_daily/*.csv`
- `logs/fetch_akshare_boards.log`
- `data_fetcher/data/failed/industry_failed.csv`
- `data_fetcher/data/failed/concept_failed.csv`

排障建议：

- 先执行 `pip install akshare --upgrade`
- 先单独测试 `--board-name 光学光电子`
- 如果仍然出现 `RemoteDisconnected`，继续降低频率
- 等待 `10-30` 分钟后再执行 `--resume-failed`
- 不要一次性高速扫全量行业或概念板块

## 二、本地行业映射表 + 自建行业代理指数

这个构建器不依赖东方财富板块接口，直接使用本地数据库中的：

- `cn_stock_daily_price`
- `cn_board_industry_member_hist`
- `cn_board_industry_master`

默认口径：

- 行业层级：`SW L1`
- 成分映射来源：`tushare_sw_l1`
- 行业名称来源：`TUSHARE_SW2021_L1`

### 生成对象

- `cn_local_industry_map_hist`
- `cn_local_industry_proxy_daily`
- `cn_local_task_status`
- `logs/local_industry_proxy.log`

### 运行命令

首次建本地映射：

```bash
python data_fetcher/build_local_industry_proxy.py --mode refresh-map
```

日常日更：

```bash
python data_fetcher/build_local_industry_proxy.py --mode daily
```

每周刷新：

```bash
python data_fetcher/build_local_industry_proxy.py --mode weekly
```

指定区间补刷：

```bash
python data_fetcher/build_local_industry_proxy.py --mode backfill --start 20240101 --end 20240430
```

做覆盖审计：

```bash
python data_fetcher/build_local_industry_proxy.py --mode audit
```

串行执行一轮：

```bash
python data_fetcher/build_local_industry_proxy.py --mode all
```

### 运行模式说明

- `refresh-map`
  - 只刷新本地行业映射表
- `daily`
  - 从代理表最新日期之后开始增量生成
- `weekly`
  - 先刷新行业映射，再重建最近 `35` 天，并修复缺失日期
- `backfill`
  - 按指定日期区间重建代理指数
- `audit`
  - 检查股票日线与代理指数之间的日期缺口和行业覆盖差异
- `all`
  - 依次执行 `refresh-map -> daily -> audit`

### 代理指标字段

`cn_local_industry_proxy_daily` 包含：

- `trade_date`
- `industry_id`
- `industry_name`
- `industry_level`
- `member_count`
- `amount_sum`
- `ret_eqw`
- `ret_amt_w`
- `proxy_close_eqw`
- `proxy_close_amt_w`
- `source`
- `updated_at`

其中：

- `ret_eqw`：等权收益
- `ret_amt_w`：按成交额加权收益
- `proxy_close_eqw`：以 `1000` 为基准累计出来的等权代理指数
- `proxy_close_amt_w`：以 `1000` 为基准累计出来的成交额加权代理指数

### 日常调度建议

- 每日股票日线入库后执行一次：
  - `python data_fetcher/build_local_industry_proxy.py --mode daily`
- 每周执行一次：
  - `python data_fetcher/build_local_industry_proxy.py --mode weekly`
- 如果怀疑历史有断档：
  - `python data_fetcher/build_local_industry_proxy.py --mode audit`
  - 再按提示区间执行 `backfill`
