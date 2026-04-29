# Event Data Pipeline Runbook

目标：从 Tushare Pro 拉取 A 股事件数据（业绩预告、业绩快报、财务指标、披露日期、分红送股、公告元数据）并落库到 `cn_market`，支持全量/增量/可审计运行。

## 依赖

- 已配置 `TUSHARE_TOKEN` 或在配置文件中可解析
- MySQL 数据库可连接（默认 `cn_market`）

## 运行方式

### 方式 1：Runner 任务

```bash
python runner.py --tasks event --asof latest --days 1
```

常用环境变量：

- `EVENT_ENABLED=1` 是否启用
- `EVENT_FULL_START=2008-01-01` 全量起始日
- `EVENT_FORCE_FULL=0` 强制全量
- `EVENT_LOOKBACK_BUFFER_DAYS=7` 增量回看缓冲
- `EVENT_WITH_ANNS=0` 是否尝试 `anns_d`
- `EVENT_WITH_SIGNAL=1` 是否生成 `cn_event_signal_daily`
- `EVENT_AUDIT_DIR=audit_reports` 质量报告输出目录
- `EVENT_MAX_SYMBOLS=0` 限制 `fina_indicator` 拉取股票数（用于烟囱测试）
- `EVENT_SYMBOLS=000001,600000` 指定 `fina_indicator` 股票清单

### 方式 2：工具脚本

```bash
python -m app.tools.sync_cn_stock_event_tushare --start 2008-01-01 --end 2026-04-08 --with-signal
```

可选参数：

- `--with-anns` 尝试 `anns_d`（权限不足会跳过）
- `--with-signal` 构建 `cn_event_signal_daily`
- `--skip-fina` 跳过 `fina_indicator`（大规模批次建议先跳过）
- `--max-symbols` 限制 `fina_indicator` 拉取股票数
- `--symbols` 指定 `fina_indicator` 股票清单
- `--audit-dir` 质量报告输出目录

## 表结构

- `cn_event_earnings_forecast`
- `cn_event_earnings_express`
- `cn_event_fina_indicator`
- `cn_event_disclosure_date`
- `cn_event_dividend`
- `cn_event_announcement_meta`
- `cn_event_signal_daily`

对应 DDL 在 `docs/DDL/` 下。

## 增量逻辑

- 默认基于每张表最大日期 + `EVENT_LOOKBACK_BUFFER_DAYS` 回看
- 不需要手工清表，可重复跑

## 数据质量报告

运行结束后自动在 `audit_reports/` 输出 JSON 报告，包含：

- 每表总行数
- 覆盖股票数
- 覆盖日期范围
- 重复键检查
- 关键字段空值计数
- 最新更新时间

---

## Fetch Strategy

This event pipeline now follows the API-first fetch strategy instead of forcing one loop shape for all endpoints.

### Symbol-first endpoints

These endpoints are fetched by `symbol/ts_code` history:

- `forecast`
- `express`
- `dividend`
- `fina_indicator`

Reason:

- these APIs are naturally aligned to single-stock historical event / statement retrieval
- Tushare official docs for `forecast` and `express` describe the standard endpoint as single-stock history first

### Date-first / month-first endpoints

These endpoints still use date-style batching:

- `disclosure_date`
- `anns_d`

Reason:

- these APIs are more naturally handled as market-wide announcement batches

### Operational rule

- do not force all event fetchers into one loop style
- use the official API's primary access pattern
- if the endpoint is single-stock history oriented, prefer `symbol-first`
- if the endpoint is market snapshot oriented, prefer `date-first`

See also:

- `docs/api_fetch_strategy.md`
