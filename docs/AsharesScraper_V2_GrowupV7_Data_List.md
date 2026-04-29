# GrowthAlpha V7 数据刷新清单

给 `AsharesScraper V2` 使用的刷数列表。目标是保证 `GrowthAlpha_V7` 的日频信号、月末筛选、回测与复盘都能正常运行。

## 1. 刷新总原则

- 日频信号主流程依赖 `日线行情 + 指数行情 + 行业日线 + 最新行业映射/股票状态`。
- 月末主筛选依赖 `monthly_basic + 财报指标`。
- 季报类表按 `公告日/披露日` 增量更新，不需要每天全量重刷。
- 本项目没有“周表”硬依赖；`weekly backtest` 仍然使用日表和月表，只是按周节奏运行策略。

## 2. P0 必刷表

这些表缺失会直接影响主流程。

| 表名 | 刷新频率 | 作用 | 最小必需字段 | 备注 |
|---|---|---|---|---|
| `cn_stock_daily_price` | 日刷，交易日收盘后 | 全市场个股日线、交易日历、涨跌宽度、技术指标、日频信号 | `SYMBOL, TRADE_DATE, OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT, CHG_PCT` | 项目最核心表 |
| `cn_index_daily_price` | 日刷，交易日收盘后 | 沪深300/指数 regime、MacroShock、PositiveSurge、基准收益 | `INDEX_CODE, TRADE_DATE, OPEN, CLOSE, HIGH, LOW, VOLUME, AMOUNT, PRE_CLOSE, CHG_PCT` | 至少保证 `sh000300` |
| `cn_sw_industry_daily` | 日刷，交易日收盘后 | 行业扩散度、板块轮动、主题检测 | `ts_code, name, trade_date, close, amount` | 用于行业强弱和 sector diffusion |
| `cn_stock_monthly_basic` | 月刷，月末最后交易日后 | 月末估值快照、主筛选估值、Regime 月度宽度代理 | `symbol, trade_date, pe_ttm, pb, total_mv, circ_mv` | 历史回测核心估值表 |
| `cn_stock_fina_indicator` | 季报/公告日驱动，建议日增量 | 利润增速、ROE、毛利率、主题/GARP/Davis/恢复类筛选 | `symbol, end_date, ann_date, report_type, netprofit_yoy, q_profit_yoy, or_yoy, eps, dt_eps, roe, roe_dt, grossprofit_margin` | 必须保留 `ann_date`，避免前视偏差 |

## 3. P1 高频辅助表

这些表强烈建议刷，缺了不一定立刻报错，但会削弱实时筛选质量。

| 表名 | 刷新频率 | 作用 | 最小必需字段 | 备注 |
|---|---|---|---|---|
| `cn_stock_daily_basic` | 日刷，交易日收盘后 | 当日估值/流通市值/量比；`get_valuation()` 首选来源 | `symbol, trade_date, pe_ttm, pb, total_mv, circ_mv, volume_ratio` | 历史不全时项目会回退到 `cn_stock_monthly_basic` |
| `cn_stock_leader_score_v2` | 日刷或至少周刷 | 最新股票名称、行业名、主题/龙头辅助映射 | `symbol, trade_date, name, industry_name` | 项目里常取“最新一日快照” |
| `cn_stock_leader_sw_l1_latest_snap` | 日刷或至少周刷 | Universe/ST 过滤时取股票名称 | `symbol, stock_name` | 用于过滤 `ST/*ST` |
| `cn_stock_universe_status_t` | 日刷或至少周刷 | 活跃股票池 | `symbol, is_active` | `is_active=1` 是主 universe 来源 |

## 4. P2 结构映射表

这些表更多服务于行业/板块归属和轮动过滤。

| 表名 | 刷新频率 | 作用 | 最小必需字段 | 备注 |
|---|---|---|---|---|
| `cn_board_industry_member_hist` | 周刷 + 变更日增量 | 股票到申万一级行业映射 | `symbol, board_id, valid_from, valid_to` | `board_id like '801%.SI'` |

## 5. P3 季报补充表

这些表主要出现在 `backtest_adaptive.py` 的增强分析/恢复链路里，建议按公告增量维护。

| 表名 | 刷新频率 | 作用 | 最小必需字段 | 备注 |
|---|---|---|---|---|
| `cn_stock_income` | 季报/公告日驱动，建议日增量 | 收入/净利润补充字段 | `symbol, end_date, ann_date, n_income, n_income_attr_p, revenue, total_revenue` | 回测增强分析使用 |
| `cn_stock_balancesheet` | 季报/公告日驱动，建议日增量 | 应收/存货等资产质量补充 | `symbol, end_date, ann_date, f_ann_date, report_type, comp_type, end_type, notes_receiv, accounts_receiv, inventories` | 财务质量过滤辅助 |
| `cn_event_disclosure_date` | 日增量 | 财报预约披露/实际披露日期 | `symbol, end_date, pre_date, actual_date, modify_date` | 事件日历辅助 |
| `cn_event_earnings_forecast` | 日增量 | 业绩预告事件 | `symbol, ann_date, end_date, forecast_type, p_change_min, p_change_max, net_profit_min, net_profit_max` | 预期变化辅助 |

## 6. 给 AsharesScraper V2 的实际执行节奏

### 日刷

- `cn_stock_daily_price`
- `cn_index_daily_price`
- `cn_sw_industry_daily`
- `cn_stock_daily_basic`
- `cn_stock_leader_score_v2`
- `cn_stock_leader_sw_l1_latest_snap`
- `cn_stock_universe_status_t`
- `cn_stock_fina_indicator` 增量抓取当日新公告
- `cn_stock_income` 增量抓取当日新公告
- `cn_stock_balancesheet` 增量抓取当日新公告
- `cn_event_disclosure_date` 增量抓取
- `cn_event_earnings_forecast` 增量抓取

建议顺序：

1. 先刷交易日行情与指数。
2. 再刷行业日线与最新股票/行业映射。
3. 最后刷财报与事件增量。

### 周刷

- `cn_board_industry_member_hist`
- `cn_stock_leader_score_v2` 全量校准一次
- `cn_stock_universe_status_t` 全量校准一次

说明：

- 项目本身没有周频独立表。
- 周刷主要是做行业归属、股票池、名称映射的全量纠偏。

### 月刷

- `cn_stock_monthly_basic`

要求：

- 必须在每月最后一个交易日收盘后生成月末快照。
- 若月末当晚未完成，次月首个交易日开盘前必须补齐，否则会影响月末候选生成。

## 7. 不建议省略的关键约束

- `cn_stock_fina_indicator` 必须保留 `ann_date`，项目按 `ann_date <= screen_date` 过滤可见财报。
- `cn_stock_monthly_basic` 不能只保留最新月，历史回测需要完整历史。
- `cn_stock_daily_price` 建议保留 `CHG_PCT`，因为 MacroShock 和 PositiveSurge 直接依赖全市场涨跌宽度统计。
- `cn_index_daily_price` 至少覆盖 `sh000300`，最好同时保留回测用到的其他基准指数。
- MySQL 数值列大量为 `DECIMAL`，下游会转数值计算，抓取时不要写成字符串。

## 8. 推荐优先级

如果 `AsharesScraper V2` 先只实现一版最小可用，请按下面顺序：

1. `cn_stock_daily_price`
2. `cn_index_daily_price`
3. `cn_stock_monthly_basic`
4. `cn_stock_fina_indicator`
5. `cn_sw_industry_daily`
6. `cn_stock_daily_basic`
7. `cn_stock_leader_score_v2`
8. `cn_stock_leader_sw_l1_latest_snap`
9. `cn_stock_universe_status_t`
10. `cn_board_industry_member_hist`
11. `cn_stock_income`
12. `cn_stock_balancesheet`
13. `cn_event_disclosure_date`
14. `cn_event_earnings_forecast`

## 9. 当前项目中的特别说明

- `cn_stock_daily_basic` 在这个项目里是“实时估值优先表”，但历史回测真正兜底依赖的是 `cn_stock_monthly_basic`。
- `weekly` 脚本不是周表策略，只是周频运行逻辑，所以不要为它单独设计周表。
- `cn_market` 只是库名/数据域概念，不是项目直接查询的业务表。

