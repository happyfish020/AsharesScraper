"# AsharesScraper" 
pip install -U playwright
python -m playwright install chromium
pip install curl_cffi


3）SP 跑某天前，先写入 GTT 参数

TRUNCATE TABLE not_secopr.GTT_TRADE_DATE_PARAM;
INSERT INTO not_secopr.GTT_TRADE_DATE_PARAM(TRADE_DATE) VALUES (DATE '2026-01-27');
COMMIT;


然后再跑你的 refresh（insert-select）即可——这时 view 链会自动只算这一天需要的历史窗口。

这招是“工程上最稳”的：不用在 view 里硬编码日期，也不用在 SP 里拼超复杂 SQL。

----
2）缺失交易日识别（以 price 为准）
-- 缺失日期清单（事实源：CN_STOCK_DAILY_PRICE）
SELECT d.trade_date
FROM (
  SELECT DISTINCT trade_date
  FROM not_secopr.CN_STOCK_DAILY_PRICE
) d
LEFT JOIN (
  SELECT DISTINCT trade_date
  FROM not_secopr.CN_SECTOR_ENERGY_SNAP_T
) e
ON e.trade_date = d.trade_date
WHERE e.trade_date IS NULL
ORDER BY d.trade_date;


这条 SQL 是“补历史缺口”的驱动器；日常跑也用它（只会返回今天缺的那一天）。

-----------------

5）每天怎么跑（生产流程）

你的 runner / task 流程里（采集完日数据后）只需要调用：

日常跑（只补缺的那一天）

BEGIN
  not_secopr.SP_BACKFILL_SECTOR_ENERGY_SNAP(NULL, 0);
END;
/


补历史缺口（例如把 2026-01-24~2026-02-06 全补齐）

BEGIN
  not_secopr.SP_BACKFILL_SECTOR_ENERGY_SNAP(DATE '2026-02-06', 0);
END;
/


强制重算（当天已有数据但你想重算）

BEGIN
  not_secopr.SP_REFRESH_SECTOR_ENERGY_SNAP(DATE '2026-02-06', 1);
END;
/


每天怎么跑（生产）
日常（推荐）
BEGIN
  not_secopr.SP_BACKFILL_SECTOR_ENERGY_SNAP(NULL, 0);
END;
/

一次性补齐历史（你现在必须跑）
BEGIN
  not_secopr.SP_BACKFILL_SECTOR_ENERGY_SNAP(DATE '2026-02-06', 0);
END;


---- 
BEGIN
  not_secopr.SP_REFRESH_ROTATION_SNAP_ALL(
    'SR_BASE_V535_EP90_XP55_XC2_MH5_RF5_K2_COST5BPS',
    DATE '2026-01-23',
    1
  );
END;


---------

最终“日更生产链路”（唯一流程）

采集数据后（你现有流程）每日按顺序跑：

Energy 日更（你 Step A 已做）

BEGIN
  not_secopr.SP_BACKFILL_SECTOR_ENERGY_SNAP(TRUNC(SYSDATE), 0);
END;
/


构建 rotation ranked + signal（latest）

BEGIN
  not_secopr.SP_BUILD_SECTOR_ROTATION_RANKED_LATEST;
  not_secopr.SP_BUILD_SECTOR_ROTATION_SIGNAL_LATEST;
END;
/


确保 BT 日历追平到最新交易日（从 price 表）

你已补到 2/06（792 行）。后续每日跑一次即可。

BEGIN
  not_secopr.SP_BACKFILL_ROT_BT_FROM_PRICE(
    'SR_BASE_V535_EP90_XP55_XC2_MH5_RF5_K2_COST5BPS',
    (SELECT MAX(trade_date) FROM not_secopr.CN_STOCK_DAILY_PRICE),
    0
  );
END;
/


生成三张 snapshot（强制=0，生产幂等）

BEGIN
  not_secopr.SP_REFRESH_ROTATION_SNAP_ALL(
    'SR_BASE_V535_EP90_XP55_XC2_MH5_RF5_K2_COST5BPS',
    (SELECT MAX(trade_date) FROM not_secopr.CN_STOCK_DAILY_PRICE),
    0
  );
END;
/


现在就算当天没有 ENTER/持仓/退出，也会落 summary 行，报告永远可读。

交付物 3：cli.py 已更新（满足“自动运行 + 可独立运行”）

你现在可用：

自动链路（推荐）

python runner.py --tasks stock --asof latest


会按顺序跑：DbInitTask -> StockLoaderTask -> RotationDailyRefreshTask

独立运行 rotation（只跑这一任务）

python runner.py --tasks rotation --asof latest


也支持显式组合：

python runner.py --tasks stock,rotation,etf --asof 20260206


Daily Refresh 的输入固定：

p_trade_date（当天交易日）

p_run_id = 'SR_LIVE_DEFAULT'

p_baseline_key = 'DEFAULT_BASELINE'

Daily Refresh 必须做的事：

读取 baseline 参数（你刚建好的 CN_BASELINE_PARAM_T）

生成当日 ENERGY（能量解释）

生成当日 SIGNAL（ENTER/EXIT/KEEP 行为）

生成当日 POS（持仓事实，供 exit 判断；或从既有 pos 表读取）

生成当日报告 snapshot（ENTRY/EXIT/HOLDING 三张清

---
面是一份可直接放到项目里的验证文档（建议命名 docs/rotation_daily_refresh_validation.md）。它只围绕你说的 4 个 SP 步骤（以及最终 snapshot）如何校验，做到“每天可验收、可审计、可定位卡点”。

Rotation Daily Refresh 验证文档（Step B）
目标

在采集系统完成 StockLoaderTask 后运行 Daily Refresh（Step B），确保：

当天 rotation ranked / signal 已生成

BT 日历轴已覆盖当天（用于 T+1 执行日）

三张 snapshot 表（ENTRY / HOLDING / EXIT）当天至少各有 1 行（明细或 summary）

报告层只需 SELECT snapshot 即可展示 Top1 / Pool / Holding / Exit

约定

:RUN_ID 生产默认：SR_BASE_V535_EP90_XP55_XC2_MH5_RF5_K2_COST5BPS

:DT 当天交易日：建议用 price 最大日作为采集完成后的 asof

取当天交易日：

SELECT MAX(trade_date) AS dt
FROM not_secopr.CN_STOCK_DAILY_PRICE;

Step 1 校验：Ranked Latest 是否生成
1.1 基础数据是否到当天（必须）
SELECT MAX(trade_date) AS eod_agg_max_dt
FROM not_secopr.CN_SECTOR_EOD_AGG_T;


期望：eod_agg_max_dt = :DT

1.2 Transition 视图当天是否有行（决定 ranked/signal 能否产）
SELECT COUNT(*) AS n_trans
FROM not_secopr.CN_SECTOR_ROTATION_TRANSITION_V
WHERE trade_date = :DT;


期望：n_trans > 0（你之前是 525）

1.3 Ranked 表当天行数是否等于 transition（通常应一致）
SELECT COUNT(*) AS n_ranked
FROM not_secopr.CN_SECTOR_ROTATION_RANKED_T
WHERE trade_date = :DT;


期望：n_ranked = n_trans（或至少 >0）

若 n_ranked=0：说明 SP_BUILD_SECTOR_ROTATION_RANKED_LATEST 没跑成功或被早退（need=0 / have=need）。

Step 2 校验：Signal Latest 是否生成（ENTER/EXIT/WATCH）
2.1 当天是否已有 signal 行（SP 会“有则 return”）
SELECT action, COUNT(*) AS n
FROM not_secopr.CN_SECTOR_ROTATION_SIGNAL_T
WHERE TRUNC(signal_date) = :DT
GROUP BY action
ORDER BY action;


期望：至少出现 1 类 action（可能只有 WATCH，这在“全市场 NO_CHANGE 日”是正常的）

2.2 如果当天只有 WATCH，必须能解释“为什么无 ENTER/EXIT”

推荐直接看 transition 分布（判断是否策略枚举触发不足）：

SELECT transition, COUNT(*) n
FROM not_secopr.CN_SECTOR_ROTATION_TRANSITION_V
WHERE trade_date = :DT
GROUP BY transition
ORDER BY n DESC FETCH FIRST 30 ROWS ONLY;


你可以进一步复刻 ENTER 条件做 sanity check（不是生产逻辑，只是验收辅助）：

SELECT COUNT(*) AS n_enter_candidate
FROM not_secopr.CN_SECTOR_ROTATION_TRANSITION_V
WHERE trade_date = :DT
  AND transition IN ('IGNITE_TO_CONFIRM','DIRECT_CONFIRM')
  AND theme_rank = 1
  AND tier = 'T1'
  AND up_ma5 >= 0.52
  AND amt_impulse >= 1.10;


期望：>0 才会出现 ENTER；若为 0，当天无 ENTER 属于策略结果，不是 bug。

Step 3 校验：BT 日历轴是否覆盖当天（用于 T+1）

注意：BT 的职责不是“交易日事实源”，而是回测/执行日历轴。
交易日集合来自 CN_STOCK_DAILY_PRICE，但 T+1 执行日、持仓状态机等依赖 BT。

3.1 BT 是否包含当天
SELECT *
FROM not_secopr.CN_SECTOR_ROT_BT_DAILY_T
WHERE run_id = :RUN_ID
  AND trade_date = :DT;


期望：返回 1 行，且：

K_USED 非空（你例子是 2）

N_POS 非空（无持仓时是 0）

EXPOSED_FLAG 非空（无持仓时是 0）

3.2 BT 是否有 next trade day（用于 T+1）
SELECT MIN(trade_date) AS next_trade_date
FROM not_secopr.CN_SECTOR_ROT_BT_DAILY_T
WHERE run_id = :RUN_ID
  AND trade_date > :DT;


期望：

若 price 数据也已入库下一交易日：这里应非空

若当前 price_max_dt 就是 :DT：这里为空是正常的（下一交易日尚未入库）

Step 4 校验：三张 Snapshot 是否落库（报告层唯一依赖）
4.1 行数必须 ≥ 1（明细或 summary）
SELECT 'ENTRY' snap, COUNT(*) n
FROM not_secopr.CN_ROTATION_ENTRY_SNAP_T
WHERE run_id=:RUN_ID AND trade_date=:DT
UNION ALL
SELECT 'HOLDING', COUNT(*)
FROM not_secopr.CN_ROTATION_HOLDING_SNAP_T
WHERE run_id=:RUN_ID AND trade_date=:DT
UNION ALL
SELECT 'EXIT', COUNT(*)
FROM not_secopr.CN_ROTATION_EXIT_SNAP_T
WHERE run_id=:RUN_ID AND trade_date=:DT;


期望：三行都 n >= 1

4.2 ENTRY：Top1 + Pool 是否可读

Top1（若无明细则返回 summary）：

SELECT sector_id, sector_name, entry_rank, energy_pct, energy_tier, source_json
FROM not_secopr.CN_ROTATION_ENTRY_SNAP_T
WHERE run_id=:RUN_ID AND trade_date=:DT
ORDER BY CASE WHEN sector_id=-1 THEN 9 ELSE 0 END, entry_rank
FETCH FIRST 1 ROWS ONLY;


Pool（rank>=2）：

SELECT sector_name, entry_rank, energy_pct, energy_tier, signal_score
FROM not_secopr.CN_ROTATION_ENTRY_SNAP_T
WHERE run_id=:RUN_ID AND trade_date=:DT
  AND sector_id<>-1
  AND entry_rank>=2
ORDER BY entry_rank;

4.3 HOLDING：无持仓必须有 summary；有持仓必须有明细

无持仓 summary（你例子会看到 NO_HOLDING_TODAY）：

SELECT sector_id, sector_name, hold_days, exit_exec_status, source_json
FROM not_secopr.CN_ROTATION_HOLDING_SNAP_T
WHERE run_id=:RUN_ID AND trade_date=:DT
ORDER BY CASE WHEN sector_id=-1 THEN 0 ELSE 1 END, hold_days DESC;

4.4 EXIT：仅对“真实持仓 + EXIT 信号”输出明细，否则 summary
SELECT sector_id, sector_name, exec_exit_date, exit_exec_status, source_json
FROM not_secopr.CN_ROTATION_EXIT_SNAP_T
WHERE run_id=:RUN_ID AND trade_date=:DT
ORDER BY CASE WHEN sector_id=-1 THEN 0 ELSE 1 END;

常见故障定位表（快速判断卡在哪一步）
A) ENTRY/HOLDING/EXIT 都是 0 行

说明 snapshot SP 未执行或异常中断

查：

SELECT object_name, status
FROM all_objects
WHERE owner='not_secopr'
  AND object_type='PROCEDURE'
  AND object_name IN (
    'SP_REFRESH_ROTATION_SNAP_ALL',
    'SP_REFRESH_ROTATION_ENTRY_SNAP',
    'SP_REFRESH_ROTATION_HOLDING_SNAP',
    'SP_REFRESH_ROTATION_EXIT_SNAP'
  );

B) ENTRY 只有 summary（NO_ENTRY_TODAY）

查当日 signal：

SELECT action, COUNT(*) n
FROM not_secopr.CN_SECTOR_ROTATION_SIGNAL_T
WHERE TRUNC(signal_date)=:DT
GROUP BY action;


若只有 WATCH：属于策略结果；用 source_json 或 transition 分布解释即可。

C) summary 里 next_trade_date / exec_exit_date 为 NULL

查 BT 是否有 trade_date > :DT：

SELECT MIN(trade_date) next_trade_date
FROM not_secopr.CN_SECTOR_ROT_BT_DAILY_T
WHERE run_id=:RUN_ID AND trade_date>:DT;


若为空且 price_max_dt=:DT：正常，等待下一交易日数据入库后再跑即可。

D) BT 插入报 ORA-01400（某列非空）

你已经按 DDL 修复过（N_POS/K_USED/EXPOSED_FLAG 必须给默认）

再次出现说明 SP 被回滚/被覆盖，重新部署正确版本。

日常验收建议（每天只跑 4 条）

每天你只需要固定跑这 4 条就能确认整条链路成功：

:DT：

SELECT MAX(trade_date) dt FROM not_secopr.CN_STOCK_DAILY_PRICE;


signal 当天分布：

SELECT action, COUNT(*) n
FROM not_secopr.CN_SECTOR_ROTATION_SIGNAL_T
WHERE TRUNC(signal_date)=:DT
GROUP BY action ORDER BY action;


bt 当天是否存在：

SELECT COUNT(*) n
FROM not_secopr.CN_SECTOR_ROT_BT_DAILY_T
WHERE run_id=:RUN_ID AND trade_date=:DT;


三张 snapshot 行数：

SELECT 'ENTRY' snap, COUNT(*) n FROM not_secopr.CN_ROTATION_ENTRY_SNAP_T WHERE run_id=:RUN_ID AND trade_date=:DT
UNION ALL
SELECT 'HOLDING', COUNT(*) FROM not_secopr.CN_ROTATION_HOLDING_SNAP_T WHERE run_id=:RUN_ID AND trade_date=:DT
UNION ALL
SELECT 'EXIT', COUNT(*) FROM not_secopr.CN_ROTATION_EXIT_SNAP_T WHERE run_id=:RUN_ID AND trade_date=:DT;

---

1) p_force：是否强制重算/重写当天 snapshot（以及相关落库）
它解决的最终需求

生产每天跑同一天时，不希望重复生成、不希望报表数据被反复覆盖（幂等）。

但你在排查/修复 bug 后，又需要对同一天重新生成（强制覆盖）。

推荐语义（你现在的链路里）

p_force = 0（默认）：
幂等模式。若当天数据（如 signal / snapshot）已经存在，则 SP 可能会 return 或跳过 delete/insert，避免重复写入。

✅ 日常自动运行：一直用 0

✅ 防止误覆盖历史、避免“每天跑多次”把审计搞乱

p_force = 1：
强制重算/重写当天（常见做法是 delete 当天记录再 insert）。

✅ 只在这几种情况用：

你修了 SP/视图逻辑（例如你刚经历的列名/绑定问题）

你补齐了上游数据（例如能量补齐后，需要让 entry_snap 的 energy 不再为空）

你手工修了某天数据，想让 snapshot 与 SP 输出重新对齐

❗不建议在日常调度中长期设为 1（容易覆盖你想保留的“当日生成痕迹”）

一句话：
0=生产日更幂等；1=修复/回放时强制重跑某一天。

2) p_refresh_energy：是否在 Daily Refresh 内顺带补能量（Step A 的那一步）
它解决的最终需求

你已经遇到过：energy 不日更会导致 snapshot 里 energy 为空，从而报告无法解释“为什么排第一”。

p_refresh_energy 的作用就是：把 Step A 的补能量动作（SP_BACKFILL_SECTOR_ENERGY_SNAP）作为可选开关，让你这个采集系统在跑 Step B 时能“顺手保证 energy 不缺”。

推荐用法

p_refresh_energy = 1（默认/推荐）：
采集系统每次做 rotation daily refresh，都顺带把当天 energy 补齐。

✅ 最稳：避免因为上游某天漏跑能量导致报告缺字段

✅ 你现在这条链路的目标就是“采集后落库可展示”，所以默认应为 1

p_refresh_energy = 0：
不在这个 SP 里补能量，假设 energy 已由其它流程保证。

✅ 适合这几种情况：

你已经有独立的能量日更调度（先跑 Step A，再跑 Step B），并且严格保证顺序

你在排查 rotation 信号/持仓逻辑，想减少变量（不让 energy 补齐影响你的调试）

你要跑一个“很快的 rotation-only refresh”，不关心 energy 字段（但你的报告需求通常关心）

一句话：
1=Daily Refresh 自带“补能量保险”；0=能量由其它流程保证时才关。

建议的生产默认组合
✅ 日常自动跑（StockLoaderTask 后）

p_force = 0

p_refresh_energy = 1

✅ 修复当天数据/你刚改完 SP 后重跑同一天

p_force = 1

p_refresh_energy = 1（通常一起开，避免 energy 缺口）

✅ 只排查 rotation（不想让 energy 参与干扰）

p_force = 1（想覆盖就 1，不覆盖就 0）

p_refresh_energy = 0

--------