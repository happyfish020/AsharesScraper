-- cn_market.cn_board_concept_eod_agg_v source

create or replace
algorithm = UNDEFINED view `cn_board_concept_eod_agg_v` as with `cons_latest` as (
select
    `t`.`CONCEPT_ID` as `concept_id`,
    `t`.`SYMBOL` as `symbol`
from
    (
    select
        `c`.`CONCEPT_ID` as `CONCEPT_ID`,
        `c`.`SYMBOL` as `SYMBOL`,
        `c`.`EXCHANGE` as `EXCHANGE`,
        `c`.`ASOF_DATE` as `ASOF_DATE`,
        `c`.`SOURCE` as `SOURCE`,
        `c`.`CREATED_AT` as `CREATED_AT`,
        row_number() over (partition by `c`.`CONCEPT_ID`,
        `c`.`SYMBOL`
    order by
        `c`.`ASOF_DATE` desc ) as `rn`
    from
        `cn_board_concept_cons` `c`) `t`
where
    (`t`.`rn` = 1)),
`px0` as (
select
    `cn_stock_daily_price`.`SYMBOL` as `symbol`,
    `cn_stock_daily_price`.`TRADE_DATE` as `trade_date`,
    `cn_stock_daily_price`.`CLOSE` as `close`,
    `cn_stock_daily_price`.`AMOUNT` as `amount`,
    `cn_stock_daily_price`.`PRE_CLOSE` as `pre_close`,
    `cn_stock_daily_price`.`CHANGE` as `change`,
    `cn_stock_daily_price`.`CHG_PCT` as `chg_pct`,
    lag(`cn_stock_daily_price`.`CLOSE`) over (partition by `cn_stock_daily_price`.`SYMBOL`
order by
    `cn_stock_daily_price`.`TRADE_DATE` ) as `lag_close`
from
    `cn_stock_daily_price`),
`px` as (
select
    `px0`.`symbol` as `symbol`,
    `px0`.`trade_date` as `trade_date`,
    `px0`.`close` as `close`,
    `px0`.`amount` as `amount`,
    coalesce(`px0`.`pre_close`,(`px0`.`close` - `px0`.`change`), `px0`.`lag_close`) as `pre_close_eff`,
    `px0`.`chg_pct` as `chg_pct`
from
    `px0`),
`px2` as (
select
    `px`.`symbol` as `symbol`,
    `px`.`trade_date` as `trade_date`,
    `px`.`close` as `close`,
    `px`.`amount` as `amount`,
    `px`.`pre_close_eff` as `pre_close_eff`,
    (case
        when ((`px`.`pre_close_eff` is not null)
            and (`px`.`pre_close_eff` <> 0)) then ((`px`.`close` / `px`.`pre_close_eff`) - 1)
        when (`px`.`chg_pct` is not null) then (case
            when (abs(`px`.`chg_pct`) > 1) then (`px`.`chg_pct` / 100)
            else `px`.`chg_pct`
        end)
        else null
    end) as `ret_eff`
from
    `px`)
select
    'CONCEPT' as `sector_type`,
    `c`.`concept_id` as `sector_id`,
    `p`.`trade_date` as `trade_date`,
    count(0) as `members`,
    sum(ifnull(`p`.`amount`, 0)) as `amount_sum`,
    avg(`p`.`ret_eff`) as `avg_ret`,
    avg(`p`.`ret_eff`) as `median_ret`,
    avg(`p`.`ret_eff`) as `eqw_ret`,
    avg((case when (`p`.`ret_eff` is null) then null when (`p`.`ret_eff` > 0) then 1 else 0 end)) as `up_ratio`
from
    (`cons_latest` `c`
join `px2` `p` on
    ((`p`.`symbol` = `c`.`symbol`)))
group by
    `c`.`concept_id`,
    `p`.`trade_date`;