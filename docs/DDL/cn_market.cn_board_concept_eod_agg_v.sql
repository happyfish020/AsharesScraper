-- cn_market.cn_board_concept_eod_agg_v source
-- Performance-oriented rewrite:
-- 1) treat CN_BOARD_CONCEPT_CONS as snapshot dimension and use latest ASOF_DATE
-- 2) avoid full-history LAG() over CN_STOCK_DAILY_PRICE

create or replace
algorithm = UNDEFINED view `cn_board_concept_eod_agg_v` as
with `asof_anchor` as (
select
    max(`c`.`ASOF_DATE`) as `max_asof_date`
from
    `cn_board_concept_cons` `c`),
`date_anchor` as (
select
    max(`p`.`TRADE_DATE`) as `max_trade_date`
from
    `cn_stock_daily_price` `p`),
`cons_latest` as (
select
    `c`.`CONCEPT_ID` as `concept_id`,
    `c`.`SYMBOL` as `symbol`
from
    `cn_board_concept_cons` `c`
join `asof_anchor` `a` on
    (`c`.`ASOF_DATE` = `a`.`max_asof_date`)),
`px` as (
select
    `p`.`SYMBOL` as `symbol`,
    `p`.`TRADE_DATE` as `trade_date`,
    `p`.`AMOUNT` as `amount`,
    (case
        when ((coalesce(`p`.`PRE_CLOSE`, (`p`.`CLOSE` - `p`.`change`)) is not null)
            and (coalesce(`p`.`PRE_CLOSE`, (`p`.`CLOSE` - `p`.`change`)) <> 0)) then ((`p`.`CLOSE` / coalesce(`p`.`PRE_CLOSE`, (`p`.`CLOSE` - `p`.`change`))) - 1)
        when (`p`.`CHG_PCT` is not null) then (case
            when (abs(`p`.`CHG_PCT`) > 1) then (`p`.`CHG_PCT` / 100)
            else `p`.`CHG_PCT`
        end)
        else null
    end) as `ret_eff`
from
    `cn_stock_daily_price` `p`
where
    (`p`.`TRADE_DATE` >= (
    select
        `d`.`max_trade_date`
    from
        `date_anchor` `d`)))
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
join `px` `p` on
    ((`p`.`symbol` = `c`.`symbol`)))
group by
    `c`.`concept_id`,
    `p`.`trade_date`;
