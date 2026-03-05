-- cn_market.cn_board_concept_eod_agg_v source
-- History-safe version using daily mapping table generated from valid_from/valid_to.

create or replace
algorithm = UNDEFINED view `cn_board_concept_eod_agg_v` as
select
    'CONCEPT' as `sector_type`,
    `m`.`sector_id` as `sector_id`,
    `p`.`trade_date` as `trade_date`,
    count(0) as `members`,
    sum(ifnull(`p`.`amount`, 0)) as `amount_sum`,
    avg((case
        when ((coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`)) is not null)
            and (coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`)) <> 0)) then ((`p`.`close` / coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`))) - 1)
        when (`p`.`chg_pct` is not null) then (case
            when (abs(`p`.`chg_pct`) > 1) then (`p`.`chg_pct` / 100)
            else `p`.`chg_pct`
        end)
        else null
    end)) as `avg_ret`,
    avg((case
        when ((coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`)) is not null)
            and (coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`)) <> 0)) then ((`p`.`close` / coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`))) - 1)
        when (`p`.`chg_pct` is not null) then (case
            when (abs(`p`.`chg_pct`) > 1) then (`p`.`chg_pct` / 100)
            else `p`.`chg_pct`
        end)
        else null
    end)) as `median_ret`,
    avg((case
        when ((coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`)) is not null)
            and (coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`)) <> 0)) then ((`p`.`close` / coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`))) - 1)
        when (`p`.`chg_pct` is not null) then (case
            when (abs(`p`.`chg_pct`) > 1) then (`p`.`chg_pct` / 100)
            else `p`.`chg_pct`
        end)
        else null
    end)) as `eqw_ret`,
    avg((case
        when ((case
            when ((coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`)) is not null)
                and (coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`)) <> 0)) then ((`p`.`close` / coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`))) - 1)
            when (`p`.`chg_pct` is not null) then (case
                when (abs(`p`.`chg_pct`) > 1) then (`p`.`chg_pct` / 100)
                else `p`.`chg_pct`
            end)
            else null
        end) is null) then null
        when ((case
            when ((coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`)) is not null)
                and (coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`)) <> 0)) then ((`p`.`close` / coalesce(`p`.`pre_close`, (`p`.`close` - `p`.`change`))) - 1)
            when (`p`.`chg_pct` is not null) then (case
                when (abs(`p`.`chg_pct`) > 1) then (`p`.`chg_pct` / 100)
                else `p`.`chg_pct`
            end)
            else null
        end) > 0) then 1
        else 0
    end)) as `up_ratio`
from
    (`cn_board_member_map_d` `m`
join `cn_stock_daily_price_active_v` `p` on
    ((`p`.`trade_date` = `m`.`trade_date`)
        and (`p`.`symbol` = (`m`.`symbol` collate utf8mb4_unicode_ci))))
where
    (`m`.`sector_type` = 'CONCEPT')
group by
    `m`.`sector_id`,
    `p`.`trade_date`;
