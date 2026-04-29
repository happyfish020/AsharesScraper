-- cn_market.cn_stock_working_capital_alert_v1
-- Latest working-capital anomaly screen using balance sheet + income + fina indicators.

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `cn_stock_working_capital_alert_v1` AS
WITH balance_ranked AS (
    SELECT
        b.*,
        ROW_NUMBER() OVER (
            PARTITION BY b.symbol, b.end_date
            ORDER BY COALESCE(b.ann_date, b.f_ann_date, b.end_date) DESC, b.report_type DESC
        ) AS rn
    FROM cn_stock_balancesheet b
),
fina_ranked AS (
    SELECT
        f.*,
        ROW_NUMBER() OVER (
            PARTITION BY f.symbol, f.end_date
            ORDER BY COALESCE(f.ann_date, f.end_date) DESC, f.report_type DESC
        ) AS rn
    FROM cn_stock_fina_indicator f
),
income_ranked AS (
    SELECT
        i.*,
        ROW_NUMBER() OVER (
            PARTITION BY i.symbol, i.end_date
            ORDER BY COALESCE(i.ann_date, i.f_ann_date, i.end_date) DESC, i.report_type DESC
        ) AS rn
    FROM cn_stock_income i
),
joined AS (
    SELECT
        b.symbol,
        b.end_date,
        b.ann_date,
        b.f_ann_date,
        b.report_type,
        b.comp_type,
        b.end_type,
        b.accounts_receiv,
        b.inventories,
        b.total_cur_assets,
        b.total_assets,
        i.revenue,
        i.total_revenue,
        COALESCE(f.or_yoy, f.tr_yoy, f.q_sales_yoy) AS revenue_yoy_pct,
        py.end_date AS prev_year_end_date,
        py.accounts_receiv AS prev_year_accounts_receiv,
        py.inventories AS prev_year_inventories,
        b.source AS balance_source,
        i.source AS income_source,
        f.source AS fina_source
    FROM balance_ranked b
    LEFT JOIN income_ranked i
      ON i.symbol = b.symbol
     AND i.end_date = b.end_date
     AND i.rn = 1
    LEFT JOIN fina_ranked f
      ON f.symbol = b.symbol
     AND f.end_date = b.end_date
     AND f.rn = 1
    LEFT JOIN balance_ranked py
      ON py.symbol = b.symbol
     AND py.end_date = DATE_SUB(b.end_date, INTERVAL 1 YEAR)
     AND py.rn = 1
    WHERE b.rn = 1
),
scored AS (
    SELECT
        j.*,
        CASE
            WHEN j.prev_year_accounts_receiv IS NULL OR j.prev_year_accounts_receiv = 0 OR j.accounts_receiv IS NULL THEN NULL
            ELSE ROUND((j.accounts_receiv - j.prev_year_accounts_receiv) / ABS(j.prev_year_accounts_receiv) * 100, 4)
        END AS accounts_receiv_yoy_pct,
        CASE
            WHEN j.prev_year_inventories IS NULL OR j.prev_year_inventories = 0 OR j.inventories IS NULL THEN NULL
            ELSE ROUND((j.inventories - j.prev_year_inventories) / ABS(j.prev_year_inventories) * 100, 4)
        END AS inventories_yoy_pct
    FROM joined j
),
latest AS (
    SELECT
        s.*,
        CASE
            WHEN s.accounts_receiv_yoy_pct IS NOT NULL AND s.revenue_yoy_pct IS NOT NULL AND s.accounts_receiv_yoy_pct > s.revenue_yoy_pct THEN 1
            ELSE 0
        END AS flag_receiv_growth_gt_revenue,
        CASE
            WHEN s.inventories_yoy_pct IS NOT NULL AND s.revenue_yoy_pct IS NOT NULL AND s.inventories_yoy_pct > s.revenue_yoy_pct THEN 1
            ELSE 0
        END AS flag_inventory_growth_gt_revenue,
        ROW_NUMBER() OVER (
            PARTITION BY s.symbol
            ORDER BY s.end_date DESC, COALESCE(s.ann_date, s.f_ann_date, s.end_date) DESC
        ) AS latest_rn
    FROM scored s
)
SELECT
    l.symbol,
    l.end_date,
    l.ann_date,
    l.f_ann_date,
    l.report_type,
    l.comp_type,
    l.end_type,
    l.accounts_receiv,
    l.prev_year_accounts_receiv,
    l.accounts_receiv_yoy_pct,
    l.inventories,
    l.prev_year_inventories,
    l.inventories_yoy_pct,
    l.revenue,
    l.total_revenue,
    l.revenue_yoy_pct,
    CASE
        WHEN l.accounts_receiv_yoy_pct IS NULL OR l.revenue_yoy_pct IS NULL THEN NULL
        ELSE ROUND(l.accounts_receiv_yoy_pct - l.revenue_yoy_pct, 4)
    END AS receiv_growth_gap_pct,
    CASE
        WHEN l.inventories_yoy_pct IS NULL OR l.revenue_yoy_pct IS NULL THEN NULL
        ELSE ROUND(l.inventories_yoy_pct - l.revenue_yoy_pct, 4)
    END AS inventory_growth_gap_pct,
    l.flag_receiv_growth_gt_revenue,
    l.flag_inventory_growth_gt_revenue,
    (
        l.flag_receiv_growth_gt_revenue
        + l.flag_inventory_growth_gt_revenue
    ) AS working_capital_alert_score,
    CASE
        WHEN (
            l.flag_receiv_growth_gt_revenue
            + l.flag_inventory_growth_gt_revenue
        ) = 2 THEN 'high'
        WHEN (
            l.flag_receiv_growth_gt_revenue
            + l.flag_inventory_growth_gt_revenue
        ) = 1 THEN 'watch'
        ELSE 'normal'
    END AS working_capital_alert_level,
    l.balance_source,
    l.income_source,
    l.fina_source
FROM latest l
WHERE l.latest_rn = 1;
