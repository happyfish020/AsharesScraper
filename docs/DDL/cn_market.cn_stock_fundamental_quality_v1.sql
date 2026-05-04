-- cn_market.cn_stock_fundamental_quality_v1
-- Latest fundamental quality screen using monthly basic + quarterly financial indicators.

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `cn_stock_fundamental_quality_v1` AS
WITH params AS (
    SELECT
        p.parameter_set,
        p.eps_min,
        p.revenue_growth_min,
        p.revenue_growth_strict_min,
        p.debt_to_eqt_max,
        p.grossprofit_margin_min
    FROM cn_fundamental_quality_param_t p
    WHERE p.is_active = 1
    ORDER BY p.updated_at DESC, p.parameter_set
    LIMIT 1
),
latest_fina AS (
    SELECT
        f.*,
        ROW_NUMBER() OVER (
            PARTITION BY f.symbol
            ORDER BY f.end_date DESC, COALESCE(f.ann_date, f.end_date) DESC
        ) AS rn
    FROM cn_stock_fina_indicator f
),
latest_cashflow AS (
    SELECT
        cf.symbol,
        cf.end_date,
        cf.n_cashflow_act,
        ROW_NUMBER() OVER (
            PARTITION BY cf.symbol
            ORDER BY cf.end_date DESC, COALESCE(cf.ann_date, cf.end_date) DESC
        ) AS rn
    FROM cn_stock_cashflow cf
),
latest_income AS (
    SELECT
        ic.symbol,
        ic.end_date,
        ic.n_income_attr_p,
        ROW_NUMBER() OVER (
            PARTITION BY ic.symbol
            ORDER BY ic.end_date DESC, COALESCE(ic.ann_date, ic.end_date) DESC
        ) AS rn
    FROM cn_stock_income ic
),
latest_basic AS (
    SELECT
        b.*,
        ROW_NUMBER() OVER (
            PARTITION BY b.symbol
            ORDER BY b.trade_date DESC
        ) AS rn
    FROM cn_stock_monthly_basic b
)
SELECT
    prm.parameter_set,
    f.symbol,
    f.end_date AS fina_end_date,
    f.ann_date,
    b.trade_date AS basic_trade_date,
    b.month_key,
    b.total_mv,
    b.circ_mv,
    b.pe,
    b.pe_ttm,
    b.pb,
    b.ps,
    b.ps_ttm,
    f.eps,
    COALESCE(f.or_yoy, f.tr_yoy, f.q_sales_yoy) AS revenue_growth_pct,
    f.or_yoy,
    f.tr_yoy,
    f.q_sales_yoy,
    f.debt_to_eqt,
    f.grossprofit_margin,
    f.netprofit_margin,
    f.roe,
    f.netprofit_yoy,
    CASE
        WHEN ic.n_income_attr_p IS NOT NULL AND ic.n_income_attr_p <> 0
        THEN cf.n_cashflow_act / ic.n_income_attr_p
        ELSE NULL
    END AS ocf_to_np,
    prm.eps_min,
    prm.revenue_growth_min,
    prm.revenue_growth_strict_min,
    prm.debt_to_eqt_max,
    prm.grossprofit_margin_min,
    CASE WHEN f.eps > prm.eps_min THEN 1 ELSE 0 END AS pass_eps_positive,
    CASE WHEN COALESCE(f.or_yoy, f.tr_yoy, f.q_sales_yoy) >= prm.revenue_growth_min THEN 1 ELSE 0 END AS pass_revenue_growth_5,
    CASE WHEN COALESCE(f.or_yoy, f.tr_yoy, f.q_sales_yoy) >= prm.revenue_growth_strict_min THEN 1 ELSE 0 END AS pass_revenue_growth_10,
    CASE WHEN f.debt_to_eqt IS NOT NULL AND f.debt_to_eqt < prm.debt_to_eqt_max THEN 1 ELSE 0 END AS pass_debt_to_eqt_lt_2,
    CASE WHEN f.grossprofit_margin IS NOT NULL AND f.grossprofit_margin > prm.grossprofit_margin_min THEN 1 ELSE 0 END AS pass_gross_margin_positive,
    (
        CASE WHEN f.eps > prm.eps_min THEN 1 ELSE 0 END
        + CASE WHEN COALESCE(f.or_yoy, f.tr_yoy, f.q_sales_yoy) >= prm.revenue_growth_min THEN 1 ELSE 0 END
        + CASE WHEN f.debt_to_eqt IS NOT NULL AND f.debt_to_eqt < prm.debt_to_eqt_max THEN 1 ELSE 0 END
    ) AS quality_core_score,
    (
        CASE WHEN f.eps > prm.eps_min THEN 1 ELSE 0 END
        + CASE WHEN COALESCE(f.or_yoy, f.tr_yoy, f.q_sales_yoy) >= prm.revenue_growth_min THEN 1 ELSE 0 END
        + CASE WHEN f.debt_to_eqt IS NOT NULL AND f.debt_to_eqt < prm.debt_to_eqt_max THEN 1 ELSE 0 END
        + CASE WHEN f.grossprofit_margin IS NOT NULL AND f.grossprofit_margin > prm.grossprofit_margin_min THEN 1 ELSE 0 END
    ) AS quality_total_score,
    CASE
        WHEN f.eps > prm.eps_min
         AND COALESCE(f.or_yoy, f.tr_yoy, f.q_sales_yoy) >= prm.revenue_growth_min
         AND f.debt_to_eqt IS NOT NULL
         AND f.debt_to_eqt < prm.debt_to_eqt_max
        THEN 1 ELSE 0
    END AS quality_pass_core,
    CASE
        WHEN f.eps > prm.eps_min
         AND COALESCE(f.or_yoy, f.tr_yoy, f.q_sales_yoy) >= prm.revenue_growth_min
         AND f.debt_to_eqt IS NOT NULL
         AND f.debt_to_eqt < prm.debt_to_eqt_max
         AND f.grossprofit_margin IS NOT NULL
         AND f.grossprofit_margin > prm.grossprofit_margin_min
        THEN 1 ELSE 0
    END AS quality_pass_with_margin,
    b.source AS basic_source,
    f.source AS fina_source,
    b.raw_payload AS basic_raw_payload,
    f.raw_payload AS fina_raw_payload
FROM latest_fina f
JOIN params prm
LEFT JOIN latest_basic b
  ON b.symbol = f.symbol
 AND b.rn = 1
LEFT JOIN latest_cashflow cf
  ON cf.symbol = f.symbol
 AND cf.end_date = f.end_date
 AND cf.rn = 1
LEFT JOIN latest_income ic
  ON ic.symbol = f.symbol
 AND ic.end_date = f.end_date
 AND ic.rn = 1
WHERE f.rn = 1;
