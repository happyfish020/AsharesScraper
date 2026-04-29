-- cn_market.cn_stock_fundamental_quality_hist_v1
-- Historical as-of monthly quality screen. Uses only financial reports known by each monthly trade_date.

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `cn_stock_fundamental_quality_hist_v1` AS
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
basic_base AS (
    SELECT
        b.symbol,
        b.trade_date,
        b.month_key,
        b.total_mv,
        b.circ_mv,
        b.pe,
        b.pe_ttm,
        b.pb,
        b.ps,
        b.ps_ttm,
        b.source AS basic_source,
        b.raw_payload AS basic_raw_payload
    FROM cn_stock_monthly_basic b
),
joined AS (
    SELECT
        b.*,
        f.end_date AS fina_end_date,
        f.ann_date,
        f.eps,
        f.or_yoy,
        f.tr_yoy,
        f.q_sales_yoy,
        f.debt_to_eqt,
        f.grossprofit_margin,
        f.netprofit_margin,
        f.source AS fina_source,
        f.raw_payload AS fina_raw_payload,
        ROW_NUMBER() OVER (
            PARTITION BY b.symbol, b.trade_date
            ORDER BY
                CASE
                    WHEN COALESCE(f.ann_date, f.end_date) <= b.trade_date THEN 0
                    ELSE 1
                END,
                COALESCE(f.ann_date, f.end_date) DESC,
                f.end_date DESC
        ) AS rn
    FROM basic_base b
    LEFT JOIN cn_stock_fina_indicator f
      ON f.symbol = b.symbol
)
SELECT
    prm.parameter_set,
    j.symbol,
    j.trade_date AS basic_trade_date,
    j.month_key,
    j.fina_end_date,
    j.ann_date,
    j.total_mv,
    j.circ_mv,
    j.pe,
    j.pe_ttm,
    j.pb,
    j.ps,
    j.ps_ttm,
    j.eps,
    COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) AS revenue_growth_pct,
    j.or_yoy,
    j.tr_yoy,
    j.q_sales_yoy,
    j.debt_to_eqt,
    j.grossprofit_margin,
    j.netprofit_margin,
    prm.eps_min,
    prm.revenue_growth_min,
    prm.revenue_growth_strict_min,
    prm.debt_to_eqt_max,
    prm.grossprofit_margin_min,
    CASE WHEN j.eps > prm.eps_min THEN 1 ELSE 0 END AS pass_eps_positive,
    CASE WHEN COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) >= prm.revenue_growth_min THEN 1 ELSE 0 END AS pass_revenue_growth_5,
    CASE WHEN COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) >= prm.revenue_growth_strict_min THEN 1 ELSE 0 END AS pass_revenue_growth_10,
    CASE WHEN j.debt_to_eqt IS NOT NULL AND j.debt_to_eqt < prm.debt_to_eqt_max THEN 1 ELSE 0 END AS pass_debt_to_eqt_lt_2,
    CASE WHEN j.grossprofit_margin IS NOT NULL AND j.grossprofit_margin > prm.grossprofit_margin_min THEN 1 ELSE 0 END AS pass_gross_margin_positive,
    (
        CASE WHEN j.eps > prm.eps_min THEN 1 ELSE 0 END
        + CASE WHEN COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) >= prm.revenue_growth_min THEN 1 ELSE 0 END
        + CASE WHEN j.debt_to_eqt IS NOT NULL AND j.debt_to_eqt < prm.debt_to_eqt_max THEN 1 ELSE 0 END
    ) AS quality_core_score,
    (
        CASE WHEN j.eps > prm.eps_min THEN 1 ELSE 0 END
        + CASE WHEN COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) >= prm.revenue_growth_min THEN 1 ELSE 0 END
        + CASE WHEN j.debt_to_eqt IS NOT NULL AND j.debt_to_eqt < prm.debt_to_eqt_max THEN 1 ELSE 0 END
        + CASE WHEN j.grossprofit_margin IS NOT NULL AND j.grossprofit_margin > prm.grossprofit_margin_min THEN 1 ELSE 0 END
    ) AS quality_total_score,
    CASE
        WHEN j.eps > prm.eps_min
         AND COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) >= prm.revenue_growth_min
         AND j.debt_to_eqt IS NOT NULL
         AND j.debt_to_eqt < prm.debt_to_eqt_max
        THEN 1 ELSE 0
    END AS quality_pass_core,
    CASE
        WHEN j.eps > prm.eps_min
         AND COALESCE(j.or_yoy, j.tr_yoy, j.q_sales_yoy) >= prm.revenue_growth_min
         AND j.debt_to_eqt IS NOT NULL
         AND j.debt_to_eqt < prm.debt_to_eqt_max
         AND j.grossprofit_margin IS NOT NULL
         AND j.grossprofit_margin > prm.grossprofit_margin_min
        THEN 1 ELSE 0
    END AS quality_pass_with_margin,
    j.basic_source,
    j.fina_source,
    j.basic_raw_payload,
    j.fina_raw_payload
FROM joined j
JOIN params prm
WHERE j.rn = 1
  AND (j.ann_date IS NULL OR j.ann_date <= j.trade_date OR j.fina_end_date <= j.trade_date);
