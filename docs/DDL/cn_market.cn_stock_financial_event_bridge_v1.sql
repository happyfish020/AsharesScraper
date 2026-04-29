-- cn_market.cn_stock_financial_event_bridge_v1
-- Latest strategy-facing financial bridge. Official income + fina_indicator first,
-- with disclosure / express / forecast attached on the same report period.

CREATE OR REPLACE
ALGORITHM = UNDEFINED VIEW `cn_stock_financial_event_bridge_v1` AS
WITH latest_income AS (
    SELECT
        i.*,
        ROW_NUMBER() OVER (
            PARTITION BY i.symbol
            ORDER BY i.end_date DESC, COALESCE(i.ann_date, i.f_ann_date, i.end_date) DESC
        ) AS rn
    FROM cn_stock_income i
),
latest_fina AS (
    SELECT
        f.*,
        ROW_NUMBER() OVER (
            PARTITION BY f.symbol, f.end_date
            ORDER BY COALESCE(f.ann_date, f.end_date) DESC, f.report_type DESC
        ) AS rn
    FROM cn_stock_fina_indicator f
),
latest_disclosure AS (
    SELECT
        d.*,
        ROW_NUMBER() OVER (
            PARTITION BY d.symbol, d.end_date
            ORDER BY COALESCE(d.actual_date, d.modify_date, d.pre_date, d.end_date) DESC
        ) AS rn
    FROM cn_event_disclosure_date d
),
latest_express AS (
    SELECT
        e.*,
        ROW_NUMBER() OVER (
            PARTITION BY e.symbol, e.end_date
            ORDER BY e.ann_date DESC
        ) AS rn
    FROM cn_event_earnings_express e
),
latest_forecast AS (
    SELECT
        f.*,
        ROW_NUMBER() OVER (
            PARTITION BY f.symbol, f.end_date
            ORDER BY f.ann_date DESC
        ) AS rn
    FROM cn_event_earnings_forecast f
)
SELECT
    i.symbol,
    i.end_date,
    i.ann_date,
    i.f_ann_date,
    COALESCE(i.ann_date, i.f_ann_date, d.actual_date, d.modify_date, d.pre_date) AS visible_ann_date,
    i.report_type AS income_report_type,
    i.comp_type,
    i.end_type,
    i.basic_eps,
    i.diluted_eps,
    i.total_revenue,
    i.revenue,
    i.operate_profit,
    i.total_profit,
    i.income_tax,
    i.n_income,
    i.n_income_attr_p,
    i.minority_gain,
    i.oth_compr_income,
    i.t_compr_income,
    i.compr_inc_attr_p,
    i.compr_inc_attr_m_s,
    i.ebit,
    i.ebitda,
    i.undist_profit,
    fi.report_type AS fina_report_type,
    fi.eps,
    fi.dt_eps,
    fi.roe,
    fi.roe_dt,
    fi.roa,
    fi.roic,
    fi.grossprofit_margin,
    fi.netprofit_margin,
    fi.profit_to_gr,
    fi.ocf_to_or,
    fi.debt_to_eqt,
    fi.debt_to_assets,
    fi.current_ratio,
    fi.quick_ratio,
    fi.or_yoy,
    fi.netprofit_yoy,
    fi.tr_yoy,
    fi.q_sales_yoy,
    fi.q_profit_yoy,
    fi.q_ocf_yoy,
    d.pre_date AS disclosure_pre_date,
    d.actual_date AS disclosure_actual_date,
    d.modify_date AS disclosure_modify_date,
    e.ann_date AS express_ann_date,
    e.revenue AS express_revenue,
    e.operate_profit AS express_operate_profit,
    e.total_profit AS express_total_profit,
    e.n_income AS express_n_income,
    e.yoy_sales AS express_yoy_sales,
    e.yoy_dedu_np AS express_yoy_dedu_np,
    fc.ann_date AS forecast_ann_date,
    fc.report_type AS forecast_report_type,
    fc.forecast_type,
    fc.p_change_min,
    fc.p_change_max,
    fc.net_profit_min,
    fc.net_profit_max,
    i.source AS income_source,
    fi.source AS fina_source,
    d.source AS disclosure_source,
    e.source AS express_source,
    fc.source AS forecast_source,
    i.raw_payload AS income_raw_payload,
    fi.raw_payload AS fina_raw_payload,
    d.raw_payload AS disclosure_raw_payload,
    e.raw_payload AS express_raw_payload,
    fc.raw_payload AS forecast_raw_payload
FROM latest_income i
LEFT JOIN latest_fina fi
  ON (fi.symbol COLLATE utf8mb4_general_ci) = (i.symbol COLLATE utf8mb4_general_ci)
 AND fi.end_date = i.end_date
 AND fi.rn = 1
LEFT JOIN latest_disclosure d
  ON (d.symbol COLLATE utf8mb4_general_ci) = (i.symbol COLLATE utf8mb4_general_ci)
 AND d.end_date = i.end_date
 AND d.rn = 1
LEFT JOIN latest_express e
  ON (e.symbol COLLATE utf8mb4_general_ci) = (i.symbol COLLATE utf8mb4_general_ci)
 AND e.end_date = i.end_date
 AND e.rn = 1
LEFT JOIN latest_forecast fc
  ON (fc.symbol COLLATE utf8mb4_general_ci) = (i.symbol COLLATE utf8mb4_general_ci)
 AND fc.end_date = i.end_date
 AND fc.rn = 1
WHERE i.rn = 1;
