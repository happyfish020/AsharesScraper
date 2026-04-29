-- cn_market.cn_fundamental_quality_param_t
-- Runtime thresholds for fundamental quality screens.

CREATE TABLE IF NOT EXISTS `cn_fundamental_quality_param_t` (
    `parameter_set` varchar(32) NOT NULL,
    `is_active` tinyint NOT NULL DEFAULT 1,
    `eps_min` decimal(18,6) NOT NULL DEFAULT 0,
    `revenue_growth_min` decimal(18,6) NOT NULL DEFAULT 5,
    `revenue_growth_strict_min` decimal(18,6) NOT NULL DEFAULT 10,
    `debt_to_eqt_max` decimal(18,6) NOT NULL DEFAULT 2,
    `grossprofit_margin_min` decimal(18,6) NOT NULL DEFAULT 0,
    `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`parameter_set`),
    KEY `idx_cn_fundamental_quality_param_active` (`is_active`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

INSERT INTO `cn_fundamental_quality_param_t` (
    `parameter_set`, `is_active`, `eps_min`, `revenue_growth_min`, `revenue_growth_strict_min`,
    `debt_to_eqt_max`, `grossprofit_margin_min`
)
SELECT
    'default', 1, 0, 5, 10, 2, 0
FROM dual
WHERE NOT EXISTS (
    SELECT 1
    FROM `cn_fundamental_quality_param_t`
    WHERE `parameter_set` = 'default'
);
