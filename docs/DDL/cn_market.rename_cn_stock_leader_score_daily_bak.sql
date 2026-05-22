-- Backup the materialized table before replacing it with a view.
-- Run this ONCE before creating the replacement view.
-- If you need to restore: RENAME TABLE cn_stock_leader_score_daily_bak TO cn_stock_leader_score_daily;

RENAME TABLE cn_stock_leader_score_daily TO cn_stock_leader_score_daily_bak;
