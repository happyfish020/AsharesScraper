# GrowthAlpha V8 — P2/P3 Final Acceptance Verdict

**Date:** 2026-05-09  
**Engineer:** Roo (Production QA)  
**Scope:** P2_MAINLINE_ENGINE_BUILD · P3_UNIFIED_ALPHA_ENGINE  

---

## 1. FINAL STATUS: ❌ FAIL

| Module | Status | Reason |
|--------|--------|--------|
| **P2 Mainline Lifecycle** | ⚠️ PARTIAL PASS | Table exists, logic ran, but all `lifecycle_state` = `UNKNOWN` — upstream mainline factor data empty/insufficient signals |
| **P3 Quality Score** | ❌ FAIL | `cn_stock_quality_score_daily` has **0 rows** — `cn_stock_fundamental_daily` has no valid data |
| **P3 Unified Alpha** | ⚠️ PARTIAL PASS | Table exists with data passing quality checks, but only **1 trade date** (2026-02-27), not reaching target 2026-03-30 |

**Overall: FAIL** — P3 Quality Score is empty, which is a blocking defect.

---

## 2. ACTUAL RESULT TABLES

| Table | Database | Exists | Rows | Date Range |
|-------|----------|--------|------|------------|
| `cn_mainline_lifecycle_daily` | `cn_market_red` | ✅ | 1,240 | 2026-01-26 ~ 2026-03-30 |
| `cn_stock_quality_score_daily` | `cn_market_red` | ✅ | **0** | N/A (empty) |
| `cn_unified_alpha_score_daily` | `cn_market_red` | ✅ | 5,175 | 2026-02-27 only |

---

## 3. TABLE STATISTICS

### P2: `cn_mainline_lifecycle_daily`

| Metric | Value |
|--------|-------|
| Row count | 1,240 |
| Unique trade dates | 40 (2026-01-26 ~ 2026-03-30) |
| Unique mainlines | 31 |
| `lifecycle_state` distribution | **ALL UNKNOWN** (1,240 / 1,240 = 100%) |
| `lifecycle_score` range | 0.50 ~ 0.50 (all 0.5, the UNKNOWN default) |
| `risk_flag` | All NULL |
| `rotation_rank` | All NULL |

**Verdict:** ⚠️ PARTIAL PASS — The build pipeline executed and wrote data correctly, but every mainline received `UNKNOWN` state because upstream mainline factor data (`cn_ga_mainline_radar_daily`, `cn_mainline_strength_daily`) contained no actionable signals for the date range. This is an **upstream data readiness issue**, not a P2 logic defect.

### P3: `cn_stock_quality_score_daily`

| Metric | Value |
|--------|-------|
| Row count | **0** |
| Columns defined | `trade_date`, `symbol`, `quality_score`, `quality_level`, `update_time` |

**Verdict:** ❌ FAIL — Table is empty. Root cause: `cn_stock_fundamental_daily` (the sole input source) has no valid data rows, so the quality score builder produces zero output.

### P3: `cn_unified_alpha_score_daily`

| Metric | Value |
|--------|-------|
| Row count | 5,175 |
| Unique trade dates | **1** (2026-02-27) |
| Unique symbols | 5,175 |
| Unique industries | 31 |
| `final_score` range | 0.4965 ~ 0.5190 |
| `final_score` mean | 0.5044 |
| `alpha_bucket` distribution | TOP_1: 156 (3.01%), TOP_5: 5,002 (96.66%), AVOID: 17 (0.33%) |
| `symbol_count` per date | 5,175 (meets >100 threshold) |

**Verdict:** ⚠️ PARTIAL PASS — Data quality checks pass (score range [0,1], buckets non-empty, symbol_count > 100), but only 1 date exists (2026-02-27), far short of the target 2026-03-30. The unified alpha depends on `cn_stock_quality_score_daily` which is empty, so the existing 5,175 rows likely came from a prior partial run.

---

## 4. UNIFIED ALPHA SUMMARY

| Factor | Weight |
|--------|--------|
| `quality_score` | 0.10 |
| `growth_acceleration_score` | 0.10 |
| `mainline_strength_score` | 0.20 |
| `capital_concentration_score` | 0.15 |
| `leader_dominance_score` | 0.15 |
| `trend_quality_score` | 0.15 |
| `lifecycle_position_score` | 0.10 |
| `risk_crowding_score` | 0.05 |

**Bucket distribution:** TOP_1=156(3.01%), TOP_5=5002(96.66%), AVOID=17(0.33%)

---

## 5. VALIDATION RESULTS

| Validation Script | Log File Found | Result |
|-------------------|----------------|--------|
| `validate_mainline_lifecycle_daily.py` | ❌ Not found | Cannot confirm automated validation ran |
| `validate_unified_alpha_score_daily.py` | ❌ Not found | Cannot confirm automated validation ran |

No validation log files exist in the project directory. Manual validation via database query was performed instead.

---

## 6. REPORT FILES

| Report | Path |
|--------|------|
| P2 Mainline Lifecycle Summary | [`reports/mainline_lifecycle/mainline_lifecycle_summary_20260101_20260330_20260509_113436.md`](../mainline_lifecycle/mainline_lifecycle_summary_20260101_20260330_20260509_113436.md) |
| P3 Unified Alpha Report | [`reports/unified_alpha/unified_alpha_2026-01-01_2026-03-30_20260509_123231.md`](../unified_alpha/unified_alpha_2026-01-01_2026-03-30_20260509_123231.md) |
| Data Asset Audit | [`reports/data_audit/data_asset_audit_20260508_194802.md`](../data_audit/data_asset_audit_20260508_194802.md) |
| This Verdict | [`reports/acceptance/P2_P3_ACCEPTANCE_VERDICT_20260509.md`](P2_P3_ACCEPTANCE_VERDICT_20260509.md) |

---

## 7. BLOCKING ISSUES (FAIL items only)

### 🔴 BLOCKER #1: `cn_stock_quality_score_daily` is empty (0 rows)

- **Root cause:** `cn_stock_fundamental_daily` has no valid data, so the quality score builder at [`scripts/build_stock_quality_score_daily.py`](../../scripts/build_stock_quality_score_daily.py) produces zero output rows.
- **Impact:** P3 Quality Score is completely non-functional. P3 Unified Alpha also cannot produce valid multi-date results without quality scores.
- **Fix required before re-acceptance:** Build `cn_stock_fundamental_daily` first, then rebuild `cn_stock_quality_score_daily`.

---

## 8. RECOMMENDED FIX ORDER

Per root cause analysis, the following execution order is required:

```
Step 1: python scripts/build_stock_fundamental_daily.py    # Build upstream fundamental data
Step 2: python scripts/build_stock_quality_score_daily.py   # Rebuild P3 Quality Score
Step 3: python scripts/build_unified_alpha_score_daily.py   # Rebuild P3 Unified Alpha
Step 4: python scripts/build_mainline_lifecycle_daily.py    # Rebuild P2 Mainline Lifecycle
```

After all four steps complete successfully, re-run acceptance checks to verify:
- `cn_stock_quality_score_daily` has rows with `quality_score` in [0,1]
- `cn_unified_alpha_score_daily` covers dates up to 2026-03-30
- `cn_mainline_lifecycle_daily` has non-UNKNOWN lifecycle states
