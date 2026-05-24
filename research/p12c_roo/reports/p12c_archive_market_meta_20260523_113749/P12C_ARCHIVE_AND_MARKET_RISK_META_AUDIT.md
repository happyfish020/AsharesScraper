# P12C Archive & Market Risk Meta-Audit

**Generated**: 2026-05-23T11:37:49.759858+00:00
**Period**: 2025-01-01 → 2026-05-01
**Status**: STOP_P12C_LINE · HOLD_RESEARCH_ONLY

---

## 1. Executive Summary

P12c (Collapse Confirmation Matrix) has been evaluated at **both mainline-level and market-level** and failed validation at both levels.

| Level | Verdict | Reason |
|-------|---------|--------|
| Mainline-level | STOP_P12C_LINE | 2/5 phases passed; TRUE_COLLAPSE = 0.04% of rows |
| Market-level | HOLD_RESEARCH_ONLY | danger_ratio noisy (94.1% days > 50%); TC too sparse (2 dates) |
| **Final** | **ARCHIVE** | Not usable for trading, policy, or market timing |

### Boundary Locks (DO NOT MODIFY)

| Lock | Value | Rationale |
|------|-------|-----------|
| `STOP_P12C_LINE` | ✅ Locked | Core signal architecture cannot produce actionable signals |
| `HOLD_RESEARCH_ONLY` | ✅ Locked | Market-level aggregate too noisy for production |
| `DO_NOT_REPAIR_TRUE_MASK` | ✅ Locked | Relaxing AND→OR would create noise, not predictive validity |
| `DO_NOT_PROMOTE_TO_POLICY` | ✅ Locked | No policy-relevant signal quality |
| `DO_NOT_CONNECT_TO_TRADING` | ✅ Locked | Would degrade P11d/P15 performance |

---

## 2. Market Risk Meta-Audit (from P12C outputs)

### 2.1 Daily Aggregate Statistics

| Metric | Value |
|--------|-------|
| Total trading days | 320 |
| Avg tracked mainlines/day | 131 |
| Avg danger_ratio | 76.52% |
| Max danger_ratio | 100.00% |
| Min danger_ratio | 6.67% |
| Avg TC ratio | 0.0643% |
| Max TC ratio | 16.67% |
| Noisy days (danger_ratio > 50.0%) | 301 / 320 (94.1%) |
| Sparse days (TC ratio < 1.0%) | 318 / 320 (99.4%) |
| Days with TC events | 2 / 320 (0.6%) |

### 2.2 Regime Breakdown (heuristically derived)

| Regime | Days | % of Total |
|--------|:----:|:----------:|
 RANGE_CHOP_HIGH | 186 | 58.1% ||  RANGE_CHOP_MID | 93 | 29.1% ||  DEFENSIVE_SHIFT | 23 | 7.2% ||  RANGE_CHOP_LOW | 18 | 5.6% |

### 2.3 Noise Analysis

- **danger_ratio** is persistently elevated: 301/320 days (94.1%) exceed 50.0%
- This is driven by DIFFUSION_WARNING classification being too broad (59.9% of all rows)
- The signal-to-noise ratio is effectively **0.06%** (TC rows / total rows)
- Even on TC event days (2026-01-09, 2026-01-26), danger_ratio was already 82.67% and 45.10% respectively

### 2.4 TC Sparsity Analysis

- Only **2 unique dates** out of 320 trading days (0.6%) have TRUE_COLLAPSE signals
- TC signals are concentrated on 2026-01-09 (25 mainlines) and 2026-01-26 (2 mainlines)
- No TC signal persistence: signals do not form multi-day trends
- TC lead/lag analysis shows mixed results (1 LEAD, 1 LAG)

### 2.5 Reusability Assessment

| Question | Answer | Evidence |
|----------|--------|----------|
| Usable for trading? | **No** | TC too sparse (2 days in 16 months); danger_ratio too noisy |
| Usable for policy? | **No** | No regime-dependent signal differentiation |
| Usable for market timing? | **No** | TC lead/lag inconsistent; danger_ratio always elevated |
| Any part reusable? | **Only as historical diagnostic** | danger_ratio as background risk context only |
| Should true_mask be repaired? | **No** | Relaxing AND→OR would create noise, not validity |

---

## 3. Failure Root Cause Analysis

### 3.1 Why P12c Failed at Mainline Level

1. **TRUE_COLLAPSE too strict**: The `true_mask` requires all 4 group conditions to fire simultaneously (AND logic). Only 36 rows (0.04%) satisfy this.
2. **DIFFUSION_WARNING too broad**: 59.9% of all rows fall into DIFFUSION_WARNING, making it a default state rather than a meaningful signal.
3. **Zero persistence**: No multi-day TRUE_COLLAPSE windows exist. Signals are isolated events.
4. **Threshold invariance**: All 36 threshold combinations produced identical results, confirming the issue is architectural, not parametric.

### 3.2 Why P12c Failed at Market Level

1. **danger_ratio too noisy**: 94.1% of trading days have danger_ratio > 50%. The metric cannot differentiate risk regimes.
2. **TC too sparse**: Only 2 TC dates in 16 months. Cannot be used for timing decisions.
3. **Regime homogeneity**: 319/320 days in RANGE_CHOP regime. No regime diversity to validate.
4. **Market-level aggregation does not fix architectural flaws**: The underlying classification issues propagate to the aggregate.

### 3.3 Why NOT to Repair true_mask

- Relaxing AND→OR would increase TC count but would likely create **false positives**
- The 4-group AND condition was intentionally strict to avoid noise
- Without a fundamentally redesigned group condition set, any relaxation is arbitrary
- The current research line should stop; a new research line (P12d or alternative) should start from scratch if needed

---

## 4. Output Files

| File | Description |
|------|-------------|
| `daily_market_risk_meta.csv` | 320-day compact diagnostic table with noise/sparsity flags |
| `P12C_ARCHIVE_AND_MARKET_RISK_META_AUDIT.md` | This report |
| `P12C_EXPERIMENT_LEDGER.md` | Experiment ledger with final boundary locks |

### Source Data

| Source File | Rows | Used For |
|-------------|:----:|----------|
| `market_collapse_daily.csv` | 320 | Daily aggregates, noise/sparsity flags |
| `market_collapse_regime.csv` | 2 | Regime breakdown |
| `market_collapse_tc_lead_lag.csv` | 2 | TC lead/lag analysis |
| `market_collapse_review.md` | — | Assessment and recommendation |

---

## 5. Final Recommendation

**Stop P12c. Do not repair. Do not promote. Do not connect.**

Continue main work in **P11d → P15** for collapse detection and adaptive trading.

The P12c experiment has been valuable in demonstrating that:
- AND-based multi-group collapse confirmation is too strict for real market data
- DIFFUSION_WARNING needs fundamental redesign before it can be useful
- Market-level aggregation does not rescue flawed mainline-level classification
- Threshold scanning is ineffective when the core logic is the bottleneck

---

*Report generated by P12C_ARCHIVE_AND_MARKET_RISK_META_AUDIT*
*No production logic was modified in this audit.*
