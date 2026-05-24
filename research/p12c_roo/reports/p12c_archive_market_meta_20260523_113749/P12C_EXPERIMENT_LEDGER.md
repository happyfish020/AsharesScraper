# P12C Experiment Ledger

## Experiment: P12c Collapse Confirmation Matrix

| Field | Value |
|-------|-------|
| **Start date** | 2026-05-22 |
| **End date** | 2026-05-23 |
| **Status** | **ARCHIVED** |
| **Script** | `research/p12c_roo/run_p12c_research.py` |
| **Core module** | `p12c_collapse_confirmation_matrix.py` |
| **Context gate** | `p12_context_gate.py` |

### Timeline

| Step | Timestamp | Result |
|------|-----------|--------|
| Phase 1: Cross-Mainline | 2026-05-22 | ✅ 11/12 themes classified |
| Phase 2: Danger-Zone | 2026-05-22 | ❌ 50.1% danger ratio |
| Phase 3: Persistence | 2026-05-22 | ❌ 0 multi-day windows |
| Phase 4: False Positive | 2026-05-22 | ✅ 0% noise rate |
| Phase 5: Philosophy | 2026-05-22 | ✅ Consistent |
| Threshold Scan (36 combos) | 2026-05-22 | ✅ All identical — architecture issue confirmed |
| Market-Level Aggregate | 2026-05-23 | ❌ danger_ratio 94.1% noisy; TC 2 dates sparse |
| **Archive & Meta-Audit** | 2026-05-23 | ✅ **FINAL** |

### Boundary Locks

```
STOP_P12C_LINE            = LOCKED  (core architecture cannot produce actionable signals)
HOLD_RESEARCH_ONLY        = LOCKED  (market aggregate too noisy for production)
DO_NOT_REPAIR_TRUE_MASK   = LOCKED  (relaxing AND→OR creates noise, not validity)
DO_NOT_PROMOTE_TO_POLICY  = LOCKED  (no policy-relevant signal quality)
DO_NOT_CONNECT_TO_TRADING = LOCKED  (would degrade P11d/P15 performance)
```

### Key Metrics (Final)

| Metric | Value |
|--------|-------|
| Total rows analyzed | 80,000 |
| TRUE_COLLAPSE rows | 36 (0.04%) |
| DIFFUSION_WARNING rows | 47,949 (59.9%) |
| TC unique dates | 2 |
| TC persistence | 0 |
| Market danger_ratio > 50% | 94.1% of days |
| Threshold combinations tested | 36 |
| Threshold combinations with different results | 0 |

### Files Preserved

- `reports/full_pipeline_20260522_201509/` — Full pipeline output
- `reports/threshold_scan_20260522_220541/` — Threshold scan results
- `reports/market_aggregate_20260523_070551/` — Market aggregate output
- `reports/p12c_archive_market_meta_20260523_113749/` — Archive & meta-audit (this)

### Next Steps (for mainline P11d→P15)

- Do NOT reference P12c in trading decisions
- Do NOT allocate compute resources to P12c maintenance
- If collapse confirmation is needed, design a new approach from scratch (P12d or alternative)
