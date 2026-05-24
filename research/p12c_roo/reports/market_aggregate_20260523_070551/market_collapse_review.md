# Market-Level P12c Collapse Aggregate
Generated: 2026-05-23T07:29:21.555017
Period: 2025-01-01 -> 2026-05-01

## Summary
- Total trading days: 320
- Total mainlines tracked: 366
- Days with TRUE_COLLAPSE signals: 2
- Days with danger_ratio > 50%: 301 (94.1%)
- Days with TC ratio > 1%: 2

## Market-Level Danger Statistics
| Metric | Value |
|--------|-------|
| Avg daily danger_ratio | 76.5% |
| Max daily danger_ratio | 100.0% (on 2026-04-23) |
| Avg daily TC ratio | 0.06% |
| Max daily TC ratio | 16.67% |
| Avg collapse_confidence | 0.5293 |
| Avg confirmation groups | 2.04 |

## Top 10 Highest Collapse Risk Days
trade_date  market_collapse_risk  tc_ratio  dw_ratio  danger_ratio
2025-10-16                 76.69      0.00     99.34         99.34
2025-01-15                 75.63      0.00     98.59         98.59
2025-09-25                 75.61      0.00     97.97         97.97
2025-01-03                 75.31      0.00     98.04         98.04
2025-02-13                 74.87      0.00     98.59         98.59
2026-01-09                 74.58     16.67     66.00         82.67
2025-12-11                 74.35      0.00     96.00         96.00
2025-11-18                 74.27      0.00     97.44         97.44
2025-01-10                 73.95      0.00     96.00         96.00
2025-04-28                 73.92      0.00     96.88         96.88

## Regime Breakdown
  market_regime  total_days  total_rows  avg_tc_ratio  avg_dw_ratio  avg_confidence  avg_group_count
DEFENSIVE_SHIFT           1          67        0.0000      100.0000          0.5307           2.0448
     RANGE_CHOP         319       41812        0.0646       78.3651          0.5387           2.0783

## TRUE_COLLAPSE Signal Analysis
- Total TC events: 27
- Unique TC dates: 2
- TC dates list: 2026-01-09, 2026-01-26

### TC Lead/Lag Analysis
trade_date  tc_mainlines  tc_ratio  danger_ratio  pre_5d_avg_danger  post_5d_avg_danger signal_lead
2026-01-09            25     16.67         82.67              81.61               81.33        LEAD
2026-01-26             2      3.92         45.10              78.48               68.15         LAG

## Assessment as Market-Level Indicator

### Strengths
- Market-level danger_ratio provides a continuous risk measure (0-100%)
- TC signals are rare (0.04% of rows) → potentially high specificity
- Zero false positive noise → clean signal

### Weaknesses
- DIFFUSION_WARNING too broad (59.9% of rows) → danger_ratio always elevated
- TC signals only appear on 2 dates → too sparse for trading decisions
- No persistence → signals don't form trends

### Recommendation
P12c as a market-level indicator is **HOLD_RESEARCH_ONLY**:
- The danger_ratio metric is too noisy (always > 50% in current regime)
- TC signals are too rare to be actionable
- Consider using only after fixing the AND 4-group condition in true_mask
