# Cross-Mainline Stability Review
Generated: 2026-05-22T21:02:35.305714
Period: 2025-01-01 -> 2026-05-01

## Summary
- Total themes analyzed: 11
- Themes with TRUE_COLLAPSE: 1
- Total research rows: 350946

## Theme Breakdown
          theme_bucket  total_rows  true_collapse_count  true_collapse_pct  diffusion_warning_count  low_diffusion_bg_count dominant_combo  avg_collapse_score
           AGRICULTURE       26622                    0               0.00                     1295                   19338              A                 0.0
     AI_INFRASTRUCTURE        7955                    0               0.00                      825                    7130              A                 0.0
        BROKER_FINANCE       22260                    0               0.00                     4963                   17297              A                 0.0
           CONSUMPTION       28836                    0               0.00                     3168                   19506              A                 0.0
INDUSTRY_MANUFACTURING        3967                    0               0.00                      599                    3368              A                 0.0
         MATERIAL_CHEM       31246                    0               0.00                     4494                   18884              A                 0.0
                 OTHER      164883                   65               0.04                    92135                   72683              A                 0.0
         PHARMA_HEALTH       38965                    0               0.00                    12604                   11715           NONE                 0.0
     POWER_ELECTRICITY        5336                    0               0.00                      869                    4467              A                 0.0
         SEMICONDUCTOR         354                    0               0.00                       98                     256              A                 0.0
       TRANSPORT_INFRA       20522                    0               0.00                     1966                   17330              A                 0.0

## Stability Observations
  OTHER: high density collapse (65 rows / 2 dates = 32.5/date)

## Assessment
Deterioration structure appears across:
  - OTHER: 65 TC / 164883 rows (0.04%)

FAIL CONDITIONS CHECK:
- Only works on AI cycle: PASS
- Highly regime-specific: FAIL - single theme
- Unstable cross-mainline behavior: PASS
