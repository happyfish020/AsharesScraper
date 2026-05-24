# Signal Repeatability Review
Generated: 2026-05-22T18:14:45.044886
Period: 2025-10-01 -> 2026-01-31

## Persistence Windows
Total windows: 2
Windows with >1 day: 0
Max window duration: 1 days

## Candidate Pool Labels
persistence_label
BACKGROUND_ONLY_AFTER_POOL        309
PRECURSOR_THEN_SINGLE_COLLAPSE     27

## Assessment
FAIL CONDITIONS CHECK:
- Isolated event clusters only: CHECK
- No persistence structure: FAIL - no multi-day windows
- Unstable transitions: WARN - no repeat collapse
