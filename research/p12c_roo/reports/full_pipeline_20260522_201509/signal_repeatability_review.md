# Signal Repeatability Review
Generated: 2026-05-22T21:31:32.906304
Period: 2025-01-01 -> 2026-05-01

## Persistence Windows
Total windows: 2
Windows with >1 day: 0
Max window duration: 1 days

## Candidate Pool Labels
persistence_label
BACKGROUND_ONLY_AFTER_POOL        339
PRECURSOR_THEN_SINGLE_COLLAPSE     27

## Assessment
FAIL CONDITIONS CHECK:
- Isolated event clusters only: CHECK
- No persistence structure: FAIL - no multi-day windows
- Unstable transitions: WARN - no repeat collapse
