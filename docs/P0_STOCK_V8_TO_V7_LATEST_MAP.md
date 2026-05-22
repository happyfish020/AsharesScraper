# P0 Stock V8 to V7 Latest Map

This step materializes the latest stock-level mapping snapshot needed before sector-rotation replacement validation.

## Inputs

- `cn_v7_v8_industry_crosswalk_latest`
- `cn_local_industry_map_hist`

## Output

- `cn_stock_v8_to_v7_sw_map_latest`

## Run

```powershell
python scripts\build_stock_v8_to_v7_sw_map_latest.py --replace --output-dir reports\analysis\stock_v8_to_v7_sw_map_latest
```

This output is the direct handoff to trading-chain replacement validation because it answers:

- for each stock on latest date
- which V8 local industry it belongs to
- which V7 SW L1 code it should map to
