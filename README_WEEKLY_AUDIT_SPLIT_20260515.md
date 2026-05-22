# Weekly audit split fix

## What changed

`v8_weekly_audit` is now a weekly-reference audit only. It checks weekly/reference outputs and no longer fails because of stock/index market coverage gaps.

Strict market coverage checks are split into standalone tasks:

- `v8_weekly_audit_stock`
- `v8_weekly_audit_index`
- `v8_weekly_audit_market`

These can be run independently without re-running `v8_weekly_refresh`.

## Commands

Run weekly refresh/audit/finalize:

```bat
weekly_split.bat --start-date 2010-01-01 --end-date 2010-12-31
```

Rerun weekly reference audit only:

```bat
python runner.py --flag tu --tasks v8_weekly_audit --start-date 2010-01-01 --end-date 2010-12-31
```

Run strict market coverage audit only:

```bat
python runner.py --flag tu --tasks v8_weekly_audit_market --start-date 2010-01-01 --end-date 2010-12-31
```

Run index-only audit/repair:

```bat
python runner.py --flag tu --tasks v8_weekly_audit_index --start-date 2010-01-01 --end-date 2010-12-31
```

## Why

Weekly pipeline refreshes board/stock-basic/periodic-event data. It should not be blocked by stock/index market coverage gaps, because those belong to daily market raw/history backfill.
