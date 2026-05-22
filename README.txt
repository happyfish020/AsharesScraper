event_backfill.bat v2

This version fixes the historical range issue.

Why v1 only fetched recent dates:
  EventLoaderTask has incremental logic:
    if EVENT_FORCE_FULL is not set, it starts from MAX(existing_date) - buffer.
  Therefore a command like --start-date 2006-01-01 still fetched only 2026 recent dates.

Fix:
  This BAT sets:
    EVENT_FORCE_FULL=1
    EVENT_FULL_START=<your --start-date>
    EVENT_LOOKBACK_BUFFER_DAYS=0

Usage:
  event_backfill.bat --start-date 2006-01-01 --end-date 2026-05-15 --replace

Optional:
  event_backfill.bat --start-date 2006-01-01 --end-date 2026-05-15 --with-anns

Notes:
  runner.py does not support global --replace.
  --replace is accepted by this BAT for consistent UX but not passed to runner.py.
