"""Check the full schema of cn_local_industry_proxy_daily to find NOT NULL columns without defaults."""
import sys
import os
import traceback

try:
    from app.settings import build_engine
    from sqlalchemy import text

    engine = build_engine()
    lines = []
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COLUMN_KEY, EXTRA
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'cn_local_industry_proxy_daily'
            ORDER BY ORDINAL_POSITION
        """)).fetchall()

        lines.append(f"{'COLUMN_NAME':30s} {'COLUMN_TYPE':30s} {'NULLABLE':10s} {'DEFAULT':20s} {'KEY':8s} {'EXTRA'}")
        lines.append("="*120)
        for r in rows:
            col_name = r[0]
            col_type = r[1]
            nullable = r[2]
            default = str(r[3]) if r[3] is not None else "NULL"
            col_key = r[4] if r[4] else ""
            extra = r[5] if r[5] else ""
            lines.append(f"{col_name:30s} {col_type:30s} {nullable:10s} {default:20s} {col_key:8s} {extra}")

        lines.append("")
        lines.append("=== NOT NULL columns WITHOUT a default value ===")
        for r in rows:
            col_name = r[0]
            col_type = r[1]
            nullable = r[2]
            default = r[3]
            if nullable == 'NO' and default is None and 'auto_increment' not in (r[5] or ''):
                lines.append(f"  {col_name:30s} {col_type:30s}")

    with open("_schema_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("SUCCESS: Output written to _schema_output.txt")
except Exception as e:
    with open("_schema_output.txt", "w", encoding="utf-8") as f:
        f.write(f"ERROR: {e}\n")
        f.write(traceback.format_exc())
    print(f"ERROR: {e}")
