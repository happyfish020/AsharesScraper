#!/usr/bin/env python3
"""Check definer on stored procedures."""
import pymysql

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'bobosky123',
    'database': 'cn_market_red',
    'charset': 'utf8mb4',
    'autocommit': True,
}

conn = pymysql.connect(**DB_CONFIG)
cursor = conn.cursor()

# Check definer and security type for all relevant SPs
print("=== Stored Procedure Definer & Security Type ===")
cursor.execute("""
    SELECT ROUTINE_NAME, DEFINER, SECURITY_TYPE, SQL_MODE
    FROM information_schema.ROUTINES
    WHERE ROUTINE_SCHEMA = 'cn_market_red'
      AND ROUTINE_NAME IN (
        'sp_build_sector_rotation_transition_batch',
        'sp_refresh_sector_eod_hist_optimized',
        'sp_build_sector_rotation_ranked_batch',
        'sp_build_sector_rotation_signal_batch',
        'sp_rebuild_rotation_year_optimized',
        'SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE',
        'SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE',
        'sp_refresh_sector_eod_hist',
        'sp_rebuild_rotation_year'
      )
    ORDER BY ROUTINE_NAME
""")
for row in cursor.fetchall():
    print(f"  {row[0]:55s} DEFINER={row[1]:25s} SECURITY={row[2]:10s}")

# Also check the original SPs that already existed
print("\n=== Original SPs definer ===")
cursor.execute("""
    SELECT ROUTINE_NAME, DEFINER, SECURITY_TYPE
    FROM information_schema.ROUTINES
    WHERE ROUTINE_SCHEMA = 'cn_market_red'
      AND ROUTINE_NAME IN (
        'sp_refresh_sector_eod_hist',
        'SP_BUILD_SECTOR_ROTATION_RANKED_BY_DATE',
        'SP_BUILD_SECTOR_ROTATION_SIGNAL_BY_DATE'
      )
    ORDER BY ROUTINE_NAME
""")
for row in cursor.fetchall():
    print(f"  {row[0]:55s} DEFINER={row[1]:25s} SECURITY={row[2]:10s}")

# Check current user and connection info
print("\n=== Current session info ===")
cursor.execute("SELECT CURRENT_USER(), USER()")
r = cursor.fetchone()
print(f"  CURRENT_USER(): {r[0]}")
print(f"  USER(): {r[1]}")

cursor.execute("SELECT CONNECTION_ID()")
print(f"  CONNECTION_ID(): {cursor.fetchone()[0]}")

cursor.close()
conn.close()
