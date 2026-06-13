"""Quarantine cn_mainline_lifecycle_daily rows generated from failed Fact V1.

This does not restore old values. It adds a visible warning by changing risk_flag
for rows whose phase_reason starts with the P0H fact-layer marker. Use before any
GrowthAlpha cutover if the failed P0G lifecycle write was executed.
"""
from __future__ import annotations
import argparse, os
from sqlalchemy import URL, create_engine, text


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--start', default='2024-01-01'); p.add_argument('--end', default='2026-06-12')
    p.add_argument('--db-name', default='cn_market_red'); p.add_argument('--db-host', default='127.0.0.1')
    p.add_argument('--db-port', type=int, default=3306); p.add_argument('--db-user', default='cn_opr_red'); p.add_argument('--db-password', default=None)
    p.add_argument('--apply', action='store_true')
    return p


def main():
    a=parser().parse_args(); pw=a.db_password or os.getenv('ASHARE_MYSQL_PASSWORD','sec_Bobo123')
    e=create_engine(URL.create('mysql+pymysql',username=a.db_user,password=pw,host=a.db_host,port=a.db_port,database=a.db_name,query={'charset':'utf8mb4'}),future=True)
    count_sql="""
      SELECT COUNT(*) FROM cn_mainline_lifecycle_daily
      WHERE trade_date BETWEEN :start AND :end
        AND (phase_reason LIKE 'fact %' OR phase_reason LIKE 'quality=%' OR phase_reason LIKE 'very weak fact-layer%')
    """
    with e.begin() as c:
        n=c.execute(text(count_sql), {'start':a.start,'end':a.end}).scalar() or 0
        print(f'[QUARANTINE CHECK] fact_lifecycle_rows={n}')
        if not a.apply:
            print('[QUARANTINE DRY-RUN] add --apply to mark rows')
            return
        c.execute(text("""
          UPDATE cn_mainline_lifecycle_daily
          SET risk_flag='INVALID_P0G_FACT_V1_DO_NOT_USE',
              phase_reason=CONCAT('[INVALID_P0G_FACT_V1] ', COALESCE(phase_reason,''))
          WHERE trade_date BETWEEN :start AND :end
            AND (phase_reason LIKE 'fact %' OR phase_reason LIKE 'quality=%' OR phase_reason LIKE 'very weak fact-layer%')
        """), {'start':a.start,'end':a.end})
        print(f'[QUARANTINE APPLIED] rows={n}')
if __name__=='__main__': main()
