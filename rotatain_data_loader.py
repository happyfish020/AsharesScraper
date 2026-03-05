from app.settings import build_engine
from sqlalchemy import text
import time

RUN_ID = "SR_BASE_V535_EP90_XP55_XC2_MH5_RF5_K2_COST5BPS"
SAMPLE_LIMIT = 20
BATCH_DAYS = 60
ENABLE_HEAVY_STATS = False

engine = build_engine()


def _date_list(rows):
    return [str(r[0]) for r in rows]


def collect_stats(conn):
    bt_dates_total = conn.execute(text("""
        select count(*)
        from (select distinct trade_date d from cn_sector_rot_bt_daily_t where run_id=:r) t
    """), {"r": RUN_ID}).scalar() or 0

    universe_date = conn.execute(text("""
        select signal_date
        from cn_sector_rotation_signal_t
        group by signal_date
        order by count(*) desc, signal_date desc
        limit 1
    """)).scalar()

    universe_count = 0
    if universe_date is not None:
        universe_count = conn.execute(text("""
            select count(*)
            from cn_sector_rotation_signal_t
            where signal_date = :d
        """), {"d": universe_date}).scalar() or 0

    signal_stats = conn.execute(text("""
        select ifnull(min(c), 0), ifnull(avg(c), 0), ifnull(max(c), 0)
        from (
            select signal_date, count(*) c
            from cn_sector_rotation_signal_t
            group by signal_date
        ) t
    """)).one()

    ranked_stats = conn.execute(text("""
        select ifnull(min(c), 0), ifnull(avg(c), 0), ifnull(max(c), 0)
        from (
            select trade_date, count(*) c
            from cn_sector_rotation_ranked_t
            group by trade_date
        ) t
    """)).one()

    signal_missing_days = 0
    ranked_missing_days = 0
    signal_full_days = 0
    ranked_full_days = 0
    signal_missing_sample = []
    ranked_missing_sample = []

    if universe_count > 0:
        signal_missing_days = conn.execute(text("""
            select count(*)
            from (
                select distinct trade_date d
                from cn_sector_rot_bt_daily_t
                where run_id=:r
            ) b
            left join (
                select signal_date, count(*) c
                from cn_sector_rotation_signal_t
                group by signal_date
            ) s on s.signal_date=b.d
            where ifnull(s.c, 0) < :uc
        """), {"r": RUN_ID, "uc": universe_count}).scalar() or 0

        ranked_missing_days = conn.execute(text("""
            select count(*)
            from (
                select distinct trade_date d
                from cn_sector_rot_bt_daily_t
                where run_id=:r
            ) b
            left join (
                select trade_date, count(*) c
                from cn_sector_rotation_ranked_t
                group by trade_date
            ) s on s.trade_date=b.d
            where ifnull(s.c, 0) < :uc
        """), {"r": RUN_ID, "uc": universe_count}).scalar() or 0

        signal_full_days = conn.execute(text("""
            select count(*)
            from (
                select b.d
                from (
                    select distinct trade_date d
                    from cn_sector_rot_bt_daily_t
                    where run_id=:r
                ) b
                left join (
                    select signal_date, count(*) c
                    from cn_sector_rotation_signal_t
                    group by signal_date
                ) s on s.signal_date=b.d
                where ifnull(s.c, 0) = :uc
            ) t
        """), {"r": RUN_ID, "uc": universe_count}).scalar() or 0

        ranked_full_days = conn.execute(text("""
            select count(*)
            from (
                select b.d
                from (
                    select distinct trade_date d
                    from cn_sector_rot_bt_daily_t
                    where run_id=:r
                ) b
                left join (
                    select trade_date, count(*) c
                    from cn_sector_rotation_ranked_t
                    group by trade_date
                ) s on s.trade_date=b.d
                where ifnull(s.c, 0) = :uc
            ) t
        """), {"r": RUN_ID, "uc": universe_count}).scalar() or 0

        signal_missing_sample = _date_list(conn.execute(text("""
            select b.d
            from (
                select distinct trade_date d
                from cn_sector_rot_bt_daily_t
                where run_id=:r
            ) b
            left join (
                select signal_date, count(*) c
                from cn_sector_rotation_signal_t
                group by signal_date
            ) s on s.signal_date=b.d
            where ifnull(s.c, 0) < :uc
            order by b.d
            limit :lim
        """), {"r": RUN_ID, "uc": universe_count, "lim": SAMPLE_LIMIT}).fetchall())

        ranked_missing_sample = _date_list(conn.execute(text("""
            select b.d
            from (
                select distinct trade_date d
                from cn_sector_rot_bt_daily_t
                where run_id=:r
            ) b
            left join (
                select trade_date, count(*) c
                from cn_sector_rotation_ranked_t
                group by trade_date
            ) s on s.trade_date=b.d
            where ifnull(s.c, 0) < :uc
            order by b.d
            limit :lim
        """), {"r": RUN_ID, "uc": universe_count, "lim": SAMPLE_LIMIT}).fetchall())

    mismatch_days = conn.execute(text("""
        select count(*)
        from (
            select b.d,
                   ifnull(s.c, 0) sc,
                   ifnull(r.c, 0) rc
            from (
                select distinct trade_date d
                from cn_sector_rot_bt_daily_t
                where run_id=:r
            ) b
            left join (
                select signal_date, count(*) c
                from cn_sector_rotation_signal_t
                group by signal_date
            ) s on s.signal_date = b.d
            left join (
                select trade_date, count(*) c
                from cn_sector_rotation_ranked_t
                group by trade_date
            ) r on r.trade_date = b.d
        ) x
        where sc <> rc
    """), {"r": RUN_ID}).scalar() or 0

    mismatch_sample_rows = conn.execute(text("""
        select d, sc, rc
        from (
            select b.d,
                   ifnull(s.c, 0) sc,
                   ifnull(r.c, 0) rc
            from (
                select distinct trade_date d
                from cn_sector_rot_bt_daily_t
                where run_id=:r
            ) b
            left join (
                select signal_date, count(*) c
                from cn_sector_rotation_signal_t
                group by signal_date
            ) s on s.signal_date = b.d
            left join (
                select trade_date, count(*) c
                from cn_sector_rotation_ranked_t
                group by trade_date
            ) r on r.trade_date = b.d
        ) x
        where sc <> rc
        order by d
        limit :lim
    """), {"r": RUN_ID, "lim": SAMPLE_LIMIT}).fetchall()

    mismatch_sample = [f"{d}|signal={sc}|ranked={rc}" for d, sc, rc in mismatch_sample_rows]

    return {
        "bt_dates_total": int(bt_dates_total),
        "universe_date": universe_date,
        "universe_count": int(universe_count),
        "signal_stats": signal_stats,
        "ranked_stats": ranked_stats,
        "signal_missing_days": int(signal_missing_days),
        "ranked_missing_days": int(ranked_missing_days),
        "signal_full_days": int(signal_full_days),
        "ranked_full_days": int(ranked_full_days),
        "mismatch_days": int(mismatch_days),
        "signal_missing_sample": signal_missing_sample,
        "ranked_missing_sample": ranked_missing_sample,
        "mismatch_sample": mismatch_sample,
    }


def fill_signal(conn, universe_date, universe_count):
    conn.execute(text("drop temporary table if exists tmp_need_dates_signal"))
    conn.execute(text("drop temporary table if exists tmp_need_dates_chunk"))
    conn.execute(text("""
        create temporary table tmp_need_dates_signal (
            d date not null primary key
        ) engine=memory
    """))
    conn.execute(text("""
        create temporary table tmp_need_dates_chunk (
            d date not null primary key
        ) engine=memory
    """))

    conn.execute(text("""
        insert into tmp_need_dates_signal (d)
        select b.d
        from (
            select distinct trade_date d
            from cn_sector_rot_bt_daily_t
            where run_id = :r
        ) b
        left join (
            select signal_date, count(*) c
            from cn_sector_rotation_signal_t
            group by signal_date
        ) s on s.signal_date = b.d
        where ifnull(s.c, 0) < :uc
    """), {"r": RUN_ID, "uc": universe_count})

    need_days = conn.execute(text("select count(*) from tmp_need_dates_signal")).scalar() or 0

    inserted_rows = 0
    batch_count = 0
    if need_days > 0:
        while True:
            conn.execute(text("truncate table tmp_need_dates_chunk"))
            conn.execute(text("""
                insert into tmp_need_dates_chunk (d)
                select d
                from tmp_need_dates_signal
                order by d
                limit :lim
            """), {"lim": BATCH_DAYS})

            this_batch_days = conn.execute(text("select count(*) from tmp_need_dates_chunk")).scalar() or 0
            if this_batch_days == 0:
                break

            res = conn.execute(text("""
                insert into cn_sector_rotation_signal_t (
                    signal_date, sector_type, sector_id, sector_name,
                    action, entry_rank, entry_cnt, weight_suggested,
                    score, state, transition, created_at
                )
                select /*+ INDEX(s2 idx_signal_sector_date) INDEX(p idx_signal_sector_date) INDEX(cur idx_signal_date_sector) */
                    pr.signal_date,
                    pr.sector_type,
                    pr.sector_id,
                    p.sector_name,
                    'WATCH' as action,
                    null as entry_rank,
                    null as entry_cnt,
                    null as weight_suggested,
                    p.score,
                    p.state,
                    'CARRY_FORWARD' as transition,
                    now(6) as created_at
                from (
                    select
                        nd.d as signal_date,
                        u.sector_type,
                        u.sector_id,
                        max(s2.signal_date) as prev_date
                    from tmp_need_dates_chunk nd
                    join (
                        select sector_type, sector_id
                        from cn_sector_rotation_signal_t
                        where signal_date = :u
                    ) u
                    join cn_sector_rotation_signal_t s2
                      on s2.sector_type = u.sector_type
                     and s2.sector_id = u.sector_id
                     and s2.signal_date < nd.d
                    group by nd.d, u.sector_type, u.sector_id
                ) pr
                join cn_sector_rotation_signal_t p
                  on p.sector_type = pr.sector_type
                 and p.sector_id = pr.sector_id
                 and p.signal_date = pr.prev_date
                left join cn_sector_rotation_signal_t cur
                       on cur.signal_date = pr.signal_date
                      and cur.sector_type = pr.sector_type
                      and cur.sector_id = pr.sector_id
                where cur.signal_date is null
            """), {"u": universe_date})
            inserted_rows += int(res.rowcount or 0)
            batch_count += 1

            conn.execute(text("""
                delete nd
                from tmp_need_dates_signal nd
                join tmp_need_dates_chunk c on c.d = nd.d
            """))

    conn.execute(text("drop temporary table if exists tmp_need_dates_signal"))
    conn.execute(text("drop temporary table if exists tmp_need_dates_chunk"))
    return int(need_days), int(inserted_rows), int(batch_count)


def fill_ranked_from_signal(conn):
    res = conn.execute(text("""
        insert into cn_sector_rotation_ranked_t (
            trade_date, sector_type, sector_id, sector_name,
            state, tier, theme_group, theme_rank, score,
            confirm_streak, amt_impulse, up_ma5, up_ratio, created_at
        )
        select /*+ INDEX(s idx_signal_date_sector) INDEX(r idx_rank_trade_sector) */
            s.signal_date as trade_date,
            s.sector_type,
            s.sector_id,
            s.sector_name,
            s.state,
            null as tier,
            null as theme_group,
            null as theme_rank,
            s.score,
            null as confirm_streak,
            null as amt_impulse,
            null as up_ma5,
            null as up_ratio,
            now(6) as created_at
        from cn_sector_rotation_signal_t s
        join (
            select distinct trade_date
            from cn_sector_rot_bt_daily_t
            where run_id = :r
        ) b on b.trade_date = s.signal_date
        left join cn_sector_rotation_ranked_t r
          on r.trade_date = s.signal_date
         and r.sector_type = s.sector_type
         and r.sector_id = s.sector_id
        where r.trade_date is null
    """), {"r": RUN_ID})
    return int(res.rowcount or 0)


def print_stats(title, stats):
    print(f"\n[{title}]")
    print(f"run_id={RUN_ID}")
    print(f"bt_dates_total={stats['bt_dates_total']}")
    print(f"universe_date={stats['universe_date']}")
    print(f"target_universe_count={stats['universe_count']}")
    print(f"signal_rows_per_day_min_avg_max={stats['signal_stats']}")
    print(f"ranked_rows_per_day_min_avg_max={stats['ranked_stats']}")
    print(f"signal_full_days={stats['signal_full_days']}/{stats['bt_dates_total']}")
    print(f"ranked_full_days={stats['ranked_full_days']}/{stats['bt_dates_total']}")
    print(f"signal_missing_days_vs_bt={stats['signal_missing_days']}")
    print(f"ranked_missing_days_vs_bt={stats['ranked_missing_days']}")
    print(f"signal_vs_ranked_count_mismatch_days={stats['mismatch_days']}")
    print("signal_missing_sample_dates=" + ",".join(stats["signal_missing_sample"]))
    print("ranked_missing_sample_dates=" + ",".join(stats["ranked_missing_sample"]))
    print("signal_ranked_mismatch_samples=" + "; ".join(stats["mismatch_sample"]))


def main():
    try:
        with engine.begin() as conn:
            before = None
            if ENABLE_HEAVY_STATS:
                before = collect_stats(conn)
                print_stats("Pre-Check", before)

            if before is not None:
                universe_date = before["universe_date"]
                universe_count = before["universe_count"]
            else:
                universe_date = conn.execute(text("""
                    select signal_date
                    from cn_sector_rotation_signal_t
                    group by signal_date
                    order by count(*) desc, signal_date desc
                    limit 1
                """)).scalar()
                universe_count = 0
                if universe_date is not None:
                    universe_count = conn.execute(text("""
                        select count(*)
                        from cn_sector_rotation_signal_t
                        where signal_date = :d
                    """), {"d": universe_date}).scalar() or 0

            if universe_date is None or universe_count == 0:
                print("No data in cn_sector_rotation_signal_t. Nothing to fill.")
                return

            t0 = time.perf_counter()
            need_days, signal_inserted, batch_count = fill_signal(
                conn,
                universe_date,
                universe_count,
            )
            t1 = time.perf_counter()
            ranked_inserted = fill_ranked_from_signal(conn)
            t2 = time.perf_counter()

            print("\n[Fill Action]")
            print(f"signal_need_days_before={need_days}")
            print(f"signal_batches={batch_count}")
            print(f"signal_rows_inserted={signal_inserted}")
            print(f"ranked_rows_inserted={ranked_inserted}")
            print(f"signal_fill_seconds={t1 - t0:.3f}")
            print(f"ranked_fill_seconds={t2 - t1:.3f}")
            print(f"total_fill_seconds={t2 - t0:.3f}")

            if ENABLE_HEAVY_STATS:
                after = collect_stats(conn)
                print_stats("Post-Check", after)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
