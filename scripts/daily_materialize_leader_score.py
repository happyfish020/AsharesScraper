"""
日级物化脚本: 通过存储过程 sp_materialize_leader_score 物化 leader score 数据

相比旧版（依赖 v1/v2 视图）的核心优化:
  1. 使用存储过程 + 临时表，只扫描目标日期范围的数据
  2. 避免每次全表扫描 cn_board_member_map_d (30.7M) + cn_stock_daily_price (16.8M)
  3. 临时表在 SP 执行期间存在，完成后自动删除
  4. 修复了 argparse --start/--end 互斥的 bug

用法:
    # 物化指定日期范围
    python scripts/daily_materialize_leader_score.py --start 2026-05-20 --end 2026-05-21

    # 物化最近 N 天（默认最近 7 天，含今天）
    python scripts/daily_materialize_leader_score.py --lookback 7

    # 物化今天的数据（收盘后运行）
    python scripts/daily_materialize_leader_score.py --today

    # 全量回填历史数据
    python scripts/daily_materialize_leader_score.py --backfill

    # 自动检测并回补缺失日期（对比 cn_stock_daily_price 中的交易日）
    python scripts/daily_materialize_leader_score.py --repair-missing

    # 回补指定日期范围内的缺失数据
    python scripts/daily_materialize_leader_score.py --repair-missing --start 2026-01-01 --end 2026-05-21

    # 回补最近 N 天的缺失数据
    python scripts/daily_materialize_leader_score.py --repair-missing --lookback 30

前置条件:
    1. cn_stock_leader_score_daily 必须是物理表
    2. sp_materialize_leader_score 存储过程必须已创建
       (执行 sql/sp_materialize_leader_score.sql)
"""

import argparse
import sys
import time
from datetime import date, datetime, timedelta

import pymysql
from pymysql.cursors import DictCursor


DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "cn_opr_red",
    "password": "sec_Bobo123",
    "database": "cn_market_red",
    "charset": "utf8mb4",
    "connect_timeout": 30,
    "read_timeout": 3600,  # 历史回填可能需要较长时间
}


def get_connection():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG)


def get_trade_dates(cur, start_date: str, end_date: str) -> list[str]:
    """获取指定日期范围内的交易日"""
    sql = """
        SELECT DISTINCT TRADE_DATE
        FROM cn_stock_daily_price
        WHERE TRADE_DATE >= %s AND TRADE_DATE <= %s
        ORDER BY TRADE_DATE
    """
    cur.execute(sql, (start_date, end_date))
    return [row[0].strftime("%Y-%m-%d") if hasattr(row[0], 'strftime') else str(row[0])
            for row in cur.fetchall()]


def check_sp_exists(cur) -> bool:
    """检查存储过程是否存在"""
    cur.execute("""
        SELECT COUNT(*) FROM information_schema.ROUTINES
        WHERE ROUTINE_SCHEMA = 'cn_market_red'
          AND ROUTINE_NAME = 'sp_materialize_leader_score'
          AND ROUTINE_TYPE = 'PROCEDURE'
    """)
    return cur.fetchone()[0] > 0


def check_table_exists(cur) -> bool:
    """检查物化表是否存在"""
    cur.execute("""
        SELECT COUNT(*) FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = 'cn_market_red' AND TABLE_NAME = 'cn_stock_leader_score_daily'
          AND TABLE_TYPE = 'BASE TABLE'
    """)
    return cur.fetchone()[0] > 0


def call_sp_materialize(cur, start_date: str, end_date: str) -> dict:
    """
    调用存储过程物化指定日期范围的数据。
    返回: { "start": str, "end": str, "inserted": int, "duration_sec": float }
    """
    t0 = time.time()

    # CALL 存储过程
    cur.callproc("sp_materialize_leader_score", (start_date, end_date))

    # 读取存储过程的 SELECT 输出
    inserted = 0
    duration_sec = 0.0
    for result in cur.fetchall():
        # 第一个结果集: (inserted_rows, duration_sec)
        if result and len(result) >= 2:
            inserted = int(result[0]) if result[0] is not None else 0
            duration_sec = float(result[1]) if result[1] is not None else 0.0
            break

    # 跳过后续结果集（progress 消息）
    while cur.nextset():
        pass

    elapsed = time.time() - t0
    return {
        "start": start_date,
        "end": end_date,
        "inserted": inserted,
        "duration_sec": round(max(duration_sec, elapsed), 2),
    }


def detect_missing_dates(cur, start_date: str, end_date: str) -> list[str]:
    """
    检测缺失日期：找出 cn_stock_daily_price 中存在但 cn_stock_leader_score_daily 中不存在的交易日。
    返回缺失日期列表（按升序排列）。
    """
    sql = """
        SELECT p.TRADE_DATE
        FROM cn_stock_daily_price p
        WHERE p.TRADE_DATE >= %s AND p.TRADE_DATE <= %s
          AND NOT EXISTS (
              SELECT 1 FROM cn_stock_leader_score_daily t
              WHERE t.trade_date = p.TRADE_DATE
          )
        GROUP BY p.TRADE_DATE
        ORDER BY p.TRADE_DATE
    """
    cur.execute(sql, (start_date, end_date))
    return [
        row[0].strftime("%Y-%m-%d") if hasattr(row[0], 'strftime') else str(row[0])
        for row in cur.fetchall()
    ]


def repair_missing_dates(cur, start_date: str, end_date: str) -> dict:
    """
    自动检测并回补指定日期范围内的缺失数据。
    返回: { "total_missing": int, "repaired": int, "duration_sec": float }
    """
    t0 = time.time()

    # 1. 检测缺失日期
    print(f"  检测缺失日期: {start_date} ~ {end_date} ...")
    missing_dates = detect_missing_dates(cur, start_date, end_date)
    print(f"  发现 {len(missing_dates)} 个缺失交易日")

    if not missing_dates:
        return {
            "total_missing": 0,
            "repaired": 0,
            "duration_sec": round(time.time() - t0, 2),
        }

    # 2. 批量物化缺失日期（按连续区间合并，减少 SP 调用次数）
    repaired = 0
    batch_start = missing_dates[0]
    batch_end = missing_dates[0]
    batch_count = 0

    def flush_batch():
        nonlocal repaired, batch_start, batch_end, batch_count
        if batch_count == 0:
            return
        print(f"    批量物化: {batch_start} ~ {batch_end} ({batch_count} 天)...")
        result = call_sp_materialize(cur, batch_start, batch_end)
        repaired += result["inserted"]
        print(f"      -> 物化 {result['inserted']} 行, 耗时 {result['duration_sec']}s")
        cur.connection.commit()

    for i, td in enumerate(missing_dates):
        td_date = datetime.strptime(td, "%Y-%m-%d").date()
        batch_end_date = datetime.strptime(batch_end, "%Y-%m-%d").date()

        if (td_date - batch_end_date).days <= 1:
            # 连续日期，扩展当前批次
            batch_end = td
            batch_count += 1
        else:
            # 不连续，刷新当前批次
            flush_batch()
            batch_start = td
            batch_end = td
            batch_count = 1

        # 进度显示
        print(f"  [{i+1}/{len(missing_dates)}] {td}")

    # 刷新最后一批
    flush_batch()

    elapsed = time.time() - t0
    return {
        "total_missing": len(missing_dates),
        "repaired": repaired,
        "duration_sec": round(elapsed, 2),
    }


def backfill_history(cur):
    """
    全量回填历史数据
    按年逐批物化，每批完成后提交
    """
    years = list(range(2010, 2027))  # 2010 ~ 2026
    total_inserted = 0
    total_duration = 0.0

    for year in years:
        start = f"{year}-01-01"
        end = f"{year}-12-31"

        print(f"  [{year}] 开始物化...")
        result = call_sp_materialize(cur, start, end)
        total_inserted += result["inserted"]
        total_duration += result["duration_sec"]
        print(f"    -> 物化 {result['inserted']} 行, 耗时 {result['duration_sec']}s")
        cur.connection.commit()

    return {
        "total_inserted": total_inserted,
        "total_duration_sec": round(total_duration, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="日级物化: 通过 SP 物化 leader score 到 cn_stock_leader_score_daily"
    )
    # --start 和 --end 不在 mutually exclusive group 中，
    # 因为 --repair-missing 模式下需要同时使用两者来限定范围
    parser.add_argument("--start", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", help="结束日期 (YYYY-MM-DD)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--today", action="store_true", help="物化今天的数据")
    group.add_argument("--lookback", type=int, default=None,
                       help="物化最近 N 天（含今天）")
    group.add_argument("--backfill", action="store_true",
                       help="全量回填历史数据 (2010~2026)")
    group.add_argument("--repair-missing", action="store_true",
                       help="自动检测并回补缺失日期（对比 cn_stock_daily_price 中的交易日）")
    parser.add_argument("--date", help="指定单个日期 (YYYY-MM-DD)")

    args = parser.parse_args()

    # 解析日期范围
    today_str = date.today().strftime("%Y-%m-%d")

    if args.backfill:
        start_date, end_date = None, None
    elif args.repair_missing:
        # --repair-missing 模式下，start/end/lookback 作为可选范围限定
        if args.start and args.end:
            start_date, end_date = args.start, args.end
        elif args.start:
            start_date, end_date = args.start, args.start
        elif args.end:
            start_date, end_date = args.end, args.end
        elif args.lookback is not None:
            start_date = (date.today() - timedelta(days=args.lookback - 1)).strftime("%Y-%m-%d")
            end_date = today_str
        else:
            # 默认检测最近 30 天的缺失
            start_date = (date.today() - timedelta(days=29)).strftime("%Y-%m-%d")
            end_date = today_str
    elif args.today:
        start_date = end_date = today_str
    elif args.date:
        start_date = end_date = args.date
    elif args.start and args.end:
        start_date, end_date = args.start, args.end
    elif args.start:
        start_date, end_date = args.start, args.start
    elif args.end:
        start_date, end_date = args.end, args.end
    elif args.lookback is not None:
        start_date = (date.today() - timedelta(days=args.lookback - 1)).strftime("%Y-%m-%d")
        end_date = today_str
    else:
        # 默认: 最近 7 天
        start_date = (date.today() - timedelta(days=6)).strftime("%Y-%m-%d")
        end_date = today_str

    print("=" * 60)
    print(f"Leader Score 日级物化脚本 (SP 模式)")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    conn = get_connection()
    cur = conn.cursor()

    try:
        # 1. 检查前置条件
        print("\n[1/4] 检查前置条件...")

        if not check_sp_exists(cur):
            print("  ERROR: sp_materialize_leader_score 存储过程不存在！")
            print("  请先执行: mysql -u cn_opr_red -p cn_market_red < sql/sp_materialize_leader_score.sql")
            sys.exit(1)
        print("  [OK] sp_materialize_leader_score 存储过程存在")

        if not check_table_exists(cur):
            print("  ERROR: cn_stock_leader_score_daily 物理表不存在！")
            print("  请先执行 sql/optimize_core_tables_partition.sql 创建表")
            sys.exit(1)
        print("  [OK] cn_stock_leader_score_daily 物理表存在")

        # 2. 执行物化
        if args.backfill:
            print("\n[2/4] 全量历史回填...")
            result = backfill_history(cur)
            print(f"\n  回填完成: 共 {result['total_inserted']} 行, "
                  f"总耗时 {result['total_duration_sec']}s")
        elif args.repair_missing:
            print(f"\n[2/4] 自动检测并回补缺失数据: {start_date} ~ {end_date}")
            result = repair_missing_dates(cur, start_date, end_date)
            print(f"\n  回补完成: 缺失={result['total_missing']}个, "
                  f"已修复={result['repaired']}行, "
                  f"耗时={result['duration_sec']}s")
        else:
            print(f"\n[2/4] 物化日期范围: {start_date} ~ {end_date}")

            # 获取交易日列表
            trade_dates = get_trade_dates(cur, start_date, end_date)
            if not trade_dates:
                print(f"  指定范围内无交易日")
                sys.exit(0)

            print(f"  交易日数量: {len(trade_dates)}")
            print(f"  首个交易日: {trade_dates[0]}")
            print(f"  最后交易日: {trade_dates[-1]}")

            # 直接调用 SP（SP 内部用临时表只处理目标范围）
            result = call_sp_materialize(cur, start_date, end_date)
            print(f"  物化完成: {result['inserted']} 行, 耗时 {result['duration_sec']}s")
            conn.commit()

        # 3. 验证
        print("\n[3/4] 验证物化结果...")
        if args.backfill:
            cur.execute("SELECT MIN(trade_date), MAX(trade_date), COUNT(*) "
                        "FROM cn_stock_leader_score_daily")
        elif args.repair_missing:
            cur.execute(
                "SELECT MIN(trade_date), MAX(trade_date), COUNT(*) "
                "FROM cn_stock_leader_score_daily "
                "WHERE trade_date >= %s AND trade_date <= %s",
                (start_date, end_date)
            )
        else:
            cur.execute(
                "SELECT MIN(trade_date), MAX(trade_date), COUNT(*) "
                "FROM cn_stock_leader_score_daily "
                "WHERE trade_date >= %s AND trade_date <= %s",
                (start_date, end_date)
            )
        row = cur.fetchone()
        print(f"  物化表日期范围: {row[0]} ~ {row[1]}")
        print(f"  物化表行数: {row[2]}")

        # 4. 检查分区使用情况
        print("\n[4/4] 分区使用情况...")
        cur.execute("""
            SELECT PARTITION_NAME, TABLE_ROWS
            FROM information_schema.PARTITIONS
            WHERE TABLE_SCHEMA = 'cn_market_red'
              AND TABLE_NAME = 'cn_stock_leader_score_daily'
              AND PARTITION_NAME IS NOT NULL
            ORDER BY PARTITION_ORDINAL_POSITION
        """)
        for r in cur.fetchall():
            print(f"  分区 {r[0]}: {r[1]} 行")

        print("\n" + "=" * 60)
        print("物化完成!")
        print("=" * 60)

    except Exception as e:
        conn.rollback()
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
