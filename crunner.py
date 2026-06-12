"""Main entry point.

Usage:
  python runner.py --tasks stock,etf,index --asof latest
  python runner.py --tasks all --asof 20260131

Notes:
- DbInitTask is always executed first.
- Task selection is handled in app/cli.py.

请修改入口 ,  可以针对不同的参数，cli.py 可以调用下列单独的 task,  DbInitTask()除外，它是必执行

      selected = ["stock", "etf", "index", "futures", "options", "audit"]
      

      只跑股票：python runner.py --tasks stock

跑股票+ETF+指数：python runner.py --tasks stock,etf,index

全部：python runner.py --tasks all

"""
from app.cli import run


if __name__ == "__main__":
    run()
