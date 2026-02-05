"""
UnifiedRisk V12 - Logging Utility
---------------------------------
提供统一的日志初始化（setup_logging）与 logger 获取（get_logger）

功能：
- 控制台输出
- 文件输出（按 market + mode 自动命名）
- 所有模块共享同一 logging 配置
"""

import logging
import os
from datetime import datetime

# 全局 logger registry
_LOGGERS = {}
_LOG_DIR = None


# ==============================================================
#  获取 logger （V12 统一接口）
# ==============================================================
def get_logger(name: str) -> logging.Logger:
    """
    获取 logger，如果不存在则创建。
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    _LOGGERS[name] = logger
    return logger


# ==============================================================
#  主初始化：setup_logging（你当前项目缺少的关键函数）
# ==============================================================
def setup_logging(market: str, mode: str):
    """
    初始化 logging：
    - 生成 logs 目录
    - 创建 log file handler
    - 设置 console handler
    - 所有 logger 共用 root handler
    """

    global _LOG_DIR

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    logs_dir = os.path.join(root_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # log path 格式：logs/cn_ashare_daily_2025-12-08-153000.log
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    log_filename = f"{market}_{mode}_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)

    _LOG_DIR = logs_dir

    # ----------------------------------------------------------
    # 创建 root logger（V12 推荐方式）
    # ----------------------------------------------------------
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 清空旧 handler（防止重复输出）
    if root_logger.handlers:
        root_logger.handlers.clear()

    # ----------------------------------------------------------
    # File Handler
    # ----------------------------------------------------------
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] (%(levelname)s) %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    # ----------------------------------------------------------
    # Console Handler
    # ----------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] (%(levelname)s) %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)

    # ----------------------------------------------------------
    # 将 handler 添加到 root
    # ----------------------------------------------------------
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # ----------------------------------------------------------
    # 初始化完成日志
    # ----------------------------------------------------------
    logger = get_logger("Logger")
    logger.info("Logging initialized: %s", log_path)

    return log_path


# ==============================================================
# 方便外部模块获取当前 log 目录
# ==============================================================
def get_log_dir() -> str:
    return _LOG_DIR
