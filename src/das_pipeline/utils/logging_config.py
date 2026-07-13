# src/das_pipeline/utils/logging_config.py

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_dir: Path | str = "outputs/logs",
    log_to_file: bool = True,
) -> None:
    """設定 root logger，同時輸出到 console 和檔案"""

    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # 避免重複呼叫 setup_logging 時 handler 疊加

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        log_file = log_dir / f"run_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Log 檔案: {log_file}")