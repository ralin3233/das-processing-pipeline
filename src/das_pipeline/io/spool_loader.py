# src/das_pipeline/io/spool_loader.py

import logging
from pathlib import Path
from typing import Iterator

import dascore as dc
import numpy as np
from das_pipeline.config import DataConfig
from das_pipeline.io.miniseed_loader import load as load_miniseed

logger = logging.getLogger(__name__)


def get_spool(config: DataConfig) -> dc.BaseSpool:
    """統一入口：不管來源是 miniseed 還是既有 hdf5，回傳一個 Spool。
    此時資料還沒被讀進記憶體，只是建立索引。
    """
    if config.format.lower() == "miniseed":
        patch = load_miniseed(config)
        spool = dc.spool(patch)
        logger.info(f"已讀入 miniSEED 資料，建立 1 個 Patch 的 spool")
    else:
        input_dir = Path(config.input_dir)
        files = sorted(input_dir.glob(config.file_pattern))

        if not files:
            raise FileNotFoundError(
                f"在 {input_dir} 找不到符合 {config.file_pattern} 的檔案"
            )

        logger.info(f"找到 {len(files)} 個檔案，格式: {config.format}")
        spool = dc.spool([str(f) for f in files])

    if config.time_range:
        logger.info(f"套用時間範圍篩選: {config.time_range}")
        spool = spool.select(time=tuple(config.time_range))

    return spool


def iter_chunks(spool: dc.BaseSpool, config: DataConfig) -> Iterator[dc.Patch]:
    """把 spool 切成固定時間長度的片段，逐段 yield 出 Patch。
    每次迭代只有「這一段」的資料被實際讀進記憶體。
    """
    chunked_spool = spool.chunk(
        time=config.chunk_duration,
        overlap=config.chunk_overlap/np.timedelta64(1, "s") if config.chunk_overlap else 0,
    )

    total = len(chunked_spool)
    logger.info(f"共切成 {total} 段，每段 {config.chunk_duration}")

    for i, patch in enumerate(chunked_spool):
        logger.info(f"處理第 {i + 1}/{total} 段")
        yield i, patch