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

        if not input_dir.is_dir():
            raise NotADirectoryError(
                f"input_dir 應為目錄路徑: {input_dir}"
            )

        # 把目錄交給 dascore 的 DirectorySpool 管理，
        # 它會自動掃描目錄下所有支援的 DAS 格式（HDF5 等），並建立 Lazy Index。
        logger.info(f"從目錄建立 spool: {input_dir} (格式: {config.format})")
        spool = dc.spool(input_dir)

    if config.time_range:
        logger.info(f"套用時間範圍篩選: {config.time_range}")
        spool = spool.select(time=tuple(config.time_range))

    return spool


def _compute_overlap_seconds(config: DataConfig) -> float:
    """根據 taper_ratio 和 filter_safety_samples 計算 chunk overlap (秒)。"""
    duration_sec = config.chunk_duration / np.timedelta64(1, "s")
    overlap_sec = 2 * config.taper_ratio * duration_sec
    if config.filter_safety_samples > 0 and config.sampling_rate is not None:
        safety_sec = config.filter_safety_samples / config.sampling_rate
        overlap_sec += safety_sec
    return overlap_sec


def _compute_core_time_range(patch: dc.Patch, config: DataConfig, is_first: bool, is_last: bool) -> tuple:
    """計算當前 chunk 的核心時間範圍（排除 taper 區域）。

    此處的 margin 計算與 _compute_overlap_seconds 一致，
    都基於 config.chunk_duration 而非 patch 實際座標，確保相鄰 chunk 無縫銜接。

    Parameters
    ----------
    patch : dc.Patch
        當前 chunk（已通過 preprocessing）。
    config : DataConfig
        設定（含 taper_ratio）。
    is_first, is_last : bool
        是否為首/尾 chunk。

    Returns
    -------
    tuple
        (core_time_start, core_time_end) 作為 select 的 time range。
    """
    time_coord = patch.get_coord("time")
    t_min: np.datetime64 = time_coord.min()  # type: ignore[assignment]
    t_max: np.datetime64 = time_coord.max()  # type: ignore[assignment]

    # 計算每邊 margin = overlap / 2
    overlap_sec = _compute_overlap_seconds(config)
    margin_sec = overlap_sec / 2.0

    # 將 margin 對齊取樣間隔（floor），確保相鄰 chunk 的 core 無縫銜接
    if config.sampling_rate is not None and config.sampling_rate > 0:
        margin_samples = int(margin_sec * config.sampling_rate)
        margin_sec = margin_samples / config.sampling_rate

    margin = np.timedelta64(int(round(margin_sec * 1e9)), "ns")

    core_start: np.datetime64 = t_min + (margin if not is_first else np.timedelta64(0, "ns"))  # type: ignore[operator]
    core_end: np.datetime64 = t_max - (margin if not is_last else np.timedelta64(0, "ns"))  # type: ignore[operator]

    if core_end <= core_start:
        logger.warning(
            f"core range 為空: [{core_start}, {core_end}]，"
            f"margin_sec={margin_sec:.3f}s。"
            f"chunk 可能太小，建議增大 chunk_duration 或降低 taper_ratio。"
        )

    return (core_start, core_end)


def iter_chunks(spool: dc.BaseSpool, config: DataConfig) -> Iterator[dc.Patch]:
    """把 spool 切成固定時間長度的片段，逐段 yield 出 Patch。
    每次迭代只有「這一段」的資料被實際讀進記憶體。
    每個 chunk 會在其 attrs 中儲存 core_time_start / core_time_end，
    供後續 merge_patches 裁切用。
    """
    overlap_sec = _compute_overlap_seconds(config)
    chunked_spool = spool.chunk(
        time=config.chunk_duration,
        overlap=overlap_sec,
        #conflict="raise",
    )

    total = len(chunked_spool)
    logger.info(
        f"共切成 {total} 段，每段 {config.chunk_duration}，"
        f"overlap={overlap_sec:.1f}s"
    )

    for i, patch in enumerate(chunked_spool): # type: ignore[arg-type]
        is_first = (i == 0)
        is_last = (i == total - 1)

        core_start, core_end = _compute_core_time_range(
            patch, config, is_first=is_first, is_last=is_last,
        )

        # 將核心範圍寫入 attrs 供 merge 裁切用
        time_coord = patch.get_coord("time")

        patch = patch.update_attrs(
            core_time_start=str(core_start),
            core_time_end=str(core_end),
            taper_ratio=config.taper_ratio,
            filter_safety_samples=config.filter_safety_samples,
            # 也存原始 chunk 邊界，便於除錯
            chunk_time_start=str(time_coord.min()),
            chunk_time_end=str(time_coord.max()),
        )

        logger.info(
            f"處理第 {i + 1}/{total} 段，"
            f"chunk=[{time_coord.min()}, {time_coord.max()}], "
            f"core=[{core_start}, {core_end}]"
        )
        yield i, patch # type: ignore[arg-type]
