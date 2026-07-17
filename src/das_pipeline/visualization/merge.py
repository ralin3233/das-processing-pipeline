# src/das_pipeline/visualization/merge.py

import logging
import re
from pathlib import Path
from typing import List, Optional
import pandas as pd
import dascore as dc
import numpy as np

logger = logging.getLogger(__name__)


def _parse_chunk_index(filename: str) -> Optional[int]:
    """從檔名中解析 chunk_index，例如 chunk0000 -> 0, chunk0042 -> 42。"""
    match = re.search(r"chunk(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def _parse_timestamp(filename: str) -> Optional[str]:
    """從檔名中解析 timestamp，例如 20250714T143000。"""
    match = re.search(r"(\d{8}T\d{6})", filename)
    if match:
        return match.group(1)
    return None


def _crop_to_core(patch: dc.Patch) -> dc.Patch:
    """根據 patch.attrs 中的 core_time_start / core_time_end 裁切至核心範圍。

    若缺少 attrs 或裁切後範圍無效，則回傳原始 patch（不裁切）。
    """
    core_start_str = patch.attrs.get("core_time_start")
    core_end_str = patch.attrs.get("core_time_end")

    if not core_start_str or not core_end_str:
        logger.debug(f"patch 缺少 core_time attrs，跳過裁切")
        return patch

    try:
        core_start = np.datetime64(core_start_str)
        core_end = np.datetime64(core_end_str)
    except Exception:
        logger.warning(f"無法解析 core_time attrs: {core_start_str}, {core_end_str}，跳過裁切")
        return patch

    if core_end <= core_start:
        logger.warning(f"core range 為空 [{core_start}, {core_end}]，跳過裁切")
        return patch

    try:
        cropped = patch.select(time=(core_start, core_end))
        return cropped
    except Exception as e:
        logger.warning(f"core range 裁切失敗: {e}，使用原始 patch")
        return patch


def merge_patches(
    file_paths: List[Path],
    sort_by: str = "chunk_index",
) -> dc.Patch:
    """將多個 chunk .h5 檔案沿時間軸合併為單一 Patch。
    合併前會依據各 chunk 儲存的 core_time_start/end attrs 裁切，
    以消除 taper 造成的空隙。

    Parameters
    ----------
    file_paths : List[Path]
        .h5 檔案路徑列表。
    sort_by : str
        排序方式，'chunk_index' 或 'timestamp'，預設 'chunk_index'。

    Returns
    -------
    dc.Patch
        合併後的完整 Patch。
    """
    if not file_paths:
        raise ValueError("file_paths 不得為空")

    # 排序檔案
    if sort_by == "chunk_index":
        file_paths = sorted(
            file_paths,
            key=lambda p: _parse_chunk_index(p.name) or 0,
        )
    elif sort_by == "timestamp":
        file_paths = sorted(
            file_paths,
            key=lambda p: _parse_timestamp(p.name) or "",
        )
    else:
        file_paths = sorted(file_paths)

    logger.info(
        f"準備合併 {len(file_paths)} 個檔案，排序方式={sort_by}"
    )
    for p in file_paths:
        logger.info(f"  - {p.name}")

    # 讀取所有 Patch
    patches: List[dc.Patch] = []
    for fp in file_paths:
        spool = dc.spool(str(fp))
        chunk_patch = spool[0]  # 每個 .h5 只有一個 Patch
        patches.append(chunk_patch)

    if len(patches) == 1:
        logger.info("只有一個檔案，無須合併")
        return patches[0]

    # 依 core_time 裁切每個 patch，消除 taper 空隙
    cropped_patches = []
    for patch in patches:
        cropped = _crop_to_core(patch)
        cropped_patches.append(cropped)

        time_coord = cropped.get_coord("time")
        logger.debug(
            f"裁切後 time 範圍: {time_coord.min()} ~ {time_coord.max()}"
        )

    # 沿時間軸拼接
    merged_spool = dc.spool(cropped_patches).concatenate(time=None)
    merged = merged_spool[0]

    time_coord = merged.get_coord("time")
    logger.info(
        f"合併完成，time 範圍: {time_coord.min()} ~ {time_coord.max()}"
    )

    return merged
