# src/das_pipeline/visualization/merge.py

import logging
import re
from pathlib import Path
from typing import List, Optional
import dascore as dc
import numpy as np
import pandas as pd

def _to_naive_utc_datetime64(ts_str: str) -> np.datetime64:
    """將時間字串（可能含 tz、numpy 字串或物件）統一轉為 tz-naive UTC 的 np.datetime64。"""
    if isinstance(ts_str, np.datetime64):
        return ts_str.astype("datetime64[ns]")

    if hasattr(ts_str, "item"):
        try:
            ts_str = ts_str.item()
        except Exception:
            pass

    if not isinstance(ts_str, str):
        ts_str = str(ts_str)

    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts)

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
        logger.debug("patch 缺少 core_time attrs，跳過裁切")
        return patch

    try:
        core_start = _to_naive_utc_datetime64(core_start_str)
        core_end = _to_naive_utc_datetime64(core_end_str)
    except Exception:
        logger.warning(f"無法解析 core_time attrs: {core_start_str}, {core_end_str}，跳過裁切")
        return patch

    if core_end <= core_start:
        # core range 無效（chunk 太短），取 chunk 中點附近的最小有效範圍，
        # 避免將 taper overlap 區域帶入合併導致時間軸非單調。
        t_coord = patch.get_coord("time")
        t_vals = np.asarray(t_coord)
        if len(t_vals) < 2:
            logger.warning(f"core range 為空且 chunk 長度不足 2，無法裁切")
            return patch
        mid = len(t_vals) // 2
        core_start = t_vals[max(0, mid - 1)]
        core_end = t_vals[min(len(t_vals) - 1, mid + 1)]
        logger.warning(
            "core range 為空，改用 chunk 中點附近 [%s, %s] 強制裁切",
            core_start, core_end,
        )

    try:
        cropped = patch.select(time=(core_start, core_end))
        return cropped
    except Exception as e:
        logger.warning("core range 裁切失敗: %s，嘗試手動裁切", e)
        # fallback: 手動用 boolean index 裁切，避免將 overlap 區域洩漏到合併
        t_vals = np.asarray(patch.get_coord("time"))
        mask = (t_vals >= core_start) & (t_vals <= core_end)
        if mask.sum() < 2:
            logger.warning("手動裁切後 data 點不足 2，回傳原始 patch")
            return patch
        data = np.asarray(patch.data)
        time_axis = patch.dims.index("time")
        if time_axis == 0:
            data = data[mask, :]
        else:
            data = data[:, mask]
        t_new = t_vals[mask]
        return dc.Patch(
            data=data,
            coords={"time": t_new, "distance": patch.get_coord("distance")},
            dims=patch.dims,
            attrs=patch.attrs,
        )


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

    # ── 驗證時間軸單調性 ──
    time_coord = merged.get_coord("time")
    t_vals = np.asarray(time_coord)

    if len(t_vals) > 1:
        try:
            t_int = t_vals.astype(np.int64)
        except Exception:
            t_int = np.array([pd.Timestamp(x).value for x in t_vals], dtype=np.int64)

        diffs = np.diff(t_int)
        if np.any(diffs <= 0):
            n_bad = int(np.sum(diffs <= 0))
            logger.warning(
                "合併後時間軸有 %d 處非單調遞增（diff <= 0），"
                "執行去重/排序修復",
                n_bad,
            )
            # 先按時間排序，再去重；若有相同時間點，保留第一次出現的樣本。
            order = np.argsort(t_int)
            t_sorted = t_vals[order]
            data = np.asarray(merged.data)
            time_axis = merged.dims.index("time")

            if time_axis == 0:
                data_sorted = data[order, :]
            else:
                data_sorted = data[:, order]

            _, unique_idx = np.unique(t_sorted, return_index=True)
            unique_idx = np.sort(unique_idx)
            t_vals = t_sorted[unique_idx]
            if time_axis == 0:
                data = data_sorted[unique_idx, :]
            else:
                data = data_sorted[:, unique_idx]

            merged = dc.Patch(
                data=data,
                coords={
                    "time": t_vals,
                    "distance": merged.get_coord("distance"),
                },
                dims=merged.dims,
                attrs=merged.attrs,
            )
            logger.info(
                "修復完成，新時間軸長度: %d (原: %d)",
                len(t_vals),
                len(t_vals) + n_bad,
            )

    time_coord = merged.get_coord("time")
    logger.info(
        f"合併完成，time 範圍: {time_coord.min()} ~ {time_coord.max()}"
    )

    return merged
