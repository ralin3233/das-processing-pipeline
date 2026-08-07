# src/das_pipeline/preprocessing/pipeline.py

import logging

import dascore as dc

from das_pipeline.config import PreprocessingConfig
from das_pipeline.preprocessing.select import select
from das_pipeline.preprocessing.detrend import detrend
from das_pipeline.preprocessing.taper import taper
from das_pipeline.preprocessing.bandpass import bandpass
from das_pipeline.preprocessing.decimate import decimate

logger = logging.getLogger(__name__)


def run_preprocessing(patch: dc.Patch, config: PreprocessingConfig) -> dc.Patch:
    """依序執行前處理各步驟：select → detrend → taper → bandpass → decimate。

    Parameters
    ----------
    patch : dc.Patch
        輸入的 Patch。強烈建議先透過 ``sanitize_nan_patch`` 清除 NaN，
        否則 IIR/FIR 濾波會將單一 NaN 擴散到整個時間序列。
    config : PreprocessingConfig
        前處理設定。

    Returns
    -------
    dc.Patch
        處理後的 Patch。
    """
    import numpy as np

    # ── NaN 防禦檢查 ──
    data_arr = np.asarray(patch.data)
    nan_mask = np.isnan(data_arr)
    if nan_mask.any():
        was_sanitized = patch.attrs.get("nan_sanitized")
        nan_pct = nan_mask.sum() / data_arr.size * 100
        if was_sanitized:
            logger.info(
                "Patch 內仍有 %.2f%% NaN（nan_sanitized=True），"
                "這些應為全 NaN channel 被保留的標記。",
                nan_pct,
            )
        else:
            logger.warning(
                "⚠️  Patch 內有 %.2f%% NaN 但未經過 sanitize_nan_patch 處理！"
                "  IIR/FIR 濾波會將 NaN 擴散至整個時間序列，"
                " 建議在 run_preprocessing 之前先呼叫 sanitize_nan_patch。",
                nan_pct,
            )

    # 1. 選取時間/距離範圍
    patch = select(patch, time_range=config.time_range, distance_range=config.distance_range)

    # 2. 去趨勢
    patch = detrend(patch, method=config.detrend)

    # 3. taper（減少濾波邊緣效應），None 則跳過
    patch = taper(patch, taper_ratio=config.taper_ratio)

    # 4. 帶通濾波
    patch = bandpass(patch, freq_range=config.bandpass)

    # 5. 降採樣
    patch = decimate(patch, factor=config.decimate_factor)

    return patch
