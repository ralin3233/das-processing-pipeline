# src/das_pipeline/preprocessing/bandpass.py

import logging
from typing import Optional
import numpy as np
import dascore as dc

logger = logging.getLogger(__name__)


def bandpass(
    patch: dc.Patch,
    freq_range: Optional[tuple[float, float]] = None,
) -> dc.Patch:
    """對時間軸進行帶通濾波。

    使用 DASCore 內建的 pass_filter，預設為 Butterworth 帶通濾波器。
    濾波前會對頭尾進行 taper 以減少邊緣效應。
    taper 比例預設為 0.05，也可從 patch.attrs["taper_ratio"] 讀取。

    Parameters
    ----------
    patch : dc.Patch
        原始 Patch。
    freq_range : tuple[float, float] or None
        帶通頻率範圍 (low_cutoff_hz, high_cutoff_hz)，如 (1.0, 20.0)。
        None 表示跳過濾波。

    Returns
    -------
    dc.Patch
        濾波後的 Patch。
    """
    if freq_range is None:
        logger.info("bandpass 已跳過")
        return patch

    low, high = freq_range
    if low <= 0 or high <= low:
        raise ValueError(f"無效的頻率範圍: [{low}, {high}]，需滿足 0 < low < high")

    # 從 attrs 讀取 taper_ratio（若無則用預設值 0.05）
    taper_ratio = patch.attrs.get("taper_ratio", 0.05)
    patch = patch.taper(time=taper_ratio)
    patch = patch.taper(distance=taper_ratio)
    logger.info("執行 bandpass 濾波: [%s, %s] Hz (taper=%.2f)", low, high, taper_ratio)
    patch = patch.pass_filter(time=(low, high))
    logger.info("bandpass 完成，shape: %s", np.asarray(patch.data).shape)
    return patch
