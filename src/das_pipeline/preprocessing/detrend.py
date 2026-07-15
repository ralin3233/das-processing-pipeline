# src/das_pipeline/preprocessing/detrend.py

import logging
from typing import Optional
import numpy as np
import dascore as dc

logger = logging.getLogger(__name__)


def detrend(patch: dc.Patch, method: Optional[str] = "linear") -> dc.Patch:
    """對時間軸進行去趨勢。

    Parameters
    ----------
    patch : dc.Patch
        原始 Patch。
    method : str or None
        去趨勢方法："linear"（線性）、"constant"（移除均值）。
        None 表示跳過去趨勢。

    Returns
    -------
    dc.Patch
        去趨勢後的 Patch。
    """
    if method is None:
        logger.info("detrend 已跳過")
        return patch

    if method not in ("linear", "constant"):
        raise ValueError(f"不支援的 detrend 方法: {method}，請使用 'linear' 或 'constant'")

    logger.info("執行 %s detrend", method)
    patch = patch.detrend(dim="time", type=method)
    logger.info("detrend 完成，shape: %s", np.asarray(patch.data).shape)
    return patch