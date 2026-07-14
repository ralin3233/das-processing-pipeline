# src/das_pipeline/preprocessing/decimate.py

import logging
from typing import Optional
import numpy as np
import dascore as dc

logger = logging.getLogger(__name__)


def decimate(patch: dc.Patch, factor: Optional[int] = None) -> dc.Patch:
    """對時間軸進行降採樣。

    使用 DASCore 內建的 decimate，已包含 anti-aliasing 濾波。

    Parameters
    ----------
    patch : dc.Patch
        原始 Patch。
    factor : int or None
        降採樣倍數（整數），如 4 表示降到 1/4 取樣率。
        None 表示跳過降採樣。

    Returns
    -------
    dc.Patch
        降採樣後的 Patch。
    """
    if factor is None:
        logger.info("decimate 已跳過")
        return patch

    if not isinstance(factor, int) or factor < 2:
        raise ValueError(f"降採樣因子必須為 ≥ 2 的整數，收到: {factor}")

    logger.info("執行 decimate，因子: %s", factor)
    patch = patch.decimate(time=factor)
    logger.info("decimate 完成，新 shape: %s", np.asarray(patch.data).shape)
    return patch