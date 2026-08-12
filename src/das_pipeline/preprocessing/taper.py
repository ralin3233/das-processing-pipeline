"""Taper 模組：對時間軸頭尾進行 cosine taper，減少濾波邊緣效應。"""

import logging
from typing import Optional
import numpy as np
import dascore as dc

logger = logging.getLogger(__name__)


def taper(patch: dc.Patch, taper_ratio: Optional[float] = None) -> dc.Patch:
    """對時間軸進行 taper。

    Parameters
    ----------
    patch : dc.Patch
        原始 Patch。
    taper_ratio : float or None
        taper 比例（頭尾各佔總長度的比例），需滿足 0 < taper_ratio < 0.5。
        None 表示跳過 taper。

    Returns
    -------
    dc.Patch
        taper 後的 Patch。

    Raises
    ------
    ValueError
        當 taper_ratio 不在 (0, 0.5) 範圍內時。
    """
    if taper_ratio is None:
        logger.info("taper 已跳過")
        return patch

    if not (0 < taper_ratio < 0.5):
        raise ValueError(
            f"無效的 taper_ratio: {taper_ratio}，需滿足 0 < taper_ratio < 0.5"
        )

    logger.info("執行 taper: ratio=%.3f", taper_ratio)
    patch = patch.taper(time=taper_ratio)
    logger.info("taper 完成，shape: %s", np.asarray(patch.data).shape)
    return patch