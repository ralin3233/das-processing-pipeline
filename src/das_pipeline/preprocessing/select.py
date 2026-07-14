# src/das_pipeline/preprocessing/select.py

import logging
from typing import Optional
import numpy as np
import dascore as dc

logger = logging.getLogger(__name__)


def select(
    patch: dc.Patch,
    time_range: Optional[tuple[float, float]] = None,
    distance_range: Optional[tuple[float, float]] = None,
) -> dc.Patch:
    """選取指定的時間與距離範圍。

    Parameters
    ----------
    patch : dc.Patch
        原始 Patch。
    time_range : tuple[float, float] or None
        時間範圍（秒），如 (0.0, 60.0)。 None 表示不裁剪時間軸。
    distance_range : tuple[float, float] or None
        距離/通道範圍，如 (0, 500)。 None 表示不裁剪距離軸。

    Returns
    -------
    dc.Patch
        選取後的 Patch。
    """
    select_kwargs = {}

    if time_range is not None:
        select_kwargs["time"] = time_range
        logger.info("選取時間範圍: %s", time_range)

    if distance_range is not None:
        select_kwargs["distance"] = distance_range
        logger.info("選取距離範圍: %s", distance_range)

    if not select_kwargs:
        logger.info("未指定任何選取範圍，直接回傳原始 Patch")
        return patch

    patch = patch.select(**select_kwargs)
    logger.info("選取完成，新 shape: %s", np.asarray(patch.data).shape)
    return patch