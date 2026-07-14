# src/das_pipeline/preprocessing/pipeline.py

import logging

import dascore as dc

from das_pipeline.config import PreprocessingConfig
from das_pipeline.preprocessing.select import select
from das_pipeline.preprocessing.detrend import detrend
from das_pipeline.preprocessing.bandpass import bandpass
from das_pipeline.preprocessing.decimate import decimate

logger = logging.getLogger(__name__)


def run_preprocessing(patch: dc.Patch, config: PreprocessingConfig) -> dc.Patch:
    """依序執行前處理各步驟：select → detrend → bandpass → decimate。

    Parameters
    ----------
    patch : dc.Patch
        輸入的 Patch。
    config : PreprocessingConfig
        前處理設定。

    Returns
    -------
    dc.Patch
        處理後的 Patch。
    """
    # 1. 選取時間/距離範圍
    patch = select(patch, time_range=config.time_range, distance_range=config.distance_range)

    # 2. 去趨勢
    patch = detrend(patch, method=config.detrend)

    # 3. 帶通濾波
    patch = bandpass(patch, freq_range=config.bandpass)

    # 4. 降採樣
    patch = decimate(patch, factor=config.decimate_factor)

    return patch