from pathlib import Path
import dascore as dc
import numpy as np
import obspy
from das_pipeline.config import DataConfig
import logging

logger = logging.getLogger(__name__)

def load(config: DataConfig) -> dc.Patch:
    """讀取 config.input_dir 底下的 MiniSEED 檔案，回傳 DASCore Patch"""
    
    input_dir = Path(config.input_dir)
    files = sorted(input_dir.glob("*.mseed"))
    
    if not files:
        logger.error(f"在 {input_dir} 找不到符合 *.mseed 的檔案")
        raise FileNotFoundError(
            f"在 {input_dir} 找不到符合 *.mseed 的檔案"
        )
    
    # ==========================================
    # 步驟 1：用 ObsPy 讀取所有 mseed
    # ==========================================
    st = obspy.read(str(input_dir / config.file_pattern))

    # 讀取所有通道
    min_length = min([len(tr.data) for tr in st])
    # 提取 2D 數據 (通道數, 時間樣本數)
    data_2d = np.vstack([tr.data[:min_length] for tr in st])

    # ==========================================
    # 步驟 2：建立時間軸與幾何座標軸
    # ==========================================
    tr_ref = st[0]
    start_time = np.datetime64(tr_ref.stats.starttime.datetime)
    delta = tr_ref.stats.delta
    time_axis = start_time + np.arange(data_2d.shape[1]) * np.timedelta64(
        int(delta * 1e6), "us"
    )

    # 讀到的站號作為距離軸
    distance_axis = np.array([int(tr.stats.station) for tr in st])

    # ==========================================
    # 步驟 3：建立 Patch
    # ==========================================
    patch = dc.Patch(
        data=data_2d,
        coords={
            "time": time_axis,
            "distance": distance_axis,
        },
        dims=("distance", "time"),
    )

    return patch