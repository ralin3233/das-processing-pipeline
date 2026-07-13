from pathlib import Path
import dascore as dc
import numpy as np
import obspy
from das_pipeline.config import DataConfig
import logging

logger = logging.getLogger(__name__)

def load(config: DataConfig) -> dc.Patch:
    # TODO: 用 dascore.read 或 obspy 讀 config.input_dir 底下的檔案
    # 先處理你熟悉的那批 MiniSEED，回傳一個 dascore.Patch
    """讀取 config.input_dir 底下的 MiniSEED 檔案，回傳 DASCore Patch"""
    
    input_dir = Path(config.input_dir)
    files = sorted(input_dir.glob("*.mseed"))
    
    if not files:
        logger.error(f"在 {input_dir} 找不到符合 *.mseed 的檔案")
        raise FileNotFoundError(
            f"在 {input_dir} 找不到符合 *.mseed 的檔案"
        )
    
    # ==========================================
    # 步驟 1：用 ObsPy 讀取並排序 mseed
    # ==========================================
    # 讀取某一小時內、所有通道的 mseed (井A)
    st = obspy.read(str(input_dir / "*.mseed"))

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

    # 讀到的站號（僅作為識別用途，不再當作距離）
    station_axis = np.array([int(tr.stats.station) for tr in st])
    
    # 如果你要的是「深度」而不是「高程」，通常要用基準面減去 elevation
    # 例如：depth_axis = reference_elevation - elev_axis
    # 如果 elevation 本身已經照順序遞減、且就是你要的距離軸，直接用它即可
    distance_axis = station_axis  # ← 這裡先用站號作為距離軸，後續會改成真正的深度值

    # ==========================================
    # 步驟 3：建立 Patch —— distance 用真正的深度值
    # ==========================================
    patch = dc.Patch(
        data=data_2d,
        coords={
            "time": time_axis,
            "distance": distance_axis,        # ← 主要空間座標改成真實深度/距離
        },
        dims=("distance", "time"),
    )

    return patch

