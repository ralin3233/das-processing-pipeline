"""MiniSEED 讀取模組：使用 ObsPy 讀取原始 MiniSEED 檔案並轉為 DASCore Patch。

處理斷訊 gap 策略：同 station 的 trace 強制合併，缺口以 NaN 填補。
"""

from pathlib import Path
import dascore as dc
import numpy as np
import obspy
from das_pipeline.config import DataConfig
import logging

logger = logging.getLogger(__name__)


def load(config: DataConfig) -> dc.Patch:
    """讀取 config.input_dir 底下的 MiniSEED 檔案，回傳 DASCore Patch

    處理斷訊（gap）的策略：
    - 同一個 station 若中間斷訊，obspy.read() 會回傳多個 Trace。
      若不先 merge，會被誤判成多個獨立 channel，造成 distance 軸重複、
      channel 數量錯位。這裡強制合併同一 station 的所有 Trace，
      缺口處以 NaN 填補，讓「缺失值」誠實地變成 NaN，而不是幽靈 channel。
    - merge 後會依 station 編號排序，確保 channel 按深度遞增排列。
    """
    input_dir = Path(config.input_dir)
    files = sorted(input_dir.glob(config.file_pattern))

    if not files:
        logger.error("在 %s 找不到符合 %s 的檔案", input_dir, config.file_pattern)
        raise FileNotFoundError(
            f"在 {input_dir} 找不到符合 {config.file_pattern} 的檔案"
        )

    # ==========================================
    # 步驟 1：用 ObsPy 讀取所有 mseed
    # ==========================================
    st = obspy.read(str(input_dir / config.file_pattern))

    # ── 合併同 station 的 trace（斷訊補 NaN）──
    n_stations_before = len({tr.stats.station for tr in st})
    st.merge(method=1, fill_value=np.nan)

    if len(st) != n_stations_before:
        raise ValueError(
            f"MiniSEED 合併後 Trace 數量 ({len(st)}) 與唯一 station 數量 "
            f"({n_stations_before}) 不一致，可能有非預期的資料結構，請檢查來源檔案。"
        )

    n_gaps = sum(
        1 for tr in st
        if np.issubdtype(tr.data.dtype, np.floating) and np.isnan(tr.data).any()
    )
    if n_gaps > 0:
        logger.warning(
            "偵測到 %d 個 channel 存在斷訊（已補 NaN），"
            "後續 pipeline 將對 NaN 進行線性插值後再處理。",
            n_gaps,
        )

    # ── 依 station 編號排序，確保 channel 按深度遞增 ──
    st.traces.sort(key=lambda tr: int(tr.stats.station))

    # 讀取所有通道
    min_length = min([len(tr.data) for tr in st])
    # 提取 2D 數據 (通道數, 時間樣本數)，顯式轉 float64 以確保 NaN 可被標記
    data_2d = np.vstack([
        np.asarray(tr.data[:min_length], dtype=np.float64) for tr in st
    ])

    # ==========================================
    # 步驟 2：建立時間軸與幾何座標軸
    # ==========================================
    tr_ref = st[0]
    start_time = np.datetime64(tr_ref.stats.starttime.datetime)
    delta = tr_ref.stats.delta
    time_axis = start_time + np.arange(data_2d.shape[1]) * np.timedelta64(
        int(delta * 1e6), "us"
    )

    # 讀到的站號作為距離軸（此時已依 station 編號排序，保證遞增）
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
