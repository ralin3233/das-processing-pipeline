"""遠震分析模組：放大倍率計算、SNR 分析與視覺化。

提供：
- ``compute_amplification``: 遠震地層放大效應
- ``compute_channel_snr``: 單一 channel 訊雜比
- ``compute_power``: 平均功率計算
- ``plot_amplification``: 放大倍率繪圖
"""

from das_pipeline.teleseismic.amplification import compute_amplification
from das_pipeline.teleseismic.snr import compute_channel_snr, compute_power
from das_pipeline.teleseismic.visualization import plot_amplification

__all__ = [
    "compute_amplification",
    "compute_channel_snr",
    "compute_power",
    "plot_amplification",
]