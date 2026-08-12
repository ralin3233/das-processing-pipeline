"""視覺化模組：瀑布圖、F-K 頻譜圖、Spectrogram 與 Patch 合併工具。

提供：
- ``plot_waterfall``: time × distance 振幅瀑布圖
- ``plot_fk_spectrum``: F-K frequency × wavenumber 功率譜
- ``plot_spectrogram``: 單通道 time × frequency PSD
- ``merge_patches``: 多 chunk 合併
"""

from das_pipeline.visualization.waterfall import plot_waterfall
from das_pipeline.visualization.fk import plot_fk_spectrum
from das_pipeline.visualization.spectrogram import plot_spectrogram
from das_pipeline.visualization.merge import merge_patches

__all__ = [
    "plot_waterfall",
    "plot_fk_spectrum",
    "plot_spectrogram",
    "merge_patches",
]