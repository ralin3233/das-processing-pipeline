# src/das_pipeline/visualization/spectrogram.py

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


def plot_spectrogram(
    patch,
    ax=None,
    *,
    channel: Optional[int] = None,
    freq_range: Optional[Tuple[float, float]] = None,
    db_range: Optional[Tuple[float, float]] = None,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    colormap: str = "viridis",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    show_colorbar: bool = True,
):
    """繪製 DAS 時頻圖 (spectrogram, time × frequency power spectral density)。

    預設選取中間通道進行 STFT 分析，也可指定特定通道。

    Parameters
    ----------
    patch : dc.Patch
        輸入的 DAS 資料。
    ax : matplotlib.axes.Axes, optional
        已存在的 Axes，若無則自動建立。
    channel : int, optional
        要分析的通道索引。若為 None 則取中間通道。
    freq_range : tuple of float, optional
        頻率範圍 [low, high] Hz。
    db_range : tuple of float, optional
        dB 顯示範圍 [min, max]，預設自動。
    nperseg : int
        STFT 的 segment 長度（樣本數），預設 256。
    noverlap : int, optional
        STFT 的 overlap 樣本數，預設 nperseg // 2。
    colormap : str
        matplotlib colormap，預設 "viridis"。
    title : str, optional
        圖表標題。
    figsize : tuple
        圖表大小，預設 (10, 5)。
    show_colorbar : bool
        是否顯示 colorbar，預設 True。

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = patch.data
    coords = patch.coords
    time_coord = coords.get_coord("time")
    dist_coord = coords.get_coord("distance")

    # 確保 data 為 (time, distance) 順序
    if patch.dims.index("time") != 0:
        data = data.T

    # 選取通道
    n_channels = data.shape[1]
    if channel is None:
        channel = n_channels // 2
    elif channel < 0 or channel >= n_channels:
        raise ValueError(
            f"channel {channel} 超出範圍 [0, {n_channels - 1}]"
        )

    ts = data[:, channel]

    # dt (秒)
    dt = np.timedelta64(time_coord[1] - time_coord[0], "s") / np.timedelta64(1, "s")
    fs = 1.0 / float(dt)

    # STFT
    if noverlap is None:
        noverlap = nperseg // 2

    f, t, Sxx = scipy_signal.spectrogram(
        ts,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
    )

    # dB scale
    Sxx_db = 10.0 * np.log10(Sxx + 1e-30)

    # 頻率範圍篩選
    if freq_range:
        f_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f = f[f_mask]
        Sxx_db = Sxx_db[f_mask, :]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.pcolormesh(
        t,
        f,
        Sxx_db,
        cmap=colormap,
        shading="auto",
    )
    if db_range:
        im.set_clim(db_range[0], db_range[1])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    channel_label = (
        coords.get_array("distance")[channel]
        if "m" in str(dist_coord.units)
        else f"ch{channel}"
    )

    if title:
        ax.set_title(f"{title} (channel {channel_label})")
    else:
        ax.set_title(f"Spectrogram (channel {channel_label})")

    if show_colorbar:
        plt.colorbar(im, ax=ax, label="PSD (dB/Hz)")

    fig.tight_layout()
    return fig