# src/das_pipeline/visualization/fk.py

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _get_spatial_axis(patch, channel_spacing: Optional[float] = None):
    """取得正確的空間軸與標籤。

    Parameters
    ----------
    patch : dc.Patch
        DAS 資料。
    channel_spacing : float, optional
        相鄰通道的物理距離（m）。若提供，將 channel index 轉換為實際距離。

    Returns
    -------
    dist_vals : np.ndarray
        空間軸數值。
    label : str
        Y 軸標籤。
    """
    dist_coord = patch.coords["distance"]
    n_channels = len(dist_coord)

    if channel_spacing is not None:
        dist_vals = np.arange(n_channels, dtype=float) * channel_spacing
        label = "Wavenumber (cycles/m)"
    elif "m" in str(dist_coord.units):
        dist_vals = dist_coord.values
        label = "Wavenumber (cycles/m)"
    else:
        dist_vals = dist_coord.values
        label = "Wavenumber (cycles/channel)"

    return dist_vals, label


def plot_fk_spectrum(
    patch,
    ax=None,
    *,
    channel_spacing: Optional[float] = None,
    freq_range: Optional[Tuple[float, float]] = None,
    db_range: Optional[Tuple[float, float]] = None,
    colormap: str = "viridis",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    show_colorbar: bool = True,
):
    """繪製 F-K 頻譜圖 (frequency × wavenumber 功率譜)。

    Parameters
    ----------
    patch : dc.Patch
        輸入的 DAS 資料。
    ax : matplotlib.axes.Axes, optional
        已存在的 Axes，若無則自動建立。
    channel_spacing : float, optional
        相鄰通道的物理距離（m）。用於將 channel index 轉換為 wavenumber。
    freq_range : tuple of float, optional
        頻率範圍 [low, high] Hz。
    db_range : tuple of float, optional
        dB 顯示範圍 [min, max]，預設自動。
    colormap : str
        matplotlib colormap，預設 "viridis"。
    title : str, optional
        圖表標題。
    figsize : tuple
        圖表大小，預設 (8, 6)。
    show_colorbar : bool
        是否顯示 colorbar，預設 True。

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = patch.data
    coords = patch.coords
    time_coord = coords["time"]
    dist_vals, dist_label = _get_spatial_axis(patch, channel_spacing)

    # 確保 data 為 (time, distance) 順序
    if patch.dims.index("time") != 0:
        data = data.T

    # dt 與 dx
    dt = np.timedelta64(time_coord[1] - time_coord[0], "s") / np.timedelta64(1, "s")
    dx = dist_vals[1] - dist_vals[0] if len(dist_vals) > 1 else 1.0

    # 2D FFT
    fft_data = np.fft.fft2(data)
    fft_data = np.fft.fftshift(fft_data)

    # 功率譜 (dB)
    power = np.abs(fft_data) ** 2
    power_db = 10.0 * np.log10(power + 1e-30)

    # 頻率軸
    n_time = data.shape[0]
    freq = np.fft.fftshift(np.fft.fftfreq(n_time, d=dt))
    # 只取正頻率
    pos_mask = freq >= 0
    freq_pos = freq[pos_mask]
    power_db_pos = power_db[pos_mask, :]

    # wavenumber 軸
    n_dist = data.shape[1]
    wavenumber = np.fft.fftshift(np.fft.fftfreq(n_dist, d=float(dx)))

    # 頻率範圍篩選
    if freq_range:
        f_mask = (freq_pos >= freq_range[0]) & (freq_pos <= freq_range[1])
        freq_pos = freq_pos[f_mask]
        power_db_pos = power_db_pos[f_mask, :]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.pcolormesh(
        wavenumber,
        freq_pos,
        power_db_pos,
        cmap=colormap,
        shading="auto",
    )
    if db_range:
        im.set_clim(db_range[0], db_range[1])

    ax.set_xlabel(dist_label)
    ax.set_ylabel("Frequency (Hz)")
    ax.axvline(0, color="white", linestyle="--", linewidth=0.5, alpha=0.5)

    if title:
        ax.set_title(title)

    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Power (dB)")

    fig.tight_layout()
    return fig