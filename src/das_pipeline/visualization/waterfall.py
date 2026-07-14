# src/das_pipeline/visualization/waterfall.py

import logging
from typing import Optional, Tuple
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_waterfall(
    patch,
    ax=None,
    *,
    time_range: Optional[Tuple[str, str]] = None,
    distance_range: Optional[Tuple[float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    clip_percentile: float = 99.0,
    colormap: str = "seismic",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    show_colorbar: bool = True,
):
    """繪製 DAS 瀑布圖 (time × distance amplitude map)。

    Parameters
    ----------
    patch : dc.Patch
        輸入的 DAS 資料。
    ax : matplotlib.axes.Axes, optional
        已存在的 Axes，若無則自動建立。
    time_range : tuple of str, optional
        時間範圍 ["start", "end"]。
    distance_range : tuple of float, optional
        距離/通道範圍 [start, end]。
    vmin, vmax : float, optional
        色彩映射的上下限，若未指定則自動以百分位數推斷。
    clip_percentile : float
        用於自動推斷 vmin/vmax 的百分位數，預設 99.0。
    colormap : str
        matplotlib colormap 名稱，預設 "seismic"。
    title : str, optional
        圖表標題。
    figsize : tuple
        圖表大小，預設 (12, 5)。
    show_colorbar : bool
        是否顯示 colorbar，預設 True。

    Returns
    -------
    matplotlib.figure.Figure
    """
    # 選取時間/距離範圍
    data = patch.data
    coords = patch.coords

    time_vals = coords.get_array("time")
    dist_vals = coords.get_array("distance")

    # 時間範圍篩選
    if time_range:
        mask = (time_vals >= np.datetime64(time_range[0])) & (
            time_vals <= np.datetime64(time_range[1])
        )
        if patch.dims.index("time") == 0:
            data = data[mask]
        else:
            data = data[:, mask]
        time_vals = time_vals[mask]

    # 距離範圍篩選
    if distance_range:
        mask = (dist_vals >= distance_range[0]) & (
            dist_vals <= distance_range[1]
        )
        axis = patch.dims.index("distance")
        if axis == 0:
            data = data[mask]
        else:
            data = data[:, mask]
        dist_vals = dist_vals[mask]

    # 自動 vmin/vmax
    if vmin is None or vmax is None:
        abs_data = np.abs(data)
        if vmin is None:
            vmin = -float(np.percentile(abs_data, clip_percentile))
        if vmax is None:
            vmax = float(np.percentile(abs_data, clip_percentile))

    # 確保 data 為 (time, distance) 順序
    if patch.dims.index("time") != 0:
        data = data.T

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.pcolormesh(
        time_vals,
        dist_vals,
        data.T,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )

    ax.set_xlabel("Time")
    if "m" in str(coords.get_coord("distance").units):
        ax.set_ylabel("Distance (m)")
    else:
        ax.set_ylabel("Channel index")

    if title:
        ax.set_title(title)

    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Amplitude")

    fig.tight_layout()
    return fig