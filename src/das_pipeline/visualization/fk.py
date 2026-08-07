# src/das_pipeline/visualization/fk.py

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


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

    使用 DASCore 的 :meth:`dascore.Patch.dft` 計算 2D 傅立葉轉換，
    並以 :func:`dascore.viz.specplot` 繪圖（其內部自動處理頻率/波數
    座標與單位標示）。

    Parameters
    ----------
    patch : dc.Patch
        輸入的 DAS 資料（time × distance，distance 預設單位為公尺）。
    ax : matplotlib.axes.Axes, optional
        已存在的 Axes，若無則自動建立。
    channel_spacing : float, optional
        相鄰通道的物理距離（m）。若提供，將 channel index 轉換為實際距離。
    freq_range : tuple of float, optional
        頻率範圍 [low, high] Hz。
    db_range : tuple of float, optional
        dB 顯示範圍 [min, max]，預設自動（IQR fence）。
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
    import dascore as dc

    # --- 確保 distance 座標為公尺（預設輸入即為公尺）---
    dist_coord = patch.get_coord("distance")
    dist_units = str(getattr(dist_coord, "units", "") or "")
    if channel_spacing is not None:
        # channel index → 實際物理距離（m）
        n_channels = len(dist_coord)
        dist_vals = np.arange(n_channels, dtype=float) * channel_spacing
        patch = patch.update_coords(
            distance=dc.get_coord(values=dist_vals, units="m")
        )
    elif "m" not in dist_units:
        # 未標示單位 → 依預設視為公尺
        patch = patch.update_coords(
            distance=dc.get_coord(values=patch.get_array("distance"), units="m")
        )

    # --- 處理 NaN channel：沿 distance 軸線性內插（FFT 無法容忍 NaN）---
    data = np.array(patch.data, copy=True)
    if np.any(np.isnan(data)):
        channel_axis = patch.dims.index("distance")
        dist_vals = patch.get_array("distance")
        other_axis = 1 - channel_axis
        n_channels = data.shape[channel_axis]
        n_interpolated = 0
        n_all_nan = 0
        for i in range(data.shape[other_axis]):
            trace = data[:, i] if channel_axis == 0 else data[i, :]
            nan_mask = np.isnan(trace)
            if np.any(nan_mask):
                valid_mask = ~nan_mask
                if np.any(valid_mask):
                    trace[nan_mask] = np.interp(
                        dist_vals[nan_mask], dist_vals[valid_mask], trace[valid_mask]
                    )
                    n_interpolated += 1
                else:
                    trace[:] = 0.0
                    n_all_nan += 1
        logger.info(
            "FK: %d / %d channels 含有 NaN：%d 個已沿 distance 軸線性內插，"
            "%d 個全 NaN channel 已以 0 替代。"
            "（內插僅供 FFT 運算，不影響原始儲存資料。）",
            n_interpolated + n_all_nan, n_channels, n_interpolated, n_all_nan,
        )
        patch = patch.new(data=data)

    # --- 2D FFT（DASCore 自動處理頻率/波數座標與單位）---
    fk_patch = patch.dft(("time", "distance")).abs()

    # --- 功率譜 (dB) ---
    power_db = 10.0 * np.log10(fk_patch.data**2 + 1e-30)
    fk_patch = fk_patch.new(data=power_db)

    # --- 只取正頻率 ---
    fk_patch = fk_patch.select(ft_time=(0, None))

    # --- 頻率範圍篩選 ---
    if freq_range:
        fk_patch = fk_patch.select(ft_time=(freq_range[0], freq_range[1]))

    # --- 轉置以匹配原始 FK 圖方向：X=Wavenumber, Y=Frequency ---
    fk_patch = fk_patch.transpose("ft_time", "ft_distance")

    # --- 繪圖 ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax = dc.viz.specplot(
        fk_patch,
        ax=ax,
        cmap=colormap,
        cbar=show_colorbar,
        scale=db_range if db_range else None,
        scale_type="absolute" if db_range else "relative",
        show=False,
    )

    # 標示零波數
    ax.axvline(0, color="white", linestyle="--", linewidth=0.5, alpha=0.5)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig
