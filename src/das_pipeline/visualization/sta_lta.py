"""STA/LTA summary visualization."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from das_pipeline.config import StaLtaConfig


def _median_by_time(patch) -> tuple[np.ndarray, np.ndarray]:
    """Return time coordinates and channel-wise median values."""
    data = np.asarray(patch.data)
    time_axis = patch.dims.index("time")
    if time_axis != 0:
        data = data.T

    median = np.ma.median(np.ma.masked_invalid(data), axis=1).filled(np.nan)
    time_values = np.asarray(patch.get_coord("time").values).ravel()
    return time_values, np.asarray(median, dtype=float)


def plot_sta_lta(
    ratio_patch,
    *,
    config: Optional[StaLtaConfig] = None,
    events: Optional[list[dict]] = None,
    title: str = "STA/LTA Ratio Channel Median",
    figsize: tuple[float, float] = (14, 6),
):
    """Plot the channel-median STA/LTA ratio time series."""
    ratio_time, ratio_median = _median_by_time(ratio_patch)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        ratio_time, ratio_median, color="tab:red", label="Ratio median",
    )

    if config is not None:
        ax.axhline(
            config.trigger_threshold, color="tab:red", linestyle="--",
            linewidth=1.0, alpha=0.7, label="Trigger",
        )
        ax.axhline(
            config.detrigger_threshold, color="tab:purple", linestyle=":",
            linewidth=1.0, alpha=0.7, label="Detrigger",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("STA/LTA ratio")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
