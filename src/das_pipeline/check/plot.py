# src/das_pipeline/check/plot.py
"""Plotting for coverage check status grid."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

import typer

from das_pipeline.check.coverage import (
    STATE_UNCOVERED,
    STATE_DATA,
    STATE_NAN,
    STATE_OVERLAP,
    STATE_COLORS,
)


def plot_status_grid(
    status: np.ndarray,
    time_edges: np.ndarray,
    dist_axis: np.ndarray,
    file_count: int,
    time_bin_sec: float,
    title: Optional[str] = None,
    save: Optional[Path] = None,
    fmt: str = "png",
    dpi: int = 150,
    no_display: bool = False,
) -> None:
    """Draw the coverage status grid.

    Parameters
    ----------
    status : np.ndarray
        Status grid, shape (n_dist, n_time_bins), dtype=np.int8.
    time_edges : np.ndarray
        Time bin edges (datetime64[ns]).
    dist_axis : np.ndarray
        Unique distance values.
    file_count : int
        Number of source files (for title).
    time_bin_sec : float
        Time bin size in seconds (for title).
    title : str, optional
        Custom chart title.
    save : Path, optional
        Directory to save the figure.
    fmt : str
        Output format (png, pdf, svg).
    dpi : int
        Image resolution.
    no_display : bool
        Suppress interactive display.
    """
    n_dist = len(dist_axis)

    cmap = ListedColormap(STATE_COLORS)
    norm = BoundaryNorm(
        [STATE_UNCOVERED, STATE_DATA, STATE_NAN, STATE_OVERLAP, STATE_OVERLAP + 1],
        ncolors=4,
    )

    n_time = status.shape[1]
    fig_w = min(max(n_time / 80, 10), 24)
    fig_h = min(max(n_dist / 40, 4), 16)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    t_edges_num = time_edges.astype("int64") / 1e9
    ax.pcolormesh(
        t_edges_num,
        np.arange(n_dist + 1),
        status,
        cmap=cmap,
        norm=norm,
        rasterized=True,
        shading="flat",
    )

    t_ref = time_edges[0].astype("int64") / 1e9
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: datetime.fromtimestamp(x, tz=timezone.utc).strftime(
                "%H:%M:%S"
            )
        )
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Distance (m)")

    if n_dist <= 50:
        y_step = max(1, n_dist // 20)
        y_ticks = np.arange(0, n_dist, y_step)
        ax.set_yticks(y_ticks + 0.5)
        ax.set_yticklabels([f"{dist_axis[i]:.0f}" for i in y_ticks])
    else:
        y_step = max(1, n_dist // 10)
        y_ticks = np.arange(0, n_dist, y_step)
        ax.set_yticks(y_ticks + 0.5)
        ax.set_yticklabels([f"{dist_axis[i]:.0f}" for i in y_ticks])
    ax.set_ylim(n_dist, 0)

    legend_patches = [
        Patch(color=STATE_COLORS[STATE_DATA], label="Data"),
        Patch(color=STATE_COLORS[STATE_NAN], label="NaN"),
        Patch(color=STATE_COLORS[STATE_UNCOVERED], label="Uncovered"),
        Patch(color=STATE_COLORS[STATE_OVERLAP], label="Overlap"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(
            f"Data Coverage ({file_count} files, time bin={time_bin_sec:.3f}s)",
            fontsize=11,
        )

    total_cells = status.size
    n_data = int(np.sum(status == STATE_DATA))
    n_nan = int(np.sum(status == STATE_NAN))
    n_uncov = int(np.sum(status == STATE_UNCOVERED))
    n_overlap = int(np.sum(status == STATE_OVERLAP))

    info_text = (
        f"Data: {n_data} ({100 * n_data / total_cells:.1f}%) | "
        f"NaN: {n_nan} ({100 * n_nan / total_cells:.1f}%) | "
        f"Overlap: {n_overlap} ({100 * n_overlap / total_cells:.1f}%) | "
        f"Uncovered: {n_uncov} ({100 * n_uncov / total_cells:.1f}%)"
    )
    ax.text(
        0.5,
        -0.12,
        info_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()

    if save:
        save_dir = Path(save)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"coverage.{fmt}"
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        typer.echo(f"✅ 已儲存: {save_path}")
        if no_display:
            plt.close(fig)
    if not save or not no_display:
        plt.show()
