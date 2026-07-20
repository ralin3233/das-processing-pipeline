"""
Overlay plot of multiple teleseismic amplification CSVs with median curve.

Each CSV is expected to have columns: channel_index, amplification, reference_amplitude
(as produced by `das-pipeline amplification --csv`).
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a single amplification CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to a teleseismic_amplification.csv.

    Returns
    -------
    channel_indices : ndarray (n_channels,)
    amplification : ndarray (n_channels,)
    """
    channels: list[int] = []
    amps: list[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            channels.append(int(row["channel_index"]))
            amps.append(float(row["amplification"]))

    return np.array(channels), np.array(amps)


def plot_overlay(
    csv_paths: list[Path],
    labels: Optional[list[str]] = None,
    save_dir: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
    show: bool = True,
) -> Optional[Path]:
    """
    Plot overlay of multiple amplification curves with a median line.

    Parameters
    ----------
    csv_paths : list[Path]
        List of paths to CSV files from ``das-pipeline amplification --csv``.
    labels : list[str], optional
        Legend labels for each CSV. If None, uses the file stem (name without extension).
    save_dir : Path, optional
        If provided, save the figure to this directory as ``amplification_overlay.png``.
    title : str, optional
        Plot title. Default: "Teleseismic Amplification Overlay".
    figsize : tuple[float, float]
        Figure size (width, height) in inches.
    dpi : int
        Figure resolution.
    show : bool
        Whether to display the plot interactively. If False and save_dir is set,
        the figure will be closed after saving.

    Returns
    -------
    Path or None
        Path to the saved figure if save_dir is provided, otherwise None.
    """
    n_files = len(csv_paths)
    if n_files == 0:
        logger.warning("No CSV files provided.")
        return None

    if labels is None:
        labels = [Path(p).stem for p in csv_paths]
    elif len(labels) != n_files:
        logger.warning(
            "Number of labels (%d) does not match number of CSV files (%d). "
            "Using file stems instead.",
            len(labels), n_files,
        )
        labels = [Path(p).stem for p in csv_paths]

    # Load all CSV data
    all_channels: list[np.ndarray] = []
    all_amps: list[np.ndarray] = []

    for i, csv_path in enumerate(csv_paths):
        try:
            channels, amps = _load_csv(csv_path)
            all_channels.append(channels)
            all_amps.append(amps)
            logger.info("Loaded %s: %d channels", csv_path.name, len(channels))
        except Exception as e:
            logger.error("Failed to load %s: %s", csv_path, e)
            continue

    if len(all_amps) == 0:
        logger.warning("No valid data loaded.")
        return None

    # Compute intersection of channel indices across all CSVs
    common_channels = all_channels[0]
    for ch in all_channels[1:]:
        common_channels = np.intersect1d(common_channels, ch)

    if len(common_channels) == 0:
        logger.warning("No common channel indices across CSVs.")
        return None

    logger.info("Common channel indices: %d channels", len(common_channels))

    # Align each event's amplification to the common channel indices
    aligned_amps: list[np.ndarray] = []
    for channels, amps in zip(all_channels, all_amps):
        mask = np.isin(channels, common_channels)
        aligned = amps[mask]
        aligned_amps.append(aligned)

    ref_channels = common_channels
    amp_stack = np.stack(aligned_amps, axis=0)       # (n_events, n_channels)
    median_amps = np.median(amp_stack, axis=0)       # (n_channels,)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Use colormap for event curves
    cmap = plt.colormaps["viridis"].resampled(len(aligned_amps))

    for i, amps in enumerate(aligned_amps):
        color = cmap(i)
        ax.plot(
            amps, ref_channels,
            color=color, linewidth=0.8, alpha=0.7,
            label=f"{labels[i]}",
        )

    # Median curve — bold dashed red line
    ax.plot(
        median_amps, ref_channels,
        color="red", linewidth=2.0, linestyle="--",
        label="Median",
    )

    # Baseline at amplification = 1
    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7,
               label="Baseline (amp=1.0)")

    ax.set_xlabel("Normalized Amplitude")
    ax.set_ylabel("Channel Index (0 = wellhead)")
    ax.set_title(title or "Teleseismic Amplification Overlay")

    # Invert y-axis so that wellhead is at the top
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    # Save or show
    saved_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / "amplification_overlay.png"
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
        logger.info("Overlay plot saved to %s", out_path)
        saved_path = out_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path