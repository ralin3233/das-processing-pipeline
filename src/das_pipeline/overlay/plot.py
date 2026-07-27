"""
Overlay plot of multiple teleseismic amplification CSVs with median curve.

Each CSV is expected to have columns: distance_m, amplification, reference_amplitude
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
    distances : ndarray (n_channels,)
        Distance in metres for each channel.
    amplification : ndarray (n_channels,)
    """
    distances: list[float] = []
    amps: list[float] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            distances.append(float(row["distance_m"]))
            amps.append(float(row["amplification"]))

    return np.array(distances), np.array(amps)


def plot_overlay(
    csv_paths: list[Path],
    labels: Optional[list[str]] = None,
    save_dir: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
    show: bool = True,
    csv_output: bool = False,
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
    all_distances: list[np.ndarray] = []
    all_amps: list[np.ndarray] = []

    for i, csv_path in enumerate(csv_paths):
        try:
            distances, amps = _load_csv(csv_path)
            all_distances.append(distances)
            all_amps.append(amps)
            logger.info("Loaded %s: %d channels", csv_path.name, len(distances))
        except Exception as e:
            logger.error("Failed to load %s: %s", csv_path, e)
            continue

    if len(all_amps) == 0:
        logger.warning("No valid data loaded.")
        return None

    # Use the distance axis from the first CSV as the common reference
    ref_distances = all_distances[0]
    n_common = len(ref_distances)
    logger.info("Reference distances: %d points", n_common)

    # Interpolate each event's amplification onto the common distance axis
    interp_amps: list[np.ndarray] = []
    for distances, amps in zip(all_distances, all_amps):
        interp = np.interp(ref_distances, distances, amps)
        interp_amps.append(interp)

    amp_stack = np.stack(interp_amps, axis=0)    # (n_events, n_channels)
    median_amps = np.median(amp_stack, axis=0)   # (n_channels,)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # Use colormap for event curves
    cmap = plt.colormaps["viridis"].resampled(len(all_amps))

    for i, (distances, amps) in enumerate(zip(all_distances, all_amps)):
        color = cmap(i)
        ax.plot(
            amps, distances,
            color=color, linewidth=0.8, alpha=0.7,
            label=f"{labels[i]}",
        )

    # Median curve — bold dashed red line
    ax.plot(
        median_amps, ref_distances,
        color="red", linewidth=2.0, linestyle="--",
        label="Median",
    )

    # Baseline at amplification = 1
    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7,
               label="Baseline (amp=1.0)")

    ax.set_xlabel("Normalized Amplitude")
    ax.set_ylabel("Distance (m)")
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

        # CSV output (data for external GUI tools)
        if csv_output:
            csv_path = save_dir / "amplification_overlay.csv"
            header = ["distance_m"] + labels + ["median"]
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for idx, dist in enumerate(ref_distances):
                    row = [f"{dist:.2f}"]
                    for event_amps in interp_amps:
                        row.append(f"{event_amps[idx]:.6f}")
                    row.append(f"{median_amps[idx]:.6f}")
                    writer.writerow(row)
            logger.info("Overlay CSV saved to %s", csv_path)

        out_path = save_dir / "amplification_overlay.png"
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
        logger.info("Overlay plot saved to %s", out_path)
        saved_path = out_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path