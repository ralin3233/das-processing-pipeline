# src/das_pipeline/check/coverage.py
"""Data loading and grid-computation for coverage check."""

from __future__ import annotations

import logging
from pathlib import Path

import dascore as dc
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------
STATE_UNCOVERED = 0   # uncovered
STATE_DATA = 1        # has data
STATE_NAN = 2         # NaN (should have data but missing)
STATE_OVERLAP = 3     # overlap (multiple files cover the same cell)

# Colours: uncovered=grey, data=green, NaN=red, overlap=orange
STATE_COLORS = ["#D3D3D3", "#4CAF50", "#F44336", "#FF9800"]


# ---------------------------------------------------------------------------
# Data loading & grid building
# ---------------------------------------------------------------------------


def load_patch_meta(file_path: Path) -> dict:
    """Load a .h5 patch and return its metadata & data.

    Returns
    -------
    dict with keys:
        path       : Path          -- file path
        time_vals  : np.ndarray    -- datetime64[ns]
        dist_vals  : np.ndarray    -- float
        data       : np.ndarray    -- shape = (n_dist, n_time)
    """
    spool = dc.spool(str(file_path))
    patch = spool[0]
    dims = patch.dims

    time_vals = np.asarray(patch.get_coord("time"))
    dist_vals = np.asarray(patch.get_coord("distance"))
    data = np.asarray(patch.data)

    # Ensure data shape is (n_distance, n_time)
    if dims[0] == "time":
        data = data.T  # (n_time, n_dist) -> (n_dist, n_time)

    return {
        "path": file_path,
        "time_vals": time_vals.astype("datetime64[ns]"),
        "dist_vals": dist_vals,
        "data": data,
    }


def auto_time_bin(metas: list[dict]) -> float:
    """Auto-compute a reasonable *time_bin* (seconds) from all files' sampling
    rates.  Targets roughly 1000 bins.
    """
    t_min = min(m["time_vals"].min() for m in metas)
    t_max = max(m["time_vals"].max() for m in metas)
    total_sec = (t_max - t_min).astype("int64") / 1e9

    if total_sec <= 0:
        return 1.0

    target_bins = 1000
    auto_bin = total_sec / target_bins

    # Do not go below the sampling interval of the first file
    dt_vals = np.diff(metas[0]["time_vals"].astype("int64"))
    min_dt = dt_vals.min() / 1e9 if len(dt_vals) > 0 else 0.01
    auto_bin = max(auto_bin, min_dt)

    if auto_bin < 0.01:
        auto_bin = round(auto_bin, 4)
    elif auto_bin < 1:
        auto_bin = round(auto_bin, 2)
    else:
        auto_bin = round(auto_bin)

    return max(auto_bin, 0.001)


def build_global_axes(
    metas: list[dict], time_bin_sec: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build global time axis (bin edges) and distance axis."""
    all_dist = np.concatenate([m["dist_vals"] for m in metas])
    dist_axis = np.unique(all_dist)
    dist_axis.sort()

    t_min = min(m["time_vals"].min() for m in metas)
    t_max = max(m["time_vals"].max() for m in metas)
    total_span_ns = int((t_max - t_min).astype("int64"))

    if total_span_ns <= 0:
        raise ValueError("Time range is empty, cannot build grid")

    bin_ns = int(time_bin_sec * 1e9)
    if bin_ns <= 0:
        bin_ns = 1
    n_bins = max(1, total_span_ns // bin_ns + 1)

    MAX_BINS = 5000
    if n_bins > MAX_BINS:
        bin_ns = total_span_ns // MAX_BINS + 1
        n_bins = MAX_BINS
        logger.warning(
            "Too many time bins, capped at %d (bin=%.3fs)", MAX_BINS, bin_ns / 1e9
        )

    time_edges = np.arange(
        t_min.astype("datetime64[ns]"),
        t_max.astype("datetime64[ns]") + np.timedelta64(bin_ns, "ns"),
        np.timedelta64(bin_ns, "ns"),
        dtype="datetime64[ns]",
    )
    if time_edges[-1] < t_max:
        time_edges = np.append(time_edges, t_max)

    return time_edges, dist_axis


# ---------------------------------------------------------------------------
# Status grid computation
# ---------------------------------------------------------------------------


def compute_status_grid(
    metas: list[dict],
    time_edges: np.ndarray,
    dist_axis: np.ndarray,
) -> np.ndarray:
    """Compute the status of every (distance, time_bin) cell.

    Overlap is detected via file-level pairwise comparison of time/distance
    ranges so that it does not depend on the grid resolution.

    Returns
    -------
    status : np.ndarray, shape = (n_dist, n_time_bins), dtype=np.int8
    """
    n_dist = len(dist_axis)
    n_time_bins = len(time_edges) - 1

    status = np.full((n_dist, n_time_bins), STATE_UNCOVERED, dtype=np.int8)

    dist_to_idx = {float(d): i for i, d in enumerate(dist_axis)}
    time_edges_int = time_edges.astype("int64")
    total_files = len(metas)

    # ---- Pass 1: per-file mark Data / NaN ----
    for fi, meta in enumerate(metas):
        logger.info(
            "Processing [%d/%d]: %s", fi + 1, total_files, meta["path"].name
        )
        data = meta["data"]
        t_vals = meta["time_vals"]
        d_vals = meta["dist_vals"]

        d_indices = np.array(
            [dist_to_idx[float(d)] for d in d_vals], dtype=np.intp
        )

        t_int = t_vals.astype("int64")
        t_bins = np.digitize(t_int, time_edges_int) - 1
        valid_t = (t_bins >= 0) & (t_bins < n_time_bins)
        t_bins = t_bins[valid_t]

        valid_t_data = data[:, valid_t]
        valid_is_nan = np.isnan(valid_t_data)

        for i_local, d_global in enumerate(d_indices):
            row_nan = valid_is_nan[i_local]

            bin_counts = np.bincount(t_bins, minlength=n_time_bins)
            nan_counts = np.bincount(
                t_bins,
                weights=row_nan.astype(np.float64),
                minlength=n_time_bins,
            )

            covered_bins = np.where(bin_counts > 0)[0]

            for tb in covered_bins:
                is_nan = nan_counts[tb] > 0
                if is_nan:
                    if status[d_global, tb] == STATE_UNCOVERED:
                        status[d_global, tb] = STATE_NAN
                else:
                    if status[d_global, tb] in (STATE_UNCOVERED, STATE_NAN):
                        status[d_global, tb] = STATE_DATA

    # ---- Pass 2: pairwise file-level overlap detection ----
    if total_files > 1:
        # Precompute per-file min/max (metadata level)
        file_ranges = []
        for meta in metas:
            t_min = meta["time_vals"].min()
            t_max = meta["time_vals"].max()
            d_min = meta["dist_vals"].min()
            d_max = meta["dist_vals"].max()
            file_ranges.append((t_min, t_max, d_min, d_max))

        for i in range(total_files):
            t_min_i, t_max_i, d_min_i, d_max_i = file_ranges[i]
            t_min_i_ns = t_min_i.astype("int64")
            t_max_i_ns = t_max_i.astype("int64")
            for j in range(i + 1, total_files):
                t_min_j, t_max_j, d_min_j, d_max_j = file_ranges[j]
                t_min_j_ns = t_min_j.astype("int64")
                t_max_j_ns = t_max_j.astype("int64")

                # Compute actual overlapping time & distance intervals
                t_ov_start = max(t_min_i_ns, t_min_j_ns)
                t_ov_end = min(t_max_i_ns, t_max_j_ns)
                d_ov_start = max(d_min_i, d_min_j)
                d_ov_end = min(d_max_i, d_max_j)

                if t_ov_start >= t_ov_end or d_ov_start >= d_ov_end:
                    continue  # no overlap

                # Map to grid range
                tb_start = max(0, np.digitize(t_ov_start, time_edges_int) - 1)
                tb_end = min(
                    n_time_bins - 1,
                    np.digitize(t_ov_end, time_edges_int) - 1,
                )
                if tb_start > tb_end:
                    continue

                d_start = max(
                    0,
                    int(np.searchsorted(dist_axis, d_ov_start, side="left")),
                )
                d_end = min(
                    n_dist - 1,
                    int(np.searchsorted(dist_axis, d_ov_end, side="right"))
                    - 1,
                )
                if d_start > d_end:
                    continue

                # Mark only cells that already have data as OVERLAP
                region = status[d_start : d_end + 1, tb_start : tb_end + 1]
                region[region != STATE_UNCOVERED] = STATE_OVERLAP

    return status
