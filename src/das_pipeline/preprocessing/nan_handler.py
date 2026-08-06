# src/das_pipeline/preprocessing/nan_handler.py

"""NaN sanitizer — interpolate NaN gaps before they enter downstream filters.

IIR/FIR bandpass and anti-aliasing decimation filters use convolution,
which spreads a single NaN across the entire time series.  This module
interpolates NaN values along the time axis *before* any filtering step,
preventing the cascade failure.

Channels that are entirely NaN (dead channels / complete sensor failure)
are NOT filled with zeros — instead their NaN values are kept, and their
indices are recorded in ``all_nan_channel_indices`` patch attrs (comma-separated).
Downstream modules (e.g. amplification.py) should read this attr and
physically remove those channels from analysis, rather than treating them
as valid data with zero amplitude.
"""

import json
import logging
from typing import Tuple

import dascore as dc
import numpy as np

logger = logging.getLogger(__name__)


def sanitize_nan_patch(patch: dc.Patch) -> Tuple[dc.Patch, dict]:
    """Interpolate NaN gaps along the time axis for every channel.

    Strategy (per channel):
    1. If a channel contains no NaN → leave untouched.
    2. If NaN ratio < 100% → linear interpolation along time axis.
    3. If NaN ratio == 100% → keep NaN, record index in attrs.

    Parameters
    ----------
    patch : dc.Patch
        Input Patch, expected shape (distance, time) or (time, distance).

    Returns
    -------
    patch : dc.Patch
        Sanitized patch (partial-NaN channels interpolated; all-NaN channels
        kept as NaN and flagged via ``all_nan_channel_indices`` attrs).
    stats : dict
        Summary dict: ``{"nan_ratio": float, "n_all_nan_channels": int,
        "all_nan_channel_indices": list[int]}``
    """
    data = np.asarray(patch.data, dtype=np.float64).copy()

    if data.size == 0:
        return patch, {"nan_ratio": 0.0, "n_all_nan_channels": 0, "all_nan_channel_indices": []}

    nan_mask = np.isnan(data)
    nan_count = nan_mask.sum()
    nan_ratio = nan_count / data.size

    if nan_count == 0:
        logger.debug("Patch contains no NaN values.")
        return patch, {"nan_ratio": 0.0, "n_all_nan_channels": 0, "all_nan_channel_indices": []}

    logger.info("Patch contains %.2f%% NaN (%d / %d values). Interpolating.",
                nan_ratio * 100, nan_count, data.size)

    # Determine which axis is "time"
    dims = patch.dims
    time_axis = dims.index("time")
    channel_axis = 1 - time_axis  # if time is 0, channel is 1; if time is 1, channel is 0

    n_channels = data.shape[channel_axis]
    n_time = data.shape[time_axis]
    all_nan_channels: list[int] = []
    bad_channel_mask = np.zeros(n_channels, dtype=bool)

    for ch in range(n_channels):
        # Slice along the channel dimension
        if channel_axis == 0:
            channel_data = data[ch, :]
        else:
            channel_data = data[:, ch]

        ch_nan = np.isnan(channel_data)
        ch_nan_ratio = ch_nan.sum() / n_time

        if ch_nan_ratio == 0:
            continue

        if ch_nan_ratio == 1.0:
            # Entire channel is NaN — keep NaN, flag as bad
            all_nan_channels.append(ch)
            bad_channel_mask[ch] = True
            # Do NOT fill with zeros — leave NaN untouched so downstream
            # modules can distinguish "no data" from "zero amplitude".
        else:
            # Partial NaN — linear interpolation
            valid_idx = np.where(~ch_nan)[0]
            valid_vals = channel_data[~ch_nan]

            if len(valid_idx) < 2:
                # Degenerate case: only 1 valid point → fill with that value
                channel_data[:] = valid_vals[0] if len(valid_idx) == 1 else 0.0
            else:
                # Use np.interp for linear interpolation (handles leading/trailing NaN
                # by extrapolation = nearest boundary value)
                channel_data[:] = np.interp(
                    np.arange(n_time), valid_idx, valid_vals
                )

        # Write back
        if channel_axis == 0:
            data[ch, :] = channel_data
        else:
            data[:, ch] = channel_data

    stats = {
        "nan_ratio": float(nan_ratio),
        "n_all_nan_channels": len(all_nan_channels),
        "all_nan_channel_indices": all_nan_channels,
    }

    if all_nan_channels:
        logger.warning(
            "%d channels are entirely NaN (indices: %s) — kept as NaN, flagged in attrs.",
            len(all_nan_channels), all_nan_channels,
        )

    # Rebuild patch with sanitized data
    new_patch = dc.Patch(
        data=data,
        coords=patch.coords,
        dims=patch.dims,
        attrs=patch.attrs,
    )
    # Store bad-channel info as comma-separated strings (DASDAE dc.spool
    # does not support attrs containing '[' or ']', so we cannot use JSON).
    new_patch = new_patch.update_attrs(
        nan_sanitized=True,
        nan_ratio_before=float(nan_ratio),
        n_all_nan_channels=len(all_nan_channels),
        all_nan_channel_indices=",".join(str(i) for i in all_nan_channels),
        bad_channel_mask=",".join(str(int(b)) for b in bad_channel_mask),
    )
    logger.info("NaN sanitization complete.")

    return new_patch, stats
