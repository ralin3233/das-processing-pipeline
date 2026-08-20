"""Utilities for handling channels flagged as entirely NaN by nan_handler.

These channel indices are stored as a comma-separated string in
``patch.attrs["all_nan_channel_indices"]`` (DASDAE spool does not support
attrs containing '[' or ']').
"""

from __future__ import annotations

import numpy as np
import dascore as dc


def get_bad_channel_indices(patch: dc.Patch) -> list[int]:
    """Read ``all_nan_channel_indices`` from patch attrs (comma-separated)."""
    raw = patch.attrs.get("all_nan_channel_indices")
    if raw is None or str(raw).strip() == "":
        return []
    try:
        return [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    except (ValueError, TypeError):
        return []


def exclude_bad_channels_from_patch(
    patch: dc.Patch,
    bad_indices: list[int],
) -> tuple[dc.Patch, int, list[int]]:
    """Physically remove entirely-NaN channels from a patch.

    Returns ``(cleaned_patch, n_excluded, local_to_original)``.
    ``local_to_original`` maps compressed channel index back to the
    original channel number.
    """
    channel_axis = patch.dims.index("distance") if "distance" in patch.dims else 0
    n_orig = patch.shape[channel_axis]
    local_to_original = list(range(n_orig))

    if not bad_indices:
        return patch, 0, local_to_original

    data = np.asarray(patch.data)
    dims = patch.dims
    keep_mask = np.ones(data.shape[channel_axis], dtype=bool)
    global_bad = np.array(bad_indices, dtype=int)
    global_bad = global_bad[(global_bad >= 0) & (global_bad < len(keep_mask))]
    keep_mask[global_bad] = False
    n_excluded = (~keep_mask).sum()

    if n_excluded == 0:
        return patch, 0, local_to_original

    if channel_axis == 0:
        data = data[keep_mask, :]
        dist_coord = patch.coords.get_array("distance")[keep_mask]
    else:
        data = data[:, keep_mask]
        dist_coord = patch.coords.get_array("distance")[keep_mask]

    local_to_original = [i for i in range(len(keep_mask)) if keep_mask[i]]

    new_patch = dc.Patch(
        data=data,
        coords={"time": patch.coords.get_array("time"), "distance": dist_coord},
        dims=dims,
        attrs=patch.attrs,
    )
    return new_patch, n_excluded, local_to_original