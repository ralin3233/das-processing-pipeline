"""Internal helpers for visualization axis labels."""

from __future__ import annotations

import numpy as np


def get_distance_label(patch, dist_vals: np.ndarray) -> str:
    """Heuristically decide whether distance axis is in metres or channel index.

    Returns ``"Distance (m)"`` or ``"Channel index"`` as the y-axis label.
    """
    coords = patch.coords
    coord = coords.get_coord("distance")
    units_str = str(getattr(coord, "units", "") or "")
    if (
        "m" in units_str
        or "m" in str(patch.attrs.get("distance_unit", ""))
        or (len(dist_vals) > 0 and np.max(dist_vals) > 10 and dist_vals.dtype.kind == "f")
    ):
        return "Distance (m)"
    return "Channel index"