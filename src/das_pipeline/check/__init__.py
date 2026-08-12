"""Data coverage check -- scan .h5 files and visualise time×distance coverage grid.

Exports state constants, grid computation helpers, and plotting.
"""

from das_pipeline.check.coverage import (
    STATE_UNCOVERED,
    STATE_DATA,
    STATE_NAN,
    STATE_OVERLAP,
    STATE_COLORS,
    load_patch_meta,
    auto_time_bin,
    build_global_axes,
    compute_status_grid,
)
from das_pipeline.check.plot import plot_status_grid

__all__ = [
    "STATE_UNCOVERED",
    "STATE_DATA",
    "STATE_NAN",
    "STATE_OVERLAP",
    "STATE_COLORS",
    "load_patch_meta",
    "auto_time_bin",
    "build_global_axes",
    "compute_status_grid",
    "plot_status_grid",
]
