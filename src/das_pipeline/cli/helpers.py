"""Shared helpers for CLI commands.

These are extracted from the original monolithic ``cli.py`` to avoid
duplicating patch-loading, file-collection, and bad-channel logic across
the individual command modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import dascore as dc
import matplotlib
import numpy as np

from das_pipeline.utils.bad_channels import get_bad_channel_indices, exclude_bad_channels_from_patch
from das_pipeline.visualization.merge import merge_patches

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# matplotlib backend
# ---------------------------------------------------------------------------


def setup_matplotlib_backend(no_display: bool) -> None:
    """Switch matplotlib to Agg if no_display is True."""
    if no_display:
        matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def collect_h5_files(path: Path, pattern: str = "*.h5") -> List[Path]:
    """Collect .h5 files from a directory or a single file path.

    Excludes hidden files (dotfiles) and files matching ``.dascore_index.h5``.
    """
    if path.is_dir():
        file_paths = sorted(path.glob(pattern))
        # Exclude hidden / index files
        file_paths = [p for p in file_paths if not p.name.startswith(".")]
        if not file_paths:
            import typer
            typer.echo(f"❌ 在 {path} 找不到符合 {pattern} 的檔案")
            raise typer.Exit(1)
        import typer
        typer.echo(f"找到 {len(file_paths)} 個檔案")
        return file_paths
    return [path]


def collect_csv_files(directory: Path, pattern: str = "*.csv") -> List[Path]:
    """Collect CSV files from a directory."""
    csv_files = sorted(directory.glob(pattern))
    if not csv_files:
        import typer
        typer.echo(f"❌ 在 {directory} 找不到符合 {pattern} 的檔案")
        raise typer.Exit(1)
    return csv_files


# ---------------------------------------------------------------------------
# Patch loading
# ---------------------------------------------------------------------------


def load_patch(
    file_paths: List[Path],
    merge: bool = False,
    sort_by: str = "chunk_index",
) -> dc.Patch:
    """Load a Patch from one or more .h5 files.

    If ``merge`` is True and more than one file is provided, calls
    :func:`merge_patches`; otherwise loads the first file via ``dc.spool``.
    """
    if merge and len(file_paths) > 1:
        import typer
        typer.echo("合併多個 chunk 檔案中...")
        return merge_patches(file_paths, sort_by=sort_by)

    spool = dc.spool(str(file_paths[0]))
    return spool[0]


def log_patch_info(patch: dc.Patch) -> None:
    """Log basic patch dimensions and time range."""
    import typer
    time_values = patch.coords.get_array("time")
    typer.echo(
        f"Patch 維度: {patch.shape}, "
        f"time: {time_values.min()} ~ {time_values.max()}"
    )


# ---------------------------------------------------------------------------
# Bad channel handling (delegates to utils.bad_channels)
# ---------------------------------------------------------------------------


def handle_bad_channels_for_detection(
    patch: dc.Patch,
) -> tuple[dc.Patch, list[int]]:
    """Warn about and remove bad channels for STA/LTA detection.

    Returns ``(cleaned_patch, local_to_original)``.
    """
    import typer
    bad_indices = get_bad_channel_indices(patch)
    if not bad_indices:
        n_ch = patch.shape[0] if "distance" in patch.dims else patch.shape[1]
        return patch, list(range(n_ch))

    n_before = patch.shape[0] if "distance" in patch.dims else patch.shape[1]
    typer.echo(
        f"⚠️  此檔案有 {len(bad_indices)} 個 channel 完全無資料"
        f"（index: {bad_indices}），將不參與 STA/LTA 檢測。"
    )
    cleaned, n_excluded, local_to_orig = exclude_bad_channels_from_patch(patch, bad_indices)
    typer.echo(f"    排除後剩餘 channel 數: {n_before - n_excluded}")
    return cleaned, local_to_orig


def warn_bad_channels_for_plot(patch: dc.Patch) -> dc.Patch:
    """Warn about bad channels and return a version with them removed.

    The original patch is returned if no bad channels exist; otherwise a
    physically pruned patch is returned (bad channels removed).
    """
    import typer
    bad_indices = get_bad_channel_indices(patch)
    if bad_indices:
        typer.echo(
            f"⚠️  此檔案有 {len(bad_indices)} 個 channel 完全無資料"
            f"（index: {bad_indices}），spectrogram 將排除這些 channel。"
        )
        cleaned, _, _ = exclude_bad_channels_from_patch(patch, bad_indices)
        return cleaned
    return patch