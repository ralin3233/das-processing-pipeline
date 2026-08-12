# src/das_pipeline/cli/commands/check.py
"""CLI command: inspect original data time×distance coverage and missingness.

Scans .hdf5/.h5 files under a directory and draws a coverage grid with
colour-coded cells: Data / NaN / Uncovered / Overlap.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from das_pipeline.check import (
    load_patch_meta,
    auto_time_bin,
    build_global_axes,
    compute_status_grid,
    plot_status_grid,
)
from das_pipeline.cli.helpers import collect_h5_files, setup_matplotlib_backend

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    @app.command()
    def check(
        directory: Annotated[
            Path,
            typer.Argument(
                ..., help="Directory containing .hdf5/.h5 files", exists=True
            ),
        ],
        pattern: Annotated[
            str,
            typer.Option("--pattern", "-p", help="glob pattern for file filtering"),
        ] = "*.h5",
        time_bin: Annotated[
            str,
            typer.Option(
                "--time-bin",
                "-t",
                help="Time bin size in seconds; 'auto' for auto-computation",
            ),
        ] = "auto",
        title: Annotated[
            Optional[str],
            typer.Option("--title", help="Custom chart title"),
        ] = None,
        save: Annotated[
            Optional[Path],
            typer.Option(
                "--save", "-s", help="Output directory; skip to show interactively"
            ),
        ] = None,
        format: Annotated[
            str,
            typer.Option("--format", help="Output format: png, pdf, svg"),
        ] = "png",
        dpi: Annotated[
            int,
            typer.Option("--dpi", help="Image resolution"),
        ] = 150,
        no_display: Annotated[
            bool,
            typer.Option("--no-display", help="Suppress interactive window when saving"),
        ] = False,
    ):
        """Inspect original data time×distance coverage and missingness.

        \b
        Scans .hdf5/.h5 files under a directory and draws a time×distance
        coverage grid.  Each cell is colour-coded:
        Data(green) / NaN(red) / Uncovered(grey) / Overlap(orange).

        \b
        Examples:
            das-pipeline check data/raw/
            das-pipeline check data/raw/ --time-bin 10
            das-pipeline check data/raw/ --save outputs/ --dpi 200
        """
        setup_matplotlib_backend(no_display)

        file_paths = collect_h5_files(directory, pattern)
        if not file_paths:
            typer.echo(f"❌ No files matching {pattern} found in {directory}")
            raise typer.Exit(1)

        typer.echo(f"Found {len(file_paths)} files, loading...")

        metas = []
        for fp in file_paths:
            try:
                meta = load_patch_meta(fp)
                metas.append(meta)
                logger.info(
                    "  %s: time=%d, dist=%d, shape=%s",
                    fp.name,
                    len(meta["time_vals"]),
                    len(meta["dist_vals"]),
                    meta["data"].shape,
                )
            except Exception as e:
                logger.warning("Cannot read %s: %s", fp.name, e)
                typer.echo(f"⚠️  Skipping {fp.name}: {e}")

        if not metas:
            typer.echo("❌ No readable files")
            raise typer.Exit(1)

        if time_bin.lower() == "auto":
            time_bin_sec = auto_time_bin(metas)
            typer.echo(f"Auto time bin: {time_bin_sec:.3f}s")
        else:
            try:
                time_bin_sec = float(time_bin)
            except ValueError:
                typer.echo(
                    f"❌ --time-bin must be a number or 'auto', got: {time_bin}"
                )
                raise typer.Exit(1)

        time_edges, dist_axis = build_global_axes(metas, time_bin_sec)
        typer.echo(
            f"Grid: {len(dist_axis)} distance × {len(time_edges) - 1} time bins "
            f"(bin={time_bin_sec:.3f}s)"
        )

        typer.echo("Computing status grid...")
        status = compute_status_grid(metas, time_edges, dist_axis)
        typer.echo("Status grid computed")

        typer.echo("Plotting...")
        plot_status_grid(
            status=status,
            time_edges=time_edges,
            dist_axis=dist_axis,
            file_count=len(metas),
            time_bin_sec=time_bin_sec,
            title=title,
            save=save,
            fmt=format,
            dpi=dpi,
            no_display=no_display,
        )
