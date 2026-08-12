"""CLI subcommand modules."""

from das_pipeline.cli.commands import convert, amplification, detect, plot, overlay, check, snr

__all__ = ["convert", "amplification", "detect", "plot", "overlay", "check", "snr"]


def register_all(app: "typer.Typer") -> None:
    """Register every subcommand on *app*."""
    convert.register(app)
    amplification.register(app)
    detect.register(app)
    plot.register(app)
    overlay.register(app)
    check.register(app)
    snr.register(app)