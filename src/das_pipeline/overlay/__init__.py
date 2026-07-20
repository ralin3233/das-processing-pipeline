"""
Overlay module for teleseismic amplification CSV overlay plotting.

Reads multiple amplification CSV files and produces an overlay plot
with each event's curve and the median curve across events.
"""

from das_pipeline.overlay.plot import plot_overlay

__all__ = ["plot_overlay"]