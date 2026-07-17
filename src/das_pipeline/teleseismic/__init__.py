# src/das_pipeline/teleseismic/__init__.py

from das_pipeline.teleseismic.amplification import compute_amplification
from das_pipeline.teleseismic.visualization import plot_amplification

__all__ = ["compute_amplification", "plot_amplification"]