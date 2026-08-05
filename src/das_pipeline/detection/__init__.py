# src/das_pipeline/detection/__init__.py

from das_pipeline.detection.sta_lta import compute_sta_lta_patch, detect_events

__all__ = ["compute_sta_lta_patch", "detect_events"]