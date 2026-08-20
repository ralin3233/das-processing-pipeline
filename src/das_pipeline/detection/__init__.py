"""STA/LTA 觸發檢測模組。

使用 DASCore 內建 stalta 計算 STA/LTA ratio，
並以空間一致性進行事件觸發檢測。
"""

from das_pipeline.detection.sta_lta import (
	compute_sta_lta_components,
	compute_sta_lta_patch,
	detect_events,
)

__all__ = ["compute_sta_lta_components", "compute_sta_lta_patch", "detect_events"]