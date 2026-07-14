import logging

import dascore as dc

from das_pipeline.config import CoordinateConfig

logger = logging.getLogger(__name__)


def align(patch: dc.Patch, config: CoordinateConfig) -> dc.Patch:
	"""暫時維持原始 Patch 不變，先讓轉檔流程可以完整跑通。

	之後若有幾何座標檔，再在這裡實作距離軸對齊、插值與 shape 檢查。
	"""

	logger.info(
		"暫不套用幾何對齊，直接沿用原始 Patch；fiber_geometry_file=%s",
		config.fiber_geometry_file,
	)
	return patch
