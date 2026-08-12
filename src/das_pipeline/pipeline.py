"""DAS Pipeline 主管線模組。

讀取原始 MiniSEED 資料、執行前處理、座標對齊，並將結果寫入 HDF5。
"""

import logging

from das_pipeline.config import ConvertConfig
from das_pipeline.io import spool_loader, coord_utils, patch_writer
from das_pipeline.preprocessing import run_preprocessing
from das_pipeline.preprocessing.nan_handler import sanitize_nan_patch

logger = logging.getLogger(__name__)


def run_convert(config: ConvertConfig):
    """執行完整轉檔流程：讀取 → NaN 清理 → 前處理 → 座標對齊 → 儲存。

    Pipeline:
        1. 從 spool 讀取資料，按 chunk_duration 分段
        2. 每個 chunk 先執行 NaN sanitization（線性插值補洞）
        3. 依序執行前處理（select / detrend / taper / bandpass / decimate）
        4. 將 distance 座標從 channel index 對齊為實際距離（米）
        5. 寫入 DASDAE 格式的 .h5 檔案

    Parameters
    ----------
    config : ConvertConfig
        包含 data, coordinate, preprocessing, output 的完整設定。

    Returns
    -------
    list[Path]
        所有產出檔案的路徑列表。
    """
    spool = spool_loader.get_spool(config.data)

    save_paths = []
    for chunk_index, patch in spool_loader.iter_chunks(
        spool,
        config.data,
        taper_ratio=config.preprocessing.taper_ratio,
    ):
        # ── NaN sanitization: interpolate gaps before filtering ──
        patch, nan_stats = sanitize_nan_patch(patch)
        if nan_stats["n_all_nan_channels"] > 0:
            logger.warning(
                "Chunk %d: %d channel(s) entirely NaN, kept as NaN and flagged in attrs.",
                chunk_index,
                nan_stats["n_all_nan_channels"],
            )

        patch = run_preprocessing(patch, config.preprocessing)
        patch = coord_utils.align(patch, config.coordinate)
        save_path = patch_writer.save(
            patch,
            config.output,
            project_name=config.project_name,
            chunk_index=chunk_index,
        )
        save_paths.append(save_path)

    return save_paths
