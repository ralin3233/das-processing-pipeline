"""Patch 儲存模組：將處理後的 DAS Patch 寫入 DASDAE 格式（HDF5）。"""

import logging
from datetime import datetime
from pathlib import Path


from das_pipeline.config import OutputConfig

logger = logging.getLogger(__name__)


def save(
    patch,
    output: OutputConfig,
    project_name: str,
    chunk_index: int = 0,
) -> Path:
    """將 Patch 寫入 DASDAE 格式的 .h5 檔案。

    使用 ``output.filename_pattern`` 產生檔名，並以當前時間戳記
    與 chunk_index 補齊模板變數。

    Parameters
    ----------
    patch : dc.Patch
        要儲存的 Patch。
    output : OutputConfig
        輸出設定（目錄、檔名模板、是否覆蓋）。
    project_name : str
        專案名稱，用於檔名模板的 ``{project_name}``。
    chunk_index : int
        Chunk 編號，用於檔名模板的 ``{chunk_index}``。

    Returns
    -------
    Path
        寫出檔案的完整路徑。

    Raises
    ------
    FileExistsError
        若目標檔案已存在且 ``output.overwrite=False``。
    """
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    filename = output.filename_pattern.format(
        project_name=project_name,
        timestamp=timestamp,
        chunk_index=chunk_index,
    )

    save_path = Path(output.save_dir) / filename
    if save_path.exists() and not output.overwrite:
        raise FileExistsError(f"{save_path} 已存在，且 overwrite=false")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    patch.io.write(str(save_path), file_format="dasdae")

    logger.info(f"成功將 Patch 儲存至 {save_path}")
    return save_path