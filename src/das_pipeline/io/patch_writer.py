import dascore as dc
from das_pipeline.config import OutputConfig
import os
import logging

logger = logging.getLogger(__name__)

def save(patch: dc.Patch, config: OutputConfig, project_name: str) -> str:
    # 確保儲存目錄存在
    os.makedirs(config.save_dir, exist_ok=True)

    # 生成檔案名稱
    filename = config.filename_pattern.format(
        project_name=project_name,
        timestamp=patch.coords.get_array("time")[0].astype("datetime64[s]").astype(str)
    )
    output_path = os.path.join(config.save_dir, filename)

    # 儲存 Patch
    patch.io.write(output_path, "dasdae")
    logger.info(f"成功將帶有地理資訊的 Patch 儲存至 {output_path}")
    return output_path