import logging
from datetime import datetime
from pathlib import Path

import dascore as dc

from das_pipeline.config import OutputConfig

logger = logging.getLogger(__name__)

def save(
    patch,
    output: OutputConfig,
    project_name: str,
    chunk_index: int = 0,
) -> Path:
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