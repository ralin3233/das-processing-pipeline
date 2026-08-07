import logging

from das_pipeline.config import ConvertConfig
from das_pipeline.io import spool_loader, coord_utils, patch_writer
from das_pipeline.preprocessing import run_preprocessing
from das_pipeline.preprocessing.nan_handler import sanitize_nan_patch

logger = logging.getLogger(__name__)


def run_convert(config: ConvertConfig):
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
