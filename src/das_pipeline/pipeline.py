from das_pipeline.config import ConvertConfig
from das_pipeline.io import spool_loader, coord_utils, patch_writer
from das_pipeline.preprocessing import run_preprocessing


def run_convert(config: ConvertConfig):
    spool = spool_loader.get_spool(config.data)

    save_paths = []
    for chunk_index, patch in spool_loader.iter_chunks(spool, config.data): # type: ignore[call-arg]
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