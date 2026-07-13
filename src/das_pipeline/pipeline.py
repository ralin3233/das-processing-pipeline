from das_pipeline.config import ConvertConfig
from das_pipeline.io import miniseed_loader, patch_writer

def run_convert(config: ConvertConfig) -> str:
    patch = miniseed_loader.load(config.data)
    return patch_writer.save(patch, config.output, config.project_name)
