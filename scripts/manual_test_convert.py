from src.das_pipeline.config import ConvertConfig
from src.das_pipeline.io import miniseed_loader, patch_writer

config = ConvertConfig.from_yaml("configs/convert_default.yaml")
patch = miniseed_loader.load(config.data)
patch_writer.save(patch, config.output, config.project_name)