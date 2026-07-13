from pydantic import BaseModel
from pathlib import Path
from typing import Optional


class DataConfig(BaseModel):
    input_dir: Path
    format: str = "miniseed"
    file_pattern: str = "*.mseed"
    channel_range: tuple[int, int]
    sampling_rate: Optional[int] = None
    time_range: Optional[tuple[str, str]] = None


class CoordinateConfig(BaseModel):
    fiber_geometry_file: Path
    interpolation: str = "linear"
    distance_unit: str = "m"
    strict_shape_check: bool = True


class OutputConfig(BaseModel):
    save_dir: Path
    filename_pattern: str = "{project_name}_{timestamp}.h5"
    format: str = "dascore_h5"
    overwrite: bool = False
    compression: Optional[str] = "gzip"


class RuntimeConfig(BaseModel):
    log_level: str = "INFO"
    save_manifest: bool = True


class ConvertConfig(BaseModel):
    project_name: str
    data: DataConfig
    coordinate: CoordinateConfig
    output: OutputConfig
    runtime: RuntimeConfig = RuntimeConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "ConvertConfig":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)