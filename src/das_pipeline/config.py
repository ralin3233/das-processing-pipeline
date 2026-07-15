import numpy as np
from pydantic import BaseModel
from pydantic import field_validator
from pathlib import Path
from typing import Optional


class DataConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    input_dir: Path
    format: str = "miniseed"
    file_pattern: str = "*.mseed"
    channel_range: tuple[int, int]
    sampling_rate: Optional[int] = None
    time_range: Optional[tuple[str, str]] = None
    chunk_duration: np.timedelta64 = np.timedelta64(10, "m")
    taper_ratio: float = 0.05
    filter_safety_samples: int = 0

    @field_validator("chunk_duration", mode="before")
    @classmethod
    def _parse_timedelta(cls, value):
        if isinstance(value, np.timedelta64):
            return value
        if value is None:
            return value

        import pandas as pd

        return pd.to_timedelta(value).to_timedelta64()


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


class PreprocessingConfig(BaseModel):
    time_range: Optional[tuple[float, float]] = None
    distance_range: Optional[tuple[float, float]] = None
    detrend: Optional[str] = "linear"
    bandpass: Optional[tuple[float, float]] = None
    decimate_factor: Optional[int] = None


class RuntimeConfig(BaseModel):
    log_level: str = "INFO"
    save_manifest: bool = True


class ConvertConfig(BaseModel):
    project_name: str
    data: DataConfig
    coordinate: CoordinateConfig
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    output: OutputConfig
    runtime: RuntimeConfig = RuntimeConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "ConvertConfig":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
