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
    filename_pattern: str = "{project_name}_{timestamp}_chunk{chunk_index:04d}.h5"
    format: str = "dascore_h5"
    overwrite: bool = False
    compression: Optional[str] = "gzip"


class PreprocessingConfig(BaseModel):
    time_range: Optional[tuple[float, float]] = None
    distance_range: Optional[tuple[float, float]] = None
    detrend: Optional[str] = "linear"
    bandpass: Optional[tuple[float, float]] = None
    decimate_factor: Optional[int] = None


class TeleseismicConfig(BaseModel):
    """遠震地層放大效應分析設定。"""
    event_distance_km: float
    event_origin_time: str          # ISO 格式，如 "2023-02-06T01:17:35"
    reference_channels: int = 10
    velocity_min: float = 2.0       # 最慢群速度 (km/s)
    velocity_max: float = 4.0       # 最快群速度 (km/s)
    skip_channels: int = 0          # 跳過前 N 個 channel（井口附近易受雜訊干擾）


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