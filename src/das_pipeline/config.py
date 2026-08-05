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
    distance_unit: str = "m"
    strict_shape_check: bool = True
    input_unit: str = "strain_rate"
    phase_strain_constant: float = 11.6e-9
    missing_channel_strategy: str = "interpolate"

    @field_validator("input_unit")
    @classmethod
    def _validate_input_unit(cls, value):
        allowed = {"strain_rate", "phase"}
        if value not in allowed:
            raise ValueError(f"input_unit 必須為 {allowed}，目前為 {value}")
        return value

    @field_validator("missing_channel_strategy")
    @classmethod
    def _validate_missing_strategy(cls, value):
        allowed = {"interpolate", "crop", "error"}
        if value not in allowed:
            raise ValueError(f"missing_channel_strategy 必須為 {allowed}，目前為 {value}")
        return value


class OutputConfig(BaseModel):
    save_dir: Path
    filename_pattern: str = "{project_name}_{timestamp}_chunk{chunk_index:04d}.h5"
    overwrite: bool = False


class PreprocessingConfig(BaseModel):
    time_range: Optional[tuple[float, float]] = None
    distance_range: Optional[tuple[float, float]] = None
    detrend: Optional[str] = "linear"
    taper_ratio: Optional[float] = None
    bandpass: Optional[tuple[float, float]] = None
    decimate_factor: Optional[int] = None

    @field_validator("taper_ratio")
    @classmethod
    def _validate_taper_ratio(cls, value):
        if value is None:
            return value
        if not (0 < value < 0.5):
            raise ValueError(
                f"taper_ratio 必須為 None 或滿足 0 < taper_ratio < 0.5，目前為 {value}"
            )
        return value


class TeleseismicConfig(BaseModel):
    """遠震地層放大效應分析設定。"""
    event_distance_km: float
    event_origin_time: str          # ISO 格式，如 "2023-02-06T01:17:35"
    reference_channels: int = 10
    reference_distance_range: Optional[tuple[float, float]] = None   # (start_m, end_m)，水平光纖用
    velocity_min: float = 2.0       # 最慢群速度 (km/s)
    velocity_max: float = 4.0       # 最快群速度 (km/s)
    skip_channels: int = 0          # 跳過前 N 個 channel（井口附近易受雜訊干擾）


class StaLtaConfig(BaseModel):
    """STA/LTA 觸發檢測設定，所有參數皆可透過 CLI 設定。"""
    sta_window_s: float = 0.5          # STA 短窗長度 (秒)
    lta_window_s: float = 10.0         # LTA 長窗長度 (秒)
    trigger_threshold: float = 3.0     # STA/LTA ratio 觸發閾值
    detrigger_threshold: float = 1.5   # STA/LTA ratio 解除觸發閾值
    min_channels_triggered: int = 3    # 最少同時觸發通道數（空間一致性）
    min_event_duration_s: float = 0.1  # 最短事件持續時間 (秒)
    merge_window_s: float = 0.0        # 合併相鄰事件閾值 (秒)，0=不合併

    @field_validator("sta_window_s", "lta_window_s")
    @classmethod
    def _validate_positive_window(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"window 必須 > 0，目前 {v}")
        return v

    @field_validator("detrigger_threshold")
    @classmethod
    def _validate_detrigger_lt_trigger(cls, v: float, info) -> float:
        trigger = info.data.get("trigger_threshold")
        if trigger is not None and v >= trigger:
            raise ValueError(
                f"detrigger_threshold ({v}) 必須 < trigger_threshold ({trigger})"
            )
        return v


class RuntimeConfig(BaseModel):
    log_level: str = "INFO"


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