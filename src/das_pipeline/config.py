"""DAS Pipeline 設定模型。

使用 Pydantic 進行參數驗證，涵蓋資料輸入、座標對齊、
前處理、遠震分析與 STA/LTA 檢測等各階段設定。
"""

import numpy as np
from pydantic import BaseModel
from pydantic import field_validator
from pathlib import Path
from typing import Optional


class DataConfig(BaseModel):
    """資料輸入設定。

    Attributes
    ----------
    input_dir : Path
        原始 MiniSEED 或 HDF5 檔案所在目錄。
    format : str
        輸入格式，``"miniseed"`` 或 ``"dasdae"``。
    file_pattern : str
        檔案 glob pattern，例如 ``"*.mseed"``。
    sampling_rate : int, optional
        取樣頻率 (Hz)，若為 None 則從資料自動推斷。
    chunk_duration : np.timedelta64
        分段時間長度，預設 10 分鐘。
    filter_safety_samples : int
        濾波安全邊界樣本數（避免濾波邊緣效應跨 chunk 擴散）。
    """

    model_config = {"arbitrary_types_allowed": True}

    input_dir: Path
    format: str = "miniseed"
    file_pattern: str = "*.mseed"
    sampling_rate: Optional[int] = None
    chunk_duration: np.timedelta64 = np.timedelta64(10, "m")
    filter_safety_samples: int = 0

    @field_validator("chunk_duration", mode="before")
    @classmethod
    def _parse_timedelta(cls, value):
        """將字串格式的時間長度（如 ``"10m"``）轉為 ``np.timedelta64``。

        Parameters
        ----------
        value : np.timedelta64, str or None
            原始欄位值。

        Returns
        -------
        np.timedelta64 or None
        """
        if isinstance(value, np.timedelta64):
            return value
        if value is None:
            return value

        import pandas as pd

        return pd.to_timedelta(value).to_timedelta64()


class CoordinateConfig(BaseModel):
    """座標對齊與單位轉換設定。

    Attributes
    ----------
    fiber_geometry_file : Path
        geometry CSV 路徑（含 channel_index, lat, lon, depth 欄位）。
    distance_unit : str
        距離單位，預設 ``"m"``。
    strict_shape_check : bool
        是否在對齊後強制檢查 data shape 與 coords 一致性。
    input_unit : str
        輸入物理量，``"strain_rate"`` 或 ``"phase"``。
    phase_strain_constant : float
        相位差轉應變率的常數（僅 input_unit="phase" 時使用）。
    missing_channel_strategy : str
        缺失 channel 的處理策略：``"interpolate"`` / ``"crop"`` / ``"error"``。
    """
    fiber_geometry_file: Path
    distance_unit: str = "m"
    strict_shape_check: bool = True
    input_unit: str = "strain_rate"
    phase_strain_constant: float = 11.6e-9
    missing_channel_strategy: str = "interpolate"

    @field_validator("input_unit")
    @classmethod
    def _validate_input_unit(cls, value):
        """驗證 input_unit 僅限 ``"strain_rate"`` 或 ``"phase"``。

        Parameters
        ----------
        value : str
            輸入值。

        Returns
        -------
        str
            驗證通過的值。

        Raises
        ------
        ValueError
            若值不在允許範圍內。
        """
        allowed = {"strain_rate", "phase"}
        if value not in allowed:
            raise ValueError(f"input_unit 必須為 {allowed}，目前為 {value}")
        return value

    @field_validator("missing_channel_strategy")
    @classmethod
    def _validate_missing_strategy(cls, value):
        """驗證 missing_channel_strategy 僅限 ``"interpolate"`` / ``"crop"`` / ``"error"``。

        Parameters
        ----------
        value : str
            輸入值。

        Returns
        -------
        str
            驗證通過的值。

        Raises
        ------
        ValueError
            若值不在允許範圍內。
        """
        allowed = {"interpolate", "crop", "error"}
        if value not in allowed:
            raise ValueError(f"missing_channel_strategy 必須為 {allowed}，目前為 {value}")
        return value


class OutputConfig(BaseModel):
    """輸出設定。

    Attributes
    ----------
    save_dir : Path
        輸出目錄路徑。
    filename_pattern : str
        檔名模板，可用變數 ``{project_name}``, ``{timestamp}``, ``{chunk_index}``。
    overwrite : bool
        是否覆蓋已存在的檔案。
    """
    save_dir: Path
    filename_pattern: str = "{project_name}_{timestamp}_chunk{chunk_index:04d}.h5"
    overwrite: bool = False


class PreprocessingConfig(BaseModel):
    """前處理設定。

    所有參數均為 Optional，None 表示跳過該步驟。

    Attributes
    ----------
    time_range : tuple[str, str], optional
        時間範圍，ISO 格式，如 ``["2023-02-06T10:24:50", "2023-02-06T10:25:00"]``。
    distance_range : tuple[float, float], optional
        距離/通道範圍。
    detrend : str, optional
        去趨勢方法：``"linear"`` / ``"constant"`` / None。
    taper_ratio : float, optional
        taper 比例，需滿足 ``0 < taper_ratio < 0.5``。
    bandpass : tuple[float, float], optional
        帶通頻率範圍 (low_hz, high_hz)。
    decimate_factor : int, optional
        降採樣倍數（≥ 2 的整數）。
    """

    time_range: Optional[tuple[str, str]] = None  # ISO 格式，如 ["2023-02-06T10:24:50", "2023-02-06T10:25:00"]
    distance_range: Optional[tuple[float, float]] = None
    detrend: Optional[str] = "linear"
    taper_ratio: Optional[float] = None
    bandpass: Optional[tuple[float, float]] = None
    decimate_factor: Optional[int] = None

    @field_validator("taper_ratio")
    @classmethod
    def _validate_taper_ratio(cls, value):
        """驗證 taper_ratio 為 None 或滿足 ``0 < value < 0.5``。

        Parameters
        ----------
        value : float or None
            輸入值。

        Returns
        -------
        float or None
            驗證通過的值。

        Raises
        ------
        ValueError
            若值不在有效範圍內。
        """
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


class SnrConfig(BaseModel):
    """單一 channel SNR（訊雜比）分析設定。"""
    event_distance_km: float
    event_origin_time: str          # ISO 格式，如 "2023-02-06T01:17:35"
    channel_index: int
    velocity_min: float = 2.0       # 最慢群速度 (km/s)
    velocity_max: float = 4.0       # 最快群速度 (km/s)
    noise_offset_s: float = 30.0    # 雜訊窗與訊號窗之間的間隔 (秒)


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
        """驗證 STA/LTA window 參數為正數。

        Parameters
        ----------
        v : float
            輸入值。

        Returns
        -------
        float
            驗證通過的值。

        Raises
        ------
        ValueError
            若值 ≤ 0。
        """
        if v <= 0:
            raise ValueError(f"window 必須 > 0，目前 {v}")
        return v

    @field_validator("detrigger_threshold")
    @classmethod
    def _validate_detrigger_lt_trigger(cls, v: float, info) -> float:
        """驗證 detrigger_threshold < trigger_threshold。

        Parameters
        ----------
        v : float
            detrigger_threshold 值。
        info : pydantic.ValidationInfo
            包含其他已驗證欄位的資訊。

        Returns
        -------
        float
            驗證通過的值。

        Raises
        ------
        ValueError
            若 detrigger_threshold ≥ trigger_threshold。
        """
        trigger = info.data.get("trigger_threshold")
        if trigger is not None and v >= trigger:
            raise ValueError(
                f"detrigger_threshold ({v}) 必須 < trigger_threshold ({trigger})"
            )
        return v


class ConvertConfig(BaseModel):
    """轉檔完整設定，組裝所有子設定模組。

    Attributes
    ----------
    project_name : str
        專案名稱，用於輸出檔名。
    data : DataConfig
        資料輸入設定。
    coordinate : CoordinateConfig
        座標對齊設定。
    preprocessing : PreprocessingConfig
        前處理設定，預設為全部跳過。
    output : OutputConfig
        輸出設定。
    """

    project_name: str
    data: DataConfig
    coordinate: CoordinateConfig
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "ConvertConfig":
        """從 YAML 設定檔建立 ConvertConfig 實例。

        Parameters
        ----------
        path : Path
            YAML 檔案路徑。

        Returns
        -------
        ConvertConfig
            解析後的設定物件。
        """
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)