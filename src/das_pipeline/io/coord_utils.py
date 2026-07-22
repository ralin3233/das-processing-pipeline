import logging
from pathlib import Path

import numpy as np
import pandas as pd
import dascore as dc
from scipy.interpolate import interp1d

from das_pipeline.config import CoordinateConfig

logger = logging.getLogger(__name__)


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """計算兩點之間的水平距離（米），使用 Haversine 公式。"""
    R = 6371000.0  # 地球半徑 (m)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def _compute_cumulative_distances(geometry_df: pd.DataFrame) -> np.ndarray:
    """計算沿光纖每個 channel 相對於第一個 channel 的累積 3D 距離（米）。

    對每對相鄰 channel：
        1. 用 Haversine 公式計算水平距離
        2. 結合深度差：3D_dist = sqrt(horizontal² + depth²)
        3. 累積加總

    Parameters
    ----------
    geometry_df : pd.DataFrame
        幾何座標表，必須包含欄位 'channel_index', 'lat', 'lon', 'depth'。
        資料已依照 channel_index 排序。

    Returns
    -------
    np.ndarray
        每個 channel 的累積距離（米），長度等於 geometry_df 的行數。
    """
    sorted_df = geometry_df.sort_values("channel_index").reset_index(drop=True)

    n = len(sorted_df)
    cumulative = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        r1 = sorted_df.iloc[i - 1]
        r2 = sorted_df.iloc[i]

        horizontal = _haversine_distance(r1["lat"], r1["lon"], r2["lat"], r2["lon"])
        depth_diff = r2["depth"] - r1["depth"]
        segment_length = np.sqrt(horizontal ** 2 + depth_diff ** 2)

        cumulative[i] = cumulative[i - 1] + segment_length

    return cumulative


def _load_geometry(filepath: Path) -> pd.DataFrame:
    """讀取 geometry CSV，回傳包含 'channel_index', 'lat', 'lon', 'depth' 的 DataFrame。

    Parameters
    ----------
    filepath : Path
        CSV 檔案路徑。

    Returns
    -------
    pd.DataFrame
        幾何座標表。
    """
    df = pd.read_csv(filepath)
    required_columns = {"channel_index", "lat", "lon", "depth"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"geometry.csv 缺少必要欄位: {missing}，"
            f"實際欄位: {list(df.columns)}"
        )
    return df


def _build_distance_map(geometry_df: pd.DataFrame) -> dict[int, float]:
    """建立 channel_index → 累積距離（米）的映射表。

    Parameters
    ----------
    geometry_df : pd.DataFrame
        幾何座標表。

    Returns
    -------
    dict[int, float]
        映射表。
    """
    cumulative = _compute_cumulative_distances(geometry_df)
    channels = geometry_df.sort_values("channel_index")["channel_index"].values
    return dict(zip(channels, cumulative))


def _map_distances_interpolate(
    patch_channels: np.ndarray,
    distance_map: dict[int, float],
) -> np.ndarray:
    """透過線性插值將 Patch channel index 對應到實際距離（米）。

    對 geometry 中缺失的 channel，利用已知的 (channel, distance) 錨點做線性插值。

    Parameters
    ----------
    patch_channels : np.ndarray
        Patch 原本的 distance 座標值（channel index）。
    distance_map : dict[int, float]
        channel_index → 累積距離（米）的映射。

    Returns
    -------
    np.ndarray
        映射後的實際距離陣列（米）。
    """
    mapped = np.array([distance_map.get(ch, np.nan) for ch in patch_channels])
    missing_mask = np.isnan(mapped)

    if not np.any(missing_mask):
        return mapped

    missing_channels = patch_channels[missing_mask]
    logger.info(
        "對 %d 個缺失 channel 進行線性插值: %s",
        len(missing_channels), list(missing_channels),
    )

    known_channels = np.array(sorted(distance_map.keys()))
    known_distances = np.array([distance_map[ch] for ch in known_channels])

    interp_func = interp1d(
        known_channels, known_distances,
        kind="linear",
        bounds_error=False,
        # scipy >= 1.1 支援 "extrapolate" 字串
        fill_value="extrapolate",  # type: ignore[arg-type]
    )
    mapped[missing_mask] = interp_func(patch_channels[missing_mask])
    return mapped


def _handle_missing_channels(
    patch_channels: np.ndarray,
    distance_map: dict[int, float],
    config: CoordinateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """處理 Patch channel 與 geometry 之間的 mismatch，回傳 (distances, keep_mask)。

    處理策略由 config.missing_channel_strategy 決定：
        - 'interpolate'：用線性插值補上缺失 channel 的距離（預設）
        - 'crop'：只保留有 mapped 的 channel
        - 'error'：遇到缺失直接報錯

    Parameters
    ----------
    patch_channels : np.ndarray
        Patch 原本的 distance 座標值（channel index）。
    distance_map : dict[int, float]
        channel_index → 累積距離（米）的映射。
    config : CoordinateConfig
        座標設定。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (distances, keep_mask)
        distances 長度與 patch_channels 相同（插值時補滿），
        但 crop 模式下 keep_mask 會對應篩選。
    """
    mapped = np.array([distance_map.get(ch, np.nan) for ch in patch_channels])
    missing_mask = np.isnan(mapped)

    if not np.any(missing_mask):
        return mapped, np.ones(len(patch_channels), dtype=bool)

    missing_channels = patch_channels[missing_mask]
    strategy = config.missing_channel_strategy

    if strategy == "error":
        raise ValueError(
            f"geometry.csv 中缺少以下 channel index 的對應資料: "
            f"{list(missing_channels)}。"
            f"請檢查 geometry.csv 是否涵蓋所有資料的 channel。"
        )

    if strategy == "crop":
        keep_mask = ~missing_mask
        logger.warning(
            "裁切 %d 個缺少 geometry 對應的 channel: %s",
            len(missing_channels), list(missing_channels),
        )
        return mapped[keep_mask], keep_mask

    # strategy == 'interpolate'
    distances = _map_distances_interpolate(patch_channels, distance_map)
    return distances, np.ones(len(patch_channels), dtype=bool)


def _convert_phase_to_strain_rate(
    patch: dc.Patch,
    config: CoordinateConfig,
) -> dc.Patch:
    """將相位差資料轉換為應變率。

    公式：strain_rate = phase_strain_constant × f × phase_data
    其中 f = 1 / dt 為取樣頻率（Hz）。

    Parameters
    ----------
    patch : dc.Patch
        原始 Patch（單位為相位差）。
    config : CoordinateConfig
        設定（含 phase_strain_constant）。

    Returns
    -------
    dc.Patch
        轉換後的 Patch（單位為應變率）。
    """
    time_coord = patch.get_coord("time")
    time_vals = time_coord.values
    if len(time_vals) < 2:
        logger.warning("時間軸長度不足 2，無法計算取樣率，跳過單位轉換")
        return patch

    # 計算取樣頻率 (Hz)
    dt_sec = (time_vals[1] - time_vals[0]) / np.timedelta64(1, "s")
    if dt_sec <= 0:
        logger.warning("無效的取樣間隔 dt=%s，跳過單位轉換", dt_sec)
        return patch

    sampling_rate = 1.0 / dt_sec
    scale_factor = config.phase_strain_constant * sampling_rate

    logger.info(
        "相位差 → 應變率轉換: constant=%e, fs=%.2f Hz, scale=%e",
        config.phase_strain_constant,
        sampling_rate,
        scale_factor,
    )

    new_data = patch.data * scale_factor

    patch = patch.update_attrs(
        input_unit="phase",
        phase_strain_constant=config.phase_strain_constant,
        sampling_rate_hz=sampling_rate,
        scale_factor=scale_factor,
    )
    patch = dc.Patch(data=new_data, coords=patch.coords, dims=patch.dims, attrs=patch.attrs)

    return patch


def align(patch: dc.Patch, config: CoordinateConfig) -> dc.Patch:
    """將 Patch 的 distance 座標從 channel index 對齊為實際距離（米）。
    並視需要將資料從相位差轉換為應變率。

    Pipeline:
        1. 讀取 geometry.csv → channel_index→(lat, lon, depth)
        2. 計算 geometry 內所有相鄰 channel 的累積 3D 距離
        3. 將 Patch 的 distance 坐標軸替換為實際累積距離
        4. 若 input_unit == "phase"，執行單位轉換

    Parameters
    ----------
    patch : dc.Patch
        原始 Patch（distance 軸為 channel index）。
    config : CoordinateConfig
        座標設定。

    Returns
    -------
    dc.Patch
        對齊後的 Patch（distance 軸為實際距離，單位米）。
    """
    # ── 步驟 1：載入幾何座標 ──
    geometry_path = Path(config.fiber_geometry_file)
    if not geometry_path.exists():
        raise FileNotFoundError(
            f"geometry.csv 不存在: {geometry_path}"
        )

    geometry_df = _load_geometry(geometry_path)
    distance_map = _build_distance_map(geometry_df)

    # ── 步驟 2：取得 Patch 目前的 channel index ──
    distance_coord = patch.get_coord("distance")

    # dascore 的 coord.values 可能是 tuple 或 ndarray，安全轉換
    patch_channels = np.asarray(distance_coord.values).ravel()

    logger.info(
        "Patch 距離軸範圍: [%s, %s]，共 %d 個 channel",
        patch_channels[0], patch_channels[-1], len(patch_channels),
    )

    # ── 步驟 3：mapping 到實際距離 ──
    new_distances, keep_mask = _handle_missing_channels(
        patch_channels, distance_map, config,
    )

    if config.missing_channel_strategy == "crop":
        # 裁切模式：先用 boolean mask 選取資料，再更新座標
        patch = patch.select(distance=keep_mask)
        logger.info(
            "裁切後保留 %d 個 channel", np.sum(keep_mask),
        )
        # 裁切後 patch_channels 長度不變，new_distances 是 subset
        # 用 new_distances 直接更新 coords（長度已一致）
        patch = patch.update_coords(distance=new_distances)
    else:
        logger.info(
            "距離軸範圍: [%.2f m, %.2f m]",
            new_distances[0], new_distances[-1],
        )
        # ── 步驟 4：更新 Patch 的 distance 座標 ──
        patch = patch.update_coords(distance=new_distances)

    # ── 步驟 5（選擇性）：相位差 → 應變率 ──
    if config.input_unit == "phase":
        patch = _convert_phase_to_strain_rate(patch, config)

    # strict shape check
    if config.strict_shape_check:
        expected_shape = tuple(
            len(patch.get_coord(dim)) for dim in patch.dims
        )
        # dascore data 可透過 .data.shape 或 patch.shape dict 取得
        data_shape = tuple(patch.data.shape[j] for j in range(patch.data.ndim))
        if data_shape != expected_shape:
            raise RuntimeError(
                f"座標對齊後 shape 不一致: data shape={data_shape}, "
                f"dims shape={expected_shape}"
            )

    logger.info("座標對齊完成")
    return patch