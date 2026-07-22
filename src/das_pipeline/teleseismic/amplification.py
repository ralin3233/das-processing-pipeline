# src/das_pipeline/teleseismic/amplification.py

import logging
from typing import Optional

import dascore as dc
import numpy as np

from das_pipeline.config import TeleseismicConfig
logger = logging.getLogger(__name__)


def _parse_origin_time(origin_time_str: str) -> np.datetime64:
    """將 ISO 時間字串轉為 numpy datetime64。"""
    return np.datetime64(origin_time_str)


def _compute_time_window(
    origin_time: np.datetime64,
    distance_km: float,
    velocity_min: float,
    velocity_max: float,
) -> tuple[np.datetime64, np.datetime64]:
    """根據表面波群速度範圍計算時間窗 [t_start, t_end]。

    遠震雷利波列因頻散而拉長，最快群速度的波最早到達 (t_start)，
    最慢群速度的波最晚到達 (t_end)。
    窗口範圍 = [D / v_max, D / v_min]，其中 D 為震央距離 (km)，
    v 為群速度 (km/s)。

    Parameters
    ----------
    origin_time : np.datetime64
        發震時刻。
    distance_km : float
        震央距離 (km)。
    velocity_min : float
        最慢群速度 (km/s)，決定窗口結束時間 t_end。
    velocity_max : float
        最快群速度 (km/s)，決定窗口開始時間 t_start。

    Returns
    -------
    tuple[np.datetime64, np.datetime64]
        (t_start, t_end) 為時間窗的起始與結束時刻。
    """
    # 換算時間（秒），單位換算: t = D / v
    t_start_sec = distance_km / velocity_max   # 最早到達 (秒)
    t_end_sec = distance_km / velocity_min     # 最晚到達 (秒)

    t_start = origin_time + np.timedelta64(int(round(t_start_sec * 1e9)), "ns")
    t_end = origin_time + np.timedelta64(int(round(t_end_sec * 1e9)), "ns")

    logger.info(
        "時間窗: [%s, %s] (D=%g km, v=[%g, %g] km/s, "
        "t_start=%.1fs, t_end=%.1fs)",
        t_start, t_end, distance_km, velocity_min, velocity_max,
        t_start_sec, t_end_sec,
    )

    return t_start, t_end


def _extract_wave_train(
    patch: dc.Patch,
    t_start: np.datetime64,
    t_end: np.datetime64,
) -> Optional[dc.Patch]:
    """從 Patch 中擷取指定時間窗內的波列資料。

    若時間窗不完全落在 patch 的時間範圍內，會自動裁切至交集區間。
    若完全無交集則回傳 None。

    Parameters
    ----------
    patch : dc.Patch
        原始 Patch。
    t_start, t_end : np.datetime64
        時間窗範圍。

    Returns
    -------
    dc.Patch or None
        擷取後的 Patch，或無交集時回傳 None。
    """
    time_coord = patch.get_coord("time")
    patch_t_min: np.datetime64 = time_coord.min()  # type: ignore[assignment]
    patch_t_max: np.datetime64 = time_coord.max()  # type: ignore[assignment]

    # 裁切至交集
    actual_start = max(t_start, patch_t_min)
    actual_end = min(t_end, patch_t_max)

    if actual_end <= actual_start:
        logger.warning(
            "時間窗 [%s, %s] 與 patch 時間範圍 [%s, %s] 無交集",
            t_start, t_end, patch_t_min, patch_t_max,
        )
        return None

    if actual_start != t_start or actual_end != t_end:
        logger.info(
            "時間窗已裁切至交集區間: [%s, %s]",
            actual_start, actual_end,
        )

    return patch.select(time=(actual_start, actual_end))


def _compute_channel_amplitudes(patch: dc.Patch) -> np.ndarray:
    """計算每個 channel 的振幅中位數（絕對值的中位數）。

    依 Patch 的 ``time`` 維度取 median，得到每個 channel 的振幅中位數。
    不假設資料維度順序，因此同時支援 ``("distance", "time")`` 與
    ``("time", "distance")`` 的 Patch。

    Parameters
    ----------
    patch : dc.Patch
        已選取時間窗的波列 Patch，shape = (n_channels, n_time)。

    Returns
    -------
    np.ndarray
        每個 channel 的振幅中位數，shape = (n_channels,)。
    """
    data = np.asarray(patch.data)
    time_axis = patch.dims.index("time")
    # 沿實際的 time 軸取 median，得到每個 channel 一個值。
    amplitudes = np.median(np.abs(data), axis=time_axis)
    logger.info("通道振幅計算完成，shape: %s", amplitudes.shape)
    return amplitudes


def _compute_reference_amplitude(
    amplitudes: np.ndarray,
    n_reference: int,
) -> float:
    """從最深 N 個 channel 計算基準振幅。

    Parameters
    ----------
    amplitudes : np.ndarray
        各通道振幅，由淺至深排列（index 0 = 井口/最淺）。
    n_reference : int
        作為基準的最深 channel 數量。

    Returns
    -------
    float
        基準振幅（最深 N 個 channel 的中位數）。
    """
    if n_reference > len(amplitudes):
        logger.warning(
            "基準 channel 數量 (%d) 大於總 channel 數 (%d)，使用全部 channel",
            n_reference, len(amplitudes),
        )
        n_reference = len(amplitudes)

    # 取最後 N 個 channel（最深處）
    ref_amplitudes = amplitudes[-n_reference:]
    reference = float(np.median(ref_amplitudes))
    logger.info(
        "基準振幅: %g (最深 %d 個 channel 的中位數)", reference, n_reference,
    )
    return reference


def _extract_distances(patch: dc.Patch) -> np.ndarray:
    """從 Patch 的 distance coord 提取實際距離值（米）。

    Parameters
    ----------
    patch : dc.Patch
        DAS Patch。

    Returns
    -------
    np.ndarray
        每個 channel 的距離值（米），shape = (n_channels,)。
    """
    dist_coord = patch.get_coord("distance")
    distances = np.asarray(dist_coord.values).ravel()
    logger.info("距離軸範圍: [%.2f, %.2f] m", distances[0], distances[-1])
    return distances


def compute_amplification(
    patch: dc.Patch,
    config: TeleseismicConfig,
) -> Optional[dict]:
    """對一個 Patch 執行遠震地層放大效應分析。

    流程：
    1. 根據震央距離與群速度計算時間窗 [D/v_max, D/v_min]
    2. 擷取時間窗內的波列
    3. 對每個 channel 計算振幅（絕對值中位數）
    4. 以最深 N 個 channel 的中位數作為基準
    5. 計算放大倍率 = channel_amplitude / reference

    Parameters
    ----------
    patch : dc.Patch
        已前處理的 DAS Patch。
    config : TeleseismicConfig
        遠震分析設定。

    Returns
    -------
    dict or None
        {
            "distances": np.ndarray,           # 每個 channel 的實際距離（米），取自 Patch distance coord
            "amplification": np.ndarray,       # 每個 channel 的放大倍率
            "reference_amplitude": float,      # 基準振幅
            "n_channels": int,                 # 總 channel 數
            "time_window": (str, str),         # 實際使用的時間窗
            "event_distance_km": float,        # 震央距離
        }
        若時間窗與 patch 無交集則回傳 None。
    """
    origin_time = _parse_origin_time(config.event_origin_time)

    t_start, t_end = _compute_time_window(
        origin_time,
        config.event_distance_km,
        config.velocity_min,
        config.velocity_max,
    )

    wave_patch = _extract_wave_train(patch, t_start, t_end)
    if wave_patch is None:
        return None

    amplitudes = _compute_channel_amplitudes(wave_patch)
    distances = _extract_distances(wave_patch)

    skip = config.skip_channels
    if skip > 0:
        logger.info("跳過前 %d 個（井口附近）channel，不參與放大倍率計算", skip)
        amplitudes = amplitudes[skip:]
        distances = distances[skip:]

    reference = _compute_reference_amplitude(
        amplitudes, config.reference_channels,
    )

    amplification = amplitudes / reference if reference > 0 else np.ones_like(amplitudes)

    n_channels = len(amplification)
    logger.info(
        "放大倍率範圍: [%g, %g], 中位數: %g, channel 數: %d",
        np.min(amplification), np.max(amplification),
        np.median(amplification), n_channels,
    )

    return {
        "distances": distances,
        "amplification": amplification,
        "reference_amplitude": reference,
        "n_channels": n_channels,
        "time_window": (str(t_start), str(t_end)),
        "event_distance_km": config.event_distance_km,
    }
