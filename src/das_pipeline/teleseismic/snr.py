# src/das_pipeline/teleseismic/snr.py

"""單一 channel 的 SNR（訊雜比）計算模組。

核心公式
--------
    P = (1/N) * Σ x_i²          （x_i = 時間窗內的振幅樣本）
    SNR_dB = 10 * log10(P_signal / P_noise)

訊號窗
------
    與遠震放大倍率分析共用相同的時間窗演算法：
    t_start = origin_time + distance / v_max
    t_end   = origin_time + distance / v_min

雜訊窗
------
    noise_end   = t_signal_start - noise_offset_s
    noise_start = max(patch_t_min, noise_end - L_signal)
    其中 L_signal = t_signal_end - t_signal_start
    長度等於訊號窗（若往前資料不足則自動縮短）。
"""

from __future__ import annotations

import logging
from typing import Optional

import dascore as dc
import numpy as np

from das_pipeline.config import SnrConfig
from das_pipeline.teleseismic.amplification import (
    _compute_time_window,
    _parse_origin_time,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def compute_power(data: np.ndarray) -> float:
    """計算訊號的平均功率。

    P = (1/N) * Σ x_i²

    Parameters
    ----------
    data : np.ndarray
        一維振幅序列（單一 channel 在時間窗內的樣本）。

    Returns
    -------
    float
        平均功率。若輸入為空或全為 NaN 則回傳 ``np.nan``。
    """
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid ** 2))


def _extract_channel_data(
    patch: dc.Patch,
    channel_index: int,
) -> np.ndarray:
    """從 Patch 中擷取指定 channel 的一維時間序列資料。

    Parameters
    ----------
    patch : dc.Patch
        已選取時間窗的 Patch，dims 為 ``["time", "distance"]``。
    channel_index : int
        要擷取的 channel 索引（對應 ``distance`` 維度）。

    Returns
    -------
    np.ndarray
        一維振幅序列，shape = (n_time_samples,)。

    Raises
    ------
    IndexError
        若 channel_index 超出 patch 的 distance 維度範圍。
    """
    data = np.asarray(patch.data)
    distance_axis = patch.dims.index("distance")
    n_channels = data.shape[distance_axis]

    if channel_index < 0 or channel_index >= n_channels:
        raise IndexError(
            f"channel_index={channel_index} 超出範圍 [0, {n_channels - 1}]"
        )

    return np.take(data, channel_index, axis=distance_axis)


def _compute_noise_window(
    t_signal_start: np.datetime64,
    t_signal_end: np.datetime64,
    patch_t_min: np.datetime64,
    noise_offset_s: float,
) -> tuple[np.datetime64, np.datetime64]:
    """計算雜訊時間窗。

    雜訊窗結束於訊號窗開始前 ``noise_offset_s`` 秒，
    往前（更早方向）取長度等於訊號窗的區間。
    若往前資料不足，則從 patch 最早時間點開始。

    Returns
    -------
    tuple[np.datetime64, np.datetime64]
        (noise_start, noise_end)。
    """
    signal_duration_s = (t_signal_end - t_signal_start) / np.timedelta64(1, "s")

    noise_end = t_signal_start - np.timedelta64(
        int(round(noise_offset_s * 1e9)), "ns"
    )
    noise_start = noise_end - np.timedelta64(
        int(round(signal_duration_s * 1e9)), "ns"
    )

    # 若雜訊窗超出 patch 範圍，自動裁切（取交集）
    actual_start = max(noise_start, patch_t_min)
    actual_end = noise_end

    if actual_end <= actual_start:
        logger.warning(
            "雜訊窗 [%s, %s] 與 patch 時間範圍無交集（patch 最早: %s）",
            noise_start, noise_end, patch_t_min,
        )
        return noise_start, noise_end

    if actual_start != noise_start:
        logger.info(
            "雜訊窗往前超出 patch 範圍，已裁切: [%s, %s] → [%s, %s]",
            noise_start, noise_end, actual_start, actual_end,
        )

    return actual_start, actual_end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_channel_snr(
    patch: dc.Patch,
    config: SnrConfig,
) -> Optional[dict]:
    """計算單一 channel 的 SNR（訊雜比）。

    流程：
    1. 根據震央距離與群速度計算訊號時間窗 [D/v_max, D/v_min]
    2. 從 patch 中擷取該 channel 在訊號窗內的資料 → 計算 P_signal
    3. 計算雜訊時間窗（訊號窗開始前 noise_offset_s 秒，往前取等長）
    4. 從 patch 中擷取該 channel 在雜訊窗內的資料 → 計算 P_noise
    5. SNR_dB = 10 * log10(P_signal / P_noise)

    Parameters
    ----------
    patch : dc.Patch
        已前處理的 DAS Patch，dims 為 ``["time", "distance"]``。
    config : SnrConfig
        SNR 分析設定。

    Returns
    -------
    dict or None
        若訊號窗或雜訊窗與 patch 時間範圍無交集則回傳 None。
        成功時回傳 dict:

        - ``snr_db`` : float
        - ``P_signal`` : float
        - ``P_noise`` : float
        - ``signal_window`` : tuple[str, str]
        - ``noise_window`` : tuple[str, str]
        - ``channel_index`` : int
        - ``event_distance_km`` : float
    """
    # ── 1. 訊號時間窗 ──
    origin_time = _parse_origin_time(config.event_origin_time)
    t_signal_start, t_signal_end = _compute_time_window(
        origin_time,
        config.event_distance_km,
        config.velocity_min,
        config.velocity_max,
    )

    time_coord = patch.get_coord("time")
    patch_t_min: np.datetime64 = time_coord.min()  # type: ignore[assignment]
    patch_t_max: np.datetime64 = time_coord.max()  # type: ignore[assignment]

    actual_signal_start = max(t_signal_start, patch_t_min)
    actual_signal_end = min(t_signal_end, patch_t_max)

    if actual_signal_end <= actual_signal_start:
        logger.warning(
            "訊號窗 [%s, %s] 與 patch 時間範圍 [%s, %s] 無交集",
            t_signal_start, t_signal_end, patch_t_min, patch_t_max,
        )
        return None

    if actual_signal_start != t_signal_start or actual_signal_end != t_signal_end:
        logger.info(
            "訊號窗已裁切至交集區間: [%s, %s]",
            actual_signal_start, actual_signal_end,
        )

    # ── 2. 擷取訊號段 → P_signal ──
    try:
        signal_patch = patch.select(time=(actual_signal_start, actual_signal_end))
    except Exception:
        logger.exception("擷取訊號窗失敗")
        return None

    try:
        signal_data = _extract_channel_data(signal_patch, config.channel_index)
    except IndexError as e:
        logger.error("無法擷取 channel %d: %s", config.channel_index, e)
        return None

    P_signal = compute_power(signal_data)
    if np.isnan(P_signal):
        logger.warning("channel %d 在訊號窗內無有效資料（全 NaN）", config.channel_index)
        return None

    logger.info(
        "channel %d P_signal = %.6e (訊號窗: [%s, %s], N=%d)",
        config.channel_index, P_signal,
        actual_signal_start, actual_signal_end, len(signal_data),
    )

    # ── 3. 雜訊時間窗 ──
    noise_start, noise_end = _compute_noise_window(
        actual_signal_start, actual_signal_end,
        patch_t_min, config.noise_offset_s,
    )

    if noise_end <= noise_start:
        logger.warning("雜訊窗 [%s, %s] 無效", noise_start, noise_end)
        return None

    # ── 4. 擷取雜訊段 → P_noise ──
    try:
        noise_patch = patch.select(time=(noise_start, noise_end))
    except Exception:
        logger.exception("擷取雜訊窗失敗")
        return None

    try:
        noise_data = _extract_channel_data(noise_patch, config.channel_index)
    except IndexError as e:
        logger.error("無法擷取 channel %d: %s", config.channel_index, e)
        return None

    P_noise = compute_power(noise_data)
    if np.isnan(P_noise):
        logger.warning("channel %d 在雜訊窗內無有效資料（全 NaN）", config.channel_index)
        return None

    logger.info(
        "channel %d P_noise  = %.6e (雜訊窗: [%s, %s], N=%d)",
        config.channel_index, P_noise,
        noise_start, noise_end, len(noise_data),
    )

    # ── 5. SNR_dB ──
    if P_noise == 0.0:
        logger.warning("P_noise = 0，SNR 無法計算（分母為零）")
        return None

    snr_db = 10.0 * np.log10(P_signal / P_noise)

    logger.info(
        "channel %d SNR = %.2f dB (P_signal=%.6e, P_noise=%.6e)",
        config.channel_index, snr_db, P_signal, P_noise,
    )

    return {
        "snr_db": snr_db,
        "P_signal": P_signal,
        "P_noise": P_noise,
        "signal_window": (str(actual_signal_start), str(actual_signal_end)),
        "noise_window": (str(noise_start), str(noise_end)),
        "channel_index": config.channel_index,
        "event_distance_km": config.event_distance_km,
    }

