"""遠震地層放大效應分析模組。

根據震央距離與表面波群速度計算時間窗、擷取雷利波列，
以基準 channel 中位數振幅為參考，計算各 channel 的放大倍率。
"""

import logging
import warnings
from typing import Optional

import dascore as dc
import numpy as np

from das_pipeline.config import TeleseismicConfig
from das_pipeline.utils.bad_channels import get_bad_channel_indices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nanmedian_of(arr: np.ndarray) -> float:
    """Compute ``np.nanmedian`` while suppressing All-NaN warnings."""
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return float(np.nanmedian(arr))


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
    """
    t_start_sec = distance_km / velocity_max
    t_end_sec = distance_km / velocity_min

    t_start = origin_time + np.timedelta64(int(round(t_start_sec * 1e9)), "ns")
    t_end = origin_time + np.timedelta64(int(round(t_end_sec * 1e9)), "ns")

    logger.info(
        "時間窗: [%s, %s] (D=%g km, v=[%g, %g] km/s, t_start=%.1fs, t_end=%.1fs)",
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
    """
    time_coord = patch.get_coord("time")
    patch_t_min: np.datetime64 = time_coord.min()  # type: ignore[assignment]
    patch_t_max: np.datetime64 = time_coord.max()  # type: ignore[assignment]

    actual_start = max(t_start, patch_t_min)
    actual_end = min(t_end, patch_t_max)

    if actual_end <= actual_start:
        logger.warning(
            "時間窗 [%s, %s] 與 patch 時間範圍 [%s, %s] 無交集",
            t_start, t_end, patch_t_min, patch_t_max,
        )
        return None

    if actual_start != t_start or actual_end != t_end:
        logger.info("時間窗已裁切至交集區間: [%s, %s]", actual_start, actual_end)

    return patch.select(time=(actual_start, actual_end))


def _compute_channel_amplitudes(patch: dc.Patch) -> np.ndarray:
    """計算每個 channel 的振幅中位數（絕對值的中位數）。

    依 Patch 的 ``time`` 維度取 median，使用 ``np.nanmedian``
    避免單一殘留 NaN 拖垮整條 channel。
    """
    data = np.asarray(patch.data)
    time_axis = patch.dims.index("time")
    amplitudes = np.apply_along_axis(_nanmedian_of, time_axis, np.abs(data))

    bad_channels = np.flatnonzero(np.isnan(amplitudes))
    if bad_channels.size > 0:
        logger.warning(
            "有 %d 個 channel 在此時間窗內完全沒有有效資料（全 NaN），"
            "channel index: %s，這些 channel 的放大倍率將為 NaN。",
            bad_channels.size, bad_channels.tolist(),
        )

    logger.info("通道振幅計算完成，shape: %s", amplitudes.shape)
    return amplitudes


def _ref_by_deepest_channels(
    amplitudes: np.ndarray,
    n_reference: int,
) -> float:
    """以最深（最後）N 個 channel 的中位數作為基準振幅。"""
    if n_reference > len(amplitudes):
        logger.warning(
            "基準 channel 數量 (%d) 大於總 channel 數 (%d)，使用全部 channel",
            n_reference, len(amplitudes),
        )
        n_reference = len(amplitudes)

    ref = amplitudes[-n_reference:]
    n_valid = np.sum(~np.isnan(ref))
    if n_valid == 0:
        raise ValueError(
            f"用來當基準的最深 {n_reference} 個 channel 全部為 NaN"
            "（可能整批斷訊/感測器異常），無法計算基準振幅。"
            "請檢查資料品質，或調整 reference_channels / skip_channels。"
        )

    reference = _nanmedian_of(ref)
    if n_valid < ref.size:
        logger.warning(
            "最深 %d 個基準 channel 中有 %d 個是 NaN，已自動排除，"
            "僅用剩餘 %d 個 channel 計算基準振幅: %g",
            n_reference, ref.size - n_valid, n_valid, reference,
        )
    else:
        logger.info(
            "基準振幅: %g (最深 %d 個 channel 的中位數)", reference, n_reference,
        )
    return reference


def _ref_by_distance_range(
    amplitudes: np.ndarray,
    distances: np.ndarray,
    distance_range: tuple[float, float],
    n_reference_fallback: int,
) -> Optional[float]:
    """以距離範圍內的 channel 中位數作為基準振幅。

    Returns ``None`` if:
    - no channels fall in the range, or
    - all selected channels are NaN.

    Caller should fall back to ``_ref_by_deepest_channels`` on ``None``.
    """
    d_min, d_max = distance_range
    mask = (distances >= d_min) & (distances <= d_max)
    ref_indices = np.flatnonzero(mask)

    if ref_indices.size == 0:
        logger.warning(
            "距離範圍 [%g, %g] m 內沒有任何 channel，fallback 使用最深 %d 個 channel",
            d_min, d_max, n_reference_fallback,
        )
        return None

    ref_amplitudes = amplitudes[ref_indices]
    n_valid = np.sum(~np.isnan(ref_amplitudes))

    if n_valid == 0:
        logger.error(
            "距離範圍 [%g, %g] m 內的 %d 個基準 channel 全部為 NaN"
            "（可能整批斷訊/感測器異常），無法計算基準振幅，"
            "fallback 使用最深 %d 個 channel",
            d_min, d_max, ref_indices.size, n_reference_fallback,
        )
        return None

    reference = _nanmedian_of(ref_amplitudes)
    if n_valid < ref_amplitudes.size:
        logger.warning(
            "距離範圍 [%g, %g] m 內 %d 個基準 channel 中有 %d 個是 NaN，"
            "已自動排除，僅用剩餘 %d 個 channel 計算基準振幅: %g",
            d_min, d_max, ref_amplitudes.size,
            ref_amplitudes.size - n_valid, n_valid, reference,
        )
    else:
        logger.info(
            "基準振幅: %g (距離範圍 [%g, %g] m 內 %d 個 channel 的中位數)",
            reference, d_min, d_max, ref_indices.size,
        )
    return reference


def _compute_reference_amplitude(
    amplitudes: np.ndarray,
    n_reference: int,
    distances: Optional[np.ndarray] = None,
    distance_range: Optional[tuple[float, float]] = None,
) -> float:
    """從指定的基準 channel 計算基準振幅。

    支援兩種指定方式：
    1. 以距離範圍指定（水平光纖）：distance_range = (start_m, end_m)
    2. 以最深 N 個 channel 指定（豎井）：distance_range 為 None 時
    """
    if distance_range is not None and distances is not None:
        result = _ref_by_distance_range(
            amplitudes, distances, distance_range, n_reference,
        )
        if result is not None:
            return result

    return _ref_by_deepest_channels(amplitudes, n_reference)


def _extract_distances(patch: dc.Patch) -> np.ndarray:
    """從 Patch 的 distance coord 提取實際距離值（米）。"""
    dist_coord = patch.get_coord("distance")
    distances = np.asarray(dist_coord.values).ravel()
    logger.info("距離軸範圍: [%.2f, %.2f] m", distances[0], distances[-1])
    return distances


def _exclude_bad_channels(
    amplitudes: np.ndarray,
    distances: np.ndarray,
    patch: dc.Patch,
) -> tuple[np.ndarray, np.ndarray, int]:
    """排除 nan_handler 標記的全 NaN channel，回傳 (amplitudes, distances, n_excluded)。"""
    bad_indices = get_bad_channel_indices(patch)
    n_excluded = 0
    if bad_indices:
        keep_mask = np.ones(len(amplitudes), dtype=bool)
        global_bad = np.array(bad_indices, dtype=int)
        global_bad = global_bad[(global_bad >= 0) & (global_bad < len(amplitudes))]
        keep_mask[global_bad] = False
        n_excluded = (~keep_mask).sum()
        if n_excluded > 0:
            logger.warning(
                "排除 %d 個不可用 channel（全 NaN），index: %s",
                n_excluded, global_bad.tolist(),
            )
        amplitudes = amplitudes[keep_mask]
        distances = distances[keep_mask]
    return amplitudes, distances, n_excluded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_amplification(
    patch: dc.Patch,
    config: TeleseismicConfig,
) -> Optional[dict]:
    """對一個 Patch 執行遠震地層放大效應分析。

    流程：
    1. 根據震央距離與群速度計算時間窗 [D/v_max, D/v_min]
    2. 擷取時間窗內的波列
    3. 對每個 channel 計算振幅（絕對值中位數）
    4. 以基準 channel 的中位數作為參考振幅
    5. 計算放大倍率 = channel_amplitude / reference

    Parameters
    ----------
    patch : dc.Patch  已前處理的 DAS Patch。
    config : TeleseismicConfig  遠震分析設定。

    Returns
    -------
    dict or None
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

    # ── Exclude channels flagged as entirely NaN by nan_handler ──
    amplitudes, distances, n_excluded = _exclude_bad_channels(amplitudes, distances, patch)

    skip = config.skip_channels
    if skip > 0:
        logger.info("跳過前 %d 個（井口附近）channel，不參與放大倍率計算", skip)
        amplitudes = amplitudes[skip:]
        distances = distances[skip:]

    reference = _compute_reference_amplitude(
        amplitudes,
        config.reference_channels,
        distances=distances,
        distance_range=config.reference_distance_range,
    )

    if reference > 0:
        amplification = amplitudes / reference
    else:
        logger.warning(
            "基準振幅為 0（%g），放大倍率無法計算，全部 channel 暫以 1.0 表示，"
            "請檢查資料是否異常。", reference,
        )
        amplification = np.ones_like(amplitudes)

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
        "n_excluded_bad_channels": n_excluded,
        "time_window": (str(t_start), str(t_end)),
        "event_distance_km": config.event_distance_km,
        "distance_unit": patch.attrs.get("distance_unit", "m"),
    }