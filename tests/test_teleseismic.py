# tests/test_teleseismic.py

import numpy as np
import dascore as dc
import pandas as pd
import pytest

from das_pipeline.config import TeleseismicConfig
from das_pipeline.teleseismic.amplification import (
    _parse_origin_time,
    _compute_time_window,
    _extract_wave_train,
    _compute_channel_amplitudes,
    _compute_reference_amplitude,
    compute_amplification,
)


def _make_dummy_patch(
    n_time: int = 2000,
    n_channels: int = 100,
    sampling_rate_hz: float = 20.0,
    start_time: str = "2023-02-06T01:17:00",
) -> dc.Patch:
    """建立一個假的 DAS Patch 供測試用。"""
    time = pd.date_range(start=start_time, periods=n_time, freq=f"{1000/sampling_rate_hz}ms")
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0, scale=1.0, size=(n_time, n_channels))

    # 模擬放大效應：前 50 個 channel 振幅放大
    data[:, :50] *= 2.0

    patch = dc.Patch(
        data=data,
        coords={
            "time": time,
            "distance": np.arange(n_channels, dtype=float),
        },
        dims=["time", "distance"],
    )
    return patch


class TestParseOriginTime:
    def test_iso_format(self):
        result = _parse_origin_time("2023-02-06T01:17:35")
        assert isinstance(result, np.datetime64)

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            _parse_origin_time("not-a-date")


class TestComputeTimeWindow:
    def test_basic_window(self):
        origin = np.datetime64("2023-02-06T01:17:35")
        t_start, t_end = _compute_time_window(origin, 3000, 2.0, 4.0)
        # D=3000km, v_max=4 km/s => t_start = 750s after origin
        # D=3000km, v_min=2 km/s => t_end = 1500s after origin
        expected_start = origin + np.timedelta64(750, "s")
        expected_end = origin + np.timedelta64(1500, "s")
        assert t_start == expected_start, f"{t_start} != {expected_start}"
        assert t_end == expected_end, f"{t_end} != {expected_end}"

    def test_zero_distance(self):
        origin = np.datetime64("2023-02-06T01:17:35")
        t_start, t_end = _compute_time_window(origin, 0, 2.0, 4.0)
        assert t_start == origin
        assert t_end == origin


class TestExtractWaveTrain:
    def test_window_within_patch(self):
        patch = _make_dummy_patch()
        origin = np.datetime64("2023-02-06T01:17:35")
        t_start = origin + np.timedelta64(10, "s")
        t_end = origin + np.timedelta64(30, "s")

        result = _extract_wave_train(patch, t_start, t_end)
        assert result is not None
        assert result.shape[0] > 0  # 有時間點

    def test_window_no_overlap(self):
        patch = _make_dummy_patch()
        origin = np.datetime64("2023-02-06T02:00:00")  # after patch
        t_start = origin
        t_end = origin + np.timedelta64(10, "s")

        result = _extract_wave_train(patch, t_start, t_end)
        assert result is None


class TestComputeChannelAmplitudes:
    def test_basic(self):
        patch = _make_dummy_patch(n_time=100, n_channels=10)
        amplitudes = _compute_channel_amplitudes(patch)
        assert amplitudes.shape == (10,)
        assert np.all(amplitudes > 0)

    def test_amplified_channels_higher(self):
        # 前 5 個 channel 有放大
        patch = _make_dummy_patch(n_time=500, n_channels=20)
        amplitudes = _compute_channel_amplitudes(patch)
        # 前 50% channel 被放大了 2.0 倍
        assert np.mean(amplitudes[:10]) > np.mean(amplitudes[10:])


class TestComputeReferenceAmplitude:
    def test_basic(self):
        amplitudes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ref = _compute_reference_amplitude(amplitudes, 3)
        assert ref == pytest.approx(5.0)  # median of [4, 5, 6]

    def test_more_ref_than_channels(self):
        amplitudes = np.array([1.0, 2.0, 3.0])
        ref = _compute_reference_amplitude(amplitudes, 10)
        assert ref == pytest.approx(2.0)  # median of all


class TestComputeAmplification:
    def test_basic(self):
        # Patch 時間範圍: 01:17:00 ~ 01:17:49.95 (50 秒, 20 Hz * 1000 samples)
        # D=10 km, v_max=4 km/s => t_start = 2.5s after origin (01:17:35 → 01:17:37.5)
        # D=10 km, v_min=2 km/s => t_end = 5.0s after origin  (01:17:35 → 01:17:40)
        patch = _make_dummy_patch(n_time=1000, n_channels=50)
        config = TeleseismicConfig(
            event_distance_km=10,                      # 小距離確保落在 patch 內
            event_origin_time="2023-02-06T01:17:35",   # 事件發生在 patch 中間
            reference_channels=10,
            velocity_min=2.0,
            velocity_max=4.0,
        )
        result = compute_amplification(patch, config)
        assert result is not None, f"result is None, patch time range might not cover the window"
        assert "amplification" in result
        assert "channel_indices" in result
        assert "reference_amplitude" in result
        assert len(result["channel_indices"]) == 50
        assert len(result["amplification"]) == 50
        assert result["reference_amplitude"] > 0
        # 確認放大的 channel 倍率 > 1（前 50% channel 被模擬放大 2x）
        assert np.any(result["amplification"] > 1.0)

    def test_no_overlap(self):
        # patch 時間在事件之前
        patch = _make_dummy_patch(
            n_time=100, start_time="2023-02-05T00:00:00",
        )
        config = TeleseismicConfig(
            event_distance_km=3000,
            event_origin_time="2023-02-06T01:17:35",
        )
        result = compute_amplification(patch, config)
        assert result is None