# tests/test_snr.py

"""Tests for the single-channel SNR computation module."""

import numpy as np
import dascore as dc
import pandas as pd
import pytest

from das_pipeline.config import SnrConfig
from das_pipeline.teleseismic.snr import (
    compute_power,
    compute_channel_snr,
    _extract_channel_data,
    _compute_noise_window,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_patch(
    n_time: int = 2000,
    n_channels: int = 100,
    sampling_rate_hz: float = 20.0,
    start_time: str = "2023-02-06T01:17:00",
) -> dc.Patch:
    """建立一個假的 DAS Patch 供 SNR 測試用。"""
    time = pd.date_range(
        start=start_time, periods=n_time,
        freq=f"{1000 / sampling_rate_hz}ms",
    )
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0, scale=1.0, size=(n_time, n_channels))
    return dc.Patch(
        data=data,
        coords={
            "time": time,
            "distance": np.arange(n_channels, dtype=float),
        },
        dims=["time", "distance"],
    )


def _make_dummy_patch_with_signal(
    n_time: int = 2000,
    n_channels: int = 100,
    start_time: str = "2023-02-06T01:17:00",
    signal_amplitude: float = 1.0,
    noise_amplitude: float = 0.1,
    signal_start_s: float = 750,
    signal_duration_s: float = 750,
):
    """建立含有區分明顯訊號段與雜訊段的 Patch。

    訊號段振幅較大，其餘時間段為背景雜訊。
    """
    sampling_rate_hz = 20.0
    time = pd.date_range(
        start=start_time, periods=n_time,
        freq=f"{1000 / sampling_rate_hz}ms",
    )
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0, scale=noise_amplitude, size=(n_time, n_channels))

    signal_start_idx = int(signal_start_s * sampling_rate_hz)
    signal_end_idx = int((signal_start_s + signal_duration_s) * sampling_rate_hz)
    signal_end_idx = min(signal_end_idx, n_time)

    data[signal_start_idx:signal_end_idx, :] = rng.normal(
        loc=0, scale=signal_amplitude,
        size=(signal_end_idx - signal_start_idx, n_channels),
    )

    return dc.Patch(
        data=data,
        coords={"time": time, "distance": np.arange(n_channels, dtype=float)},
        dims=["time", "distance"],
    )



# ---------------------------------------------------------------------------
# Tests: compute_power
# ---------------------------------------------------------------------------


class TestComputePower:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0])
        P = compute_power(data)
        assert P == pytest.approx(14.0 / 3.0)

    def test_zeros(self):
        data = np.zeros(100)
        assert compute_power(data) == 0.0

    def test_all_nan(self):
        data = np.array([np.nan, np.nan, np.nan])
        assert np.isnan(compute_power(data))

    def test_mixed_nan(self):
        data = np.array([1.0, np.nan, 3.0])
        P = compute_power(data)
        assert P == pytest.approx((1.0 + 9.0) / 2.0)

    def test_all_inf(self):
        data = np.array([np.inf, -np.inf])
        assert np.isnan(compute_power(data))

    def test_single_value(self):
        data = np.array([5.0])
        assert compute_power(data) == 25.0


# ---------------------------------------------------------------------------
# Tests: _extract_channel_data
# ---------------------------------------------------------------------------


class TestExtractChannelData:
    def test_basic(self):
        patch = _make_dummy_patch(n_time=100, n_channels=10)
        ch_data = _extract_channel_data(patch, channel_index=3)
        assert ch_data.shape == (100,)

    def test_out_of_bounds_negative(self):
        patch = _make_dummy_patch(n_time=100, n_channels=10)
        with pytest.raises(IndexError, match="超出範圍"):
            _extract_channel_data(patch, channel_index=-1)

    def test_out_of_bounds_too_large(self):
        patch = _make_dummy_patch(n_time=100, n_channels=10)
        with pytest.raises(IndexError, match="超出範圍"):
            _extract_channel_data(patch, channel_index=10)

    def test_first_channel(self):
        patch = _make_dummy_patch(n_time=100, n_channels=10)
        ch_data = _extract_channel_data(patch, channel_index=0)
        assert ch_data.shape == (100,)


# ---------------------------------------------------------------------------
# Tests: _compute_noise_window
# ---------------------------------------------------------------------------


class TestComputeNoiseWindow:
    def test_basic(self):
        t_signal_start = np.datetime64("2023-02-06T01:30:05")
        t_signal_end = np.datetime64("2023-02-06T01:42:35")
        patch_t_min = np.datetime64("2023-02-06T01:17:00")

        noise_start, noise_end = _compute_noise_window(
            t_signal_start, t_signal_end, patch_t_min, noise_offset_s=30.0,
        )

        assert noise_end == np.datetime64("2023-02-06T01:29:35")
        assert noise_start == np.datetime64("2023-02-06T01:17:05")

    def test_clamped_to_patch_min(self):
        t_signal_start = np.datetime64("2023-02-06T01:30:05")
        t_signal_end = np.datetime64("2023-02-06T01:42:35")
        patch_t_min = np.datetime64("2023-02-06T01:25:00")

        noise_start, noise_end = _compute_noise_window(
            t_signal_start, t_signal_end, patch_t_min, noise_offset_s=30.0,
        )

        assert noise_start == patch_t_min
        assert noise_end == np.datetime64("2023-02-06T01:29:35")

    def test_no_overlap(self):
        t_signal_start = np.datetime64("2023-02-06T01:30:05")
        t_signal_end = np.datetime64("2023-02-06T01:42:35")
        patch_t_min = np.datetime64("2023-02-06T01:35:00")




# ---------------------------------------------------------------------------
# Tests: compute_channel_snr
# ---------------------------------------------------------------------------


class TestComputeChannelSnr:
    def test_signal_window_no_overlap(self):
        """Patch 時間完全在訊號窗之前 → 回傳 None。"""
        patch = _make_dummy_patch(
            n_time=100, start_time="2023-02-05T00:00:00",
        )
        config = SnrConfig(
            event_distance_km=3000,
            event_origin_time="2023-02-06T01:17:35",
            channel_index=0,
        )
        result = compute_channel_snr(patch, config)
        assert result is None

    def test_channel_out_of_bounds(self):
        """channel_index 超出範圍 → 回傳 None。"""
        patch = _make_dummy_patch(n_time=1000, n_channels=10)
        config = SnrConfig(
            event_distance_km=3000,
            event_origin_time="2023-02-06T01:17:35",
            channel_index=100,
        )
        result = compute_channel_snr(patch, config)
        assert result is None

    def test_basic_snr_positive(self):
        """訊號振幅 >> 雜訊振幅 → SNR 應為正值。"""
        patch = _make_dummy_patch_with_signal(
            n_time=2000, n_channels=10,
            start_time="2023-02-06T01:17:00",
            signal_amplitude=1.0,
            noise_amplitude=0.1,
            signal_start_s=47.5,
            signal_duration_s=12.5,
        )
        config = SnrConfig(
            event_distance_km=50,
            event_origin_time="2023-02-06T01:17:35",
            channel_index=3,
            velocity_min=2.0,
            velocity_max=4.0,
            noise_offset_s=30.0,
        )
        result = compute_channel_snr(patch, config)
        assert result is not None
        assert result["channel_index"] == 3
        assert result["snr_db"] > 0
        assert result["P_signal"] > result["P_noise"]

    def test_snr_result_keys(self):
        """回傳的 dict 包含所有必要欄位。"""
        patch = _make_dummy_patch_with_signal(
            n_time=2000, n_channels=10,
            start_time="2023-02-06T01:17:00",
            signal_amplitude=1.0,
            noise_amplitude=0.1,
            signal_start_s=47.5,
            signal_duration_s=12.5,
        )
        config = SnrConfig(
            event_distance_km=50,
            event_origin_time="2023-02-06T01:17:35",
            channel_index=5,
        )
        result = compute_channel_snr(patch, config)
        assert result is not None
        for key in ("snr_db", "P_signal", "P_noise",
                     "signal_window", "noise_window",
                     "channel_index", "event_distance_km"):
            assert key in result, f"Missing key: {key}"

    def test_noise_window_before_patch(self):
        """雜訊窗完全在 patch 之前 → 回傳 None。"""
        patch = _make_dummy_patch(
            n_time=1000, n_channels=10,
            start_time="2023-02-06T01:30:00",
        )
        config = SnrConfig(
            event_distance_km=1,
            event_origin_time="2023-02-06T01:30:05",
            channel_index=0,
            noise_offset_s=30.0,
        )
        result = compute_channel_snr(patch, config)
        assert result is None
