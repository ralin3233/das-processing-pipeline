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

    def test_by_distance_range(self):
        """以距離範圍指定基準，取距離落在範圍內的 channel。"""
        amplitudes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        distances = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ref = _compute_reference_amplitude(
            amplitudes,
            10,
            distances=distances,
            distance_range=(2.0, 4.0),
        )
        # indices 2, 3, 4 → amplitudes [3, 4, 5] → median = 4.0
        assert ref == pytest.approx(4.0)

    def test_by_distance_range_partial(self):
        """距離範圍只對應到部分 channel（含邊界）。"""
        amplitudes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        distances = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
        ref = _compute_reference_amplitude(
            amplitudes,
            2,
            distances=distances,
            distance_range=(0.0, 2.0),
        )
        # indices 0, 1, 2 → amplitudes [1, 2, 3] → median = 2.0
        assert ref == pytest.approx(2.0)

    def test_distance_range_fallback_when_empty(self):
        """距離範圍內沒有任何 channel 時，fallback 到最深 N 個 channel。"""
        amplitudes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        distances = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ref = _compute_reference_amplitude(
            amplitudes,
            2,
            distances=distances,
            distance_range=(100.0, 200.0),
        )
        # fallback → 最後 2 個 → [5, 6] → median = 5.5
        assert ref == pytest.approx(5.5)

    def test_distance_range_ignored_without_distances(self):
        """未提供 distances 時，distance_range 不啟用，沿用最深 N 個 channel。"""
        amplitudes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ref = _compute_reference_amplitude(
            amplitudes,
            3,
            distance_range=(2.0, 4.0),
        )
        assert ref == pytest.approx(5.0)  # median of [4, 5, 6]


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
        assert "distances" in result
        assert "reference_amplitude" in result
        assert len(result["amplification"]) == 50
        assert len(result["distances"]) == 50
        assert result["reference_amplitude"] > 0
        # 確認放大的 channel 倍率 > 1（前 50% channel 被模擬放大 2x）
        assert np.any(result["amplification"] > 1.0)

    def test_distances_equals_patch_coord(self):
        """確認 distances 取自 Patch 的 distance coord，且沒有被重新編排。"""
        patch = _make_dummy_patch(n_time=1000, n_channels=50)
        config = TeleseismicConfig(
            event_distance_km=10,
            event_origin_time="2023-02-06T01:17:35",
        )
        result = compute_amplification(patch, config)
        assert result is not None
        # 檢查距離等於 Patch distance coord
        expected_distances = np.arange(50, dtype=float)
        np.testing.assert_array_equal(result["distances"], expected_distances)

    def test_distances_not_reindexed(self):
        """若 Patch distance coord 有 gap（如 1,2,5），distances 應保持原始值。"""
        time = pd.date_range(start="2023-02-06T01:17:00", periods=1000, freq="50ms")
        rng = np.random.default_rng(42)
        data = rng.normal(loc=0, scale=1.0, size=(1000, 3))
        patch = dc.Patch(
            data=data,
            coords={
                "time": time,
                "distance": np.array([1235, 1236, 1240], dtype=float),
            },
            dims=["time", "distance"],
        )
        config = TeleseismicConfig(
            event_distance_km=10,
            event_origin_time="2023-02-06T01:17:35",
        )
        result = compute_amplification(patch, config)
        assert result is not None
        # distances 應保持原始值 [1235, 1236, 1240]，而非重新編排成 [0, 1, 2]
        np.testing.assert_array_equal(
            result["distances"],
            np.array([1235, 1236, 1240], dtype=float),
        )

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

    def test_skip_channels_preserves_distances(self):
        """skip_channels 裁切後 distances 仍應保持原始值。"""
        patch = _make_dummy_patch(n_time=1000, n_channels=50)
        config = TeleseismicConfig(
            event_distance_km=10,
            event_origin_time="2023-02-06T01:17:35",
            skip_channels=10,
        )

        result = compute_amplification(patch, config)

        assert result is not None
        # 跳過前 10 個 channel（距離 0~9），應保留距離 10~49
        np.testing.assert_array_equal(
            result["distances"],
            np.arange(10.0, 50.0),
        )

    def test_reference_by_distance_range(self):
        """reference_distance_range 指定基準段，落在範圍外的 channel 倍率 > 1。"""
        # D=50 km, v=[2,4] km/s → 時間窗 12.5s~25s（250 個 samples/channel），
        # 統計上中位數較穩定；patch 需涵蓋到 01:18:00。
        patch = _make_dummy_patch(n_time=2000, n_channels=50)  # 01:17:00~01:18:39.95
        config = TeleseismicConfig(
            event_distance_km=50,
            event_origin_time="2023-02-06T01:17:35",
            reference_distance_range=(40.0, 49.0),  # 後段 channel（全部放大 2x）
        )

        result = compute_amplification(patch, config)

        assert result is not None
        assert "reference_amplitude" in result
        assert result["reference_amplitude"] > 0
        # 所有 channel 皆被放大 2x 且基準段也在其中，因此整體倍率應接近 1
        assert abs(np.median(result["amplification"]) - 1.0) < 0.05

    def test_reference_by_distance_range_amplified(self):
        """基準段未放大時，放大的 channel 倍率 > 1。"""
        n_channels = 50
        n_time = 2000
        time = pd.date_range(start="2023-02-06T01:17:00", periods=n_time, freq="50ms")
        rng = np.random.default_rng(42)
        data = rng.normal(loc=0, scale=1.0, size=(n_time, n_channels))
        # 模擬水平光纖：前 30 個 channel 放大 2x，後 20 個 channel 未放大（作為基準）
        data[:, :30] *= 2.0
        patch = dc.Patch(
            data=data,
            coords={
                "time": time,
                "distance": np.arange(n_channels, dtype=float),
            },
            dims=["time", "distance"],
        )
        config = TeleseismicConfig(
            event_distance_km=50,  # 時間窗 12.5s~25s（250 samples/channel）較穩定
            event_origin_time="2023-02-06T01:17:35",
            reference_distance_range=(30.0, 49.0),  # 後段未放大的 channel 為基準
        )

        result = compute_amplification(patch, config)

        assert result is not None
        # 前 30 個（距離 0~29）放大 2x → 倍率 ≈ 2
        amplified = result["amplification"][:30]
        assert np.median(amplified) == pytest.approx(2.0, rel=0.1)
        # 後 20 個（距離 30~49，基準段）→ 倍率 ≈ 1
        reference_seg = result["amplification"][30:]
        assert np.median(reference_seg) == pytest.approx(1.0, rel=0.1)
