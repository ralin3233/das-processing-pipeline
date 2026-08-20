# tests/test_sta_lta.py

"""Tests for STA/LTA trigger detection powered by DASCore.

Covers:
  1. compute_sta_lta_patch: DASCore stalta() integration
  2. detect_events: trigger/detrigger logic with synthetic pulses
  3. Spatial consistency filtering (single-channel spike rejected)
  4. Short spike rejection (min_event_duration_s)
  5. Event merging (merge_window_s)
  6. Threshold boundary (no chatter)
  7. DASCore stalta() direct regression tests
  8. Config validation
"""

import numpy as np
import pandas as pd
import pytest

import dascore as dc

from das_pipeline.config import StaLtaConfig
from das_pipeline.detection.sta_lta import (
    compute_sta_lta_patch,
    detect_events,
    _merge_nearby_events,
)
from das_pipeline.cli.helpers import handle_bad_channels_for_detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_patch(
    n_time: int = 2000,
    n_channels: int = 5,
    sampling_rate_hz: float = 100.0,
    start_time: str = "2023-02-06T01:00:00",
    seed: int = 42,
    noise_scale: float = 1.0,
    *,
    pulse_start_s: float | None = None,
    pulse_duration_s: float = 0.5,
    pulse_amplitude: float = 10.0,
    pulse_channels: slice = slice(None),
) -> dc.Patch:
    """Create a DAS Patch with white noise + optional Gaussian pulse.

    Pulse is injected during construction (before DASCore freezes data).
    """
    sr = sampling_rate_hz
    time = pd.date_range(
        start=start_time,
        periods=n_time,
        freq=f"{1000 / sr}ms",
    )
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=noise_scale, size=(n_time, n_channels))
    data = np.ascontiguousarray(data)

    if pulse_start_s is not None:
        t_start = int(pulse_start_s * sr)
        dur = max(1, int(pulse_duration_s * sr))
        t_env = np.arange(dur)
        envelope = np.exp(-0.5 * ((t_env - dur / 2) / (dur / 6)) ** 2)
        pulse_wave = pulse_amplitude * envelope
        t_end = min(t_start + dur, n_time)
        actual_len = t_end - t_start
        data[t_start:t_end, pulse_channels] += pulse_wave[:actual_len, np.newaxis]

    return dc.Patch(
        data=data,
        coords={"time": time, "distance": np.arange(n_channels, dtype=float)},
        dims=["time", "distance"],
    )


def get_sampling_rate(patch: dc.Patch) -> float:
    time_vals = patch.coords.get_array("time")
    return 1.0 / ((time_vals[1] - time_vals[0]) / np.timedelta64(1, "s"))


def _make_ratio_patch(
    data: np.ndarray,
    sampling_rate_hz: float = 10.0,
    start_time: str = "2023-02-06T01:00:00",
) -> dc.Patch:
    """Create a Patch containing a synthetic STA/LTA ratio matrix."""
    time = pd.date_range(
        start=start_time,
        periods=data.shape[0],
        freq=f"{1000 / sampling_rate_hz}ms",
    )
    return dc.Patch(
        data=np.asarray(data, dtype=float),
        coords={"time": time, "distance": np.arange(data.shape[1], dtype=float)},
        dims=["time", "distance"],
    )


class TestIgnoreLeadingChannels:
    def test_ignores_original_leading_channels_time_first(self):
        data = np.ones((4, 6))
        patch = dc.Patch(
            data=data,
            coords={
                "time": pd.date_range("2023-02-06T01:00:00", periods=4, freq="100ms"),
                "distance": np.arange(6, dtype=float),
            },
            dims=["time", "distance"],
            attrs={"all_nan_channel_indices": "4"},
        )

        cleaned, local_to_orig = handle_bad_channels_for_detection(
            patch, ignore_leading_channels=2,
        )

        assert cleaned.shape == (4, 3)
        assert local_to_orig == [2, 3, 5]
        np.testing.assert_array_equal(
            cleaned.coords.get_array("distance"), np.array([2.0, 3.0, 5.0]),
        )

    def test_rejects_invalid_count(self):
        patch = _make_ratio_patch(np.ones((4, 3)))

        with pytest.raises(ValueError, match="不可超過"):
            handle_bad_channels_for_detection(patch, ignore_leading_channels=4)


# ===================================================================
# compute_sta_lta_patch — DASCore integration
# ===================================================================

class TestComputeStaLtaPatch:
    def test_returns_patch_same_time_length(self):
        """DASCore stalta uses mode='same' → preserves # time samples."""
        patch = _make_dummy_patch(n_time=2000, n_channels=4)
        config = StaLtaConfig(sta_window_s=0.5, lta_window_s=10.0)
        result = compute_sta_lta_patch(patch, config)

        assert isinstance(result, dc.Patch)
        assert "time" in result.dims
        assert result.shape[result.dims.index("time")] == 2000

    def test_ratio_near_one_for_noise(self):
        """Pure noise → median ratio ≈ 1 (skip LTA NaN ramp-up)."""
        patch = _make_dummy_patch(n_time=5000, n_channels=4, sampling_rate_hz=100.0)
        config = StaLtaConfig(sta_window_s=0.2, lta_window_s=5.0)
        result = compute_sta_lta_patch(patch, config)

        data = np.asarray(result.data)
        sr = get_sampling_rate(patch)
        lta_samples = int(config.lta_window_s * sr)
        valid = data[lta_samples:]  # skip NaN ramp (time is axis 0)
        valid = valid[np.isfinite(valid)]
        median_ratio = float(np.median(valid))
        assert 0.7 < median_ratio < 1.3, f"median ratio = {median_ratio:.3f}"

    def test_pulse_elevates_ratio(self):
        """Pulse causes max(ratio) > 5."""
        patch = _make_dummy_patch(
            n_time=3000, n_channels=3, sampling_rate_hz=100.0,
            pulse_start_s=20.0, pulse_duration_s=0.5, pulse_amplitude=30.0,
        )
        config = StaLtaConfig(sta_window_s=0.5, lta_window_s=5.0)
        result = compute_sta_lta_patch(patch, config)
        data = np.asarray(result.data)
        assert np.nanmax(data) > 5.0, f"max ratio = {np.nanmax(data):.2f}"

    def test_matches_dascore_direct(self):
        """Our wrapper == DASCore direct stalta call."""
        patch = _make_dummy_patch(
            n_time=2000, n_channels=3,
            pulse_start_s=15.0, pulse_duration_s=0.5, pulse_amplitude=20.0,
        )
        config = StaLtaConfig(sta_window_s=0.5, lta_window_s=5.0)
        ours = compute_sta_lta_patch(patch, config)
        ref = patch.abs().stalta(time=(0.5, 5.0))
        np.testing.assert_allclose(
            np.asarray(ours.data), np.asarray(ref.data), rtol=1e-12, atol=1e-14,
        )


# ===================================================================
# detect_events — synthetic signal tests
# ===================================================================

class TestDetectEventsSynthetic:
    def test_pure_noise_no_trigger(self):
        patch = _make_dummy_patch(n_time=4000, n_channels=4, sampling_rate_hz=200.0, seed=7)
        config = StaLtaConfig(
            sta_window_s=0.5, lta_window_s=5.0,
            trigger_threshold=3.0, detrigger_threshold=1.5,
            min_channels_triggered=2, min_event_duration_s=0.1,
        )
        sr = get_sampling_rate(patch)
        sta_lta = compute_sta_lta_patch(patch, config)
        events = detect_events(sta_lta, config, sr)
        assert len(events) == 0

    def test_below_min_channels_filtered(self):
        """Pulse on 1/5 channels → filtered (min_channels_triggered=3)."""
        patch = _make_dummy_patch(
            n_time=4000, n_channels=5, sampling_rate_hz=200.0, seed=11,
            pulse_start_s=8.0, pulse_duration_s=0.4, pulse_amplitude=30.0,
            pulse_channels=slice(0, 1),
        )
        config = StaLtaConfig(
            sta_window_s=0.3, lta_window_s=3.0,
            trigger_threshold=5.0, detrigger_threshold=2.0,
            min_channels_triggered=3, min_event_duration_s=0.05,
        )
        sr = get_sampling_rate(patch)
        sta_lta = compute_sta_lta_patch(patch, config)
        events = detect_events(sta_lta, config, sr)
        assert len(events) == 0

    def test_multi_channel_trigger(self):
        """Pulse on 4/5 channels → 1 event, correct channel list."""
        patch = _make_dummy_patch(
            n_time=5000, n_channels=5, sampling_rate_hz=200.0, seed=13,
            pulse_start_s=12.0, pulse_duration_s=0.5, pulse_amplitude=20.0,
            pulse_channels=slice(0, 4),
        )
        config = StaLtaConfig(
            sta_window_s=0.5, lta_window_s=4.0,
            trigger_threshold=4.0, detrigger_threshold=1.5,
            min_channels_triggered=3, min_event_duration_s=0.1,
        )
        sr = get_sampling_rate(patch)
        sta_lta = compute_sta_lta_patch(patch, config)
        events = detect_events(sta_lta, config, sr)

        assert len(events) == 1
        evt = events[0]
        for c in [0, 1, 2, 3]:
            assert c in evt["triggered_channels"]
        assert 4 not in evt["triggered_channels"]

    def test_short_pulse_filtered(self):
        """Pulse < min_event_duration_s → 0 events."""
        patch = _make_dummy_patch(
            n_time=4000, n_channels=5, sampling_rate_hz=200.0, seed=17,
            pulse_start_s=10.0, pulse_duration_s=0.01, pulse_amplitude=50.0,
        )
        config = StaLtaConfig(
            sta_window_s=0.1, lta_window_s=2.0,
            trigger_threshold=4.0, detrigger_threshold=2.0,
            min_channels_triggered=3, min_event_duration_s=0.05,
        )
        sr = get_sampling_rate(patch)
        sta_lta = compute_sta_lta_patch(patch, config)
        events = detect_events(sta_lta, config, sr)
        assert len(events) == 0

    def test_trigger_time_accuracy(self):
        """Detected start time within sta_window_s of true arrival."""
        true_arrival_s = 20.0
        patch = _make_dummy_patch(
            n_time=6000, n_channels=5, sampling_rate_hz=200.0, seed=99,
            pulse_start_s=true_arrival_s, pulse_duration_s=0.5, pulse_amplitude=25.0,
        )
        config = StaLtaConfig(
            sta_window_s=0.3, lta_window_s=3.0,
            trigger_threshold=5.0, detrigger_threshold=2.0,
            min_channels_triggered=3, min_event_duration_s=0.05,
        )
        sr = get_sampling_rate(patch)
        sta_lta = compute_sta_lta_patch(patch, config)
        events = detect_events(sta_lta, config, sr)

        assert len(events) >= 1
        detected_start = np.datetime64(events[0]["start_time"])
        time_vals = patch.coords.get_array("time")
        true_start = time_vals[0] + np.timedelta64(int(true_arrival_s * 1e9), "ns")
        error_s = abs((detected_start - true_start) / np.timedelta64(1, "s"))
        assert error_s <= config.sta_window_s + 0.15


# ===================================================================
# detect_events — threshold boundary
# ===================================================================

class TestDetectEventsBoundary:
    def test_no_chatter(self):
        """Clean pulse → exactly 1 event."""
        patch = _make_dummy_patch(
            n_time=6000, n_channels=4, sampling_rate_hz=200.0, seed=21,
            pulse_start_s=15.0, pulse_duration_s=1.0, pulse_amplitude=12.0,
        )
        config = StaLtaConfig(
            sta_window_s=0.5, lta_window_s=5.0,
            trigger_threshold=3.0, detrigger_threshold=1.5,
            min_channels_triggered=2, min_event_duration_s=0.1,
        )
        sr = get_sampling_rate(patch)
        sta_lta = compute_sta_lta_patch(patch, config)
        events = detect_events(sta_lta, config, sr)
        assert len(events) == 1

    def test_hysteresis_works(self):
        """detrigger < trigger prevents on/off chatter."""
        patch = _make_dummy_patch(
            n_time=6000, n_channels=4, sampling_rate_hz=200.0, seed=23,
            pulse_start_s=15.0, pulse_duration_s=0.8, pulse_amplitude=8.0,
        )
        config = StaLtaConfig(
            sta_window_s=0.5, lta_window_s=5.0,
            trigger_threshold=2.5, detrigger_threshold=1.2,
            min_channels_triggered=2, min_event_duration_s=0.1,
        )
        sr = get_sampling_rate(patch)
        sta_lta = compute_sta_lta_patch(patch, config)
        events = detect_events(sta_lta, config, sr)
        assert len(events) <= 2

    def test_lta_warmup_does_not_trigger(self):
        """Samples before the first complete LTA window are ignored."""
        data = np.full((30, 3), 10.0)
        config = StaLtaConfig(
            sta_window_s=0.5, lta_window_s=2.0,
            trigger_threshold=3.0, detrigger_threshold=1.5,
            min_channels_triggered=2, min_event_duration_s=0.1,
        )
        patch = _make_ratio_patch(data, sampling_rate_hz=10.0)

        events = detect_events(patch, config, sampling_rate=10.0)

        assert len(events) == 1
        first_complete_lta = int(config.lta_window_s * 10.0) - 1
        assert np.datetime64(events[0]["start_time"]) == patch.coords.get_array("time")[first_complete_lta]

    def test_detrigger_uses_spatial_count(self):
        """Event ends when fewer than the minimum channels exceed detrigger."""
        data = np.full((20, 3), 1.0)
        data[2:8, :] = 4.0
        data[8:12, 0] = 2.0
        data[8:12, 1:] = 1.0
        config = StaLtaConfig(
            sta_window_s=0.5, lta_window_s=0.5,
            trigger_threshold=3.0, detrigger_threshold=1.5,
            min_channels_triggered=2, min_event_duration_s=0.2,
        )
        patch = _make_ratio_patch(data, sampling_rate_hz=10.0)

        events = detect_events(patch, config, sampling_rate=10.0)

        assert len(events) == 1
        assert events[0]["end_time"] == str(patch.coords.get_array("time")[7])


# ===================================================================
# Event merging
# ===================================================================

class TestEventMerging:
    def test_merge_close_events(self):
        events = [
            {"start_time": "2023-02-06T01:00:00.000", "end_time": "2023-02-06T01:00:02.000",
             "peak_time": "2023-02-06T01:00:01.000", "peak_ratio": 5.0,
             "triggered_channels": [0, 1, 2], "num_triggered_channels": 3, "duration_s": 2.0},
            {"start_time": "2023-02-06T01:00:02.500", "end_time": "2023-02-06T01:00:04.000",
             "peak_time": "2023-02-06T01:00:03.000", "peak_ratio": 6.0,
             "triggered_channels": [1, 2, 3], "num_triggered_channels": 3, "duration_s": 1.5},
        ]
        merged = _merge_nearby_events(events, merge_window_s=1.0)
        assert len(merged) == 1
        assert merged[0]["peak_ratio"] == 6.0
        assert set(merged[0]["triggered_channels"]) == {0, 1, 2, 3}

    def test_no_merge_far_events(self):
        events = [
            {"start_time": "2023-02-06T01:00:00.000", "end_time": "2023-02-06T01:00:01.000",
             "peak_time": "2023-02-06T01:00:00.500", "peak_ratio": 5.0,
             "triggered_channels": [0], "num_triggered_channels": 1, "duration_s": 1.0},
            {"start_time": "2023-02-06T01:00:05.000", "end_time": "2023-02-06T01:00:06.000",
             "peak_time": "2023-02-06T01:00:05.500", "peak_ratio": 4.0,
             "triggered_channels": [1], "num_triggered_channels": 1, "duration_s": 1.0},
        ]
        merged = _merge_nearby_events(events, merge_window_s=1.0)
        assert len(merged) == 2

    def test_merge_window_zero(self):
        events = [
            {"start_time": "2023-02-06T01:00:00.000", "end_time": "2023-02-06T01:00:01.000",
             "peak_time": "2023-02-06T01:00:00.500", "peak_ratio": 3.0,
             "triggered_channels": [0], "num_triggered_channels": 1, "duration_s": 1.0},
            {"start_time": "2023-02-06T01:00:01.100", "end_time": "2023-02-06T01:00:02.000",
             "peak_time": "2023-02-06T01:00:01.500", "peak_ratio": 4.0,
             "triggered_channels": [1], "num_triggered_channels": 1, "duration_s": 0.9},
        ]
        merged = _merge_nearby_events(events, merge_window_s=0.0)
        assert len(merged) == 2

    def test_empty_list(self):
        assert _merge_nearby_events([], 1.0) == []


# ===================================================================
# Event dict shape
# ===================================================================

class TestEventDictShape:
    def test_event_keys_and_types(self):
        patch = _make_dummy_patch(
            n_time=5000, n_channels=4, sampling_rate_hz=200.0, seed=31,
            pulse_start_s=15.0, pulse_duration_s=0.5, pulse_amplitude=15.0,
        )
        config = StaLtaConfig(
            sta_window_s=0.5, lta_window_s=3.0,
            trigger_threshold=3.0, detrigger_threshold=1.0,
            min_channels_triggered=2, min_event_duration_s=0.1,
        )
        sr = get_sampling_rate(patch)
        sta_lta = compute_sta_lta_patch(patch, config)
        events = detect_events(sta_lta, config, sr)
        assert len(events) == 1

        evt = events[0]
        required = {"start_time", "end_time", "peak_time", "peak_ratio",
                     "triggered_channels", "num_triggered_channels", "duration_s"}
        assert required <= set(evt.keys())
        assert isinstance(evt["start_time"], str)
        assert isinstance(evt["peak_ratio"], float)
        assert isinstance(evt["triggered_channels"], list)
        assert isinstance(evt["duration_s"], float)
        assert evt["duration_s"] > 0
        assert len(evt["triggered_channels"]) == evt["num_triggered_channels"]


# ===================================================================
# StaLtaConfig validation
# ===================================================================

class TestStaLtaConfig:
    def test_defaults(self):
        cfg = StaLtaConfig()
        assert cfg.sta_window_s == 0.5
        assert cfg.lta_window_s == 10.0
        assert cfg.trigger_threshold == 3.0
        assert cfg.detrigger_threshold == 1.5

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match="window"):
            StaLtaConfig(sta_window_s=-1.0)

    def test_detrigger_not_less_than_trigger_raises(self):
        with pytest.raises(ValueError, match="detrigger_threshold"):
            StaLtaConfig(trigger_threshold=3.0, detrigger_threshold=5.0)


# ===================================================================
# DASCore stalta() direct regression
# ===================================================================

class TestDascoreStaltaDirect:
    def test_stalta_preserves_time_length(self):
        patch = _make_dummy_patch(n_time=1000, n_channels=3, sampling_rate_hz=100.0)
        result = patch.abs().stalta(time=(0.5, 5.0))
        assert result.shape[result.dims.index("time")] == 1000

    def test_stalta_noise_median_near_one(self):
        patch = _make_dummy_patch(n_time=5000, n_channels=2, sampling_rate_hz=200.0)
        result = patch.abs().stalta(time=(0.5, 10.0))
        data = np.asarray(result.data)
        lta_samples = int(10.0 * 200)
        valid = data[lta_samples:]
        valid = valid[np.isfinite(valid)]
        med = float(np.median(valid))
        assert 0.7 < med < 1.3, f"median={med:.3f}"

    def test_stalta_output_is_readonly(self):
        patch = _make_dummy_patch(n_time=1000, n_channels=3, sampling_rate_hz=100.0)
        result = patch.abs().stalta(time=(0.5, 5.0))
        assert not result.data.flags.writeable