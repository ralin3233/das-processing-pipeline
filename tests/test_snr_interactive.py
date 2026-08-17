# tests/test_snr_interactive.py

"""Tests for the interactive SNR window-picking module.

互動選窗的圖形介面無法在無顯示環境測試，此處針對不依賴 GUI 的
時間轉換 helper、``_SnrWindowPicker`` 的計算邏輯與公開 API 流程做驗證。
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import dascore as dc
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from das_pipeline.teleseismic.snr_interactive import (
    _to_epoch_seconds,
    _from_epoch_seconds,
    _fmt_time,
    _SnrWindowPicker,
    pick_snr_windows_interactive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_patch_with_signal(
    n_time: int = 2000,
    n_channels: int = 10,
    start_time: str = "2023-02-06T01:17:00",
    signal_amplitude: float = 1.0,
    noise_amplitude: float = 0.1,
) -> dc.Patch:
    """建立含高振幅訊號段與低振幅雜訊段的 Patch（20 Hz）。"""
    sampling_rate_hz = 20.0
    time = pd.date_range(
        start=start_time, periods=n_time,
        freq=f"{1000 / sampling_rate_hz}ms",
    )
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0, scale=noise_amplitude, size=(n_time, n_channels))

    # 訊號段：47.5s ~ 60.0s 振幅較大
    signal_start_idx = int(47.5 * sampling_rate_hz)
    signal_end_idx = int(60.0 * sampling_rate_hz)
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
# Tests: time conversion helpers
# ---------------------------------------------------------------------------


class TestTimeConversion:
    def test_epoch_roundtrip(self):
        t = np.datetime64("2023-02-06T01:17:35")
        assert _from_epoch_seconds(_to_epoch_seconds(t)) == t

    def test_epoch_seconds_value(self):
        t = np.datetime64("1970-01-01T00:00:10")
        assert _to_epoch_seconds(t) == pytest.approx(10.0)

    def test_fmt_time(self):
        assert _fmt_time(0.0) == "00:00:00"


# ---------------------------------------------------------------------------
# Tests: _SnrWindowPicker computation logic (no GUI interaction)
# ---------------------------------------------------------------------------


class TestSnrWindowPicker:
    def test_compute_result_positive_snr(self):
        patch = _make_dummy_patch_with_signal()
        picker = _SnrWindowPicker(
            patch,
            channel_index=3,
            default_signal_window=None,
            default_noise_window=None,
            event_distance_km=50.0,
        )
        try:
            t0 = _to_epoch_seconds(np.datetime64("2023-02-06T01:17:00"))
            picker.signal_window = (t0 + 47.5, t0 + 60.0)
            picker.noise_window = (t0 + 5.0, t0 + 20.0)

            result = picker._compute_result()

            assert result is not None
            assert result["channel_index"] == 3
            assert result["snr_db"] > 0
            assert result["P_signal"] > result["P_noise"]
            assert "signal_window" in result
            assert "noise_window" in result
        finally:
            plt.close(picker.fig)

    def test_result_none_without_windows(self):
        patch = _make_dummy_patch_with_signal()
        picker = _SnrWindowPicker(
            patch, channel_index=0,
            default_signal_window=None, default_noise_window=None,
        )
        try:
            assert picker.result is None
            assert picker._compute_result() is None
        finally:
            plt.close(picker.fig)

    def test_default_windows_clamped_to_patch(self):
        """預設窗超出 patch 範圍時，應裁切至 patch 時間內。"""
        n_time = 200
        time = pd.date_range(
            start="2023-02-06T01:17:00", periods=n_time, freq="50ms",
        )
        data = np.random.default_rng(0).normal(size=(n_time, 4))
        patch = dc.Patch(
            data=data,
            coords={"time": time, "distance": np.arange(4, dtype=float)},
            dims=["time", "distance"],
        )
        # patch: 2023-02-06T01:17:00 ~ +10s
        default_signal = (
            np.datetime64("2023-02-06T01:17:05"),
            np.datetime64("2023-02-06T01:18:00"),
        )
        picker = _SnrWindowPicker(
            patch, channel_index=0,
            default_signal_window=default_signal,
            default_noise_window=None,
        )
        try:
            t0 = float(picker.t_epoch[0])
            t1 = float(picker.t_epoch[-1])
            assert picker.signal_window is not None
            assert picker.signal_window[0] >= t0
            assert picker.signal_window[1] <= t1
            assert picker.signal_window[0] < picker.signal_window[1]
        finally:
            plt.close(picker.fig)

    def test_xlim_stays_within_data_range(self):
        """x 軸應貼近資料時間範圍，不可被 SpanSelector 拉到 epoch 0。

        回歸測試：SpanSelector 會建立 x=0（資料座標）的隱形矩形，
        若未固定 x 軸，autoscale 會把 x 軸展開數十年，波形被壓成垂直線。
        """
        patch = _make_dummy_patch_with_signal(n_time=2000)
        picker = _SnrWindowPicker(
            patch, channel_index=3,
            default_signal_window=None, default_noise_window=None,
            event_distance_km=50.0,
        )
        try:
            t0 = float(picker.t_epoch[0])
            t1 = float(picker.t_epoch[-1])
            x0, x1 = picker.ax.get_xlim()
            # 資料長度約 100 秒；x 軸跨度不應遠大於此（錯誤時會是數十年）
            assert x1 - x0 <= 2 * (t1 - t0), f"x 軸範圍異常: [{x0}, {x1}]"
            assert x0 >= t0 - 0.1 * (t1 - t0)
            assert x1 <= t1 + 0.1 * (t1 - t0)
        finally:
            plt.close(picker.fig)


# ---------------------------------------------------------------------------
# Tests: public API flow (with plt.show monkeypatched)
# ---------------------------------------------------------------------------


class TestPickSnrWindowsInteractive:
    def test_pick_with_default_windows(self, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)
        patch = _make_dummy_patch_with_signal()
        default_signal = (
            np.datetime64("2023-02-06T01:17:47.5"),
            np.datetime64("2023-02-06T01:18:00"),
        )
        default_noise = (
            np.datetime64("2023-02-06T01:17:05"),
            np.datetime64("2023-02-06T01:17:20"),
        )

        try:
            signal, noise, result = pick_snr_windows_interactive(
                patch,
                channel_index=3,
                default_signal_window=default_signal,
                default_noise_window=default_noise,
                event_distance_km=50.0,
            )
        finally:
            plt.close("all")

        assert isinstance(signal, tuple) and len(signal) == 2
        assert isinstance(noise, tuple) and len(noise) == 2
        assert result is not None
        assert result["snr_db"] > 0
