# tests/test_visualization.py

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import dascore as dc

from das_pipeline.visualization.waterfall import plot_waterfall
from das_pipeline.visualization.fk import plot_fk_spectrum, _get_spatial_axis
from das_pipeline.visualization.spectrogram import plot_spectrogram
from das_pipeline.visualization.merge import (
    merge_patches,
    _parse_chunk_index,
    _parse_timestamp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_patch(
    n_time: int = 200,
    n_distance: int = 10,
    sampling_rate: float = 100.0,
    distance_start: float = 0.0,
    distance_step: float = 10.0,
    dims: tuple = ("distance", "time"),
    use_datetime: bool = True,
) -> dc.Patch:
    """建立合成測試 Patch。

    Parameters
    ----------
    use_datetime : bool
        True 表示 time 軸使用 datetime64（與真實資料相同），
        False 表示使用 float seconds（某些測試需要）。
    """
    time_axis = np.arange(n_time) / sampling_rate
    distance_axis = distance_start + np.arange(n_distance) * distance_step

    if use_datetime:
        # 使用 datetime64 以符合真實資料格式
        time_axis = (
            np.datetime64("1970-01-01T00:00:00")
            + (time_axis * 1e9).astype("timedelta64[ns]")
        )

    # 2 Hz + 15 Hz sin waves + noise
    rng = np.random.default_rng(42)
    t_float = np.arange(n_time) / sampling_rate
    t = t_float[np.newaxis, :]  # (1, n_time)
    # 預設 data shape = (distance, time)
    data = (
        np.sin(2 * np.pi * 2.0 * t)          # 2 Hz
        + 0.5 * np.sin(2 * np.pi * 15.0 * t)  # 15 Hz
        + 0.1 * rng.normal(size=(n_distance, n_time))
    )

    # 若 dims 為 ("time", "distance")，data 需轉置
    if dims == ("time", "distance"):
        data = data.T

    test_patch = dc.Patch(
        data=data,
        coords={
            "time": time_axis,
            "distance": distance_axis,
        },
        dims=dims,
    )
    return test_patch


# ---------------------------------------------------------------------------
# merge helpers
# ---------------------------------------------------------------------------

class TestMergeHelpers(unittest.TestCase):
    def test_parse_chunk_index_normal(self):
        self.assertEqual(_parse_chunk_index("data_chunk0000.h5"), 0)
        self.assertEqual(_parse_chunk_index("data_chunk0042.h5"), 42)
        self.assertEqual(_parse_chunk_index("chunk9999.h5"), 9999)

    def test_parse_chunk_index_no_match(self):
        self.assertIsNone(_parse_chunk_index("data_20250714.h5"))
        self.assertIsNone(_parse_chunk_index("no_chunk_here.h5"))

    def test_parse_chunk_index_empty(self):
        self.assertIsNone(_parse_chunk_index(""))

    def test_parse_timestamp_normal(self):
        self.assertEqual(
            _parse_timestamp("data_20250714T143000.h5"),
            "20250714T143000",
        )
        self.assertEqual(
            _parse_timestamp("20260101T000000_chunk0000.h5"),
            "20260101T000000",
        )

    def test_parse_timestamp_no_match(self):
        self.assertIsNone(_parse_timestamp("data_without_ts.h5"))

    def test_parse_timestamp_invalid_format(self):
        self.assertIsNone(_parse_timestamp("data_20250714.h5"))
        self.assertIsNone(_parse_timestamp(""))


# ---------------------------------------------------------------------------
# merge_patches
# ---------------------------------------------------------------------------

class TestMergePatches(unittest.TestCase):
    def test_empty_paths_raises(self):
        with self.assertRaises(ValueError):
            merge_patches([])

    @patch("das_pipeline.visualization.merge.dc.spool")
    def test_single_file_returns_directly(self, mock_spool):
        """單一檔案應直接回傳，不經過 concat。"""
        mock_patch_obj = MagicMock(spec=dc.Patch)
        mock_spool_instance = MagicMock()
        mock_spool_instance.__getitem__.return_value = mock_patch_obj
        mock_spool.return_value = mock_spool_instance

        result = merge_patches([Path("single.h5")])
        self.assertIs(result, mock_patch_obj)
        # concat 不應該被呼叫
        mock_spool_instance.concatenate.assert_not_called()

    @patch("das_pipeline.visualization.merge.dc.spool")
    def test_multiple_files_calls_concat(self, mock_spool):
        """多個檔案應呼叫 spool(...).concatenate(time=None)，且參數要正確。"""
        # The actual call path in merge_patches is:
        #   dc.spool(patches).concatenate(time=None)[0]
        merged_patch = MagicMock(spec=dc.Patch)
        merged_spool = MagicMock()
        merged_spool.concatenate.return_value.__getitem__.return_value = merged_patch

        # mock individual file reads: 依呼叫順序回傳不同的 spool
        file_spools = [MagicMock(), MagicMock(), MagicMock()]
        for sp in file_spools:
            sp.__getitem__.return_value = MagicMock()

        call_log = []

        def spool_side_effect(arg):
            call_log.append(arg)
            if isinstance(arg, list):
                return merged_spool
            return file_spools.pop(0)

        mock_spool.side_effect = spool_side_effect

        result = merge_patches([
            Path("chunk0000.h5"),
            Path("chunk0001.h5"),
            Path("chunk0002.h5"),
        ])

        self.assertIs(result, merged_patch)
        # 驗證合併呼叫時的參數是否正確，而不是只看回傳值
        merged_spool.concatenate.assert_called_once_with(time=None)
        # 驗證讀取每個檔案時傳入的是字串路徑（依目前實作）
        file_read_calls = [c for c in call_log if not isinstance(c, list)]
        self.assertEqual(len(file_read_calls), 3)
        for c in file_read_calls:
            self.assertIsInstance(c, str)

    @patch("das_pipeline.visualization.merge.dc.spool")
    def test_files_read_in_sorted_order(self, mock_spool):
        """驗證 merge_patches 實際依 chunk_index 排序後才讀檔，而不只是複製排序邏輯來測。"""
        merged_patch = MagicMock(spec=dc.Patch)
        merged_spool = MagicMock()
        merged_spool.concatenate.return_value.__getitem__.return_value = merged_patch

        read_order = []

        def spool_side_effect(arg):
            if isinstance(arg, list):
                return merged_spool
            read_order.append(arg)
            m = MagicMock()
            m.__getitem__.return_value = MagicMock()
            return m

        mock_spool.side_effect = spool_side_effect

        # 故意打亂順序傳入
        merge_patches([
            Path("chunk0002.h5"),
            Path("chunk0000.h5"),
            Path("chunk0001.h5"),
        ])

        self.assertEqual(
            read_order,
            ["chunk0000.h5", "chunk0001.h5", "chunk0002.h5"],
        )

    def test_sort_chunk_index(self):
        """測試 chunk_index 排序邏輯本身（純函式行為，與 merge_patches 內部邏輯分開驗證）。"""
        paths = [
            Path("chunk0002.h5"),
            Path("chunk0000.h5"),
            Path("chunk0001.h5"),
        ]
        sorted_paths = sorted(
            paths,
            key=lambda p: _parse_chunk_index(p.name) or 0,
        )
        self.assertEqual(
            [p.name for p in sorted_paths],
            ["chunk0000.h5", "chunk0001.h5", "chunk0002.h5"],
        )

    def test_sort_timestamp(self):
        paths = [
            Path("data_20250714T143002.h5"),
            Path("data_20250714T143000.h5"),
            Path("data_20250714T143001.h5"),
        ]
        sorted_paths = sorted(
            paths,
            key=lambda p: _parse_timestamp(p.name) or "",
        )
        self.assertEqual(
            [p.name for p in sorted_paths],
            [
                "data_20250714T143000.h5",
                "data_20250714T143001.h5",
                "data_20250714T143002.h5",
            ],
        )


# ---------------------------------------------------------------------------
# plot_waterfall
# ---------------------------------------------------------------------------

class TestPlotWaterfall(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_basic_returns_figure(self):
        test_patch = _make_test_patch()
        fig = plot_waterfall(test_patch)
        self.assertIsInstance(fig, Figure)
        # fig.axes = [main_axes, colorbar_axes]
        self.assertGreaterEqual(len(fig.axes), 1)

    def test_with_existing_axes(self):
        test_patch = _make_test_patch()
        fig, ax = plt.subplots()
        result_fig = plot_waterfall(test_patch, ax=ax)
        self.assertIs(result_fig, fig)
        self.assertIs(result_fig.axes[0], ax)

    def test_time_range_filter(self):
        test_patch = _make_test_patch(n_time=200, sampling_rate=100.0)
        fig = plot_waterfall(
            test_patch,
            time_range=("1970-01-01T00:00:00.500", "1970-01-01T00:00:01.500"),
        )
        self.assertIsInstance(fig, Figure)

    def test_distance_range_filter(self):
        test_patch = _make_test_patch(n_distance=20, distance_step=10.0)
        fig = plot_waterfall(test_patch, distance_range=(10.0, 100.0))
        self.assertIsInstance(fig, Figure)

    def test_custom_vmin_vmax(self):
        test_patch = _make_test_patch()
        fig = plot_waterfall(test_patch, vmin=-1.0, vmax=1.0)
        self.assertIsInstance(fig, Figure)

    def test_auto_vmin_vmax(self):
        """clip_percentile 自動推斷 vmin/vmax，不應拋錯。"""
        test_patch = _make_test_patch()
        fig = plot_waterfall(test_patch, clip_percentile=95.0)
        self.assertIsInstance(fig, Figure)

    def test_custom_colormap(self):
        test_patch = _make_test_patch()
        fig = plot_waterfall(test_patch, colormap="viridis")
        self.assertIsInstance(fig, Figure)

    def test_custom_title(self):
        test_patch = _make_test_patch()
        fig = plot_waterfall(test_patch, title="Test Waterfall")
        self.assertIsInstance(fig, Figure)
        self.assertEqual(fig.axes[0].get_title(), "Test Waterfall")

    def test_custom_figsize(self):
        test_patch = _make_test_patch()
        fig = plot_waterfall(test_patch, figsize=(6, 3))
        self.assertIsInstance(fig, Figure)

    def test_no_colorbar(self):
        test_patch = _make_test_patch()
        fig = plot_waterfall(test_patch, show_colorbar=False)
        self.assertIsInstance(fig, Figure)
        # 沒有 colorbar 時只有一個 axes
        self.assertEqual(len(fig.axes), 1)

    def test_dims_time_first(self):
        """(time, distance) 順序的 Patch。"""
        test_patch = _make_test_patch(
            n_time=100, n_distance=5, dims=("time", "distance"),
            use_datetime=False,
        )
        fig = plot_waterfall(test_patch)
        self.assertIsInstance(fig, Figure)

    def test_no_m_in_distance_units(self):
        """當 distance 沒有指定單位（units=None）時，ylabel 應顯示 'Channel index'。"""
        # 未指定 distance step → float array, units = None
        test_patch = _make_test_patch(
            n_time=100, n_distance=5, distance_step=1.0,
            use_datetime=False,
        )
        fig = plot_waterfall(test_patch)
        ylabel = fig.axes[0].get_ylabel()
        # units 未設定時應明確走「無單位」分支，而不是兩者皆可
        self.assertEqual(ylabel, "Channel index")


# ---------------------------------------------------------------------------
# plot_fk_spectrum
# ---------------------------------------------------------------------------

class TestFkHelpers(unittest.TestCase):
    def test_get_spatial_axis_with_spacing(self):
        test_patch = _make_test_patch(n_distance=10, distance_step=10.0)
        dist_vals, label = _get_spatial_axis(test_patch, channel_spacing=5.0)
        self.assertEqual(len(dist_vals), 10)
        self.assertEqual(dist_vals[1], 5.0)
        self.assertEqual(label, "Wavenumber (cycles/m)")

    def test_get_spatial_axis_without_m_units(self):
        """distance 單位為 None 時，label 應為 Wavenumber (cycles/channel)。"""
        test_patch = _make_test_patch(n_distance=10, distance_step=10.0)
        dist_vals, label = _get_spatial_axis(test_patch, channel_spacing=None)
        # units = None → "m" not in str(None) → else branch
        self.assertEqual(label, "Wavenumber (cycles/channel)")

    def test_get_spatial_axis_with_m_units_explicit(self):
        """channel_spacing 明確指定時，即使 units=None 也應是 cycles/m。"""
        test_patch = _make_test_patch(n_distance=10, distance_step=10.0)
        dist_vals, label = _get_spatial_axis(test_patch, channel_spacing=5.0)
        self.assertEqual(label, "Wavenumber (cycles/m)")


class TestPlotFkSpectrum(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_basic_returns_figure(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_fk_spectrum(test_patch)
        self.assertIsInstance(fig, Figure)
        self.assertGreaterEqual(len(fig.axes), 1)

    def test_with_existing_axes(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig, ax = plt.subplots()
        result_fig = plot_fk_spectrum(test_patch, ax=ax)
        self.assertIs(result_fig, fig)

    def test_with_channel_spacing(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_fk_spectrum(test_patch, channel_spacing=5.0)
        self.assertIsInstance(fig, Figure)

    def test_freq_range_filter(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10, sampling_rate=100.0)
        fig = plot_fk_spectrum(test_patch, freq_range=(1.0, 30.0))
        self.assertIsInstance(fig, Figure)

    def test_db_range(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_fk_spectrum(test_patch, db_range=(0, 40))
        self.assertIsInstance(fig, Figure)

    def test_custom_colormap(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_fk_spectrum(test_patch, colormap="plasma")
        self.assertIsInstance(fig, Figure)

    def test_custom_title(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_fk_spectrum(test_patch, title="FK Test")
        self.assertIsInstance(fig, Figure)
        self.assertEqual(fig.axes[0].get_title(), "FK Test")

    def test_no_colorbar(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_fk_spectrum(test_patch, show_colorbar=False)
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 1)

    def test_invalid_dt_raises(self):
        """單一時間點時 dt 無法計算，應拋錯（不論是建構期還是函式內部）。"""
        with self.assertRaises((ValueError, IndexError)):
            time_axis = np.array([np.datetime64("1970-01-01T00:00:00")])
            distance_axis = np.array([0.0, 10.0])
            data = np.random.default_rng(42).normal(size=(1, 2))
            bad_patch = dc.Patch(
                data=data,
                coords={"time": time_axis, "distance": distance_axis},
                dims=("time", "distance"),
            )
            plot_fk_spectrum(bad_patch)

    def test_zero_dx_raises(self):
        """所有 distance 相同時 dx=0，應拋錯——不論是 Patch 建構期或函式內部防呆。"""
        with self.assertRaises(ValueError):
            time_axis = np.arange(100) / 100.0
            time_axis = (
                np.datetime64("1970-01-01T00:00:00")
                + (time_axis * 1e9).astype("timedelta64[ns]")
            )
            distance_axis = np.array([0.0, 0.0, 0.0])
            data = np.random.default_rng(42).normal(size=(100, 3))
            bad_patch = dc.Patch(
                data=data,
                coords={"time": time_axis, "distance": distance_axis},
                dims=("time", "distance"),
            )
            plot_fk_spectrum(bad_patch)


# ---------------------------------------------------------------------------
# plot_spectrogram
# ---------------------------------------------------------------------------

class TestPlotSpectrogram(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_basic_returns_figure(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_spectrogram(test_patch)
        self.assertIsInstance(fig, Figure)
        self.assertGreaterEqual(len(fig.axes), 1)

    def test_with_existing_axes(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig, ax = plt.subplots()
        result_fig = plot_spectrogram(test_patch, ax=ax)
        self.assertIs(result_fig, fig)

    def test_specific_channel(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_spectrogram(test_patch, channel=0)
        self.assertIsInstance(fig, Figure)

    def test_channel_out_of_range_low(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        with self.assertRaises(ValueError):
            plot_spectrogram(test_patch, channel=-1)

    def test_channel_out_of_range_high(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        with self.assertRaises(ValueError):
            plot_spectrogram(test_patch, channel=999)

    def test_freq_range_filter(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10, sampling_rate=100.0)
        fig = plot_spectrogram(test_patch, freq_range=(1.0, 30.0))
        self.assertIsInstance(fig, Figure)

    def test_db_range(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_spectrogram(test_patch, db_range=(-20, 0))
        self.assertIsInstance(fig, Figure)

    def test_custom_nperseg(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_spectrogram(test_patch, nperseg=128)
        self.assertIsInstance(fig, Figure)

    def test_custom_noverlap(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_spectrogram(test_patch, nperseg=256, noverlap=128)
        self.assertIsInstance(fig, Figure)

    def test_custom_colormap(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_spectrogram(test_patch, colormap="inferno")
        self.assertIsInstance(fig, Figure)

    def test_custom_title(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_spectrogram(test_patch, title="Spec Test")
        self.assertIsInstance(fig, Figure)
        self.assertIn("Spec Test", fig.axes[0].get_title())

    def test_no_colorbar(self):
        test_patch = _make_test_patch(n_time=200, n_distance=10)
        fig = plot_spectrogram(test_patch, show_colorbar=False)
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 1)

    def test_invalid_dt_raises(self):
        """單一時間點時 dt 無法計算，應拋錯（不論是建構期還是函式內部）。"""
        with self.assertRaises((ValueError, IndexError)):
            time_axis = np.array([np.datetime64("1970-01-01T00:00:00")])
            distance_axis = np.array([0.0, 10.0])
            data = np.random.default_rng(42).normal(size=(1, 2))
            bad_patch = dc.Patch(
                data=data,
                coords={"time": time_axis, "distance": distance_axis},
                dims=("time", "distance"),
            )
            plot_spectrogram(bad_patch)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()