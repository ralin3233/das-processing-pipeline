from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from das_pipeline.config import CoordinateConfig
from das_pipeline.io.coord_utils import (
    _haversine_distance,
    _compute_cumulative_distances,
    _load_geometry,
    _build_distance_map,
    _map_distances_interpolate,
    _handle_missing_channels,
    align,
)

import dascore as dc


def _make_geometry_csv(
    path: Path, channels: list[int], lat_start: float = 23.5,
    lon_start: float = 120.5, depth_start: float = 0.0,
    lat_step: float = 0.001, lon_step: float = 0.001, depth_step: float = 10.0,
) -> None:
    """建立測試用的 geometry.csv。"""
    rows = []
    for i, ch in enumerate(channels):
        rows.append({
            "channel_index": ch,
            "lat": lat_start + i * lat_step,
            "lon": lon_start + i * lon_step,
            "depth": depth_start + i * depth_step,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_test_patch(
    channels: list[int],
    n_times: int = 50,
    delta_us: int = 100_000,
) -> dc.Patch:
    """建立測試用的 dascore Patch（distance 軸為 channel index）。"""
    start_time = np.datetime64("2023-02-06T10:24:50")
    time_axis = start_time + np.arange(n_times) * np.timedelta64(delta_us, "us")
    distance_axis = np.array(channels)

    data = np.random.randn(len(channels), n_times).astype(np.float64)

    return dc.Patch(
        data=data,
        coords={"time": time_axis, "distance": distance_axis},
        dims=("distance", "time"),
    )


class TestHaversineDistance(unittest.TestCase):
    def test_known_distance(self) -> None:
        """同一點距離為 0。"""
        d = _haversine_distance(23.5, 120.5, 23.5, 120.5)
        self.assertAlmostEqual(d, 0.0, places=4)

    def test_one_degree_lat(self) -> None:
        """約 1° 緯度 ≈ 111 km（簡化檢驗）。"""
        d = _haversine_distance(23.5, 120.5, 24.5, 120.5)
        self.assertAlmostEqual(d / 111_000, 1.0, delta=0.05)

    def test_zero_distance(self) -> None:
        """兩組相同經緯度距離為 0。"""
        d = _haversine_distance(23.5, 120.5, 23.5, 120.5)
        self.assertAlmostEqual(d, 0.0, places=4)


class TestComputeCumulativeDistances(unittest.TestCase):
    def test_single_channel(self) -> None:
        """只有一個 channel，累積距離為 0。"""
        df = pd.DataFrame({
            "channel_index": [100],
            "lat": [23.5],
            "lon": [120.5],
            "depth": [0.0],
        })
        dists = _compute_cumulative_distances(df)
        np.testing.assert_array_equal(dists, [0.0])

    def test_two_channels(self) -> None:
        """兩個 channel，應大於 0。"""
        df = pd.DataFrame({
            "channel_index": [100, 101],
            "lat": [23.5, 23.5001],
            "lon": [120.5, 120.5001],
            "depth": [0.0, 10.0],
        })
        dists = _compute_cumulative_distances(df)
        self.assertEqual(len(dists), 2)
        self.assertAlmostEqual(dists[0], 0.0)
        self.assertGreater(dists[1], 10.0)  # 至少大於深度差

    def test_unsorted_channels(self) -> None:
        """即使 channel_index 未排序，函式內部會自動排序。"""
        df = pd.DataFrame({
            "channel_index": [102, 100, 101],
            "lat": [23.5002, 23.5, 23.5001],
            "lon": [120.5002, 120.5, 120.5001],
            "depth": [20.0, 0.0, 10.0],
        })
        dists = _compute_cumulative_distances(df)
        self.assertEqual(len(dists), 3)
        self.assertAlmostEqual(dists[0], 0.0)
        self.assertGreater(dists[2], dists[1])


class TestLoadGeometry(unittest.TestCase):
    def test_valid_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "geometry.csv"
            _make_geometry_csv(csv_path, channels=[100, 101, 102])

            df = _load_geometry(csv_path)
            self.assertListEqual(list(df.columns), ["channel_index", "lat", "lon", "depth"])
            self.assertEqual(len(df), 3)

    def test_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "bad.csv"
            pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "缺少必要欄位"):
                _load_geometry(csv_path)


class TestBuildDistanceMap(unittest.TestCase):
    def test_returns_dict(self) -> None:
        df = pd.DataFrame({
            "channel_index": [100, 101, 102],
            "lat": [23.5, 23.5001, 23.5002],
            "lon": [120.5, 120.5001, 120.5002],
            "depth": [0.0, 10.0, 20.0],
        })
        dmap = _build_distance_map(df)
        self.assertIn(100, dmap)
        self.assertIn(101, dmap)
        self.assertEqual(dmap[100], 0.0)
        self.assertGreater(dmap[102], dmap[101])


class TestMapDistancesInterpolate(unittest.TestCase):
    def test_all_matched(self) -> None:
        channels = np.array([100, 101, 102])
        dmap = {100: 0.0, 101: 15.0, 102: 30.0}
        result = _map_distances_interpolate(channels, dmap)
        np.testing.assert_array_almost_equal(result, [0.0, 15.0, 30.0])

    def test_missing_interpolated(self) -> None:
        """channel 101 不在 geometry 中，應被插值。"""
        channels = np.array([100, 101, 102])
        dmap = {100: 0.0, 102: 30.0}
        result = _map_distances_interpolate(channels, dmap)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[2], 30.0)
        # 101 應在 100 與 102 中間 => 15.0（線性插值中點，應為精確值）
        self.assertAlmostEqual(result[1], 15.0, places=6)


class TestHandleMissingChannels(unittest.TestCase):
    def setUp(self):
        self.config_interpolate = CoordinateConfig(
            fiber_geometry_file=Path("/tmp/dummy.csv"),
            missing_channel_strategy="interpolate",
        )
        self.config_crop = CoordinateConfig(
            fiber_geometry_file=Path("/tmp/dummy.csv"),
            missing_channel_strategy="crop",
        )
        self.config_error = CoordinateConfig(
            fiber_geometry_file=Path("/tmp/dummy.csv"),
            missing_channel_strategy="error",
        )

    def test_no_missing(self) -> None:
        channels = np.array([100, 101])
        dmap = {100: 0.0, 101: 15.0}
        dists, keep = _handle_missing_channels(channels, dmap, self.config_interpolate)
        np.testing.assert_array_almost_equal(dists, [0.0, 15.0])
        self.assertTrue(keep.all())

    def test_interpolate(self) -> None:
        channels = np.array([100, 101, 102])
        dmap = {100: 0.0, 102: 30.0}
        dists, keep = _handle_missing_channels(channels, dmap, self.config_interpolate)
        self.assertEqual(len(dists), 3)
        self.assertTrue(keep.all())
        self.assertAlmostEqual(dists[1], 15.0, delta=0.1)

    def test_crop(self) -> None:
        channels = np.array([100, 101, 102])
        dmap = {100: 0.0, 102: 30.0}
        dists, keep = _handle_missing_channels(channels, dmap, self.config_crop)
        self.assertEqual(len(dists), 2)
        np.testing.assert_array_equal(dists, [0.0, 30.0])
        np.testing.assert_array_equal(keep, [True, False, True])

    def test_error(self) -> None:
        channels = np.array([100, 101, 102])
        dmap = {100: 0.0, 102: 30.0}
        with self.assertRaisesRegex(ValueError, "缺少以下"):
            _handle_missing_channels(channels, dmap, self.config_error)




class TestAlignWithInterpolate(unittest.TestCase):
    def test_all_channels_mapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "geometry.csv"
            # geometry 涵蓋 channel 100~103
            _make_geometry_csv(csv_path, channels=[100, 101, 102, 103])

            config = CoordinateConfig(fiber_geometry_file=csv_path)
            # Patch 只有 100~102（部分對應）
            patch = _make_test_patch(channels=[100, 101, 102])

            result = align(patch, config)

            # distance 軸應變成實際距離
            dist_coord = result.get_coord("distance")
            dist_vals = np.asarray(dist_coord)
            self.assertEqual(len(dist_vals), 3)
            self.assertGreater(dist_vals[1], 0)  # 非零距離
            self.assertEqual(result.dims, ("distance", "time"))

    def test_phase_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "geometry.csv"
            _make_geometry_csv(csv_path, channels=[100, 101, 102])

            config = CoordinateConfig(
                fiber_geometry_file=csv_path,
                input_unit="phase",
            )
            patch = _make_test_patch(channels=[100, 101, 102], n_times=10, delta_us=1_000_000)

            result = align(patch, config)

            # 精確驗證 scale factor：phase_strain_constant * fs
            # dt = 1_000_000 us = 1s => fs = 1 Hz
            expected_scale = config.phase_strain_constant * 1.0
            result_data = np.asarray(result.data)
            patch_data = np.asarray(patch.data)
            np.testing.assert_allclose(result_data, patch_data * expected_scale, rtol=1e-10)
            # attrs 應記錄轉換參數
            self.assertEqual(result.attrs.get("input_unit"), "strain_rate")
            self.assertEqual(result.attrs.get("phase_strain_constant"), config.phase_strain_constant)
            self.assertAlmostEqual(float(result.attrs.get("sampling_rate_hz")), 1.0)
            self.assertAlmostEqual(float(result.attrs.get("scale_factor")), float(expected_scale))


class TestAlignWithCrop(unittest.TestCase):
    def test_crop_missing_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "geometry.csv"
            _make_geometry_csv(csv_path, channels=[100, 102, 103])

            config = CoordinateConfig(
                fiber_geometry_file=csv_path,
                missing_channel_strategy="crop",
            )
            # Patch 有 100, 101（101 在 geometry 缺失）
            patch = _make_test_patch(channels=[100, 101])

            # 記錄原始資料以便比對
            orig_data = np.asarray(patch.data)
            orig_ch0_data = orig_data[0, :].copy()

            result = align(patch, config)

            # 裁切後應只剩 channel 100 的資料
            dist_coord = result.get_coord("distance")
            dist_vals = np.asarray(dist_coord)
            self.assertEqual(len(dist_vals), 1)
            self.assertAlmostEqual(float(dist_vals[0]), 0.0)  # channel 100 距離 = 0

            # 驗證留下的 data 等同原本 channel 100 的資料（索引 0）
            result_data = np.asarray(result.data)
            self.assertEqual(result_data.shape, (1, orig_ch0_data.shape[0]))
            np.testing.assert_array_equal(result_data[0, :], orig_ch0_data)


class TestAlignWithError(unittest.TestCase):
    def test_error_on_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "geometry.csv"
            _make_geometry_csv(csv_path, channels=[100, 102])

            config = CoordinateConfig(
                fiber_geometry_file=csv_path,
                missing_channel_strategy="error",
            )
            patch = _make_test_patch(channels=[100, 101])

            with self.assertRaises(ValueError):
                align(patch, config)


# ---- 缺失分支覆蓋測試 ----


class TestAlignGeometryNotFound(unittest.TestCase):
    """geometry.csv 不存在的路徑。"""

    def test_no_geometry_skips_alignment(self) -> None:
        """geometry 檔案不存在時不報錯，維持原始 channel index。"""
        config = CoordinateConfig(
            fiber_geometry_file=Path("/tmp/nonexistent_geometry.csv"),
        )
        patch = _make_test_patch(channels=[100, 101, 102])

        result = align(patch, config)

        dist_vals = np.asarray(result.get_coord("distance"))
        np.testing.assert_array_equal(dist_vals, [100, 101, 102])

    def test_no_geometry_with_phase_unit(self) -> None:
        """geometry 不存在 + input_unit=phase：不對齊距離但仍做單位轉換。"""
        config = CoordinateConfig(
            fiber_geometry_file=Path("/tmp/nonexistent_geometry.csv"),
            input_unit="phase",
        )
        patch = _make_test_patch(channels=[100, 101, 102], n_times=10, delta_us=1_000_000)

        result = align(patch, config)

        expected_scale = config.phase_strain_constant * 1.0
        result_data = np.asarray(result.data)
        patch_data = np.asarray(patch.data)
        np.testing.assert_allclose(result_data, patch_data * expected_scale, rtol=1e-10)
        self.assertEqual(result.attrs.get("input_unit"), "strain_rate")
        # distance 軸維持原始 channel index
        dist_vals = np.asarray(result.get_coord("distance"))
        np.testing.assert_array_equal(dist_vals, [100, 101, 102])


class TestAlignPhaseConversionEdgeCases(unittest.TestCase):
    """Phase 轉換的邊界狀況。"""

    def test_time_axis_too_short_skips_phase_conversion(self) -> None:
        """時間軸只有 1 點時跳過單位轉換。"""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "geometry.csv"
            _make_geometry_csv(csv_path, channels=[100, 101])

            config = CoordinateConfig(
                fiber_geometry_file=csv_path,
                input_unit="phase",
            )
            patch = _make_test_patch(channels=[100, 101], n_times=1, delta_us=1_000_000)

            result = align(patch, config)
            # 資料不變
            result_data = np.asarray(result.data)
            patch_data = np.asarray(patch.data)
            np.testing.assert_array_equal(result_data, patch_data)


class TestAlignStrictShapeCheck(unittest.TestCase):
    """strict_shape_check=True 的 RuntimeError 路徑。"""

    def test_strict_shape_check_passes(self) -> None:
        """正常狀況下 strict_shape_check 不報錯（預設 True）。"""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "geometry.csv"
            _make_geometry_csv(csv_path, channels=[100, 101, 102])

            config = CoordinateConfig(
                fiber_geometry_file=csv_path,
                strict_shape_check=True,
            )
            patch = _make_test_patch(channels=[100, 101, 102])

            result = align(patch, config)
            result_data = np.asarray(result.data)
            self.assertEqual(
                result_data.shape,
                (len(result.get_coord("distance")), len(result.get_coord("time"))),
            )


class TestHandleMissingChannelsEdgeCases(unittest.TestCase):
    """_handle_missing_channels 邊界狀況。"""

    def setUp(self):
        self.config_crop = CoordinateConfig(
            fiber_geometry_file=Path("/tmp/dummy.csv"),
            missing_channel_strategy="crop",
        )

    def test_crop_all_channels_missing(self) -> None:
        """所有 channel 都不在 geometry 中時，crop 後為空。"""
        channels = np.array([100, 101, 102])
        dmap = {200: 10.0, 201: 20.0}  # 完全無重疊
        dists, keep = _handle_missing_channels(channels, dmap, self.config_crop)
        self.assertEqual(len(dists), 0)
        np.testing.assert_array_equal(keep, [False, False, False])


class TestAlignIntegrationWithPreprocessing(unittest.TestCase):
    """整合測試：align() 串接在 preprocessing 管線之後（模擬 run_convert 的實際流程）。"""

    def test_align_after_select_bandpass_decimate(self) -> None:
        """模擬 run_convert 路徑: select → bandpass → decimate → align。"""
        from das_pipeline.preprocessing.bandpass import bandpass
        from das_pipeline.preprocessing.decimate import decimate

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "geometry.csv"
            # geometry 涵蓋 channel 0~9
            _make_geometry_csv(csv_path, channels=list(range(10)), depth_step=5.0)

            config = CoordinateConfig(fiber_geometry_file=csv_path)
            # Patch 使用較小的 channel 子集，模擬 select 後的距離裁切
            patch = _make_test_patch(channels=[0, 1, 2, 3, 4], n_times=200, delta_us=5_000)

            # Step 1: bandpass (低通 1-20 Hz, fs=200 Hz)
            patched_bp = bandpass(patch, freq_range=(1.0, 20.0))

            # Step 2: decimate factor 2 (200 Hz → 100 Hz)
            patched_dec = decimate(patched_bp, factor=2)

            # Step 3: align
            result = align(patched_dec, config)

            # 驗證 distance 軸已轉為實際距離（米）
            dist_vals = np.asarray(result.get_coord("distance"))
            self.assertEqual(len(dist_vals), 5)
            self.assertAlmostEqual(float(dist_vals[0]), 0.0)
            self.assertGreater(float(dist_vals[4]), float(dist_vals[0]))

            # 驗證時間軸已被 decimate 縮短（約一半）
            time_vals = np.asarray(result.get_coord("time"))
            orig_time_len = len(np.asarray(patch.get_coord("time")))
            self.assertLess(len(time_vals), orig_time_len)
            self.assertGreater(len(time_vals), orig_time_len // 2 - 5)

            # 驗證 data shape 一致
            result_data = np.asarray(result.data)
            self.assertEqual(result_data.shape[0], len(dist_vals))
            self.assertEqual(result_data.shape[1], len(time_vals))
            self.assertEqual(result.dims, ("distance", "time"))


class TestMapDistancesInterpolateEdgeCases(unittest.TestCase):
    """_map_distances_interpolate 邊界狀況（外推）。"""

    def test_extrapolate_beyond_known_range(self) -> None:
        """channel 超出 geometry 範圍時做外推。"""
        channels = np.array([98, 99, 100, 101, 102, 103, 104])
        dmap = {100: 0.0, 102: 30.0}
        result = _map_distances_interpolate(channels, dmap)
        self.assertEqual(len(result), 7)
        # 內插區域：101 應在 0~30 之間
        self.assertAlmostEqual(result[3], 15.0, places=6)  # 101
        # 外推區域：98 應 < 0（往負方向外推）
        self.assertLess(result[0], 0.0)
        # 外推區域：104 應 > 30（往正方向外推）
        self.assertGreater(result[6], 30.0)


if __name__ == "__main__":
    unittest.main()
