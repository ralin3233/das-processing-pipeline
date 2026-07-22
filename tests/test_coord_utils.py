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
        # 101 應在 100 與 102 中間 => ~15.0
        self.assertAlmostEqual(result[1], 15.0, delta=0.1)


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


class TestAlignMissingGeometryFile(unittest.TestCase):
    def test_file_not_found(self) -> None:
        config = CoordinateConfig(fiber_geometry_file=Path("/nonexistent/geometry.csv"))
        patch = _make_test_patch(channels=[100, 101])
        with self.assertRaisesRegex(FileNotFoundError, "不存在"):
            align(patch, config)


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
            dist_vals = np.asarray(dist_coord.values).ravel()
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

            # 數據應已被縮放（scale > 1）
            self.assertTrue(np.any(result.data != patch.data))
            # attrs 應記錄轉換參數
            self.assertEqual(result.attrs.get("input_unit"), "phase")


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

            result = align(patch, config)

            # 裁切後應只剩 100
            dist_coord = result.get_coord("distance")
            dist_vals = np.asarray(dist_coord.values).ravel()
            self.assertEqual(len(dist_vals), 1)


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


if __name__ == "__main__":
    unittest.main()