# tests/test_preprocessing.py

import unittest

import numpy as np
import dascore as dc

from das_pipeline.config import PreprocessingConfig
from das_pipeline.preprocessing import run_preprocessing
from das_pipeline.preprocessing.select import select
from das_pipeline.preprocessing.detrend import detrend
from das_pipeline.preprocessing.bandpass import bandpass
from das_pipeline.preprocessing.decimate import decimate


def _make_test_patch(
    n_distance: int = 5,
    n_time: int = 200,
    sampling_rate: float = 100.0,
    distance_start: float = 0.0,
    distance_step: float = 10.0,
) -> dc.Patch:
    """建立一個簡單的合成測試 Patch。"""
    time_axis = np.arange(n_time) / sampling_rate
    distance_axis = distance_start + np.arange(n_distance) * distance_step

    # 用 sin 波疊加雜訊
    t = time_axis[np.newaxis, :]
    data = (
        np.sin(2 * np.pi * 2.0 * t)          # 2 Hz 訊號
        + 0.5 * np.sin(2 * np.pi * 15.0 * t)  # 15 Hz 訊號
        + 0.1 * np.random.default_rng(42).normal(size=(n_distance, n_time))
    )

    patch = dc.Patch(
        data=data,
        coords={
            "time": time_axis,
            "distance": distance_axis,
        },
        dims=("distance", "time"),
    )
    return patch


class TestSelect(unittest.TestCase):
    def test_select_time_range(self):
        patch = _make_test_patch(n_time=200, sampling_rate=100.0)
        result = select(patch, time_range=(0.5, 1.5))
        self.assertIsNotNone(result)
        # 0.5s ~ 1.5s at 100 Hz → ~100 samples
        self.assertGreater(np.asarray(result.data).shape[1], 50)
        self.assertLess(np.asarray(result.data).shape[1], 150)

    def test_select_distance_range(self):
        patch = _make_test_patch(n_distance=10, distance_step=10.0)
        result = select(patch, distance_range=(10.0, 50.0))
        self.assertIsNotNone(result)
        self.assertGreater(np.asarray(result.data).shape[0], 1)

    def test_select_no_op_when_none(self):
        patch = _make_test_patch()
        result = select(patch)
        self.assertIs(result, patch)

    def test_select_with_both_ranges(self):
        patch = _make_test_patch(n_distance=10, n_time=200)
        result = select(
            patch,
            time_range=(0.0, 1.0),
            distance_range=(0.0, 50.0),
        )
        self.assertIsNotNone(result)
        self.assertLess(np.asarray(result.data).shape[0], np.asarray(patch.data).shape[0])
        self.assertLess(np.asarray(result.data).shape[1], np.asarray(patch.data).shape[1])


class TestDetrend(unittest.TestCase):
    def test_detrend_linear_removes_trend(self):
        """加入線性趨勢後，detrend 應將其移除（均值接近 0）。"""
        n_time = 100
        x = np.arange(n_time, dtype=float).reshape(1, -1) * 10.0  # 強線性趨勢
        patch = dc.Patch(
            data=x,
            coords={"time": np.arange(n_time) / 100.0, "distance": np.array([0.0])},
            dims=("distance", "time"),
        )
        result = detrend(patch, method="linear")
        self.assertAlmostEqual(np.asarray(result.data).mean(), 0.0, delta=0.5)

    def test_detrend_constant_removes_mean(self):
        data = np.ones((1, 50)) * 100.0
        patch = dc.Patch(
            data=data,
            coords={"time": np.arange(50) / 100.0, "distance": np.array([0.0])},
            dims=("distance", "time"),
        )
        result = detrend(patch, method="constant")
        self.assertAlmostEqual(np.asarray(result.data).mean(), 0.0, delta=1e-6)

    def test_detrend_none_skips(self):
        patch = _make_test_patch()
        result = detrend(patch, method=None)
        self.assertIs(result, patch)

    def test_detrend_invalid_method(self):
        patch = _make_test_patch()
        with self.assertRaises(ValueError):
            detrend(patch, method="polynomial")


class TestBandpass(unittest.TestCase):
    def test_bandpass_preserves_energy(self):
        """濾波後形狀與原始相同，且不應是 NaN。"""
        patch = _make_test_patch(n_time=200)
        result = bandpass(patch, freq_range=(1.0, 10.0))
        self.assertEqual(np.asarray(result.data).shape, np.asarray(patch.data).shape)
        self.assertFalse(np.isnan(np.asarray(result.data)).any())

    def test_bandpass_none_skips(self):
        patch = _make_test_patch()
        result = bandpass(patch, freq_range=None)
        self.assertIs(result, patch)

    def test_bandpass_invalid_range(self):
        patch = _make_test_patch()
        with self.assertRaises(ValueError):
            bandpass(patch, freq_range=(0.0, 10.0))
        with self.assertRaises(ValueError):
            bandpass(patch, freq_range=(20.0, 10.0))


class TestDecimate(unittest.TestCase):
    def test_decimate_reduces_time_samples(self):
        patch = _make_test_patch(n_time=200, sampling_rate=100.0)
        result = decimate(patch, factor=2)
        self.assertEqual(np.asarray(result.data).shape[1], 100)

    def test_decimate_none_skips(self):
        patch = _make_test_patch()
        result = decimate(patch, factor=None)
        self.assertIs(result, patch)

    def test_decimate_invalid_factor(self):
        patch = _make_test_patch()
        with self.assertRaises(ValueError):
            decimate(patch, factor=1)
        with self.assertRaises(ValueError):
            decimate(patch, factor=0)


class TestRunPreprocessing(unittest.TestCase):
    def test_full_pipeline_with_config(self):
        config = PreprocessingConfig(
            distance_range=(0.0, 50.0),
            detrend="linear",
            bandpass=(1.0, 10.0),
            decimate_factor=2,
        )
        patch = _make_test_patch(n_distance=10, n_time=200, sampling_rate=100.0)
        result = run_preprocessing(patch, config)
        self.assertIsNotNone(result)
        # decimate_factor=2: 200→100; distance_range chosen to keep ~5 channels
        self.assertEqual(np.asarray(result.data).shape[1], 100)

    def test_empty_config_passthrough(self):
        config = PreprocessingConfig()
        patch = _make_test_patch()
        result = run_preprocessing(patch, config)
        # 所有參數都是預設值：select/time_range=None → 不裁
        # detrend="linear" → 有執行，bandpass=None → 跳過, decimate=None → 跳過
        self.assertEqual(np.asarray(result.data).shape, np.asarray(patch.data).shape)