from pathlib import Path
import tempfile
import unittest

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from das_pipeline.config import DataConfig
from das_pipeline.io.miniseed_loader import load


START_TIME = UTCDateTime("2023-02-06T10:24:50")


def write_mseed(path: Path, station: str, data: np.ndarray, delta: float = 0.1) -> None:
    trace = Trace(data=data)
    trace.stats.station = station
    trace.stats.starttime = START_TIME
    trace.stats.delta = delta
    Stream([trace]).write(str(path), format="MSEED")


class TestMiniSeedLoader(unittest.TestCase):
    def test_load_returns_patch_and_truncates_to_shortest_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange: 建立兩個長度不同的 miniSEED 檔案，讓 loader 需要裁切到最短長度。
            input_dir = Path(temp_dir)
            write_mseed(input_dir / "001.mseed", "001", np.array([0, 1, 2, 3, 4], dtype=np.int32))
            write_mseed(input_dir / "002.mseed", "002", np.array([10, 11, 12], dtype=np.int32))

            # Act: 以資料設定載入資料，取得 DASCore Patch。
            config = DataConfig(input_dir=input_dir, channel_range=(0, 1))

            patch = load(config)

            # Assert: 確認維度、shape、距離軸與時間軸都符合預期。
            self.assertEqual(patch.dims, ("distance", "time"))
            self.assertEqual(patch.data.shape, (2, 3))
            np.testing.assert_array_equal(
                patch.data,
                np.array([[0, 1, 2], [10, 11, 12]], dtype=np.int32),
            )
            np.testing.assert_array_equal(
                np.asarray(patch.coords.get_array("distance")),
                np.array([1, 2]),
            )
            expected_time = np.datetime64("2023-02-06T10:24:50.000000") + np.arange(3) * np.timedelta64(100_000, "us")
            np.testing.assert_array_equal(
                np.asarray(patch.coords.get_array("time")),
                expected_time,
            )

    def test_load_raises_file_not_found_when_no_mseed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange: 提供空目錄，模擬沒有任何 *.mseed 檔案的情況。
            input_dir = Path(temp_dir)
            config = DataConfig(input_dir=input_dir, channel_range=(0, 1))

            # Assert: loader 應該明確回報找不到檔案。
            with self.assertRaisesRegex(FileNotFoundError, r"找不到符合 \*\.mseed 的檔案"):
                load(config)
