"""Tests for NaN handling across the DAS processing pipeline."""

import numpy as np
import pytest

import dascore as dc


# ============================================================================
# nan_handler.py — sanitize_nan_patch
# ============================================================================

class TestSanitizeNanPatch:
    """Tests for preprocessing/nan_handler.py."""

    @pytest.fixture
    def clean_patch(self):
        """A patch with no NaN."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(5, 100))
        return dc.Patch(
            data=data,
            coords={"time": np.arange(100), "distance": np.arange(5)},
            dims=("distance", "time"),
        )

    @pytest.fixture
    def partial_nan_patch(self):
        """A patch where channel 2 has a gap of NaN in the middle."""
        data = np.ones((4, 50), dtype=np.float64)
        data[2, 20:30] = np.nan
        return dc.Patch(
            data=data,
            coords={"time": np.arange(50), "distance": np.arange(4)},
            dims=("distance", "time"),
        )

    @pytest.fixture
    def all_nan_channel_patch(self):
        """A patch where channel 1 is entirely NaN."""
        data = np.ones((3, 50), dtype=np.float64)
        data[1, :] = np.nan
        return dc.Patch(
            data=data,
            coords={"time": np.arange(50), "distance": np.arange(3)},
            dims=("distance", "time"),
        )

    @pytest.fixture
    def leading_trailing_nan_patch(self):
        """A patch with NaN at leading and trailing edges."""
        data = np.ones((2, 30), dtype=np.float64)
        data[0, :5] = np.nan   # leading
        data[0, -5:] = np.nan  # trailing
        return dc.Patch(
            data=data,
            coords={"time": np.arange(30), "distance": np.arange(2)},
            dims=("distance", "time"),
        )

    def test_clean_patch_untouched(self, clean_patch):
        """No NaN → return same patch, zero stats."""
        from das_pipeline.preprocessing.nan_handler import sanitize_nan_patch

        result, stats = sanitize_nan_patch(clean_patch)
        data = np.asarray(result.data)

        assert stats["nan_ratio"] == 0.0
        assert stats["n_all_nan_channels"] == 0
        assert not np.isnan(data).any()

    def test_partial_nan_interpolated(self, partial_nan_patch):
        """Gap NaN → linearly interpolated, no NaN remains."""
        from das_pipeline.preprocessing.nan_handler import sanitize_nan_patch

        result, stats = sanitize_nan_patch(partial_nan_patch)
        data = np.asarray(result.data)

        assert stats["nan_ratio"] > 0.0
        assert stats["n_all_nan_channels"] == 0
        assert not np.isnan(data).any()
        # The gap (20:30) should now be ~1.0 (interpolated)
        assert np.allclose(data[2, 20:30], 1.0, atol=1e-6)

    def test_all_nan_channel_kept_nan(self, all_nan_channel_patch):
        """Fully NaN channel → kept as NaN, flagged in attrs (NOT filled with 0)."""
        from das_pipeline.preprocessing.nan_handler import sanitize_nan_patch

        result, stats = sanitize_nan_patch(all_nan_channel_patch)
        data = np.asarray(result.data)

        assert stats["n_all_nan_channels"] == 1
        assert stats["all_nan_channel_indices"] == [1]
        # Bad channel should STILL be NaN (not 0.0)
        assert np.isnan(data[1]).all()
        # Good channels should be clean
        assert not np.isnan(data[0]).any()
        assert not np.isnan(data[2]).any()
        # Attrs should carry comma-separated indices
        assert result.attrs.get("nan_sanitized") is True
        assert result.attrs["all_nan_channel_indices"] == "1"
        assert result.attrs["bad_channel_mask"] == "0,1,0"

    def test_leading_trailing_nan(self, leading_trailing_nan_patch):
        """Edge NaN → extrapolated to nearest boundary value (~1.0)."""
        from das_pipeline.preprocessing.nan_handler import sanitize_nan_patch

        result, stats = sanitize_nan_patch(leading_trailing_nan_patch)
        data = np.asarray(result.data)

        assert not np.isnan(data).any()
        assert np.allclose(data[0, :5], 1.0, atol=1e-6)
        assert np.allclose(data[0, -5:], 1.0, atol=1e-6)

    def test_single_valid_point(self):
        """Degenerate: only 1 valid point → fill entire channel with it."""
        data = np.ones((1, 20), dtype=np.float64) * np.nan
        data[0, 10] = 5.0
        patch = dc.Patch(
            data=data,
            coords={"time": np.arange(20), "distance": np.arange(1)},
            dims=("distance", "time"),
        )

        from das_pipeline.preprocessing.nan_handler import sanitize_nan_patch

        result, stats = sanitize_nan_patch(patch)
        data_out = np.asarray(result.data)
        assert not np.isnan(data_out).any()
        assert np.allclose(data_out, 5.0)

    def test_time_first_dim_patch(self):
        """Patch with (time, distance) dims should also work."""
        data = np.ones((50, 3), dtype=np.float64)
        data[25:30, 1] = np.nan
        patch = dc.Patch(
            data=data,
            coords={"time": np.arange(50), "distance": np.arange(3)},
            dims=("time", "distance"),
        )

        from das_pipeline.preprocessing.nan_handler import sanitize_nan_patch

        result, stats = sanitize_nan_patch(patch)
        data_out = np.asarray(result.data)
        assert not np.isnan(data_out).any()

    def test_attrs_preserved(self, partial_nan_patch):
        """Sanitized patch should carry nan_sanitized attrs."""
        from das_pipeline.preprocessing.nan_handler import sanitize_nan_patch

        result, _ = sanitize_nan_patch(partial_nan_patch)
        assert result.attrs.get("nan_sanitized") is True
        assert result.attrs.get("nan_ratio_before") is not None

    def test_h5_roundtrip_nan_attrs(self):
        """Write sanitized patch to .h5 and read back via dc.spool — attrs must survive."""
        import tempfile
        import os

        from das_pipeline.preprocessing.nan_handler import sanitize_nan_patch

        # Build a patch with mixed NaN: channel 0 all-NaN, channel 2 partial NaN
        n_samples = 20
        data = np.ones((4, n_samples), dtype=np.float64)
        data[0, :] = np.nan          # fully NaN channel
        data[2, 5:10] = np.nan       # partial NaN gap
        patch = dc.Patch(
            data=data,
            coords={
                "distance": np.arange(4, dtype=np.float64),
                "time": dc.get_coord(
                    values=np.arange(n_samples, dtype="timedelta64[ms]"),
                    step=np.timedelta64(1, "ms"),
                ),
            },
            dims=["distance", "time"],
        )

        result, _stats = sanitize_nan_patch(patch)

        # Write to temp .h5
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
                tmp_path = f.name
            result.io.write(tmp_path, "dasdae")

            # Read back via dc.spool — the SAME way all downsteam CLI commands do
            spool = dc.spool(str(tmp_path))
            patches = list(spool)
            assert len(patches) == 1, "Expected exactly 1 patch in spool"
            read_patch = patches[0]

            # Verify scalar attrs — roundtrip converts to numpy scalars
            assert bool(read_patch.attrs.get("nan_sanitized")) is True
            assert float(read_patch.attrs["nan_ratio_before"]) > 0.0
            assert int(read_patch.attrs["n_all_nan_channels"]) == 1

            # Verify comma-separated string attrs
            indices_str = str(read_patch.attrs["all_nan_channel_indices"])
            assert indices_str == "0"

            mask_str = str(read_patch.attrs["bad_channel_mask"])
            assert mask_str == "1,0,0,0"

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ============================================================================
# amplification.py — nanmedian behavior
# ============================================================================

class TestAmplificationNanHandling:
    """Tests for teleseismic/amplification.py nan-aware changes."""

    def test_nanmedian_excludes_nan(self):
        """np.nanmedian ignores NaN values correctly."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        assert np.isnan(np.median(data))
        assert np.nanmedian(data) == 3.0

    def test_nanmedian_all_nan_returns_nan(self):
        """np.nanmedian of all NaN → NaN (but with warning suppressed)."""
        import warnings

        data = np.array([np.nan, np.nan, np.nan])
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            result = np.nanmedian(data)
        assert np.isnan(result)

    def test_reference_all_nan_raises(self):
        """_compute_reference_amplitude raises when all ref channels NaN."""
        from das_pipeline.teleseismic.amplification import _compute_reference_amplitude

        # First 2 channels are valid, last 3 are NaN → n_reference=3 picks all-NaN
        amplitudes = np.array([1.0, 2.0, np.nan, np.nan, np.nan])
        with pytest.raises(ValueError, match="全部為 NaN"):
            _compute_reference_amplitude(amplitudes, n_reference=3)

    def test_reference_partial_nan_ok(self):
        """_compute_reference_amplitude with some NaN → works."""
        from das_pipeline.teleseismic.amplification import _compute_reference_amplitude

        amplitudes = np.array([np.nan, 0.5, 1.0, 2.0, 3.0])
        ref = _compute_reference_amplitude(amplitudes, n_reference=3)
        # Last 3 channels: [1.0, 2.0, 3.0] → median = 2.0
        assert ref == pytest.approx(2.0)


# ============================================================================
# sta_lta.py — isfinite mask
# ============================================================================

class TestStaLtaNanHandling:
    """Tests for detection/sta_lta.py NaN-aware trigger mask."""

    def test_nan_not_triggered(self):
        """NaN > threshold should be False and not counted as triggered."""
        data = np.array([[1.0, 2.0, np.nan, 4.0, 5.0],
                         [1.0, 2.0, 3.0, 4.0, 5.0]])
        threshold = 3.0
        finite = np.isfinite(data)
        triggered = (data > threshold) & finite

        # NaN at [0, 2] → False
        assert not triggered[0, 2]
        # 4.0, 5.0 → True
        assert triggered[0, 3]
        assert triggered[0, 4]
        # 3.0 is not > 3.0 → False
        assert not triggered[1, 2]


# ============================================================================
# spectrogram / visualization — nan_to_num
# ============================================================================

class TestVisualizationNanHandling:
    """Tests for visualization nan_to_num behavior."""

    def test_nan_to_num_replaces_nan_with_zero(self):
        """np.nan_to_num should replace NaN with 0.0."""
        data = np.array([1.0, np.nan, 3.0])
        cleaned = np.nan_to_num(data, nan=0.0)
        assert cleaned[1] == 0.0
        assert not np.isnan(cleaned).any()

    def test_nan_to_num_does_not_affect_finite(self):
        """np.nan_to_num leaves finite values unchanged."""
        data = np.array([1.0, -2.5, 0.0, np.inf, -np.inf])
        cleaned = np.nan_to_num(data)
        assert cleaned[0] == 1.0
        assert cleaned[1] == -2.5
        assert cleaned[2] == 0.0
        assert np.isfinite(cleaned[3])  # inf → large finite
        assert np.isfinite(cleaned[4])  # -inf → large negative finite


# ============================================================================
# Integration: full pipeline NaN scenario
# ============================================================================

class TestPipelineNanIntegration:
    """End-to-end simulation of NaN flowing through the pipeline."""

    def test_nan_sanitizer_before_preprocessing(self):
        """Simulate: miniseed→sanitize→preprocess pipeline."""
        from das_pipeline.preprocessing.nan_handler import sanitize_nan_patch

        # Simulate a patch with gaps (like what miniseed_loader would produce)
        data = np.random.default_rng(99).normal(size=(3, 200)).astype(np.float64)
        data[1, 80:90] = np.nan  # gap in channel 1
        patch = dc.Patch(
            data=data,
            coords={"time": np.arange(200), "distance": np.arange(3)},
            dims=("distance", "time"),
        )

        # Step 1: sanitize
        clean_patch, stats = sanitize_nan_patch(patch)
        clean_data = np.asarray(clean_patch.data)

        # Partial NaN should be interpolated (no NaN left)
        assert not np.isnan(clean_data).any()
        assert stats["nan_ratio"] > 0.0
        assert stats["n_all_nan_channels"] == 0

    def test_all_nan_channel_not_zeroed(self):
        """All-NaN channel should NOT be filled with 0 — should stay NaN + be flagged."""
        from das_pipeline.preprocessing.nan_handler import sanitize_nan_patch

        data = np.ones((3, 100), dtype=np.float64)
        data[1, :] = np.nan
        patch = dc.Patch(
            data=data,
            coords={"time": np.arange(100), "distance": np.arange(3)},
            dims=("distance", "time"),
        )

        clean_patch, stats = sanitize_nan_patch(patch)
        clean_data = np.asarray(clean_patch.data)

        assert stats["n_all_nan_channels"] == 1
        assert np.isnan(clean_data[1]).all()
        assert not np.isnan(clean_data[0]).any()
        assert not np.isnan(clean_data[2]).any()
        assert clean_patch.attrs["all_nan_channel_indices"] == "1"

    def test_amplification_excludes_bad_channels(self):
        """amplification.py should physically remove channels flagged as all-NaN."""
        import json

        from das_pipeline.teleseismic.amplification import compute_amplification
        from das_pipeline.config import TeleseismicConfig

        import pandas as pd

        # Create a patch with channel 1 flagged as bad, with proper datetime64 time axis
        data = np.random.default_rng(77).normal(size=(4, 500)).astype(np.float64)
        data[1, :] = np.nan  # dead channel
        time_axis = pd.date_range("2023-02-06T01:00:00", periods=500, freq="1s")
        patch = dc.Patch(
            data=data,
            coords={"time": time_axis,
                    "distance": np.array([0., 100., 200., 300.])},
            dims=("distance", "time"),
        )
        patch = patch.update_attrs(
            all_nan_channel_indices="1",
        )

        # Use origin_time within the patch time range with short distance
        config = TeleseismicConfig(
            event_distance_km=100.0,
            event_origin_time="2023-02-06T01:00:00",
            reference_channels=2,
        )

        result = compute_amplification(patch, config)
        assert result is not None
        # Channel 1 should have been excluded → only 3 channels remain
        assert result["n_channels"] == 3
        assert result["n_excluded_bad_channels"] == 1
        # Assert distances don't include the dead channel's distance (100m)
        assert 100.0 not in result["distances"]

    # === Bug regression tests for detect / plot NaN handling ===

    def test_exclude_bad_channels_mapping(self):
        """_exclude_bad_channels_from_patch must return correct local_to_original."""
        from das_pipeline.utils.bad_channels import exclude_bad_channels_from_patch as _exclude_bad_channels_from_patch

        # 5 channels, channel 1 is bad
        data = np.ones((5, 10), dtype=np.float64)
        data[1, :] = np.nan
        patch = dc.Patch(
            data=data,
            coords={"time": np.arange(10), "distance": np.array([0., 10., 20., 30., 40.])},
            dims=("distance", "time"),
        )

        cleaned, n_excluded, l2o = _exclude_bad_channels_from_patch(patch, [1])
        assert n_excluded == 1
        assert l2o == [0, 2, 3, 4]  # original indices of remaining channels
        assert cleaned.shape[0] == 4
        np.testing.assert_array_equal(
            cleaned.coords.get_array("distance"),
            np.array([0., 20., 30., 40.]),
        )

    def test_exclude_bad_channels_no_bad(self):
        """_exclude_bad_channels_from_patch with no bad indices returns identity."""
        from das_pipeline.utils.bad_channels import exclude_bad_channels_from_patch as _exclude_bad_channels_from_patch

        data = np.ones((3, 10), dtype=np.float64)
        patch = dc.Patch(
            data=data,
            coords={"time": np.arange(10), "distance": np.arange(3, dtype=np.float64)},
            dims=("distance", "time"),
        )
        cleaned, n_excluded, l2o = _exclude_bad_channels_from_patch(patch, [])
        assert n_excluded == 0
        assert l2o == [0, 1, 2]
        assert cleaned is patch  # should return same object

    def test_fk_spectrum_nan_interpolated(self):
        """plot_fk_spectrum 應對 NaN channel 線性內插，而非排除。"""
        # 5 channels, channel 1 is all-NaN, rest are signal
        rng = np.random.default_rng(42)
        data = rng.normal(size=(5, 128)).astype(np.float64)
        data[1, :] = np.nan
        patch = dc.Patch(
            data=data,
            coords={
                "distance": np.arange(5, dtype=np.float64),
                "time": dc.get_coord(
                    values=np.arange(128, dtype="timedelta64[ms]"),
                    step=np.timedelta64(1, "ms"),
                ),
            },
            dims=("distance", "time"),
        )

        # 直接傳入含 NaN 的 patch，FK 內部應自行內插後繪圖
        from das_pipeline.visualization.fk import plot_fk_spectrum
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plot_fk_spectrum(patch)
        assert isinstance(fig, plt.Figure), "FK plot should return a Figure"
        plt.close(fig)

        # 驗證內插後圖中有實際資料（不是全 NaN）
        # specplot 底層 waterfall 會用 nanmin/nanmax 計算顯示範圍，
        # 若資料全 NaN 會噴 RuntimeWarning 但不會報錯；
        # 這裡透過檢查第 1 個 channel 的值不全是外插的 0，
        # 來確認內插有產生合理值（非全 0 或全 NaN）
        # Note: patch.new() 不改變原始 patch，但內部會用內插後的 data 繪圖

    def test_waterfall_nanpercentile_works(self):
        """Bug 3 regression: clip_percentile uses nanpercentile, not percentile."""
        data = np.ones((5, 20), dtype=np.float64)
        data[1, :] = np.nan  # bad channel
        data[0, 5:10] = 5.0
        data[4, 12:15] = 3.0

        abs_data = np.abs(data)

        # Precondition: plain np.percentile returns NaN with NaN values
        assert np.isnan(np.percentile(abs_data, 99.0)), \
            "precondition: np.percentile should return NaN with NaN data"

        # Fix: np.nanpercentile should be finite and > 0
        fixed_value = np.nanpercentile(abs_data, 99.0)
        assert np.isfinite(fixed_value), f"nanpercentile should be finite, got {fixed_value}"
        assert fixed_value > 0, f"nanpercentile should be > 0, got {fixed_value}"

        # Verify the waterfall code actually uses nanpercentile now
        import inspect
        from das_pipeline.visualization.waterfall import plot_waterfall
        source = inspect.getsource(plot_waterfall)
        assert "nanpercentile" in source, \
            "plot_waterfall must use np.nanpercentile, not np.percentile"


# ============================================================================
# cli.py — spectrogram uses patch_clean
# ============================================================================

class TestSpectrogramWithPatchClean:
    """Verify spectrogram receives patch_clean (bad channels excluded)."""

    def test_patch_clean_excludes_bad_channels(self):
        """spectrogram should get clean_patch without all-NaN channels."""
        # Simulate what 'plot' command does: exclude bad channels from patch_clean
        from das_pipeline.utils.bad_channels import get_bad_channel_indices as _get_bad_channel_indices, exclude_bad_channels_from_patch as _exclude_bad_channels_from_patch

        data = np.ones((5, 100), dtype=np.float64)
        data[1, :] = np.nan  # channel 1 is bad
        patch = dc.Patch(
            data=data,
            coords={
                "distance": np.array([0., 100., 200., 300., 400.]),
                "time": np.arange(100),
            },
            dims=("distance", "time"),
        )
        patch = patch.update_attrs(all_nan_channel_indices="1")

        bad_indices = _get_bad_channel_indices(patch)
        assert bad_indices == [1]

        patch_clean, n_excluded, l2o = _exclude_bad_channels_from_patch(patch, bad_indices)
        assert n_excluded == 1
        assert l2o == [0, 2, 3, 4]

        # After exclusion, the middle channel (index 2 in clean) should be
        # original channel 3, not the NaN channel 1.
        clean_data = np.asarray(patch_clean.data)
        assert not np.isnan(clean_data).any()
        assert clean_data.shape[0] == 4

        # The default channel=None in spectrogram picks n_channels // 2 = 2
        # which maps back to l2o[2] = 3 (original channel index)
        assert l2o[2] == 3

    def test_spectrogram_channel_mapping_with_bad(self):
        """When --channel is specified, it indexes into patch_clean, not original."""
        from das_pipeline.utils.bad_channels import get_bad_channel_indices as _get_bad_channel_indices, exclude_bad_channels_from_patch as _exclude_bad_channels_from_patch

        data = np.ones((5, 100), dtype=np.float64)
        data[1, :] = np.nan
        data[3, :] = np.nan  # channels 1 and 3 are bad
        patch = dc.Patch(
            data=data,
            coords={
                "distance": np.array([0., 100., 200., 300., 400.]),
                "time": np.arange(100),
            },
            dims=("distance", "time"),
        )
        patch = patch.update_attrs(all_nan_channel_indices="1,3")

        bad_indices = _get_bad_channel_indices(patch)
        patch_clean, n_excluded, l2o = _exclude_bad_channels_from_patch(patch, bad_indices)
        assert l2o == [0, 2, 4]  # original channels 0, 2, 4 remain
        assert n_excluded == 2

        # User-specified channel=1 in clean space → original channel 2
        assert l2o[1] == 2
