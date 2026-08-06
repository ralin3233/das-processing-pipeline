# src/das_pipeline/detection/sta_lta.py

"""STA/LTA trigger detection for DAS data, powered by DASCore.

Uses ``patch.stalta(time=(sta_s, lta_s))`` for the STA/LTA ratio
computation (boxcar STA / boxcar LTA on absolute amplitude),
then performs per-channel spatial-consistency trigger detection.

Core algorithm:
  1. Compute STA/LTA ratio via DASCore's built-in transform.
  2. Apply per-time-step spatial consistency check
     (num triggered channels >= min_channels_triggered).
  3. Follow trigger/detrigger hysteresis with debounce.
  4. Merge adjacent events within merge_window_s.
"""

import logging
from typing import Optional

import dascore as dc
import numpy as np

from das_pipeline.config import StaLtaConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STA/LTA ratio via DASCore
# ---------------------------------------------------------------------------

def compute_sta_lta_patch(
    patch: dc.Patch,
    config: StaLtaConfig,
) -> dc.Patch:
    """Compute STA/LTA ratio on a Patch using DASCore's built-in transform.

    Steps:
      1. ``patch.abs()`` to get absolute amplitude.
      2. ``.stalta(time=(sta_window_s, lta_window_s))`` for ratio.

    Parameters
    ----------
    patch : dc.Patch
        Input DAS Patch with ``time`` dimension.
    config : StaLtaConfig
        STA/LTA window parameters.

    Returns
    -------
    dc.Patch
        STA/LTA ratio Patch.  The ``time`` coordinate is preserved but
        shortened by ``lta_samples - 1`` samples (boxcar edge effect).
    """
    # DASCore stalta uses boxcar rolling mean, ours is energy-based on
    # absolute amplitude.  Historically the pipeline used raw strain_rate;
    # taking abs() makes it amplitude-driven, consistent with classic STA/LTA.
    abs_patch = patch.abs()

    sta_lta_patch = abs_patch.stalta(time=(config.sta_window_s, config.lta_window_s))

    # Check for NaN contamination after STA/LTA computation
    data = np.asarray(sta_lta_patch.data)
    nan_frac = np.mean(np.isnan(data))
    if nan_frac > 0.5:
        logger.warning(
            "STA/LTA ratio 中有 %.1f%% NaN（>50%%），"
            "資料可能因斷訊導致濾波擴散，檢測結果可能不可靠。", nan_frac * 100,
        )
    elif nan_frac > 0.0:
        logger.info("STA/LTA ratio 中有 %.1f%% NaN。", nan_frac * 100)

    logger.info(
        "DASCore STA/LTA: sta=%gs, lta=%gs, output shape=%s",
        config.sta_window_s,
        config.lta_window_s,
        sta_lta_patch.shape,
    )
    return sta_lta_patch


# ---------------------------------------------------------------------------
# Event detection with hysteresis
# ---------------------------------------------------------------------------

def _merge_nearby_events(
    events: list[dict],
    merge_window_s: float,
) -> list[dict]:
    """Merge events whose inter-event gap < merge_window_s."""
    if not events or merge_window_s <= 0:
        return events

    merged: list[dict] = []
    current = events[0].copy()
    for evt in events[1:]:
        gap = (
            np.datetime64(evt["start_time"])
            - np.datetime64(current["end_time"])
        ) / np.timedelta64(1, "s")
        if gap < merge_window_s:
            current["end_time"] = evt["end_time"]
            current["duration_s"] = (
                np.datetime64(current["end_time"])
                - np.datetime64(current["start_time"])
            ) / np.timedelta64(1, "s")
            if evt["peak_ratio"] > current["peak_ratio"]:
                current["peak_time"] = evt["peak_time"]
                current["peak_ratio"] = evt["peak_ratio"]
            current["triggered_channels"] = sorted(
                set(current["triggered_channels"]) | set(evt["triggered_channels"])
            )
            current["num_triggered_channels"] = len(current["triggered_channels"])
        else:
            merged.append(current)
            current = evt.copy()
    merged.append(current)
    logger.info("Merged %d events into %d", len(events), len(merged))
    return merged


def detect_events(
    sta_lta_patch: dc.Patch,
    config: StaLtaConfig,
    sampling_rate: float,
) -> list[dict]:
    """Detect trigger events from a STA/LTA ratio Patch.

    Uses per-channel + spatial-consistency: for each time step,
    count channels where ratio > trigger_threshold.  If the count
    is >= min_channels_triggered, enter trigger state.  Detrigger
    requires sustained drop below detrigger_threshold.

    Parameters
    ----------
    sta_lta_patch : dc.Patch
        STA/LTA ratio Patch from :func:`compute_sta_lta_patch`.
        Dimensions: (time, distance) or (distance, time).
    config : StaLtaConfig
        Trigger/detrigger parameters.
    sampling_rate : float
        Sampling rate in Hz (used for min_event_duration conversion).

    Returns
    -------
    list[dict]
        Each event dict:
        - start_time / end_time / peak_time: str (ISO)
        - peak_ratio: float
        - triggered_channels: list[int]
        - num_triggered_channels: int
        - duration_s: float
    """
    # Extract data and time axis from the patch
    data = np.asarray(sta_lta_patch.data)
    time_coord = sta_lta_patch.get_coord("time")
    time_aligned = np.asarray(time_coord.values).ravel()

    # Ensure data is (n_channels, n_time)
    dims = sta_lta_patch.dims
    time_axis_dim = dims.index("time")
    if time_axis_dim != data.ndim - 1:
        data = data.T

    n_channels, n_valid = data.shape
    min_dur_samp = max(1, int(round(config.min_event_duration_s * sampling_rate)))

    # Per-time-step triggered channels
    # Exclude NaN explicitly: NaN > threshold = False, but make intent clear.
    finite_data = np.isfinite(data)
    triggered_mask = (data > config.trigger_threshold) & finite_data  # (n_channels, n_valid)
    detriggered_mask = (data < config.detrigger_threshold) & finite_data
    triggered_count = np.sum(triggered_mask, axis=0)  # (n_valid,)

    events: list[dict] = []
    in_event = False
    event_start_idx: Optional[int] = None
    below_duration = 0

    for i in range(n_valid):
        enough = triggered_count[i] >= config.min_channels_triggered

        if not in_event:
            if enough:
                in_event = True
                event_start_idx = i
                below_duration = 0
        else:
            if enough:
                below_duration = 0
            else:
                triggered_now = np.flatnonzero(triggered_mask[:, i])
                all_below = (
                    np.all(detriggered_mask[triggered_now, i])
                    if len(triggered_now) > 0
                    else True
                )
                if all_below:
                    below_duration += 1
                else:
                    below_duration = 0

            if below_duration >= min_dur_samp:
                event_end_idx = i - below_duration
                if event_start_idx is not None and event_end_idx > event_start_idx:
                    events.append(
                        _build_event(
                            data, triggered_mask, time_aligned,
                            event_start_idx, event_end_idx,
                        )
                    )
                in_event = False
                event_start_idx = None
                below_duration = 0

    # Trailing event
    if in_event and event_start_idx is not None and event_start_idx < n_valid - 1:
        events.append(
            _build_event(
                data, triggered_mask, time_aligned,
                event_start_idx, n_valid - 1,
            )
        )

    events = _merge_nearby_events(events, config.merge_window_s)
    logger.info("Detected %d events from STA/LTA", len(events))
    return events


def _build_event(
    ratio: np.ndarray,
    triggered_mask: np.ndarray,
    time_aligned: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> dict:
    """Build a single event dict."""
    window_ratio = ratio[:, start_idx : end_idx + 1]
    peak_flat = np.argmax(window_ratio)
    peak_chan, peak_t_idx = np.unravel_index(peak_flat, window_ratio.shape)

    chan_triggered = np.flatnonzero(
        np.any(triggered_mask[:, start_idx : end_idx + 1], axis=1)
    )

    start_time = time_aligned[start_idx]
    end_time = time_aligned[end_idx]
    duration_s = (end_time - start_time) / np.timedelta64(1, "s")

    return {
        "start_time": str(start_time),
        "end_time": str(end_time),
        "peak_time": str(time_aligned[start_idx + peak_t_idx]),
        "peak_ratio": float(window_ratio[peak_chan, peak_t_idx]),
        "triggered_channels": chan_triggered.tolist(),
        "num_triggered_channels": int(len(chan_triggered)),
        "duration_s": float(duration_s),
    }