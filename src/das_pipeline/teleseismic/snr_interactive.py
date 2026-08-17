# src/das_pipeline/teleseismic/snr_interactive.py

"""互動式 SNR 視窗選取模組。

在 matplotlib 圖窗中顯示單一 channel 的波形，讓使用者拖曳圈選
訊號窗（綠色）與雜訊窗（紅色），並即時顯示計算出的 SNR（dB）。

操作方式：:

    - 按 ``s`` 切換為「訊號窗」選取模式
    - 按 ``n`` 切換為「雜訊窗」選取模式
    - 在圖上按住滑鼠左鍵拖曳即設定目前模式的視窗
    - 按 ``q`` 或直接關閉圖窗完成選取
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import dascore as dc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import SpanSelector

from das_pipeline.teleseismic.snr import compute_power, _extract_channel_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time conversion helpers
# ---------------------------------------------------------------------------


def _to_epoch_seconds(t: np.datetime64) -> float:
    """將 datetime64 轉為 Unix epoch 秒（float）。

    Parameters
    ----------
    t : np.datetime64
        單一時間點。

    Returns
    -------
    float
        對應的 epoch 秒數。
    """
    return float(np.datetime64(t, "ns").astype("int64") / 1e9)


def _from_epoch_seconds(x: float) -> np.datetime64:
    """將 Unix epoch 秒（float）轉為 ``datetime64[ns]``。

    Parameters
    ----------
    x : float
        epoch 秒數。

    Returns
    -------
    np.datetime64
        對應的 ``datetime64[ns]`` 時間點。
    """
    return np.datetime64(int(round(x * 1e9)), "ns")


def _fmt_time(seconds: float) -> str:
    """將 epoch 秒格式化為 ``HH:MM:SS``（UTC）。"""
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Interactive picker
# ---------------------------------------------------------------------------


class _SnrWindowPicker:
    """matplotlib 互動式 SNR 訊號／雜訊窗選取器。"""

    def __init__(
        self,
        patch: dc.Patch,
        channel_index: int,
        default_signal_window: Optional[tuple[np.datetime64, np.datetime64]] = None,
        default_noise_window: Optional[tuple[np.datetime64, np.datetime64]] = None,
        event_distance_km: float = float("nan"),
    ) -> None:
        """建立選取器與圖窗。

        Parameters
        ----------
        patch : dc.Patch
            已前處理的 DAS Patch，dims 為 ``["time", "distance"]``。
        channel_index : int
            要顯示與分析的 channel 索引。
        default_signal_window : tuple[np.datetime64, np.datetime64] or None
            初始訊號窗（供使用者參考，可再拖曳修改）。
        default_noise_window : tuple[np.datetime64, np.datetime64] or None
            初始雜訊窗。
        event_distance_km : float
            震央距離（僅供標題與結果紀錄）。
        """
        self.patch = patch
        self.channel_index = channel_index
        self.event_distance_km = event_distance_km

        # 目前選取模式: 'signal' 或 'noise'
        self.mode = "signal"

        # 訊號窗 / 雜訊窗，單位為 epoch 秒 (float, float)
        self.signal_window: Optional[tuple[float, float]] = None
        self.noise_window: Optional[tuple[float, float]] = None
        self.result: Optional[dict] = None

        # 擷取 channel 波形與時間軸（epoch 秒）
        time_vals = np.asarray(patch.get_coord("time")).astype("datetime64[ns]")
        self.t_epoch = time_vals.astype("int64") / 1e9
        self.data = _extract_channel_data(patch, channel_index)

        t0, t1 = float(self.t_epoch[0]), float(self.t_epoch[-1])
        if default_signal_window is not None:
            s0 = _to_epoch_seconds(default_signal_window[0])
            s1 = _to_epoch_seconds(default_signal_window[1])
            s0, s1 = min(s0, s1), max(s0, s1)
            # 裁切至 patch 時間範圍內，避免初始陰影落在圖外
            if s1 > t0 and s0 < t1:
                self.signal_window = (max(s0, t0), min(s1, t1))

        if default_noise_window is not None:
            n0 = _to_epoch_seconds(default_noise_window[0])
            n1 = _to_epoch_seconds(default_noise_window[1])
            n0, n1 = min(n0, n1), max(n0, n1)
            if n1 > t0 and n0 < t1:
                self.noise_window = (max(n0, t0), min(n1, t1))

        # ── 建立圖窗 ──
        self.fig, self.ax = plt.subplots(figsize=(12, 5))

        # 顯示用降採樣，避免超長 patch 拖慢互動
        step = max(1, len(self.t_epoch) // 200_000)
        self.ax.plot(self.t_epoch[::step], self.data[::step],
                     color="black", linewidth=0.6)
        self.ax.set_xlabel("Time (UTC)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)
        self.ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: _fmt_time(float(x)))
        )

        # 三個標題：中＝channel/距離，左＝目前模式，右＝即時 SNR
        self.ax.set_title(
            f"Channel {channel_index}  (D={event_distance_km:g} km)"
        )
        self._mode_text = self.ax.set_title("", loc="left")
        self._result_text = self.ax.set_title("", loc="right")

        self._signal_span = None
        self._noise_span = None

        # 水平拖曳圈選
        self.span = SpanSelector(
            self.ax,
            self._onselect,
            "horizontal",
            useblit=False,
            props=dict(facecolor="0.7", alpha=0.4),
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # 固定 x 軸為資料時間範圍。SpanSelector 水平選取會建立一個
        # x=0（資料座標）的隱形矩形，若任由 autoscale 會把 x 軸從
        # epoch 0 一路展開到資料末端，波形會被壓成一條垂直線。
        self.ax.set_xlim(self.t_epoch[0], self.t_epoch[-1])

        self._update_title()
        self._redraw()

    # -- 互動回呼 ---------------------------------------------------------

    def _onselect(self, xmin: float, xmax: float) -> None:
        """SpanSelector 圈選完成後的回呼。"""
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        if self.mode == "signal":
            self.signal_window = (float(xmin), float(xmax))
        else:
            self.noise_window = (float(xmin), float(xmax))

        self._update_title()
        self._redraw()

    def _on_key(self, event) -> None:
        """鍵盤事件：``s``/``n`` 切換模式，``q``/``escape`` 結束。"""
        if event.key == "s":
            self.mode = "signal"
            self._update_title()
            self._redraw()
        elif event.key == "n":
            self.mode = "noise"
            self._update_title()
            self._redraw()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    # -- 計算與繪圖 -------------------------------------------------------

    def _compute_result(self) -> Optional[dict]:
        """以目前選取的兩個窗計算 SNR，結果存入 ``self.result``。"""
        if self.signal_window is None or self.noise_window is None:
            self.result = None
            return None

        signal_start = _from_epoch_seconds(self.signal_window[0])
        signal_end = _from_epoch_seconds(self.signal_window[1])
        noise_start = _from_epoch_seconds(self.noise_window[0])
        noise_end = _from_epoch_seconds(self.noise_window[1])

        try:
            signal_patch = self.patch.select(time=(signal_start, signal_end))
            noise_patch = self.patch.select(time=(noise_start, noise_end))
        except Exception:
            logger.exception("擷取訊號／雜訊窗失敗")
            self.result = None
            return None

        try:
            signal_data = _extract_channel_data(signal_patch, self.channel_index)
            noise_data = _extract_channel_data(noise_patch, self.channel_index)
        except IndexError as e:
            logger.error("無法擷取 channel %d: %s", self.channel_index, e)
            self.result = None
            return None

        P_signal = compute_power(signal_data)
        P_noise = compute_power(noise_data)

        if np.isnan(P_signal) or np.isnan(P_noise):
            self.result = None
            return None

        if P_noise == 0.0:
            logger.warning("P_noise = 0，SNR 無法計算（分母為零）")
            self.result = None
            return None

        snr_db = 10.0 * np.log10(P_signal / P_noise)

        self.result = {
            "snr_db": float(snr_db),
            "P_signal": float(P_signal),
            "P_noise": float(P_noise),
            "signal_window": (str(signal_start), str(signal_end)),
            "noise_window": (str(noise_start), str(noise_end)),
            "channel_index": self.channel_index,
            "event_distance_km": self.event_distance_km,
        }
        return self.result

    def _update_title(self) -> None:
        """Update the current selection mode indicator (top-left)."""
        mode_label = "signal" if self.mode == "signal" else "noise"
        self._mode_text.set_text(f"Mode: {mode_label}")

    def _update_result_text(self) -> None:
        """更新右上角的即時 SNR 顯示。"""
        if self.result is None:
            self._result_text.set_text("SNR: N/A")
        else:
            self._result_text.set_text(f"SNR: {self.result['snr_db']:+.2f} dB")

    def _redraw(self) -> None:
        """重繪訊號／雜訊窗陰影並重新計算 SNR。"""
        for span in (self._signal_span, self._noise_span):
            if span is not None:
                span.remove()
        self._signal_span = None
        self._noise_span = None

        if self.signal_window is not None:
            self._signal_span = self.ax.axvspan(
                self.signal_window[0], self.signal_window[1],
                color="green", alpha=0.25, label="signal",
            )
        if self.noise_window is not None:
            self._noise_span = self.ax.axvspan(
                self.noise_window[0], self.noise_window[1],
                color="red", alpha=0.25, label="noise",
            )

        self._compute_result()
        self._update_result_text()
        self.fig.canvas.draw_idle()

    @staticmethod
    def _fmt_span(span: tuple[float, float]) -> str:
        """將 epoch 秒視窗格式化為可讀字串。"""
        return f"{_fmt_time(span[0])} ~ {_fmt_time(span[1])}"

    # -- 主流程 -----------------------------------------------------------

    def run(
        self,
    ) -> tuple[
        tuple[np.datetime64, np.datetime64],
        tuple[np.datetime64, np.datetime64],
        Optional[dict],
    ]:
        """顯示互動圖窗，關閉後回傳選取結果。

        Returns
        -------
        tuple
            ``(signal_window, noise_window, result)``，其中 signal_window /
            noise_window 為 ``(start, end)`` 的 ``datetime64`` 元組，
            result 為 SNR 計算結果 dict（若資料無效則為 None）。

        Raises
        ------
        ValueError
            若關閉圖窗前未選取訊號窗或雜訊窗。
        """
        plt.show()

        if self.signal_window is None:
            raise ValueError(
                "No signal window selected. "
                "Drag to select the signal segment before closing the window."
            )
        if self.noise_window is None:
            raise ValueError(
                "No noise window selected. "
                "Drag to select the noise segment before closing the window."
            )

        signal = (
            _from_epoch_seconds(self.signal_window[0]),
            _from_epoch_seconds(self.signal_window[1]),
        )
        noise = (
            _from_epoch_seconds(self.noise_window[0]),
            _from_epoch_seconds(self.noise_window[1]),
        )

        # 確保結果與最終選取一致
        self._compute_result()

        return signal, noise, self.result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pick_snr_windows_interactive(
    patch: dc.Patch,
    channel_index: int,
    default_signal_window: Optional[tuple[np.datetime64, np.datetime64]] = None,
    default_noise_window: Optional[tuple[np.datetime64, np.datetime64]] = None,
    event_distance_km: float = float("nan"),
) -> tuple[
    tuple[np.datetime64, np.datetime64],
    tuple[np.datetime64, np.datetime64],
    Optional[dict],
]:
    """互動式選取 SNR 的訊號窗與雜訊窗。

    顯示指定 channel 的波形圖，讓使用者以滑鼠拖曳圈選訊號窗與
    雜訊窗（按 ``s``／``n`` 切換模式），關閉圖窗後回傳選取結果。

    Parameters
    ----------
    patch : dc.Patch
        已前處理的 DAS Patch，dims 為 ``["time", "distance"]``。
    channel_index : int
        要分析的 channel 索引。
    default_signal_window : tuple[np.datetime64, np.datetime64] or None
        初始訊號窗（可由 ``_compute_time_window`` 依距離／速度算出）。
    default_noise_window : tuple[np.datetime64, np.datetime64] or None
        初始雜訊窗（可由 ``_compute_noise_window`` 算出）。
    event_distance_km : float
        震央距離（km），僅供標題與結果紀錄。

    Returns
    -------
    tuple
        ``(signal_window, noise_window, result)``。
    """
    picker = _SnrWindowPicker(
        patch=patch,
        channel_index=channel_index,
        default_signal_window=default_signal_window,
        default_noise_window=default_noise_window,
        event_distance_km=event_distance_km,
    )
    return picker.run()
