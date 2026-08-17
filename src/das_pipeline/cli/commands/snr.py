# src/das_pipeline/cli/commands/snr.py

"""CLI 命令：單一 channel SNR 分析。

用法::

    das-pipeline snr data/processed/event.h5 -c 100 -d 3000 -o "2023-02-06T01:17:35"
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from typing_extensions import Annotated

from das_pipeline.cli.helpers import (
    collect_h5_files,
    load_patch,
    log_patch_info,
)
from das_pipeline.config import SnrConfig
from das_pipeline.teleseismic.amplification import (
    _compute_time_window,
    _parse_origin_time,
)
from das_pipeline.teleseismic.snr import compute_channel_snr, _compute_noise_window
from das_pipeline.teleseismic.snr_interactive import pick_snr_windows_interactive

logger = logging.getLogger(__name__)


def _compute_default_windows(
    patch,
    config: SnrConfig,
) -> tuple[
    tuple[np.datetime64, np.datetime64],
    tuple[np.datetime64, np.datetime64],
]:
    """依震央距離與群速度計算預設訊號窗與雜訊窗。

    Parameters
    ----------
    patch : dc.Patch
        已載入的 DAS Patch。
    config : SnrConfig
        SNR 分析設定。

    Returns
    -------
    tuple
        ``(default_signal_window, default_noise_window)``，皆為
        ``(start, end)`` 的 ``datetime64`` 元組。
    """
    origin_time = _parse_origin_time(config.event_origin_time)
    t_signal_start, t_signal_end = _compute_time_window(
        origin_time,
        config.event_distance_km,
        config.velocity_min,
        config.velocity_max,
    )
    time_coord = patch.get_coord("time")
    patch_t_min: np.datetime64 = time_coord.min()  # type: ignore[assignment]

    default_noise = _compute_noise_window(
        t_signal_start, t_signal_end, patch_t_min, config.noise_offset_s,
    )
    return (t_signal_start, t_signal_end), default_noise


def _run_interactive_snr(patch, config: SnrConfig):
    """啟動互動式選窗，回傳 SNR 結果 dict（或 None）。"""
    default_signal, default_noise = _compute_default_windows(patch, config)
    _, _, result = pick_snr_windows_interactive(
        patch=patch,
        channel_index=config.channel_index,
        default_signal_window=default_signal,
        default_noise_window=default_noise,
        event_distance_km=config.event_distance_km,
    )
    return result


def register(app: typer.Typer) -> None:
    @app.command()
    def snr(
        path: Annotated[
            Path,
            typer.Argument(..., help="已處理的 .h5 檔案路徑或資料夾路徑", exists=True),
        ],
        channel: Annotated[
            int,
            typer.Option("--channel", "-c", help="要分析的 channel 索引"),
        ],
        distance: Annotated[
            float,
            typer.Option("--distance", "-d", help="震央距離 (km)"),
        ],
        origin_time: Annotated[
            str,
            typer.Option(
                "--origin-time", "-o",
                help="發震時刻 (ISO 格式, e.g. 2023-02-06T01:17:35)",
            ),
        ],
        vmin: Annotated[
            float,
            typer.Option("--vmin", help="最慢群速度 (km/s)"),
        ] = 2.0,
        vmax: Annotated[
            float,
            typer.Option("--vmax", help="最快群速度 (km/s)"),
        ] = 4.0,
        noise_offset: Annotated[
            float,
            typer.Option("--noise-offset", help="雜訊窗與訊號窗之間隔 (秒)"),
        ] = 30.0,
        merge: Annotated[
            bool,
            typer.Option("--merge", "-m", help="啟用批次合併模式"),
        ] = False,
        pattern: Annotated[
            str,
            typer.Option("--pattern", "-p", help="批次合併的 glob pattern"),
        ] = "*.h5",
        sort_by: Annotated[
            str,
            typer.Option("--sort-by", help="合併排序方式: chunk_index, timestamp"),
        ] = "chunk_index",
        save: Annotated[
            Optional[Path],
            typer.Option("--save", "-s", help="輸出 JSON 結果到指定目錄"),
        ] = None,
        interactive: Annotated[
            bool,
            typer.Option(
                "--interactive", "-i",
                help="互動式選窗模式：在波形圖上拖曳圈選訊號／雜訊窗",
            ),
        ] = False,
    ):
        """計算單一 channel 的 SNR（訊雜比，單位 dB）。

        根據震央距離與表面波群速度計算訊號時間窗，
        並以訊號窗開始前一段時間作為雜訊窗，
        利用 P = (1/N) Σ x_i² 計算訊號/雜訊平均功率，
        最終輸出 SNR_dB = 10 * log10(P_signal / P_noise)。

        \\b
        使用範例：
            das-pipeline snr data/processed/event1.h5 \\
                -c 100 -d 3000 -o "2023-02-06T01:17:35"

            das-pipeline snr data/processed/ \\
                -c 100 -d 3000 -o "2023-02-06T01:17:35" \\
                --merge --save results/

            互動式選窗（在波形圖上拖曳圈選訊號／雜訊窗）：
            das-pipeline snr data/processed/event1.h5 \\
                -c 100 -d 3000 -o "2023-02-06T01:17:35" --interactive
        """
        # --- 收集檔案 ---
        file_paths = collect_h5_files(path, pattern)

        # --- 載入 Patch ---
        patch = load_patch(file_paths, merge=merge, sort_by=sort_by)
        log_patch_info(patch)

        # --- 檢查 channel 是否在範圍內 ---
        if "distance" in patch.dims:
            n_channels = patch.shape[patch.dims.index("distance")]
        else:
            typer.echo("❌ Patch 沒有 distance 維度，無法依 channel 分析")
            raise typer.Exit(1)

        if channel < 0 or channel >= n_channels:
            typer.echo(f"❌ channel={channel} 超出範圍 [0, {n_channels - 1}]")
            raise typer.Exit(1)

        # --- 建立 Config ---
        config = SnrConfig(
            event_distance_km=distance,
            event_origin_time=origin_time,
            channel_index=channel,
            velocity_min=vmin,
            velocity_max=vmax,
            noise_offset_s=noise_offset,
        )

        typer.echo(
            f"SNR 分析設定: channel={channel}, D={distance} km, "
            f"origin={origin_time}, v=[{vmin}, {vmax}] km/s, "
            f"noise_offset={noise_offset}s"
        )

        # --- 計算 SNR ---
        if interactive:
            typer.echo(
                "互動式選窗：按 s 切換訊號窗、n 切換雜訊窗，"
                "拖曳圈選後按 q 或直接關閉視窗完成。"
            )
            result = _run_interactive_snr(patch, config)
        else:
            result = compute_channel_snr(patch, config)

        if result is None:
            typer.echo("❌ 無法計算 SNR（時間窗無交集或資料無效）")
            raise typer.Exit(1)

        # --- 輸出結果 ---
        snr_db = result["snr_db"]
        P_signal = result["P_signal"]
        P_noise = result["P_noise"]
        sig_start, sig_end = result["signal_window"]
        noi_start, noi_end = result["noise_window"]

        typer.echo()
        typer.echo("=" * 60)
        typer.echo(f"  Channel {channel} SNR 分析結果")
        typer.echo("=" * 60)
        typer.echo(f"  SNR        : {snr_db:+.2f} dB")
        typer.echo(f"  P_signal   : {P_signal:.6e}")
        typer.echo(f"  P_noise    : {P_noise:.6e}")
        typer.echo(f"  訊號窗     : [{sig_start}, {sig_end}]")
        typer.echo(f"  雜訊窗     : [{noi_start}, {noi_end}]")
        typer.echo(f"  P_signal/P_noise : {P_signal / P_noise:.2f}")
        typer.echo("=" * 60)

        # --- 可選存檔 ---
        if save:
            save_dir = Path(save)
            save_dir.mkdir(parents=True, exist_ok=True)
            json_path = save_dir / "snr_result.json"
            serializable = {
                k: v for k, v in result.items()
                if isinstance(v, (int, float, str, list, tuple))
            }
            with open(json_path, "w") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            typer.echo(f"\n✅ JSON 已儲存: {json_path}")
