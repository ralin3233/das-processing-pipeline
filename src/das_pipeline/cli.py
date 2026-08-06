# src/das_pipeline/cli.py

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import typer
from typing_extensions import Annotated

app = typer.Typer(help="DAS Processing Pipeline CLI")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_bad_channel_indices(patch) -> list[int]:
    """Read ``all_nan_channel_indices`` from patch attrs (comma-separated)."""
    raw = patch.attrs.get("all_nan_channel_indices")
    if raw is None or str(raw).strip() == "":
        return []
    try:
        return [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    except (ValueError, TypeError):
        return []


def _exclude_bad_channels_from_patch(patch, bad_indices: list[int]):
    """Physically remove entirely-NaN channels from a patch.

    Returns (cleaned_patch, n_excluded, local_to_original).
    ``local_to_original`` maps compressed channel index back to the
    original channel number.
    """
    import dascore as dc

    n_orig = patch.shape[0] if "distance" in patch.dims else patch.shape[1]
    local_to_original = list(range(n_orig))

    if not bad_indices:
        return patch, 0, local_to_original

    data = np.asarray(patch.data)
    dims = patch.dims
    channel_axis = dims.index("distance") if "distance" in dims else 0

    keep_mask = np.ones(data.shape[channel_axis], dtype=bool)
    global_bad = np.array(bad_indices, dtype=int)
    global_bad = global_bad[(global_bad >= 0) & (global_bad < len(keep_mask))]
    keep_mask[global_bad] = False
    n_excluded = (~keep_mask).sum()

    if n_excluded == 0:
        return patch, 0, local_to_original

    if channel_axis == 0:
        data = data[keep_mask, :]
        dist_coord = patch.coords.get_array("distance")[keep_mask]
    else:
        data = data[:, keep_mask]
        dist_coord = patch.coords.get_array("distance")[keep_mask]

    local_to_original = [i for i in range(len(keep_mask)) if keep_mask[i]]

    new_patch = dc.Patch(
        data=data,
        coords={"time": patch.coords.get_array("time"), "distance": dist_coord},
        dims=dims,
        attrs=patch.attrs,
    )
    return new_patch, n_excluded, local_to_original


@app.callback()
def main():
    """DAS Processing Pipeline"""
    pass


@app.command()
def convert(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, help="YAML 設定檔路徑"),
    ],
):
    from das_pipeline.config import ConvertConfig
    from das_pipeline.pipeline import run_convert
    from das_pipeline.utils.logging_config import setup_logging

    cfg = ConvertConfig.from_yaml(config)
    setup_logging(cfg.runtime.log_level)

    save_paths = run_convert(cfg)
    typer.echo(f"✅ 轉檔完成，共產生 {len(save_paths)} 個檔案")
    for p in save_paths:
        typer.echo(f"   - {p}")


@app.command()
def amplification(
    path: Annotated[
        Path,
        typer.Argument(..., help="已處理的 .h5 檔案路徑或資料夾路徑", exists=True),
    ],
    distance: Annotated[
        float,
        typer.Option("--distance", "-d", help="震央距離 (km)"),
    ],
    origin_time: Annotated[
        str,
        typer.Option("--origin-time", "-o", help="發震時刻 (ISO 格式, e.g. 2023-02-06T01:17:35)"),
    ],
    ref_channels: Annotated[
        int,
        typer.Option("--ref-channels", help="基準 channel 數（井底最深 N 個）"),
    ] = 10,
    vmin: Annotated[
        float,
        typer.Option("--vmin", help="最慢群速度 (km/s)"),
    ] = 2.0,
    vmax: Annotated[
        float,
        typer.Option("--vmax", help="最快群速度 (km/s)"),
    ] = 4.0,
    merge: Annotated[
        bool,
        typer.Option("--merge", "-m", help="啟用批次合併模式（合併多個 .h5 為一個 Patch）"),
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
        typer.Option("--save", "-s", help="輸出圖片/CSV 目錄"),
    ] = None,
    event_label: Annotated[
        Optional[str],
        typer.Option("--event-label", "-l", help="事件標籤（用於圖例）"),
    ] = None,
    title: Annotated[
        Optional[str],
        typer.Option("--title", "-t", help="圖表自訂標題"),
    ] = None,
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="圖片解析度"),
    ] = 150,
    csv: Annotated[
        bool,
        typer.Option("--csv", help="同時輸出各 channel 放大倍率至 CSV"),
    ] = False,
    ref_distance_range: Annotated[
        Optional[Tuple[float, float]],
        typer.Option(
            "--ref-distance-range",
            help="基準距離範圍 (m)，水平光纖用。例: --ref-distance-range 500 600",
        ),
    ] = None,
    skip_channels: Annotated[
        int,
        typer.Option("--skip-channels", help="跳過前 N 個 channel（井口附近易受雜訊干擾）"),
    ] = 0,
    no_display: Annotated[
        bool,
        typer.Option("--no-display", help="存檔模式下不彈出視窗"),
    ] = False,
):
    """對已前處理的 .h5 檔案進行遠震地層放大效應分析。

    根據震央距離與表面波群速度計算時間窗，擷取雷利波列，
    以最深 N 個 channel 的中位數振幅作為基準，
    計算各 channel 的放大倍率並繪圖。

    使用範例：
    \b
        # 單一檔案分析
        das-pipeline amplification data/processed/event1.h5 \\
            --distance 3000 --origin-time "2023-02-06T01:17:35" \\
            --save results/

        # 多檔案合併後分析
        das-pipeline amplification data/processed/ \\
            --distance 3000 --origin-time "2023-02-06T01:17:35" \\
            --merge --pattern "*.h5" --save results/

        # 水平光纖：以距離 500~600 米段落為基準
        das-pipeline amplification data/processed/event1.h5 \\
            --distance 3000 --origin-time "2023-02-06T01:17:35" \\
            --ref-distance-range 500 600 --save results/
    """
    import logging

    import matplotlib

    if no_display:
        matplotlib.use("Agg")

    import dascore as dc

    from das_pipeline.config import TeleseismicConfig
    from das_pipeline.teleseismic import compute_amplification, plot_amplification
    from das_pipeline.visualization.merge import merge_patches

    logger = logging.getLogger(__name__)

    # --- 收集檔案 ---
    if path.is_dir():
        file_paths = sorted(path.glob(pattern))
        if not file_paths:
            typer.echo(f"❌ 在 {path} 找不到符合 {pattern} 的檔案")
            raise typer.Exit(1)
        typer.echo(f"找到 {len(file_paths)} 個 .h5 檔案")
    else:
        file_paths = [path]

    # --- 載入 Patch ---
    if merge and len(file_paths) > 1:
        typer.echo("合併多個 chunk 檔案中...")
        patch = merge_patches(file_paths, sort_by=sort_by)
    else:
        spool = dc.spool(str(file_paths[0]))
        patch = spool[0]

    time_values = patch.coords.get_array("time")
    typer.echo(
        f"Patch 維度: {patch.shape}, "
        f"time: {time_values.min()} ~ {time_values.max()}"
    )

    # --- 建立 Config ---
    config = TeleseismicConfig(
        event_distance_km=distance,
        event_origin_time=origin_time,
        reference_channels=ref_channels,
        reference_distance_range=ref_distance_range,
        velocity_min=vmin,
        velocity_max=vmax,
        skip_channels=skip_channels,
    )

    ref_info = (
        f"ref_distance_range=[{ref_distance_range[0]}, {ref_distance_range[1]}] m"
        if ref_distance_range else f"ref_channels={ref_channels}"
    )
    typer.echo(
        f"遠震分析設定: D={distance} km, origin={origin_time}, "
        f"v=[{vmin}, {vmax}] km/s, {ref_info}"
        f"{', skip_channels=' + str(skip_channels) if skip_channels else ''}"
    )

    # --- 計算放大倍率 ---
    result = compute_amplification(patch, config)
    if result is None:
        typer.echo("❌ 時間窗與檔案時間範圍無交集，無法分析")
        raise typer.Exit(1)

    typer.echo(
        f"✅ 放大倍率範圍: [{np.min(result['amplification']):.3f}, "
        f"{np.max(result['amplification']):.3f}], "
        f"中位數: {np.median(result['amplification']):.3f}"
    )

    # --- 輸出 CSV ---
    if csv and save is not None:
        import csv as csv_module

        save_dir = Path(save)
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / "teleseismic_amplification.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv_module.writer(f)
            writer.writerow(["distance_m", "amplification", "reference_amplitude"])
            for dist, amp in zip(result["distances"], result["amplification"]):
                writer.writerow([f"{dist:.2f}", f"{amp:.6f}", f"{result['reference_amplitude']:.6e}"])
        typer.echo(f"✅ CSV 已儲存: {csv_path}")

    # --- 繪圖 ---
    labels = [event_label] if event_label else None
    plot_amplification(
        [result],
        save_dir=Path(save) if save else None,
        labels=labels,
        title=title,
        dpi=dpi,
        show=not no_display,
    )

    if save:
        typer.echo(f"✅ 圖表已儲存至: {save}/teleseismic_amplification.png")


@app.command()
def detect(
    path: Annotated[
        Path,
        typer.Argument(..., help="已處理的 .h5 檔案路徑", exists=True),
    ],
    sta_window: Annotated[
        float,
        typer.Option("--sta-window", help="STA 短窗長度 (秒)"),
    ] = 0.5,
    lta_window: Annotated[
        float,
        typer.Option("--lta-window", help="LTA 長窗長度 (秒)"),
    ] = 10.0,
    trigger_threshold: Annotated[
        float,
        typer.Option("--trigger-threshold", help="STA/LTA ratio 觸發閾值"),
    ] = 3.0,
    detrigger_threshold: Annotated[
        float,
        typer.Option("--detrigger-threshold", help="STA/LTA ratio 解除觸發閾值"),
    ] = 1.5,
    min_channels: Annotated[
        int,
        typer.Option("--min-channels", help="最少同時觸發的通道數（空間一致性）"),
    ] = 30,
    min_duration: Annotated[
        float,
        typer.Option("--min-duration", help="最短事件持續時間 (秒)"),
    ] = 0.1,
    merge_window: Annotated[
        float,
        typer.Option("--merge-window", help="合併相鄰事件閾值 (秒)，0=不合併"),
    ] = 0.0,
    save: Annotated[
        Optional[Path],
        typer.Option("--save", "-s", help="輸出目錄路徑"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="輸出格式: csv, json (預設兩者皆輸出)"),
    ] = "all",
):
    """對已處理的 .h5 檔案執行 STA/LTA 觸發檢測（使用 DASCore stalta）。

    逐通道計算 STA/LTA ratio，並以空間一致性篩選真實訊號事件。

    \b
    使用範例：
        das-pipeline detect data/processed/event.h5 \\
            --sta-window 0.5 --lta-window 10.0 \\
            --trigger-threshold 3.0 --save results/
    """
    import json
    import csv as csv_module
    import logging

    import dascore as dc
    import numpy as np

    from das_pipeline.config import StaLtaConfig
    from das_pipeline.detection import compute_sta_lta_patch, detect_events

    logger = logging.getLogger(__name__)

    # --- 載入 Patch ---
    spool = dc.spool(str(path))
    patch = spool[0]

    time_values = patch.coords.get_array("time")
    sampling_rate = 1.0 / (
        (time_values[1] - time_values[0]) / np.timedelta64(1, "s")
    )

    typer.echo(
        f"Patch 維度: {patch.shape}, "
        f"sampling_rate: {sampling_rate:.2f} Hz, "
        f"time: {time_values[0]} ~ {time_values[-1]}"
    )

    # --- Config ---
    config = StaLtaConfig(
        sta_window_s=sta_window,
        lta_window_s=lta_window,
        trigger_threshold=trigger_threshold,
        detrigger_threshold=detrigger_threshold,
        min_channels_triggered=min_channels,
        min_event_duration_s=min_duration,
        merge_window_s=merge_window,
    )

    typer.echo(
        f"STA/LTA 設定: sta={sta_window}s, lta={lta_window}s, "
        f"trigger={trigger_threshold}, detrigger={detrigger_threshold}, "
        f"min_channels={min_channels}, min_duration={min_duration}s"
    )

    # --- 排除不可用 channel（全 NaN，由 nan_handler 標記）---
    local_to_orig: list[int] = []
    bad_indices = _get_bad_channel_indices(patch)
    if bad_indices:
        n_before = patch.shape[0] if "distance" in patch.dims else patch.shape[1]
        typer.echo(
            f"⚠️  此檔案有 {len(bad_indices)} 個 channel 完全無資料"
            f"（index: {bad_indices}），將不參與 STA/LTA 檢測。"
        )
        patch, n_excluded, local_to_orig = _exclude_bad_channels_from_patch(patch, bad_indices)
        typer.echo(f"    排除後剩餘 channel 數: {n_before - n_excluded}")
    else:
        local_to_orig = list(range(patch.shape[0] if "distance" in patch.dims else patch.shape[1]))

    # --- STA/LTA via DASCore ---
    sta_lta_patch = compute_sta_lta_patch(patch, config)
    typer.echo(f"STA/LTA ratio shape: {sta_lta_patch.shape}")

    # --- Event detection ---
    events = detect_events(sta_lta_patch, config, sampling_rate)
    typer.echo(f"檢測到 {len(events)} 個事件")

    if len(events) == 0:
        typer.echo("⚠️  未檢測到任何觸發事件")
        return

    for i, evt in enumerate(events):
        typer.echo(
            f"  [{i}] {evt['start_time']} ~ {evt['end_time']} "
            f"({evt['duration_s']:.2f}s), "
            f"peak={evt['peak_ratio']:.2f}, "
            f"channels={evt['num_triggered_channels']}"
        )

    # --- 輸出 (map local channel indices back to original) ---
    if save is not None:
        save_dir = Path(save)
        save_dir.mkdir(parents=True, exist_ok=True)

        export = []
        for e in events:
            local_chs = e["triggered_channels"]
            orig_chs = [local_to_orig[ch] for ch in local_chs if ch < len(local_to_orig)]
            export.append({
                "start_time": e["start_time"],
                "end_time": e["end_time"],
                "peak_time": e["peak_time"],
                "peak_ratio": e["peak_ratio"],
                "triggered_channels": orig_chs,
                "num_triggered_channels": len(orig_chs),
                "duration_s": e["duration_s"],
            })

        write_json = format in ("json", "all")
        write_csv = format in ("csv", "all")

        if write_json:
            json_path = save_dir / "detections.json"
            with open(json_path, "w") as f:
                json.dump(export, f, indent=2, ensure_ascii=False)
            typer.echo(f"✅ JSON: {json_path}")

        if write_csv:
            csv_path = save_dir / "detections.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv_module.writer(f)
                writer.writerow([
                    "start_time", "end_time", "peak_time", "peak_ratio",
                    "num_triggered_channels", "triggered_channels", "duration_s",
                ])
                for e in export:
                    writer.writerow([
                        e["start_time"], e["end_time"], e["peak_time"],
                        f"{e['peak_ratio']:.6f}",
                        e["num_triggered_channels"],
                        ",".join(str(c) for c in e["triggered_channels"]),
                        f"{e['duration_s']:.6f}",
                    ])
            typer.echo(f"✅ CSV: {csv_path}")


@app.command()
def plot(
    path: Annotated[
        Path,
        typer.Argument(..., help=".h5 檔案路徑或資料夾路徑", exists=True),
    ],
    type: Annotated[
        List[str],
        typer.Option(
            "--type", "-t",
            help="圖表類型: waterfall, fk, spectrogram (可複選)",
        ),
    ] = ["waterfall"],
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
    channel: Annotated[
        Optional[int],
        typer.Option("--channel", help="Spectrogram 要分析的通道索引"),
    ] = None,
    time_range: Annotated[
        Optional[Tuple[str, str]],
        typer.Option("--time-range", help="時間範圍 [start, end] (ISO 格式, e.g. 2023-02-06T10:30:00)"),
    ] = None,
    distance_range: Annotated[
        Optional[Tuple[float, float]],
        typer.Option(
            "--distance-range", "--dist-range",
            help="距離/通道範圍 [start, end]",
        ),
    ] = None,
    freq_range: Annotated[
        Optional[Tuple[float, float]],
        typer.Option("--freq-range", help="頻率範圍 [low, high] Hz"),
    ] = None,
    channel_spacing: Annotated[
        Optional[float],
        typer.Option(
            "--channel-spacing",
            help="相鄰通道的物理距離 (m)，用於 FK 正確 wavenumber",
        ),
    ] = None,
    save: Annotated[
        Optional[Path],
        typer.Option(
            "--save", "-s",
            help="存檔目錄路徑，不指定則互動式顯示",
        ),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", help="存檔格式: png, pdf, svg"),
    ] = "png",
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="圖片解析度"),
    ] = 150,
    colormap: Annotated[
        str,
        typer.Option("--colormap", help="matplotlib colormap 名稱"),
    ] = "seismic",
    title: Annotated[
        Optional[str],
        typer.Option("--title", help="圖表自訂標題"),
    ] = None,
    no_display: Annotated[
        bool,
        typer.Option("--no-display", help="存檔模式下不彈出視窗"),
    ] = False,
):
    """對已處理的 .h5 檔案進行視覺化分析。

    支援 Waterfall、F-K 頻譜圖、Spectrogram 時頻圖，以及批次合併多檔案繪圖。
    """
    import logging

    import matplotlib

    if no_display:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import dascore as dc

    from das_pipeline.visualization import (
        plot_waterfall,
        plot_fk_spectrum,
        plot_spectrogram,
        merge_patches,
    )

    logger = logging.getLogger(__name__)

    # --- 收集檔案 ---
    if path.is_dir():
        file_paths = sorted(path.glob(pattern))
        if not file_paths:
            typer.echo(f"❌ 在 {path} 找不到符合 {pattern} 的檔案")
            raise typer.Exit(1)
        typer.echo(f"找到 {len(file_paths)} 個檔案")
    else:
        file_paths = [path]

    # --- 載入 Patch ---
    if merge and len(file_paths) > 1:
        typer.echo("合併多個 chunk 檔案中...")
        patch = merge_patches(file_paths, sort_by=sort_by)
    else:
        spool = dc.spool(str(file_paths[0]))
        patch = spool[0]

    time_values = patch.coords.get_array("time")
    typer.echo(
        f"Patch 維度: {patch.shape}, "
        f"time: {time_values.min()} ~ {time_values.max()}"
    )

    # --- 排除不可用 channel（全 NaN，由 nan_handler 標記）---
    # Waterfall 保留 NaN 標示無訊號區；FK/spectrogram 需排除避免 NaN 污染 FFT/STFT。
    bad_indices = _get_bad_channel_indices(patch)
    if bad_indices:
        typer.echo(
            f"⚠️  此檔案有 {len(bad_indices)} 個 channel 完全無資料"
            f"（index: {bad_indices}），FK/spectrogram 將排除這些 channel。"
        )
        patch_clean, _n_ex, _l2o = _exclude_bad_channels_from_patch(patch, bad_indices)
    else:
        patch_clean = patch

    # --- 繪圖 ---
    type_set = {t.lower() for t in type}
    fig_axes = []

    if "waterfall" in type_set:
        fig, ax = plt.subplots(figsize=(12, 5))
        try:
            fig = plot_waterfall(
                patch, ax=ax,  # waterfall keeps original patch (NaN = no signal)
                time_range=time_range,
                distance_range=distance_range,
                colormap=colormap,
                title=title,
            )
            fig_axes.append(("waterfall", fig))
        except Exception as e:
            logger.error(f"繪製 waterfall 失敗: {e}")
            plt.close(fig)
            typer.echo(f"❌ 繪製 waterfall 失敗: {e}")

    if "fk" in type_set:
        _cmap = colormap if colormap != "seismic" else "viridis"
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            fig = plot_fk_spectrum(
                patch_clean, ax=ax,  # use cleaned patch (bad channels removed)
                channel_spacing=channel_spacing,
                freq_range=freq_range,
                colormap=_cmap,
                title=title,
            )
            fig_axes.append(("fk", fig))
        except Exception as e:
            logger.error(f"繪製 F-K 失敗: {e}")
            plt.close(fig)
            typer.echo(f"❌ 繪製 F-K 頻譜圖失敗: {e}")

    if "spectrogram" in type_set:
        _cmap = colormap if colormap != "seismic" else "viridis"
        fig, ax = plt.subplots(figsize=(10, 5))
        try:
            fig = plot_spectrogram(
                patch, ax=ax,
                channel=channel,
                freq_range=freq_range,
                colormap=_cmap,
                title=title,
            )
            fig_axes.append(("spectrogram", fig))
        except Exception as e:
            logger.error(f"繪製 spectrogram 失敗: {e}")
            plt.close(fig)
            typer.echo(f"❌ 繪製 spectrogram 失敗: {e}")

    if not fig_axes:
        typer.echo("❌ 沒有成功繪製任何圖表")
        raise typer.Exit(1)

    # --- 存檔或顯示 ---
    if save:
        save_dir = Path(save)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in fig_axes:
            save_path = save_dir / f"{name}.{format}"
            fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
            typer.echo(f"✅ 已儲存: {save_path}")
            if no_display:
                plt.close(fig)
        if not no_display:
            plt.show()
    else:
        plt.show()


@app.command()
def overlay(
    dir: Annotated[
        Path,
        typer.Argument(..., help="包含 CSV 檔案的目錄路徑", exists=True),
    ],
    pattern: Annotated[
        str,
        typer.Option("--pattern", "-p", help="CSV 檔案 glob pattern"),
    ] = "*.csv",
    labels: Annotated[
        Optional[str],
        typer.Option("--labels", "-l", help="圖例標籤，逗號分隔（預設為檔名）"),
    ] = None,
    save: Annotated[
        Optional[Path],
        typer.Option("--save", "-s", help="輸出圖片目錄"),
    ] = None,
    title: Annotated[
        Optional[str],
        typer.Option("--title", "-t", help="圖表自訂標題"),
    ] = None,
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="圖片解析度"),
    ] = 150,
    csv: Annotated[
        bool,
        typer.Option("--csv", help="同時輸出疊圖資料至 CSV（各事件 + 中位數）"),
    ] = False,
    no_display: Annotated[
        bool,
        typer.Option("--no-display", help="存檔模式下不彈出視窗"),
    ] = False,
):
    """疊加顯示多個事件的遠震放大倍率曲線，並繪製中位數線。

    讀取多個 ``das-pipeline amplification --csv`` 輸出的 CSV 檔案，
    在同一張圖上疊加各事件的放大倍率曲線，並以紅色虛線繪製中位數線。

    \b
    使用範例：
        das-pipeline overlay results/ \\
            --pattern "*.csv" --labels "M6.5,M7.0,M7.2" \\
            --save results/ --title "Amplification Overlay"
    """
    import logging

    import matplotlib

    if no_display:
        matplotlib.use("Agg")

    from das_pipeline.overlay import plot_overlay

    logger = logging.getLogger(__name__)

    # 收集 CSV 檔案
    csv_files = sorted(dir.glob(pattern))
    if not csv_files:
        typer.echo(f"❌ 在 {dir} 找不到符合 {pattern} 的檔案")
        raise typer.Exit(1)
    typer.echo(f"找到 {len(csv_files)} 個 CSV 檔案")

    # 解析圖例標籤
    label_list: Optional[list[str]] = None
    if labels is not None:
        label_list = [lbl.strip() for lbl in labels.split(",")]

    result = plot_overlay(
        csv_paths=csv_files,
        labels=label_list,
        save_dir=Path(save) if save else None,
        title=title,
        dpi=dpi,
        show=not no_display,
        csv_output=csv,
    )

    if result is not None:
        typer.echo(f"✅ 疊圖已儲存: {result}")
        if csv and save is not None:
            typer.echo(f"✅ 疊圖 CSV 已儲存: {save}/amplification_overlay.csv")
    elif save is None:
        typer.echo("✅ 疊圖完成，已顯示於視窗")
    else:
        typer.echo("❌ 疊圖失敗，請檢查 CSV 檔案與 log")


if __name__ == "__main__":
    app()
