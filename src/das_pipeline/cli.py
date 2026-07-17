# src/das_pipeline/cli.py

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import typer
from typing_extensions import Annotated

app = typer.Typer(help="DAS Processing Pipeline CLI")


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
    """
    import logging

    import matplotlib

    if no_display:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
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
        velocity_min=vmin,
        velocity_max=vmax,
        skip_channels=skip_channels,
    )

    typer.echo(
        f"遠震分析設定: D={distance} km, origin={origin_time}, "
        f"v=[{vmin}, {vmax}] km/s, ref_channels={ref_channels}"
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
            writer.writerow(["channel_index", "amplification", "reference_amplitude"])
            for ch, amp in zip(result["channel_indices"], result["amplification"]):
                writer.writerow([ch, f"{amp:.6f}", f"{result['reference_amplitude']:.6e}"])
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

    # --- 繪圖 ---
    type_set = {t.lower() for t in type}
    fig_axes = []

    if "waterfall" in type_set:
        fig, ax = plt.subplots(figsize=(12, 5))
        try:
            fig = plot_waterfall(
                patch, ax=ax,
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
                patch, ax=ax,
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


if __name__ == "__main__":
    app()
