# src/das_pipeline/cli.py

from pathlib import Path
from typing import List, Optional, Tuple

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
        typer.Option("--time-range", help="時間範圍 [start, end]"),
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

    typer.echo(
        f"Patch 維度: {patch.shape}, "
        f"time: {patch.coords['time'].min()} ~ {patch.coords['time'].max()}"
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