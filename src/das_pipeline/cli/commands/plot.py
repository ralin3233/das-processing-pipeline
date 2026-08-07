# src/das_pipeline/cli/commands/plot.py

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import typer
from typing_extensions import Annotated

from das_pipeline.cli.helpers import (
    collect_h5_files,
    load_patch,
    log_patch_info,
    setup_matplotlib_backend,
    warn_bad_channels_for_plot,
)

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
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
            typer.Option("--save", "-s", help="存檔目錄路徑，不指定則互動式顯示"),
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
        import matplotlib
        setup_matplotlib_backend(no_display)

        from das_pipeline.visualization import (
            plot_waterfall,
            plot_fk_spectrum,
            plot_spectrogram,
        )

        # --- 收集檔案 ---
        file_paths = collect_h5_files(path, pattern)

        # --- 載入 Patch ---
        patch = load_patch(file_paths, merge=merge, sort_by=sort_by)
        log_patch_info(patch)

        # --- 排除不可用 channel ---
        patch_clean = warn_bad_channels_for_plot(patch)

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
                    patch_clean, ax=ax,
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