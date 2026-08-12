"""CLI 命令：視覺化繪圖工具（plot 子命令群組）。

提供 waterfall、F-K 頻譜圖、spectrogram 三種繪圖功能。
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

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


# ---------------------------------------------------------------------------
# shared helper: load & clean a patch from the common path/merge/pattern options
# ---------------------------------------------------------------------------


def _load_and_clean(
    path: Path,
    merge: bool,
    pattern: str,
    sort_by: str,
):
    """Load (and optionally merge) a Patch and remove bad channels for plot."""
    file_paths = collect_h5_files(path, pattern)
    patch = load_patch(file_paths, merge=merge, sort_by=sort_by)
    log_patch_info(patch)
    patch_clean = warn_bad_channels_for_plot(patch)
    return patch, patch_clean


def _save_or_show(
    fig,
    name: str,
    save: Optional[Path],
    format: str,
    dpi: int,
    no_display: bool,
) -> None:
    """Save figure to disk and/or display it interactively."""
    if save:
        save_dir = Path(save)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{name}.{format}"
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        typer.echo(f"✅ 已儲存: {save_path}")
        if no_display:
            plt.close(fig)
    if not save or not no_display:
        plt.show()


def register(app: typer.Typer) -> None:
    plot_app = typer.Typer(help="視覺化繪圖工具")

    # ── waterfall ──────────────────────────────────────────────────────

    @plot_app.command()
    def waterfall(
        path: Annotated[
            Path,
            typer.Argument(..., help=".h5 檔案路徑或資料夾路徑", exists=True),
        ],
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
        time_range: Annotated[
            Optional[Tuple[str, str]],
            typer.Option(
                "--time-range",
                help="時間範圍 [start, end] (ISO 格式)",
            ),
        ] = None,
        distance_range: Annotated[
            Optional[Tuple[float, float]],
            typer.Option(
                "--distance-range", "--dist-range",
                help="距離/通道範圍 [start, end]",
            ),
        ] = None,
        colormap: Annotated[
            str,
            typer.Option("--colormap", help="matplotlib colormap 名稱"),
        ] = "seismic",
        title: Annotated[
            Optional[str],
            typer.Option("--title", help="圖表自訂標題"),
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
        no_display: Annotated[
            bool,
            typer.Option("--no-display", help="存檔模式下不彈出視窗"),
        ] = False,
    ):
        """繪製 DAS 瀑布圖 (time x distance amplitude map)。"""
        setup_matplotlib_backend(no_display)

        from das_pipeline.visualization import plot_waterfall

        _, patch = _load_and_clean(path, merge, pattern, sort_by)

        fig, ax = plt.subplots(figsize=(12, 5))
        try:
            fig = plot_waterfall(
                patch, ax=ax,
                time_range=time_range,
                distance_range=distance_range,
                colormap=colormap,
                title=title,
            )
        except Exception as e:
            logger.error(f"繪製 waterfall 失敗: {e}")
            plt.close(fig)
            typer.echo(f"❌ 繪製 waterfall 失敗: {e}")
            raise typer.Exit(1)

        _save_or_show(fig, "waterfall", save, format, dpi, no_display)

    # ── fk ─────────────────────────────────────────────────────────────

    @plot_app.command()
    def fk(
        path: Annotated[
            Path,
            typer.Argument(..., help=".h5 檔案路徑或資料夾路徑", exists=True),
        ],
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
        channel_spacing: Annotated[
            Optional[float],
            typer.Option(
                "--channel-spacing",
                help="相鄰通道的物理距離 (m)，用於 FK 正確 wavenumber",
            ),
        ] = None,
        freq_range: Annotated[
            Optional[Tuple[float, float]],
            typer.Option("--freq-range", help="頻率範圍 [low, high] Hz"),
        ] = None,
        colormap: Annotated[
            str,
            typer.Option("--colormap", help="matplotlib colormap 名稱"),
        ] = "viridis",
        title: Annotated[
            Optional[str],
            typer.Option("--title", help="圖表自訂標題"),
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
        no_display: Annotated[
            bool,
            typer.Option("--no-display", help="存檔模式下不彈出視窗"),
        ] = False,
    ):
        """繪製 F-K 頻譜圖 (frequency x wavenumber 功率譜)。"""
        setup_matplotlib_backend(no_display)

        from das_pipeline.visualization import plot_fk_spectrum

        patch, _ = _load_and_clean(path, merge, pattern, sort_by)

        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            fig = plot_fk_spectrum(
                patch, ax=ax,
                channel_spacing=channel_spacing,
                freq_range=freq_range,
                colormap=colormap,
                title=title,
            )
        except Exception as e:
            logger.error(f"繪製 F-K 失敗: {e}")
            plt.close(fig)
            typer.echo(f"❌ 繪製 F-K 頻譜圖失敗: {e}")
            raise typer.Exit(1)

        _save_or_show(fig, "fk", save, format, dpi, no_display)

    # ── spectrogram ────────────────────────────────────────────────────

    @plot_app.command()
    def spectrogram(
        path: Annotated[
            Path,
            typer.Argument(..., help=".h5 檔案路徑或資料夾路徑", exists=True),
        ],
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
        freq_range: Annotated[
            Optional[Tuple[float, float]],
            typer.Option("--freq-range", help="頻率範圍 [low, high] Hz"),
        ] = None,
        colormap: Annotated[
            str,
            typer.Option("--colormap", help="matplotlib colormap 名稱"),
        ] = "viridis",
        title: Annotated[
            Optional[str],
            typer.Option("--title", help="圖表自訂標題"),
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
        no_display: Annotated[
            bool,
            typer.Option("--no-display", help="存檔模式下不彈出視窗"),
        ] = False,
    ):
        """繪製 DAS 時頻圖 (spectrogram, time x frequency PSD)。"""
        setup_matplotlib_backend(no_display)

        from das_pipeline.visualization import plot_spectrogram

        _, patch_clean = _load_and_clean(path, merge, pattern, sort_by)

        fig, ax = plt.subplots(figsize=(10, 5))
        try:
            fig = plot_spectrogram(
                patch_clean, ax=ax,
                channel=channel,
                freq_range=freq_range,
                colormap=colormap,
                title=title,
            )
        except Exception as e:
            logger.error(f"繪製 spectrogram 失敗: {e}")
            plt.close(fig)
            typer.echo(f"❌ 繪製 spectrogram 失敗: {e}")
            raise typer.Exit(1)

        _save_or_show(fig, "spectrogram", save, format, dpi, no_display)

    app.add_typer(plot_app, name="plot")