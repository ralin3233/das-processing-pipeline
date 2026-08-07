# src/das_pipeline/cli/commands/overlay.py

import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from das_pipeline.cli.helpers import collect_csv_files, setup_matplotlib_backend

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
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
        import matplotlib
        setup_matplotlib_backend(no_display)

        from das_pipeline.overlay import plot_overlay

        # 收集 CSV 檔案
        csv_files = collect_csv_files(dir, pattern)
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