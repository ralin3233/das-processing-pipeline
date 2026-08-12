"""CLI 命令：從 YAML 設定檔執行 MiniSEED → HDF5 轉檔。

讀取 YAML 設定檔，執行完整轉檔管線（讀取 → 前處理 → 座標對齊 → 儲存）。
"""

from pathlib import Path

import typer
from typing_extensions import Annotated


def register(app: typer.Typer) -> None:
    @app.command()
    def convert(
        config: Annotated[
            Path,
            typer.Option("--config", "-c", exists=True, help="YAML 設定檔路徑"),
        ],
    ):
        from das_pipeline.config import ConvertConfig
        from das_pipeline.pipeline import run_convert

        cfg = ConvertConfig.from_yaml(config)

        save_paths = run_convert(cfg)
        typer.echo(f"✅ 轉檔完成，共產生 {len(save_paths)} 個檔案")
        for p in save_paths:
            typer.echo(f"   - {p}")