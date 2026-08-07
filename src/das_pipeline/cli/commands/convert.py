# src/das_pipeline/cli/commands/convert.py

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
        from das_pipeline.utils.logging_config import setup_logging

        cfg = ConvertConfig.from_yaml(config)
        setup_logging(cfg.runtime.log_level)

        save_paths = run_convert(cfg)
        typer.echo(f"✅ 轉檔完成，共產生 {len(save_paths)} 個檔案")
        for p in save_paths:
            typer.echo(f"   - {p}")