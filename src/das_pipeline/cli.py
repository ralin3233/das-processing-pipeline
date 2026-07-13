# src/das_pipeline/cli.py

from pathlib import Path
import typer

from das_pipeline.config import ConvertConfig
from das_pipeline.pipeline import run_convert
from das_pipeline.utils.logging_config import setup_logging

app = typer.Typer(help="DAS Processing Pipeline CLI")

@app.callback()
def main():
    """DAS Processing Pipeline"""
    pass

@app.command()
def convert(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="設定檔路徑 (YAML)",
        exists=True,          # 檔案不存在會直接報錯，不用自己 check
    ),
):
    """執行 MiniSEED -> Patch 轉檔"""
    cfg = ConvertConfig.from_yaml(config)
    setup_logging(cfg.runtime.log_level)

    save_path = run_convert(cfg)
    typer.echo(f"✅ 轉檔完成: {save_path}")


if __name__ == "__main__":
    app()