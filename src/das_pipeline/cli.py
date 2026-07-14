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
@app.command()
def convert(config: Path = typer.Option(..., "--config", "-c", exists=True)):
    cfg = ConvertConfig.from_yaml(config)
    setup_logging(cfg.runtime.log_level)

    save_paths = run_convert(cfg)
    typer.echo(f"✅ 轉檔完成，共產生 {len(save_paths)} 個檔案")
    for p in save_paths:
        typer.echo(f"   - {p}")


if __name__ == "__main__":
    app()