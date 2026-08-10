# src/das_pipeline/cli/__init__.py

import typer
from typing_extensions import Annotated

app = typer.Typer(help="DAS Processing Pipeline CLI")


@app.callback()
def main(
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Log 等級: DEBUG | INFO | WARNING | ERROR",
        ),
    ] = "INFO",
):
    """DAS Processing Pipeline"""
    from das_pipeline.utils.logging_config import setup_logging

    setup_logging(log_level)


# Register all subcommands
from das_pipeline.cli.commands import register_all

register_all(app)


if __name__ == "__main__":
    app()