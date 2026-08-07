# src/das_pipeline/cli/__init__.py

import typer

app = typer.Typer(help="DAS Processing Pipeline CLI")


@app.callback()
def main():
    """DAS Processing Pipeline"""
    pass


# Register all subcommands
from das_pipeline.cli.commands import register_all
register_all(app)


if __name__ == "__main__":
    app()