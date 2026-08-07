# src/das_pipeline/cli/commands/amplification.py

import csv as csv_module
import logging
from pathlib import Path
from typing import Optional, Tuple

import dascore as dc
import numpy as np
import typer
from typing_extensions import Annotated

from das_pipeline.cli.helpers import (
    collect_h5_files,
    load_patch,
    log_patch_info,
    setup_matplotlib_backend,
)

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    @app.command()
    def amplification(
        path: Annotated[
            Path,
            typer.Argument(..., help="已處理的 .h5 檔案路徑或資料夾路徑", exists=True),
        ],
        distance: Annotated[
            float,
            typer.Option("--distance", "-d", help="震央距離 (km)"),
        ],
        origin_time: Annotated[
            str,
            typer.Option("--origin-time", "-o", help="發震時刻 (ISO 格式, e.g. 2023-02-06T01:17:35)"),
        ],
        ref_channels: Annotated[
            int,
            typer.Option("--ref-channels", help="基準 channel 數（井底最深 N 個）"),
        ] = 10,
        vmin: Annotated[
            float,
            typer.Option("--vmin", help="最慢群速度 (km/s)"),
        ] = 2.0,
        vmax: Annotated[
            float,
            typer.Option("--vmax", help="最快群速度 (km/s)"),
        ] = 4.0,
        merge: Annotated[
            bool,
            typer.Option("--merge", "-m", help="啟用批次合併模式（合併多個 .h5 為一個 Patch）"),
        ] = False,
        pattern: Annotated[
            str,
            typer.Option("--pattern", "-p", help="批次合併的 glob pattern"),
        ] = "*.h5",
        sort_by: Annotated[
            str,
            typer.Option("--sort-by", help="合併排序方式: chunk_index, timestamp"),
        ] = "chunk_index",
        save: Annotated[
            Optional[Path],
            typer.Option("--save", "-s", help="輸出圖片/CSV 目錄"),
        ] = None,
        event_label: Annotated[
            Optional[str],
            typer.Option("--event-label", "-l", help="事件標籤（用於圖例）"),
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
            typer.Option("--csv", help="同時輸出各 channel 放大倍率至 CSV"),
        ] = False,
        ref_distance_range: Annotated[
            Optional[Tuple[float, float]],
            typer.Option(
                "--ref-distance-range",
                help="基準距離範圍 (m)，水平光纖用。例: --ref-distance-range 500 600",
            ),
        ] = None,
        skip_channels: Annotated[
            int,
            typer.Option("--skip-channels", help="跳過前 N 個 channel（井口附近易受雜訊干擾）"),
        ] = 0,
        no_display: Annotated[
            bool,
            typer.Option("--no-display", help="存檔模式下不彈出視窗"),
        ] = False,
    ):
        """對已前處理的 .h5 檔案進行遠震地層放大效應分析。

        根據震央距離與表面波群速度計算時間窗，擷取雷利波列，
        以最深 N 個 channel 的中位數振幅作為基準，
        計算各 channel 的放大倍率並繪圖。

        使用範例：
        \b
            # 單一檔案分析
            das-pipeline amplification data/processed/event1.h5 \\
                --distance 3000 --origin-time "2023-02-06T01:17:35" \\
                --save results/

            # 多檔案合併後分析
            das-pipeline amplification data/processed/ \\
                --distance 3000 --origin-time "2023-02-06T01:17:35" \\
                --merge --pattern "*.h5" --save results/

            # 水平光纖：以距離 500~600 米段落為基準
            das-pipeline amplification data/processed/event1.h5 \\
                --distance 3000 --origin-time "2023-02-06T01:17:35" \\
                --ref-distance-range 500 600 --save results/
        """
        import matplotlib
        setup_matplotlib_backend(no_display)

        from das_pipeline.config import TeleseismicConfig
        from das_pipeline.teleseismic import compute_amplification, plot_amplification

        # --- 收集檔案 ---
        file_paths = collect_h5_files(path, pattern)

        # --- 載入 Patch ---
        patch = load_patch(file_paths, merge=merge, sort_by=sort_by)
        log_patch_info(patch)

        # --- 建立 Config ---
        config = TeleseismicConfig(
            event_distance_km=distance,
            event_origin_time=origin_time,
            reference_channels=ref_channels,
            reference_distance_range=ref_distance_range,
            velocity_min=vmin,
            velocity_max=vmax,
            skip_channels=skip_channels,
        )

        ref_info = (
            f"ref_distance_range=[{ref_distance_range[0]}, {ref_distance_range[1]}] m"
            if ref_distance_range else f"ref_channels={ref_channels}"
        )
        typer.echo(
            f"遠震分析設定: D={distance} km, origin={origin_time}, "
            f"v=[{vmin}, {vmax}] km/s, {ref_info}"
            f"{', skip_channels=' + str(skip_channels) if skip_channels else ''}"
        )

        # --- 計算放大倍率 ---
        result = compute_amplification(patch, config)
        if result is None:
            typer.echo("❌ 時間窗與檔案時間範圍無交集，無法分析")
            raise typer.Exit(1)

        typer.echo(
            f"✅ 放大倍率範圍: [{np.min(result['amplification']):.3f}, "
            f"{np.max(result['amplification']):.3f}], "
            f"中位數: {np.median(result['amplification']):.3f}"
        )

        # --- 輸出 CSV ---
        if csv and save is not None:
            save_dir = Path(save)
            save_dir.mkdir(parents=True, exist_ok=True)
            csv_path = save_dir / "teleseismic_amplification.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv_module.writer(f)
                writer.writerow(["distance_m", "amplification", "reference_amplitude"])
                for dist, amp in zip(result["distances"], result["amplification"]):
                    writer.writerow([f"{dist:.2f}", f"{amp:.6f}", f"{result['reference_amplitude']:.6e}"])
            typer.echo(f"✅ CSV 已儲存: {csv_path}")

        # --- 繪圖 ---
        labels = [event_label] if event_label else None
        plot_amplification(
            [result],
            save_dir=Path(save) if save else None,
            labels=labels,
            title=title,
            dpi=dpi,
            show=not no_display,
        )

        if save:
            typer.echo(f"✅ 圖表已儲存至: {save}/teleseismic_amplification.png")