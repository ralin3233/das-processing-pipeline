"""CLI 命令：STA/LTA 觸發檢測。

對已處理的 .h5 檔案逐通道計算 STA/LTA ratio，
並以空間一致性篩選真實訊號事件，輸出 JSON/CSV。
"""

import csv as csv_module
import json
import logging
from pathlib import Path
from typing import List, Optional

import dascore as dc
import numpy as np
import typer
from typing_extensions import Annotated

from das_pipeline.cli.helpers import (
    collect_h5_files,
    handle_bad_channels_for_detection,
    load_patch,
    log_patch_info,
)

logger = logging.getLogger(__name__)


def register(app: typer.Typer) -> None:
    @app.command()
    def detect(
        path: Annotated[
            Path,
            typer.Argument(..., help="已處理的 .h5 檔案路徑或資料夾路徑", exists=True),
        ],
        sta_window: Annotated[
            float,
            typer.Option("--sta-window", help="STA 短窗長度 (秒)"),
        ] = 0.5,
        lta_window: Annotated[
            float,
            typer.Option("--lta-window", help="LTA 長窗長度 (秒)"),
        ] = 10.0,
        trigger_threshold: Annotated[
            float,
            typer.Option("--trigger-threshold", help="STA/LTA ratio 觸發閾值"),
        ] = 3.0,
        detrigger_threshold: Annotated[
            float,
            typer.Option("--detrigger-threshold", help="STA/LTA ratio 解除觸發閾值"),
        ] = 1.5,
        min_channels: Annotated[
            int,
            typer.Option("--min-channels", help="最少同時觸發的通道數（空間一致性）"),
        ] = 30,
        min_duration: Annotated[
            float,
            typer.Option("--min-duration", help="最短事件持續時間 (秒)"),
        ] = 0.1,
        merge_window: Annotated[
            float,
            typer.Option("--merge-window", help="合併相鄰事件閾值 (秒)，0=不合併"),
        ] = 0.0,
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
            typer.Option("--save", "-s", help="輸出目錄路徑"),
        ] = None,
        format: Annotated[
            str,
            typer.Option("--format", "-f", help="輸出格式: csv, json (預設兩者皆輸出)"),
        ] = "all",
    ):
        """對已處理的 .h5 檔案執行 STA/LTA 觸發檢測（使用 DASCore stalta）。

        逐通道計算 STA/LTA ratio，並以空間一致性篩選真實訊號事件。

        \b
        使用範例：
            # 單一檔案檢測
            das-pipeline detect data/processed/event.h5 \\
                --sta-window 0.5 --lta-window 10.0 \\
                --trigger-threshold 3.0 --save results/

            # 多檔案合併後檢測
            das-pipeline detect data/processed/ \\
                --sta-window 0.5 --lta-window 10.0 \\
                --merge --pattern "*.h5" --save results/
        """
        from das_pipeline.config import StaLtaConfig
        from das_pipeline.detection import compute_sta_lta_patch, detect_events

        # --- 載入 Patch ---
        file_paths = collect_h5_files(path, pattern)
        patch = load_patch(file_paths, merge=merge, sort_by=sort_by)

        time_values = patch.coords.get_array("time")
        sampling_rate = 1.0 / (
            (time_values[1] - time_values[0]) / np.timedelta64(1, "s")
        )

        typer.echo(
            f"Patch 維度: {patch.shape}, "
            f"sampling_rate: {sampling_rate:.2f} Hz, "
            f"time: {time_values[0]} ~ {time_values[-1]}"
        )

        # --- Config ---
        config = StaLtaConfig(
            sta_window_s=sta_window,
            lta_window_s=lta_window,
            trigger_threshold=trigger_threshold,
            detrigger_threshold=detrigger_threshold,
            min_channels_triggered=min_channels,
            min_event_duration_s=min_duration,
            merge_window_s=merge_window,
        )

        typer.echo(
            f"STA/LTA 設定: sta={sta_window}s, lta={lta_window}s, "
            f"trigger={trigger_threshold}, detrigger={detrigger_threshold}, "
            f"min_channels={min_channels}, min_duration={min_duration}s"
        )

        # --- 排除不可用 channel ---
        patch, local_to_orig = handle_bad_channels_for_detection(patch)

        # --- STA/LTA via DASCore ---
        sta_lta_patch = compute_sta_lta_patch(patch, config)
        typer.echo(f"STA/LTA ratio shape: {sta_lta_patch.shape}")

        # --- Event detection ---
        events = detect_events(sta_lta_patch, config, sampling_rate)
        typer.echo(f"檢測到 {len(events)} 個事件")

        if len(events) == 0:
            typer.echo("⚠️  未檢測到任何觸發事件")
            return

        for i, evt in enumerate(events):
            typer.echo(
                f"  [{i}] {evt['start_time']} ~ {evt['end_time']} "
                f"({evt['duration_s']:.2f}s), "
                f"peak={evt['peak_ratio']:.2f}, "
                f"channels={evt['num_triggered_channels']}"
            )

        # --- 輸出 ---
        if save is not None:
            save_dir = Path(save)
            save_dir.mkdir(parents=True, exist_ok=True)

            export = []
            for e in events:
                local_chs = e["triggered_channels"]
                orig_chs = [local_to_orig[ch] for ch in local_chs if ch < len(local_to_orig)]
                export.append({
                    "start_time": e["start_time"],
                    "end_time": e["end_time"],
                    "peak_time": e["peak_time"],
                    "peak_ratio": e["peak_ratio"],
                    "triggered_channels": orig_chs,
                    "num_triggered_channels": len(orig_chs),
                    "duration_s": e["duration_s"],
                })

            write_json = format in ("json", "all")
            write_csv = format in ("csv", "all")

            if write_json:
                json_path = save_dir / "detections.json"
                with open(json_path, "w") as f:
                    json.dump(export, f, indent=2, ensure_ascii=False)
                typer.echo(f"✅ JSON: {json_path}")

            if write_csv:
                csv_path = save_dir / "detections.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv_module.writer(f)
                    writer.writerow([
                        "start_time", "end_time", "peak_time", "peak_ratio",
                        "num_triggered_channels", "triggered_channels", "duration_s",
                    ])
                    for e in export:
                        writer.writerow([
                            e["start_time"], e["end_time"], e["peak_time"],
                            f"{e['peak_ratio']:.6f}",
                            e["num_triggered_channels"],
                            ",".join(str(c) for c in e["triggered_channels"]),
                            f"{e['duration_s']:.6f}",
                        ])
                typer.echo(f"✅ CSV: {csv_path}")