# src/das_pipeline/cli/commands/check.py
"""檢視原始資料的時間距離分布與缺失狀況。

掃描目錄下的 .hdf5 檔案，繪製時間（X 軸）× 距離（Y 軸）的覆蓋網格圖，
以顏色標記每個資料點的狀態：有資料 / NaN / 未覆蓋 / 重疊。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import dascore as dc
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing_extensions import Annotated

from das_pipeline.cli.helpers import collect_h5_files, setup_matplotlib_backend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 狀態常數
# ---------------------------------------------------------------------------
STATE_UNCOVERED = 0   # 未覆蓋
STATE_DATA = 1        # 有資料
STATE_NAN = 2         # NaN（理論上應有但無）
STATE_OVERLAP = 3     # 重疊（多個檔案覆蓋同一點）

# 顏色對應：未覆蓋=灰, 有資料=綠, NaN=紅, 重疊=橘
STATE_COLORS = ["#D3D3D3", "#4CAF50", "#F44336", "#FF9800"]



# ---------------------------------------------------------------------------
# 資料讀取與網格建構
# ---------------------------------------------------------------------------


def _load_patch_meta(file_path: Path) -> dict:
    """載入一個 .h5 的 patch 並回傳 metadata 與 data。

    回傳 dict 包含:
        path: 檔案路徑
        time_vals: np.ndarray (datetime64[ns])
        dist_vals: np.ndarray (float)
        data: np.ndarray，shape = (n_dist, n_time)
    """
    spool = dc.spool(str(file_path))
    patch = spool[0]
    dims = patch.dims

    time_vals = np.asarray(patch.get_coord("time"))
    dist_vals = np.asarray(patch.get_coord("distance"))
    data = np.asarray(patch.data)

    # 確保 data shape 為 (n_distance, n_time)
    if dims[0] == "time":
        data = data.T  # (n_time, n_dist) -> (n_dist, n_time)

    return {
        "path": file_path,
        "time_vals": time_vals.astype("datetime64[ns]"),
        "dist_vals": dist_vals,
        "data": data,
    }


def _auto_time_bin(metas: list[dict]) -> float:
    """根據所有檔案的取樣率自動決定合理的 time_bin（秒）。目標 ~1000 個 bin。"""
    t_min = min(m["time_vals"].min() for m in metas)
    t_max = max(m["time_vals"].max() for m in metas)
    total_sec = (t_max - t_min).astype("int64") / 1e9

    if total_sec <= 0:
        return 1.0

    target_bins = 1000
    auto_bin = total_sec / target_bins

    # 不小於資料的取樣間隔
    dt_vals = np.diff(metas[0]["time_vals"].astype("int64"))
    min_dt = dt_vals.min() / 1e9 if len(dt_vals) > 0 else 0.01
    auto_bin = max(auto_bin, min_dt)

    if auto_bin < 0.01:
        auto_bin = round(auto_bin, 4)
    elif auto_bin < 1:
        auto_bin = round(auto_bin, 2)
    else:
        auto_bin = round(auto_bin)

    return max(auto_bin, 0.001)


def _build_global_axes(
    metas: list[dict], time_bin_sec: float
) -> tuple[np.ndarray, np.ndarray]:
    """建立全域時間軸 (bin edges) 與距離軸。"""
    all_dist = np.concatenate([m["dist_vals"] for m in metas])
    dist_axis = np.unique(all_dist)
    dist_axis.sort()

    t_min = min(m["time_vals"].min() for m in metas)
    t_max = max(m["time_vals"].max() for m in metas)
    total_span_ns = int((t_max - t_min).astype("int64"))

    if total_span_ns <= 0:
        raise ValueError("時間範圍為空，無法建立網格")

    bin_ns = int(time_bin_sec * 1e9)
    if bin_ns <= 0:
        bin_ns = 1
    n_bins = max(1, total_span_ns // bin_ns + 1)

    MAX_BINS = 5000
    if n_bins > MAX_BINS:
        bin_ns = total_span_ns // MAX_BINS + 1
        n_bins = MAX_BINS
        logger.warning(f"時間 bin 數過多，限制為 {MAX_BINS}（bin={bin_ns/1e9:.3f}s）")

    time_edges = np.arange(
        t_min.astype("datetime64[ns]"),
        t_max.astype("datetime64[ns]") + np.timedelta64(bin_ns, "ns"),
        np.timedelta64(bin_ns, "ns"),
        dtype="datetime64[ns]",
    )
    if time_edges[-1] < t_max:
        time_edges = np.append(time_edges, t_max)

    return time_edges, dist_axis



def _compute_status_grid(
    metas: list[dict],
    time_edges: np.ndarray,
    dist_axis: np.ndarray,
) -> np.ndarray:
    """計算每個 (distance, time_bin) 格點的狀態。

    Overlap 判斷透過檔案層級 metadata 兩兩比對時間/距離範圍，
    避免因網格太粗導致誤判。

    Returns
    -------
    status : np.ndarray, shape = (n_dist, n_time_bins), dtype=np.int8
    """
    n_dist = len(dist_axis)
    n_time_bins = len(time_edges) - 1

    status = np.full((n_dist, n_time_bins), STATE_UNCOVERED, dtype=np.int8)

    dist_to_idx = {float(d): i for i, d in enumerate(dist_axis)}
    time_edges_int = time_edges.astype("int64")
    total_files = len(metas)

    # ---- Pass 1: 逐檔標記 Data / NaN ----
    for fi, meta in enumerate(metas):
        logger.info(f"處理 [{fi+1}/{total_files}]: {meta['path'].name}")
        data = meta["data"]
        t_vals = meta["time_vals"]
        d_vals = meta["dist_vals"]

        d_indices = np.array([dist_to_idx[float(d)] for d in d_vals], dtype=np.intp)

        t_int = t_vals.astype("int64")
        t_bins = np.digitize(t_int, time_edges_int) - 1
        valid_t = (t_bins >= 0) & (t_bins < n_time_bins)
        t_bins = t_bins[valid_t]

        valid_t_data = data[:, valid_t]
        valid_is_nan = np.isnan(valid_t_data)

        for i_local, d_global in enumerate(d_indices):
            row_data = valid_t_data[i_local]
            row_nan = valid_is_nan[i_local]

            bin_counts = np.bincount(t_bins, minlength=n_time_bins)
            nan_counts = np.bincount(
                t_bins, weights=row_nan.astype(np.float64), minlength=n_time_bins
            )

            covered_bins = np.where(bin_counts > 0)[0]

            for tb in covered_bins:
                is_nan = nan_counts[tb] > 0
                if is_nan:
                    if status[d_global, tb] == STATE_UNCOVERED:
                        status[d_global, tb] = STATE_NAN
                else:
                    if status[d_global, tb] in (STATE_UNCOVERED, STATE_NAN):
                        status[d_global, tb] = STATE_DATA

    # ---- Pass 2: 檔案層級兩兩比對 overlap ----
    if total_files > 1:
        # 預先取出每個檔案的 min/max（metadata 層級）
        file_ranges = []
        for meta in metas:
            t_min = meta["time_vals"].min()
            t_max = meta["time_vals"].max()
            d_min = meta["dist_vals"].min()
            d_max = meta["dist_vals"].max()
            file_ranges.append((t_min, t_max, d_min, d_max))

        for i in range(total_files):
            t_min_i, t_max_i, d_min_i, d_max_i = file_ranges[i]
            t_min_i_ns = t_min_i.astype("int64")
            t_max_i_ns = t_max_i.astype("int64")
            for j in range(i + 1, total_files):
                t_min_j, t_max_j, d_min_j, d_max_j = file_ranges[j]
                t_min_j_ns = t_min_j.astype("int64")
                t_max_j_ns = t_max_j.astype("int64")

                # 計算時間與距離的實際重疊區間
                t_ov_start = max(t_min_i_ns, t_min_j_ns)
                t_ov_end = min(t_max_i_ns, t_max_j_ns)
                d_ov_start = max(d_min_i, d_min_j)
                d_ov_end = min(d_max_i, d_max_j)

                if t_ov_start >= t_ov_end or d_ov_start >= d_ov_end:
                    continue  # 無重疊

                # 對應到網格範圍
                tb_start = max(0, np.digitize(t_ov_start, time_edges_int) - 1)
                tb_end = min(n_time_bins - 1,
                             np.digitize(t_ov_end, time_edges_int) - 1)
                if tb_start > tb_end:
                    continue

                d_start = max(0, int(np.searchsorted(dist_axis, d_ov_start,
                                                     side="left")))
                d_end = min(n_dist - 1,
                            int(np.searchsorted(dist_axis, d_ov_end,
                                                side="right")) - 1)
                if d_start > d_end:
                    continue

                # 僅標記有資料的 cell 為 OVERLAP
                region = status[d_start:d_end + 1, tb_start:tb_end + 1]
                region[region != STATE_UNCOVERED] = STATE_OVERLAP

    return status



# ---------------------------------------------------------------------------
# 繪圖
# ---------------------------------------------------------------------------


def _plot_status_grid(
    status: np.ndarray,
    time_edges: np.ndarray,
    dist_axis: np.ndarray,
    file_count: int,
    time_bin_sec: float,
    title: Optional[str],
    save: Optional[Path],
    fmt: str,
    dpi: int,
    no_display: bool,
) -> None:
    """繪製狀態網格圖。"""
    from datetime import datetime, timezone
    from matplotlib.patches import Patch

    n_dist = len(dist_axis)

    cmap = ListedColormap(STATE_COLORS)
    norm = BoundaryNorm(
        [STATE_UNCOVERED, STATE_DATA, STATE_NAN, STATE_OVERLAP, STATE_OVERLAP + 1],
        ncolors=4,
    )

    n_time = status.shape[1]
    fig_w = min(max(n_time / 80, 10), 24)
    fig_h = min(max(n_dist / 40, 4), 16)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    t_edges_num = time_edges.astype("int64") / 1e9
    ax.pcolormesh(
        t_edges_num,
        np.arange(n_dist + 1),
        status,
        cmap=cmap,
        norm=norm,
        rasterized=True,
        shading="flat",
    )

    t_ref = time_edges[0].astype("int64") / 1e9
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: datetime.fromtimestamp(x, tz=timezone.utc).strftime(
                "%H:%M:%S"
            )
        )
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Distance (m)")

    if n_dist <= 50:
        y_step = max(1, n_dist // 20)
        y_ticks = np.arange(0, n_dist, y_step)
        ax.set_yticks(y_ticks + 0.5)
        ax.set_yticklabels([f"{dist_axis[i]:.0f}" for i in y_ticks])
    else:
        y_step = max(1, n_dist // 10)
        y_ticks = np.arange(0, n_dist, y_step)
        ax.set_yticks(y_ticks + 0.5)
        ax.set_yticklabels([f"{dist_axis[i]:.0f}" for i in y_ticks])
    ax.set_ylim(n_dist, 0)

    legend_patches = [
        Patch(color=STATE_COLORS[STATE_DATA], label="Data"),
        Patch(color=STATE_COLORS[STATE_NAN], label="NaN"),
        Patch(color=STATE_COLORS[STATE_UNCOVERED], label="Uncovered"),
        Patch(color=STATE_COLORS[STATE_OVERLAP], label="Overlap"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    if title:
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(
            f"Data Coverage ({file_count} files, time bin={time_bin_sec:.3f}s)",
            fontsize=11,
        )

    total_cells = status.size
    n_data = int(np.sum(status == STATE_DATA))
    n_nan = int(np.sum(status == STATE_NAN))
    n_uncov = int(np.sum(status == STATE_UNCOVERED))
    n_overlap = int(np.sum(status == STATE_OVERLAP))

    info_text = (
        f"Data: {n_data} ({100*n_data/total_cells:.1f}%) | "
        f"NaN: {n_nan} ({100*n_nan/total_cells:.1f}%) | "
        f"Overlap: {n_overlap} ({100*n_overlap/total_cells:.1f}%) | "
        f"Uncovered: {n_uncov} ({100*n_uncov/total_cells:.1f}%)"
    )
    ax.text(
        0.5, -0.12, info_text,
        transform=ax.transAxes, ha="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()

    if save:
        save_dir = Path(save)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"coverage.{fmt}"
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        typer.echo(f"✅ 已儲存: {save_path}")
        if no_display:
            plt.close(fig)
    if not save or not no_display:
        plt.show()



# ---------------------------------------------------------------------------
# CLI 註冊
# ---------------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    @app.command()
    def check(
        directory: Annotated[
            Path,
            typer.Argument(
                ..., help="包含多個 .hdf5/.h5 的目錄路徑", exists=True
            ),
        ],
        pattern: Annotated[
            str,
            typer.Option("--pattern", "-p", help="glob pattern 篩選檔案"),
        ] = "*.h5",
        time_bin: Annotated[
            str,
            typer.Option(
                "--time-bin",
                "-t",
                help="時間 bin 大小（秒），設為 'auto' 自動計算",
            ),
        ] = "auto",
        title: Annotated[
            Optional[str],
            typer.Option("--title", help="圖表自訂標題"),
        ] = None,
        save: Annotated[
            Optional[Path],
            typer.Option(
                "--save", "-s", help="存檔目錄路徑，不指定則互動式顯示"
            ),
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
        """檢視原始資料的時間距離分布與缺失狀況。

        \b
        掃描目錄下的 .hdf5/.h5 檔案，繪製時間 × 距離的覆蓋網格圖。
        每個格點以顏色標記：有資料(綠) / NaN(紅) / 未覆蓋(灰) / 重疊(橘)。

        \b
        使用範例:
            das-pipeline check data/raw/
            das-pipeline check data/raw/ --time-bin 10
            das-pipeline check data/raw/ --save outputs/ --dpi 200
        """
        setup_matplotlib_backend(no_display)

        file_paths = collect_h5_files(directory, pattern)
        if not file_paths:
            typer.echo(f"❌ 在 {directory} 找不到符合 {pattern} 的檔案")
            raise typer.Exit(1)

        typer.echo(f"找到 {len(file_paths)} 個檔案，讀取中...")

        metas = []
        for fp in file_paths:
            try:
                meta = _load_patch_meta(fp)
                metas.append(meta)
                logger.info(
                    f"  {fp.name}: time={len(meta['time_vals'])}, "
                    f"dist={len(meta['dist_vals'])}, "
                    f"shape={meta['data'].shape}"
                )
            except Exception as e:
                logger.warning(f"無法讀取 {fp.name}: {e}")
                typer.echo(f"⚠️  跳過 {fp.name}: {e}")

        if not metas:
            typer.echo("❌ 沒有可讀取的檔案")
            raise typer.Exit(1)

        if time_bin.lower() == "auto":
            time_bin_sec = _auto_time_bin(metas)
            typer.echo(f"自動 time bin: {time_bin_sec:.3f}s")
        else:
            try:
                time_bin_sec = float(time_bin)
            except ValueError:
                typer.echo(
                    f"❌ --time-bin 必須是數字或 'auto'，收到: {time_bin}"
                )
                raise typer.Exit(1)

        time_edges, dist_axis = _build_global_axes(metas, time_bin_sec)
        typer.echo(
            f"網格: {len(dist_axis)} 距離 × {len(time_edges)-1} 時間 bin "
            f"(bin={time_bin_sec:.3f}s)"
        )

        typer.echo("計算狀態網格中...")
        status = _compute_status_grid(metas, time_edges, dist_axis)
        typer.echo("狀態網格計算完成")

        typer.echo("繪圖中...")
        _plot_status_grid(
            status=status,
            time_edges=time_edges,
            dist_axis=dist_axis,
            file_count=len(metas),
            time_bin_sec=time_bin_sec,
            title=title,
            save=save,
            fmt=format,
            dpi=dpi,
            no_display=no_display,
        )

