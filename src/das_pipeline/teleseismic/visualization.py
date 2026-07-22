# src/das_pipeline/teleseismic/visualization.py

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_amplification(
    results: list[dict],
    save_dir: Optional[Path] = None,
    labels: Optional[list[str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 150,
    show: bool = True,
) -> Optional[Path]:
    """繪製遠震地層放大倍率 vs. 實際距離 圖。

    橫軸為正規化振幅（放大倍率），縱軸為距離（米）。
    若結果有 "distances" 欄位則使用距離，否則退回使用 channel index。
    支援多事件疊圖，不同事件用不同顏色區分。

    Parameters
    ----------
    results : list[dict]
        compute_amplification() 回傳的結果字典列表，每個元素包含
        "channel_indices", "distances", "amplification", "event_distance_km" 等。
    save_dir : Path or None
        若指定，將圖片存到該目錄。
    labels : list[str] or None
        每個事件的圖例標籤（例如事件名稱或編號）。
    title : str or None
        圖表標題。
    figsize : tuple[float, float]
        圖表尺寸 (width, height) inches。
    dpi : int
        圖片解析度。
    show : bool
        是否顯示圖片（若 save_dir 有指定且 show=False 則不顯示）。

    Returns
    -------
    Path or None
        若 save_dir 有指定，回傳存檔路徑；否則回傳 None。
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_events = len(results)

    if labels is None:
        labels = [f"Event {i+1}" for i in range(n_events)]

    if n_events == 0:
        logger.warning("無任何結果可繪製")
        plt.close(fig)
        return None

    for i, result in enumerate(results):
        if result is None:
            logger.warning("第 %d 個結果為 None，跳過", i + 1)
            continue

        amp = result["amplification"]
        distances = result["distances"]
        dist_km = result.get("event_distance_km", "?")

        label = f"{labels[i]} (D={dist_km} km)"
        ax.plot(amp, distances, label=label, linewidth=1.5)
        # 在每個事件曲線上標記起點（最深處 = 最大距離）
        ax.scatter(amp[0], distances[0], s=15, zorder=3)

    # 基準線
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
               label="Baseline (amp=1.0)")

    ax.set_xlabel("Normalized Amplitude")
    ax.set_ylabel("Distance (m)")
    ax.set_title(title or "Teleseismic Amplification")

    # 縱軸反轉（讓井口/小距離在上方，井底/大距離在下方）
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    saved_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "teleseismic_amplification.png"
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        logger.info("圖表已儲存: %s", save_path)
        saved_path = save_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved_path
