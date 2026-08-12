"""前處理模組：提供完整的 DAS 資料前處理管線。

將個別前處理步驟（選取、去趨勢、taper、濾波、降採樣）
組合成統一管線 ``run_preprocessing``。
"""

from das_pipeline.preprocessing.pipeline import run_preprocessing

__all__ = ["run_preprocessing"]