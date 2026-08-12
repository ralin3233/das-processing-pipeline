"""IO 模組：MiniSEED 讀取、Spool 管理、座標對齊與 Patch 儲存。

提供統一的資料讀寫介面：
- ``miniseed_loader``: 讀取原始 MiniSEED 檔案
- ``spool_loader``: 管理 DASCore Spool（含分段迭代）
- ``coord_utils``: 座標對齊與單位轉換
- ``patch_writer``: 輸出處理後的 Patch
"""

from das_pipeline.io.coord_utils import align as coord_align
from das_pipeline.io.patch_writer import save as patch_save
from das_pipeline.io.spool_loader import get_spool, iter_chunks