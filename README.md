# DAS Pipeline

DAS（Distributed Acoustic Sensing，分佈式光纖聲學感測）資料處理管線，提供從 MiniSEED 原始資料讀取、座標對齊、前處理、分段切割到輸出 DASDAE HDF5 格式的完整流程，並支援多種視覺化分析工具。

## 功能簡介

- **多格式讀取**：支援 MiniSEED 及其他 DASCore 相容格式（HDF5、SEGY 等）
- **自動分段**：依照指定的時間長度（chunk duration）將長時間連續資料自動切割
- **前處理管線**：時間/距離範圍選取、去趨勢（detrend）、帶通濾波（bandpass）、降採樣（decimate）
- **座標對齊**：支援外部光纖幾何座標檔，進行距離軸對齊與插值
- **標準化輸出**：以 DASDAE HDF5（`.h5`）格式儲存，保留完整座標與屬性資訊
- **YAML 設定驅動**：所有參數集中於 YAML 設定檔，易於版本控管與重現
- **CLI 介面**：透過 `das-pipeline convert` 執行轉檔流程，`das-pipeline plot` 進行視覺化分析
- **視覺化分析**：支援 Waterfall、F-K 頻譜圖、Spectrogram 時頻圖，以及多檔案批次合併繪圖

## 專案結構

```
das-pipeline/
├── configs/                        # YAML 設定檔
│   ├── convert_default.yaml         # 轉檔流程的預設設定
├── scripts/                         # 手動測試與輔助腳本
│   └── manual_test_convert.py
├── src/das_pipeline/                # 主要套件原始碼
│   ├── cli.py                       # Typer CLI 入口點（convert / plot）
│   ├── config.py                    # Pydantic 設定模型
│   ├── pipeline.py                  # 核心管線邏輯
│   ├── io/
│   │   ├── miniseed_loader.py       # MiniSEED 讀取器
│   │   ├── spool_loader.py          # 通用 Spool 載入與分段
│   │   ├── coord_utils.py           # 座標對齊工具
│   │   └── patch_writer.py          # Patch 輸出寫入
│   ├── preprocessing/               # 前處理模組
│   │   ├── pipeline.py              # 前處理管線編排
│   │   ├── select.py                # 時間/距離範圍選取
│   │   ├── detrend.py               # 去趨勢
│   │   ├── bandpass.py              # 帶通濾波
│   │   └── decimate.py              # 降採樣
│   ├── visualization/               # 視覺化模組
│   │   ├── waterfall.py             # Waterfall 時空圖
│   │   ├── fk.py                    # F-K 頻譜圖
│   │   ├── spectrogram.py           # 時頻圖（Spectrogram）
│   │   └── merge.py                 # 多檔案批次合併
│   └── utils/
│       └── logging_config.py        # 日誌設定
├── tests/                           # 單元測試
│   └── test_miniseed_loader.py
│   └── test_preprocessing.py
│   └── test_visualization.py
├── logs/                            # 日誌輸出目錄
├── pyproject.toml                   # 專案設定與相依套件
└── README.md
```

## 環境需求

- Python ≥ 3.10
- 相依套件（詳見 `pyproject.toml`）：
  - [DASCore](https://github.com/DASDAE/dascore) — DAS 資料核心處理
  - [ObsPy](https://github.com/obspy/obspy) — MiniSEED 讀取
  - [Typer](https://typer.tiangolo.com/) — CLI 框架
  - [Pydantic](https://docs.pydantic.dev/) — 設定驗證
  - [PyYAML](https://pyyaml.org/) — YAML 解析
  - [Matplotlib](https://matplotlib.org/) — 圖表繪製
  - [SciPy](https://scipy.org/) — 訊號處理（濾波、降採樣）

## 安裝

```bash
# 複製專案
git clone <repo-url>
cd das-pipeline

# 建立虛擬環境（建議）
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 安裝套件（開發模式）
pip install -e .

# 若需執行測試
pip install -e ".[test]"
```

## 快速開始

### 1. 準備設定檔

編輯 `configs/convert_default.yaml`，將 `data.input_dir` 指向你的 MiniSEED 檔案所在目錄：

```yaml
data:
  input_dir: "/path/to/your/miniseed/files/"
  format: "miniseed"
  chunk_duration: "10min"
  # ...
```

### 2. 執行轉檔

```bash
das-pipeline convert --config configs/convert_default.yaml
```

執行成功後，處理完的 `.h5` 檔案會出現在 `output.save_dir` 所指定的目錄中（預設為 `data/processed/`）。

### 3. 查看輸出

```bash
ls data/processed/
# 輸出範例: das_convert_default_20250218T143000_chunk0000.h5
```

## CLI 指令

```bash
# 查看所有可用指令
das-pipeline --help

# 執行轉檔流程
das-pipeline convert --config <path-to-yaml>

# 執行視覺化繪圖
das-pipeline plot <path> [options]
```

### convert — 轉檔流程

| 參數 | 說明 |
|------|------|
| `--config`, `-c` | YAML 設定檔路徑（必要） |

### plot — 視覺化繪圖

對已處理的 `.h5` 檔案進行視覺化分析，可指定檔案路徑或資料夾（搭配 glob pattern 批次載入）。

| 參數 | 類型 | 說明 |
|------|------|------|
| `path` | path | `.h5` 檔案路徑或資料夾路徑（必要） |
| `--type`, `-t` | list[str] | 圖表類型：`waterfall`, `fk`, `spectrogram`（可複選，預設 `waterfall`） |
| `--merge`, `-m` | bool | 啟用批次合併模式，將多個 chunk 合併後再繪圖 |
| `--pattern`, `-p` | str | 批次合併的 glob pattern（預設 `*.h5`） |
| `--sort-by` | str | 合併排序方式：`chunk_index`, `timestamp`（預設 `chunk_index`） |
| `--channel` | int | Spectrogram 要分析的通道索引 |
| `--time-range` | [str, str] | 時間範圍 `[start, end]`（ISO 格式，如 `2023-02-06T10:30:00`） |
| `--distance-range`, `--dist-range` | [float, float] | 距離/通道範圍 `[start, end]` |
| `--freq-range` | [float, float] | 頻率範圍 `[low, high]` Hz |
| `--channel-spacing` | float | 相鄰通道的物理距離（m），用於 FK 正確 wavenumber |
| `--save`, `-s` | path | 存檔目錄路徑，不指定則互動式顯示 |
| `--format` | str | 存檔格式：`png`, `pdf`, `svg`（預設 `png`） |
| `--dpi` | int | 圖片解析度（預設 `150`） |
| `--colormap` | str | matplotlib colormap 名稱（預設 `seismic`） |
| `--title` | str | 圖表自訂標題 |
| `--no-display` | bool | 存檔模式下不彈出視窗 |

## 設定檔說明

設定檔採用 YAML 格式，由五個主要區塊組成：

| 區塊 | 說明 |
|------|------|
| `project_name` | 專案名稱，用於輸出檔名 |
| `data` | 資料來源設定：路徑、格式、時間範圍、分段長度等 |
| `coordinate` | 座標對齊設定：幾何座標檔路徑、插值方法等 |
| `preprocessing` | 前處理設定：範圍選取、去趨勢、濾波、降採樣等 |
| `output` | 輸出設定：輸出路徑、檔名格式、壓縮方式等 |
| `runtime` | 執行期設定：日誌層級、是否儲存 manifest 等 |

### DataConfig 重點參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `input_dir` | path | MiniSEED 檔案所在目錄 |
| `format` | str | 輸入格式，目前支援 `miniseed` 及 dascore 相容格式 |
| `file_pattern` | str | 檔案 glob pattern，預設 `*.mseed` |
| `channel_range` | [int, int] | 要讀取的 channel 範圍 `[start, end]` |
| `sampling_rate` | int or null | 取樣率（Hz），MiniSEED header 未記錄時可手動指定 |
| `time_range` | [str, str] or null | 可選的時間範圍篩選，如 `["2023-02-06T10:00:00", "2023-02-06T11:00:00"]` |
| `chunk_duration` | str | 每段時間長度，如 `"10min"`、`"1h"` |
| `taper_ratio` | float | 濾波前 taper 比例，預設 `0.05`（頭尾各 5%） |
| `filter_safety_samples` | int | 濾波器額外安全邊際（樣本數），`0` 表示不增加 |

### PreprocessingConfig 重點參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `time_range` | [float, float] or null | 限定時間範圍 |
| `distance_range` | [float, float] or null | 限定距離/通道範圍 |
| `detrend` | str or null | 去趨勢方法：`linear`, `constant`, `null`（關閉） |
| `bandpass` | [float, float] or null | 帶通濾波頻率範圍 `[low_cutoff, high_cutoff]` Hz |
| `decimate_factor` | int or null | 降採樣倍數，`null` 則跳過 |

### OutputConfig 重點參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `save_dir` | path | 輸出目錄 |
| `filename_pattern` | str | 檔名格式，可用 `{project_name}`、`{timestamp}`、`{chunk_index}` |
| `format` | str | 輸出格式，預設 `"dascore_h5"`（DASDAE HDF5） |
| `overwrite` | bool | 是否覆蓋已存在的檔案 |
| `compression` | str or null | HDF5 壓縮方式，如 `"gzip"` |

## 執行測試

```bash
pytest tests/ -v
```

## 開發狀態

本專案仍在積極開發中，以下是目前的功能狀態：

- [x] MiniSEED 讀取
- [x] 時間分段（chunking）
- [x] DASDAE HDF5 輸出
- [x] YAML 設定驅動
- [x] CLI 介面
- [x] 前處理管線（選取 / 去趨勢 / 帶通濾波 / 降採樣）
- [x] 視覺化繪圖（Waterfall / F-K 頻譜 / Spectrogram / 批次合併）
- [ ] 座標對齊（目前 stub，待實作）
- [ ] SEGY 輸入支援
- [ ] 平行處理

## License

待定