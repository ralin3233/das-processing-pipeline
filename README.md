# DAS Pipeline

以 [DASCore](https://github.com/DASDAE/dascore) 為核心的 Distributed Acoustic Sensing（DAS，分散式光纖聲學感測）資料處理工具。它可將 MiniSEED 或既有 DASCore 相容資料切分為時間 chunk、套用前處理，並輸出 DASDAE HDF5（`.h5`）；也提供 Waterfall、F-K 與 Spectrogram 繪圖指令。

## 功能

- MiniSEED 讀取，以及由 DASCore 載入目錄中的相容格式（例如 HDF5）
- 依時間分段處理；相鄰 chunk 可保留 overlap，降低濾波邊界效應
- 前處理：時間／距離選取、去趨勢、帶通濾波、降採樣
- 輸出 DASDAE HDF5，並在 chunk 屬性中保存核心時間範圍，供後續合併
- CLI 視覺化：Waterfall、F-K spectrum、Spectrogram，以及多個 chunk 的合併繪圖
- YAML 設定檔與 Pydantic 驗證

> 座標幾何對齊尚未實作。目前 `coordinate` 區塊仍為必要設定，但轉檔時會保留輸入 Patch 原有的 distance 座標。

## 安裝

需求：Python 3.10 以上。

```bash
git clone <repo-url>
cd das-processing-pipeline

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .

# 開發／測試環境
pip install -e ".[test]"
```

## 快速開始

### 1. 建立設定檔

以範例檔為起點，並依自己的資料路徑與格式修改：

```bash
cp configs/config.yaml.example configs/config.yaml
```

最少需要確認下列欄位：

```yaml
project_name: "my_das_run"

data:
  input_dir: "/path/to/input"
  format: "hdf5"             # "miniseed" 或 DASCore 可讀的格式
  file_pattern: "*.hdf5"     # MiniSEED 時例如 "*.mseed"
  chunk_duration: "10min"

coordinate:
  fiber_geometry_file: "data/raw/geometry.csv" # 目前僅為必要欄位，尚未套用

output:
  save_dir: "data/processed"
```

完整選項請參考 [configs/config.yaml.example](configs/config.yaml.example)。請勿直接使用範例中的絕對路徑。

### 2. 轉檔

```bash
das-pipeline convert --config configs/config.yaml
```

每個輸出檔預設依 `output.filename_pattern` 命名，例如：

```text
data/processed/my_das_run_20260717T143000_chunk0000.h5
```

若同名檔案已存在且 `output.overwrite: false`，流程會停止並回報錯誤，避免意外覆寫。

### 3. 繪圖

```bash
# 繪製單一檔案的 Waterfall
das-pipeline plot data/processed/my_das_run_20260717T143000_chunk0000.h5

# 同時產生三種圖並存檔，不開啟視窗
das-pipeline plot data/processed/my_das_run_20260717T143000_chunk0000.h5 \
  --type waterfall --type fk --type spectrogram \
  --save figures --no-display

# 合併資料夾中的 chunk 後繪圖
das-pipeline plot data/processed --merge --pattern "*.h5" \
  --type waterfall --save figures --no-display
```

合併模式會使用轉檔時寫入的 `core_time_start` 與 `core_time_end` 裁掉 overlap 區域，再沿時間軸串接。

## CLI 參考

```bash
das-pipeline --help
das-pipeline convert --help
das-pipeline plot --help
```

### `convert`

| 參數 | 說明 |
| --- | --- |
| `--config`, `-c` | YAML 設定檔路徑（必要） |

### `plot`

`path` 可為單一 `.h5` 檔或目錄。目錄模式會依 `--pattern` 收集檔案；只有加上 `--merge` 才會將多個檔案合併。

| 參數 | 說明 |
| --- | --- |
| `--type`, `-t` | `waterfall`、`fk`、`spectrogram`；可重複指定，預設 `waterfall` |
| `--merge`, `-m` | 合併多個 chunk 後繪圖 |
| `--pattern`, `-p` | 目錄模式的 glob，預設 `*.h5` |
| `--sort-by` | 合併排序：`chunk_index`（預設）或 `timestamp` |
| `--channel` | Spectrogram 的通道索引；未指定時使用中間通道 |
| `--time-range` | Waterfall 時間範圍：起訖 ISO 時間 |
| `--distance-range`, `--dist-range` | Waterfall 距離／通道範圍 |
| `--freq-range` | F-K 或 Spectrogram 的頻率範圍（Hz） |
| `--channel-spacing` | F-K 的通道間距（m） |
| `--save`, `-s` | 圖檔輸出目錄；未指定時顯示互動式視窗 |
| `--format` | 圖檔格式：`png`（預設）、`pdf`、`svg` |
| `--dpi` | 圖片解析度，預設 150 |
| `--colormap` | Matplotlib colormap；預設 `seismic` |
| `--title` | 自訂圖表標題 |
| `--no-display` | 存檔後不顯示視窗 |

## 設定檔參考

設定檔包含六個區塊：

| 區塊 | 主要用途 |
| --- | --- |
| `project_name` | 輸出檔名使用的專案名稱 |
| `data` | 輸入資料、格式、時間篩選與 chunk 設定 |
| `coordinate` | 未來的幾何對齊設定；目前不會改變資料座標 |
| `preprocessing` | 範圍裁切、detrend、bandpass、decimate |
| `output` | 儲存路徑與輸出檔名 |
| `runtime` | 日誌層級與 manifest 預留設定 |

### `data`

| 欄位 | 說明 |
| --- | --- |
| `input_dir` | 輸入目錄（必要） |
| `format` | `miniseed` 時使用 ObsPy 讀取；其他值交由 DASCore 建立 spool |
| `file_pattern` | MiniSEED 的檔案 glob；非 MiniSEED 目錄由 DASCore 掃描 |
| `sampling_rate` | Hz。若提供，也會用於將濾波安全邊際換算為秒 |
| `time_range` | 輸入資料的 ISO 起訖時間篩選 |
| `chunk_duration` | 每個 chunk 的時間，例如 `"10min"`、`"1h"` |
| `taper_ratio` | 每個 chunk 用於 overlap 與濾波 taper 的比例，預設 `0.05` |
| `filter_safety_samples` | 額外 overlap 樣本數；需同時設定 `sampling_rate` 才會生效 |

### `preprocessing`

處理順序固定為：`select → detrend → bandpass → decimate`。

| 欄位 | 說明 |
| --- | --- |
| `time_range` | 時間範圍；目前對應資料的時間座標篩選 |
| `distance_range` | 距離／通道範圍 |
| `detrend` | `linear`、`constant` 或 `null` |
| `bandpass` | `[low_cutoff, high_cutoff]` Hz；設為 `null` 跳過 |
| `decimate_factor` | 整數且至少為 2；設為 `null` 跳過 |

### `output`

| 欄位 | 說明 |
| --- | --- |
| `save_dir` | 輸出目錄（必要） |
| `filename_pattern` | 可使用 `{project_name}`、`{timestamp}`、`{chunk_index:04d}` |
| `format` | 設定模型保留的欄位；目前寫出格式固定為 DASDAE HDF5 |
| `overwrite` | 是否允許覆寫已存在檔案 |
| `compression` | 設定模型保留的欄位；目前寫出器尚未將它傳給 DASCore |

## 專案結構

```text
├── configs/config.yaml.example  # 設定檔範例
├── src/das_pipeline/
│   ├── cli.py                   # convert / plot 指令
│   ├── config.py                # Pydantic 設定模型
│   ├── pipeline.py              # 轉檔流程
│   ├── io/                      # 載入、分段、輸出與座標處理
│   ├── preprocessing/           # 前處理步驟
│   └── visualization/           # 圖表與 chunk 合併
├── tests/
└── pyproject.toml
```

## 測試

```bash
pytest tests/ -v
```

## 已知限制

- `coordinate` 的幾何檔案、插值與 shape 檢查尚未套用。
- `output.format` 與 `output.compression` 已可在 YAML 驗證，但尚未影響目前固定的 DASDAE 輸出寫入方式。
- 目前未提供平行處理。

## License

請參閱 [LICENSE](LICENSE)。
