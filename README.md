# DAS Pipeline

以 [DASCore](https://github.com/DASDAE/dascore) 為核心的 Distributed Acoustic Sensing（DAS，分散式光纖聲學感測）資料處理工具。它可將 MiniSEED 或既有 DASCore 相容資料切分為時間 chunk、套用前處理與座標對齊，並輸出 DASDAE HDF5（`.h5`）；也提供遠震地層放大效應分析、多事件疊圖，以及 Waterfall、F-K 與 Spectrogram 繪圖指令。

## 功能

- MiniSEED 讀取，以及由 DASCore 載入目錄中的相容格式（例如 HDF5）
- 依時間分段處理；相鄰 chunk 可保留 overlap，降低濾波邊界效應
- 前處理：時間／距離選取、去趨勢、帶通濾波、降採樣
- 座標對齊：讀取光纖幾何座標檔，以 Haversine + 深度差計算沿光纖的累積 3D 距離，將 channel index 映射為實際距離（米）；支援相位差 → 應變率轉換
- 輸出 DASDAE HDF5，並在 chunk 屬性中保存核心時間範圍，供後續合併
- 遠震地層放大效應分析：依震央距離與表面波群速度計算時間窗，以井底最深 N 個 channel（或水平光纖指定距離範圍）為基準，計算各 channel 的放大倍率並繪圖
- 多事件疊圖：將多個放大倍率 CSV 疊加顯示，並繪製中位數曲線
- STA/LTA 觸發檢測：逐通道計算 STA/LTA ratio，以空間一致性篩選真實地震事件，輸出 CSV/JSON
- 單一 Channel SNR 分析：計算訊號窗與雜訊窗的平均功率比，輸出 SNR（dB）
- 資料覆蓋率檢查：掃描 .h5 目錄，繪製時間 × 距離覆蓋網格圖，標記 Data/NaN/未覆蓋/重疊
- CLI 視覺化：Waterfall、F-K spectrum、Spectrogram，以及多個 chunk 的合併繪圖
- YAML 設定檔與 Pydantic 驗證

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
  fiber_geometry_file: "data/raw/geometry.csv"
  input_unit: "phase"        # "phase"（相位差）或 "strain_rate"（應變率）

output:
  save_dir: "data/processed"
```

> 若 `fiber_geometry_file` 指向的檔案不存在，會略過座標對齊並沿用原始 distance 軸（channel index）。若 `input_unit: "phase"`，會依 `phase_strain_constant` 與取樣頻率自動轉換為應變率。

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

### 3. 遠震地層放大效應分析

對已前處理的 `.h5` 檔案，根據震央距離與發震時刻計算雷利波列時間窗，分析各 channel 的放大倍率：

```bash
# 單一檔案分析（井底最深 10 個 channel 為基準）
das-pipeline amplification data/processed/my_das_run_20260717T143000_chunk0000.h5 \
  --distance 3000 --origin-time "2023-02-06T01:17:35" \
  --save results/

# 水平光纖：以距離 500~600 米段落為基準
das-pipeline amplification data/processed/my_das_run_20260717T143000_chunk0000.h5 \
  --distance 3000 --origin-time "2023-02-06T01:17:35" \
  --ref-distance-range 500 600 --save results/

# 多檔案合併後分析，並同時輸出 CSV
das-pipeline amplification data/processed/ \
  --distance 3000 --origin-time "2023-02-06T01:17:35" \
  --merge --pattern "*.h5" --csv --save results/
```

`--csv` 會額外輸出 `teleseismic_amplification.csv`（各 channel 的距離、放大倍率與基準振幅），供後續疊圖使用。

### 4. 多事件疊圖

將多個 `amplification --csv` 輸出的 CSV 疊加顯示，並以紅色虛線繪製中位數曲線：

```bash
das-pipeline overlay results/ \
  --pattern "*.csv" --labels "M6.5,M7.0,M7.2" \
  --save results/ --title "Amplification Overlay"
```

`--labels` 以逗號分隔，用於圖例；未指定時使用檔名。

### 5. 基本繪圖

```bash
# 繪製單一檔案的 Waterfall
das-pipeline plot waterfall data/processed/my_das_run_20260717T143000_chunk0000.h5

# Waterfall 配合時間與距離範圍篩選，存檔不顯示視窗
das-pipeline plot waterfall data/processed/my_das_run_20260717T143000_chunk0000.h5 \
  --time-range "2023-02-06T10:24:50" "2023-02-06T10:25:00" \
  --save figures --no-display

# 合併資料夾中的 chunk 後繪製 Waterfall
das-pipeline plot waterfall data/processed --merge --pattern "*.h5" \
  --save figures --no-display

# F-K 頻譜圖
das-pipeline plot fk data/processed/my_das_run_20260717T143000_chunk0000.h5 \
  --channel-spacing 1.0 --save figures

# Spectrogram
das-pipeline plot spectrogram data/processed/my_das_run_20260717T143000_chunk0000.h5 \
  --channel 100 --save figures
```

合併模式會使用轉檔時寫入的 `core_time_start` 與 `core_time_end` 裁掉 overlap 區域，再沿時間軸串接。

### 6. STA/LTA 觸發檢測

對已處理的 `.h5` 檔案逐通道計算 STA/LTA ratio，再透過空間一致性篩選真實地震事件：

```bash
# 單一檔案檢測
das-pipeline detect data/processed/event.h5 \
  --sta-window 0.5 --lta-window 10.0 \
  --trigger-threshold 3.0 --save results/

# 多檔案合併後檢測
das-pipeline detect data/processed/ \
  --sta-window 0.5 --lta-window 10.0 \
  --merge --pattern "*.h5" --save results/
```

依 `--format`（`csv`、`json` 或 `all`）輸出事件清單，包含起訖時間、峰值 ratio 與觸發 channel 列表。

### 7. 單一 Channel SNR 分析

根據震央距離與表面波群速度計算訊號／雜訊時間窗，輸出特定 channel 的 SNR（dB）：

```bash
# 單一檔案
das-pipeline snr data/processed/event.h5 \
  -c 100 -d 3000 -o "2023-02-06T01:17:35"

# 多檔案合併後分析，輸出 JSON
das-pipeline snr data/processed/ \
  -c 100 -d 3000 -o "2023-02-06T01:17:35" \
  --merge --save results/

# 互動式選窗：在波形圖上拖曳圈選訊號／雜訊窗後自動計算 SNR
das-pipeline snr data/processed/event.h5 \
  -c 100 -d 3000 -o "2023-02-06T01:17:35" --interactive
```

互動式選窗模式下，會以距離／速度先算出建議的訊號窗（綠色）與雜訊窗（紅色）
作為初始值，接著在波形圖上：

- 按 `s` 切換到訊號窗選取模式、按 `n` 切換到雜訊窗選取模式
- 按住滑鼠左鍵拖曳即設定目前模式的視窗
- 按 `q` 或直接關閉視窗完成選取，程式會即時算出並輸出 SNR（dB）

### 8. 資料覆蓋率檢查

掃描目錄下多個 `.h5` 檔案，繪製時間 × 距離的覆蓋網格圖，以顏色標記 Data／NaN／未覆蓋／重疊：

```bash
# 自動時間 bin
das-pipeline check data/raw/

# 指定時間 bin 並存檔
das-pipeline check data/raw/ --time-bin 10 --save outputs/ --dpi 200
```

## CLI 參考

```bash
das-pipeline --help
das-pipeline convert --help
das-pipeline amplification --help
das-pipeline detect --help
das-pipeline snr --help
das-pipeline plot --help
das-pipeline overlay --help
das-pipeline check --help
```

所有子命令均支援全域選項 `--log-level DEBUG|INFO|WARNING|ERROR`（預設 `INFO`）。

### `convert`

| 參數 | 說明 |
| --- | --- |
| `--config`, `-c` | YAML 設定檔路徑（必要） |

### `amplification`

`path` 可為單一 `.h5` 檔或目錄。目錄模式會依 `--pattern` 收集檔案；只有加上 `--merge` 才會將多個檔案合併。

| 參數 | 說明 |
| --- | --- |
| `--distance`, `-d` | 震央距離（km，必要） |
| `--origin-time`, `-o` | 發震時刻，ISO 格式，例如 `2023-02-06T01:17:35`（必要） |
| `--ref-channels` | 基準 channel 數（井底最深 N 個），預設 10 |
| `--ref-distance-range` | 基準距離範圍（m），水平光纖用，例如 `--ref-distance-range 500 600` |
| `--vmin` | 最慢群速度（km/s），預設 2.0 |
| `--vmax` | 最快群速度（km/s），預設 4.0 |
| `--skip-channels` | 跳過前 N 個 channel（井口附近易受雜訊干擾），預設 0 |
| `--merge`, `-m` | 合併多個 chunk 後分析 |
| `--pattern`, `-p` | 目錄模式的 glob，預設 `*.h5` |
| `--sort-by` | 合併排序：`chunk_index`（預設）或 `timestamp` |
| `--csv` | 同時輸出各 channel 放大倍率至 CSV |
| `--event-label`, `-l` | 事件標籤（用於圖例） |
| `--save`, `-s` | 輸出圖片／CSV 目錄；未指定時顯示互動式視窗 |
| `--title`, `-t` | 圖表自訂標題 |
| `--dpi` | 圖片解析度，預設 150 |
| `--no-display` | 存檔模式下不彈出視窗 |

### `plot waterfall`

`path` 可為單一 `.h5` 檔或目錄。目錄模式會依 `--pattern` 收集檔案；只有加上 `--merge` 才會將多個檔案合併。

| 參數 | 說明 |
| --- | --- |
| `--merge`, `-m` | 合併多個 chunk 後繪圖 |
| `--pattern`, `-p` | 目錄模式的 glob，預設 `*.h5` |
| `--sort-by` | 合併排序：`chunk_index`（預設）或 `timestamp` |
| `--time-range` | 時間範圍：起訖 ISO 時間 |
| `--distance-range`, `--dist-range` | 距離／通道範圍 |
| `--colormap` | Matplotlib colormap，預設 `seismic` |
| `--title` | 自訂圖表標題 |
| `--save`, `-s` | 圖檔輸出目錄；未指定時顯示互動式視窗 |
| `--format` | 圖檔格式：`png`（預設）、`pdf`、`svg` |
| `--dpi` | 圖片解析度，預設 150 |
| `--no-display` | 存檔後不顯示視窗 |

### `plot fk`

| 參數 | 說明 |
| --- | --- |
| `--merge`, `-m` | 合併多個 chunk 後繪圖 |
| `--pattern`, `-p` | 目錄模式的 glob，預設 `*.h5` |
| `--sort-by` | 合併排序：`chunk_index`（預設）或 `timestamp` |
| `--channel-spacing` | 通道間距（m），必要 |
| `--freq-range` | 頻率範圍（Hz） |
| `--colormap` | Matplotlib colormap，預設 `hot` |
| `--title` | 自訂圖表標題 |
| `--save`, `-s` | 圖檔輸出目錄；未指定時顯示互動式視窗 |
| `--format` | 圖檔格式：`png`（預設）、`pdf`、`svg` |
| `--dpi` | 圖片解析度，預設 150 |
| `--no-display` | 存檔後不顯示視窗 |

### `plot spectrogram`

| 參數 | 說明 |
| --- | --- |
| `--merge`, `-m` | 合併多個 chunk 後繪圖 |
| `--pattern`, `-p` | 目錄模式的 glob，預設 `*.h5` |
| `--sort-by` | 合併排序：`chunk_index`（預設）或 `timestamp` |
| `--channel` | 通道索引；未指定時使用中間通道 |
| `--freq-range` | 頻率範圍（Hz） |
| `--colormap` | Matplotlib colormap，預設 `viridis` |
| `--title` | 自訂圖表標題 |
| `--save`, `-s` | 圖檔輸出目錄；未指定時顯示互動式視窗 |
| `--format` | 圖檔格式：`png`（預設）、`pdf`、`svg` |
| `--dpi` | 圖片解析度，預設 150 |
| `--no-display` | 存檔後不顯示視窗 |

### `overlay`

| 參數 | 說明 |
| --- | --- |
| `--pattern`, `-p` | CSV 檔案 glob，預設 `*.csv` |
| `--labels`, `-l` | 圖例標籤，逗號分隔（預設為檔名） |
| `--save`, `-s` | 輸出圖片目錄 |
| `--title`, `-t` | 圖表自訂標題 |
| `--dpi` | 圖片解析度，預設 150 |
| `--csv` | 同時輸出疊圖資料至 CSV（各事件 + 中位數） |
| `--no-display` | 存檔模式下不彈出視窗 |

### `detect`

`path` 可為單一 `.h5` 檔或目錄。目錄模式會依 `--pattern` 收集檔案；只有加上 `--merge` 才會將多個檔案合併。

| 參數 | 說明 |
| --- | --- |
| `--sta-window` | STA 短窗長度（秒），預設 0.5 |
| `--lta-window` | LTA 長窗長度（秒），預設 10.0 |
| `--trigger-threshold` | STA/LTA ratio 觸發閾值，預設 3.0 |
| `--detrigger-threshold` | STA/LTA ratio 解除觸發閾值，預設 1.5 |
| `--min-channels` | 最少同時觸發通道數（空間一致性），預設 30 |
| `--min-duration` | 最短事件持續時間（秒），預設 0.1 |
| `--merge-window` | 合併相鄰事件閾值（秒），0=不合併 |
| `--merge`, `-m` | 合併多個 chunk 後檢測 |
| `--pattern`, `-p` | 目錄模式的 glob，預設 `*.h5` |
| `--sort-by` | 合併排序：`chunk_index`（預設）或 `timestamp` |
| `--save`, `-s` | 輸出目錄路徑 |
| `--format`, `-f` | 輸出格式：`csv`、`json` 或 `all`（預設） |

### `snr`

`path` 可為單一 `.h5` 檔或目錄。目錄模式會依 `--pattern` 收集檔案；只有加上 `--merge` 才會將多個檔案合併。

| 參數 | 說明 |
| --- | --- |
| `--channel`, `-c` | 要分析的 channel 索引（必要） |
| `--distance`, `-d` | 震央距離（km，必要） |
| `--origin-time`, `-o` | 發震時刻，ISO 格式（必要） |
| `--vmin` | 最慢群速度（km/s），預設 2.0 |
| `--vmax` | 最快群速度（km/s），預設 4.0 |
| `--noise-offset` | 雜訊窗與訊號窗之間隔（秒），預設 30.0 |
| `--merge`, `-m` | 合併多個 chunk 後分析 |
| `--pattern`, `-p` | 目錄模式的 glob，預設 `*.h5` |
| `--sort-by` | 合併排序：`chunk_index`（預設）或 `timestamp` |
| `--save`, `-s` | 輸出 JSON 結果到指定目錄 |
| `--interactive`, `-i` | 互動式選窗模式：在波形圖上拖曳圈選訊號／雜訊窗 |

### `check`

| 參數 | 說明 |
| --- | --- |
| `--pattern`, `-p` | 檔案 glob pattern，預設 `*.h5` |
| `--time-bin`, `-t` | 時間 bin 大小（秒），設為 `auto` 自動計算（預設） |
| `--title` | 圖表自訂標題 |
| `--save`, `-s` | 輸出圖片目錄；未指定時互動式顯示 |
| `--format` | 圖檔格式：`png`（預設）、`pdf`、`svg` |
| `--dpi` | 圖片解析度，預設 150 |
| `--no-display` | 存檔模式下不彈出視窗 |

## 設定檔參考

設定檔包含五個區塊：

| 區塊 | 主要用途 |
| --- | --- |
| `project_name` | 輸出檔名使用的專案名稱 |
| `data` | 輸入資料、格式與 chunk 設定 |
| `coordinate` | 光纖幾何座標對齊與單位轉換設定 |
| `preprocessing` | 範圍裁切、detrend、taper、bandpass、decimate |
| `output` | 儲存路徑與輸出檔名 |

> 日誌層級（log level）由全域 CLI 選項 `--log-level` 控制，不再透過設定檔。

### `data`

| 欄位 | 說明 |
| --- | --- |
| `input_dir` | 輸入目錄（必要） |
| `format` | `miniseed` 時使用 ObsPy 讀取；其他值交由 DASCore 建立 spool |
| `file_pattern` | MiniSEED 的檔案 glob；非 MiniSEED 目錄由 DASCore 掃描 |
| `sampling_rate` | Hz。若提供，也會用於將濾波安全邊際換算為秒 |
| `chunk_duration` | 每個 chunk 的時間，例如 `"10min"`、`"1h"` |
| `filter_safety_samples` | 額外 overlap 樣本數；需同時設定 `sampling_rate` 才會生效 |

### `coordinate`

| 欄位 | 說明 |
| --- | --- |
| `fiber_geometry_file` | 光纖幾何座標檔（CSV），需包含欄位 `channel_index`、`lat`、`lon`、`depth` |
| `distance_unit` | 座標檔案裡距離的單位，預設 `m` |
| `strict_shape_check` | 對齊後是否檢查資料與座標 shape 一致，預設 `true` |
| `input_unit` | 原始資料單位：`strain_rate`（預設）或 `phase`（相位差） |
| `phase_strain_constant` | 相位差 → 應變率常數，預設 `11.6e-9`；公式 `ε̇ = constant × f × ΔΦ` |
| `missing_channel_strategy` | channel 缺失處理：`interpolate`（預設）、`crop` 或 `error` |

座標對齊流程：

1. 讀取 geometry.csv，以 `channel_index` 排序
2. 用 Haversine 公式計算相鄰 channel 的水平距離，結合深度差得到 3D 段長，累積為每個 channel 的實際距離（米）
3. 將 Patch 的 distance 座標由 channel index 替換為實際累積距離；缺失 channel 依 `missing_channel_strategy` 處理
4. 若 `input_unit: "phase"`，將資料轉換為應變率

若 geometry.csv 不存在，會略過座標對齊，沿用原始 channel index 距離軸。

### `preprocessing`

處理順序固定為：`select → detrend → bandpass → decimate`。

| 欄位 | 說明 |
| --- | --- |
| `time_range` | 時間範圍；目前對應資料的時間座標篩選 |
| `distance_range` | 距離／通道範圍 |
| `detrend` | `linear`、`constant` 或 `null` |
| `bandpass` | `[low_cutoff, high_cutoff]` Hz；設為 `null` 跳過 |
| `taper_ratio` | 濾波前 taper 比例（頭尾各半），設為 `null` 跳過 |
| `decimate_factor` | 整數且至少為 2；設為 `null` 跳過 |

### `output`

| 欄位 | 說明 |
| --- | --- |
| `save_dir` | 輸出目錄（必要） |
| `filename_pattern` | 可使用 `{project_name}`、`{timestamp}`、`{chunk_index:04d}` |
| `overwrite` | 是否允許覆寫已存在檔案 |

## 專案結構

```text
├── configs/config.yaml.example  # 設定檔範例
├── src/das_pipeline/
│   ├── config.py                # Pydantic 設定模型
│   ├── pipeline.py              # 轉檔流程
│   ├── cli/                     # CLI 入口與子命令（convert / amplification / detect / snr / plot / overlay / check）
│   ├── io/                      # 載入、分段、輸出與座標對齊
│   ├── preprocessing/           # 前處理步驟（select / detrend / bandpass / decimate / taper / NaN 處理）
│   ├── detection/               # STA/LTA 觸發檢測
│   ├── teleseismic/             # 遠震地層放大效應分析與 SNR 計算
│   ├── overlay/                 # 多事件放大倍率疊圖
│   ├── visualization/           # Waterfall / F-K / Spectrogram 繪圖與 chunk 合併
│   └── utils/                   # 共用工具（日誌設定、壞道排除等）
├── tests/
└── pyproject.toml
```

## 測試

```bash
pytest tests/ -v
```

## 已知限制

- 座標對齊固定使用線性插值與米（m）。
- 輸出格式固定為 DASDAE HDF5。
- 目前未提供平行處理。

## License

請參閱 [LICENSE](LICENSE)。