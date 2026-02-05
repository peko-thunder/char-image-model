# 文字画像分類モデル

## プロジェクト概要

転移学習を用いた文字画像分類モデル。VGG16、EfficientNetB0、EfficientNetB7に対応。

## ディレクトリ構成

```
char-image-model/
├── src/
│   ├── lib/
│   │   └── generate_dataset/   # データセット生成
│   │       ├── __main__.py     # メインエントリポイント
│   │       ├── char.py         # 文字読込
│   │       ├── font.py         # フォント処理
│   │       ├── image.py        # 画像生成
│   │       └── log.py          # ログ出力
│   ├── train/
│   │   ├── vgg16/
│   │   │   └── main.py         # VGG16モデル訓練
│   │   ├── efficientnetb0/
│   │   │   └── main.py         # EfficientNetB0モデル訓練
│   │   └── efficientnetb7/
│   │       └── main.py         # EfficientNetB7モデル訓練
│   ├── utils/
│   │   ├── config.py           # 設定クラス定義
│   │   ├── dataset.py          # データセット読込
│   │   └── validate.py         # モデル検証（共通）
│   └── viewer.py               # 検証結果ビューア（Streamlit）
├── chars/                      # 文字セット定義
│   ├── jis_x_0208_level1_kanji.txt
│   ├── jis_x_0208_level2_kanji.txt
│   ├── jis_x_0208_level3_kanji.txt
│   ├── jis_x_0208_level4_kanji.txt
│   └── name_kanji.txt
├── fonts/                      # フォントファイル
│   ├── ipamjm00601/            # IPA明朝
│   ├── Noto_Sans_JP/           # Noto Sans JP
│   └── Noto_Serif_JP/          # Noto Serif JP
├── tests/                      # テスト
│   └── lib/generate_dataset/
├── dataset/                    # 訓練データ（クラスごとのサブディレクトリ）
├── logs/                       # データセット生成ログ
└── workspace/                  # 実験ワークスペース
```

## ワークスペース

各実験はワークスペースディレクトリで管理する。

```
workspace/<experiment_name>/
├── config.json             # 入力：全設定
├── model.keras             # 出力：訓練済みモデル
└── validation_results.csv  # 出力：検証結果
```

### config.json の構造

```json
{
  "dataset": {
    "dataset_dir": "dataset/default",
    "image_size": [50, 50],
    "batch_size": 128,
    "validation_split": 0.2,
    "seed": 123
  },
  "training": {
    "epochs": 100,
    "dropout_rate": 0.5
  }
}
```

## 使い方

### 1. データセット生成

```bash
python -m src.lib.generate_dataset
```

- `chars/` 内の文字セットファイルと `fonts/` 内のフォントを使用して画像を生成
- 出力先: `<dataset_path>/<unicode_hex>/<font_name>.jpg`（デフォルト: `dataset/tmp`）
- 画像サイズ: 50x50ピクセル
- フォントが対応していない文字はスキップされ、`logs/` にログ出力

### 2. ワークスペースの作成

```bash
mkdir -p workspace/exp001
```

`config.json`を作成して設定を記述。

### 3. 訓練

```bash
python -m src.train.vgg16.main workspace/exp001
python -m src.train.efficientnetb0.main workspace/exp001
python -m src.train.efficientnetb7.main workspace/exp001
```

出力:
- `workspace/exp001/model.keras` - 訓練済みモデル

### 4. 検証

```bash
python -m src.utils.validate workspace/exp001
python -m src.utils.validate workspace/exp001 -o results.csv
```

出力:
- `validation_results.csv` - 各画像の予測結果（画像パス、正解ラベル、予測ラベル、信頼度、結果）

### 5. 検証結果の閲覧

```bash
streamlit run src/viewer.py
```

ブラウザで検証結果をテーブル表示。詳細は「検証結果ビューア」セクション参照。

## データセット生成 (`src/lib/generate_dataset/`)

### モジュール構成

| ファイル | 説明 |
|---------|------|
| `__main__.py` | メインエントリポイント。フォントと文字の組み合わせから画像を生成 |
| `char.py` | 文字セットファイルの読込 |
| `font.py` | フォントが文字をサポートしているか確認 |
| `image.py` | PILを使用した文字画像生成 |
| `log.py` | ログファイル出力（`logs/` に日時付きで保存） |

### 文字セットファイル (`chars/`)

| ファイル | 説明 |
|---------|------|
| `jis_x_0208_level1_kanji.txt` | JIS第1水準漢字 |
| `jis_x_0208_level2_kanji.txt` | JIS第2水準漢字 |
| `jis_x_0208_level3_kanji.txt` | JIS第3水準漢字 |
| `jis_x_0208_level4_kanji.txt` | JIS第4水準漢字 |
| `name_kanji.txt` | 人名用漢字 |

フォーマット: 1行1文字の改行区切りテキスト

### フォント (`fonts/`)

| フォント | 説明 |
|---------|------|
| `ipamjm00601/ipamjm.ttf` | IPA明朝（明朝体） |
| `Noto_Sans_JP/` | Noto Sans JP（ゴシック体、各ウェイト） |
| `Noto_Serif_JP/` | Noto Serif JP（明朝体、各ウェイト） |

### 依存関係

```bash
pip install Pillow fonttools
```

## 検証結果ビューア (`src/viewer.py`)

Streamlitベースの検証結果閲覧ツール。

### 依存関係

```bash
pip install streamlit
```

### 機能

- **ワークスペース選択**: `validation_results.csv`を含むワークスペースを自動検出
- **画像表示**: 予測対象の画像を表示（スケール調整可能）
- **ラベル表示**: コードポイント（16進数）と実際の文字を併記
- **統計情報**: Total / Correct / Wrong / Accuracy を表示
- **ページネーション**: 表示件数を調整可能

### フィルター機能

| フィルター | 説明 |
|-----------|------|
| 文字検索 | 任意の文字で検索（例: `あ`）。True/Predicted Labelに一致する行を表示 |
| Result | CORRECT / WRONG で絞り込み |
| Dataset Type | train / val で絞り込み |
| True Label | 正解ラベルで絞り込み |
| Predicted Label | 予測ラベルで絞り込み |

### 表示設定

| 設定 | 説明 |
|-----|------|
| 画像スケール | 0.5〜5.0倍（デフォルト: 1.0）。config.jsonのimage_size[1]を基準 |
| 表示件数 | 10〜100件（デフォルト: 20） |

## 設定クラス (`src/utils/config.py`)

### DatasetConfig
```python
@dataclass
class DatasetConfig:
    dataset_dir: str                    # データセットのパス
    image_size: tuple[int, int]         # 画像サイズ (H, W)
    batch_size: int                     # バッチサイズ
    validation_split: float | None      # 検証データの割合
    seed: Any | None                    # 乱数シード
```

### TrainingConfig
```python
@dataclass
class TrainingConfig:
    epochs: int                         # エポック数
    dropout_rate: float                 # ドロップアウト率
```

### WorkspaceConfig
```python
@dataclass
class WorkspaceConfig:
    dataset: DatasetConfig
    training: TrainingConfig
```

### 読込
```python
from src.utils.config import load_workspace_config

config = load_workspace_config("workspace/exp001/config.json")
```

## モデルアーキテクチャ

### VGG16 (`src/train/vgg16/main.py`)
- **ベースモデル**: VGG16（ImageNet事前学習済み、凍結）
- **データ拡張**: RandomZoom（訓練時のみ適用）
- **カスタムヘッド**: GlobalAveragePooling2D → Dropout → Dense(softmax)

### 訓練設定（共通）

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| optimizer | Adam | オプティマイザ |
| loss | categorical_crossentropy | 損失関数 |
| EarlyStopping | patience=5 | 過学習防止 |
| ReduceLROnPlateau | patience=3, factor=0.5 | 学習率自動調整 |

## 精度改善のオプション

### データ拡張 (`_create_model`内)

```python
data_augmentation = keras.Sequential([
    keras.layers.RandomRotation(0.05),      # ±18度
    keras.layers.RandomZoom(0.05),          # ±5%
    keras.layers.RandomTranslation(0.05, 0.05),  # ±5%
])
```

文字画像では`RandomFlip`は使用しない（左右反転で別文字になるため）。

### 画像サイズ

VGG16は224x224で学習されているため、大きいサイズほど精度向上の可能性あり。

### Fine-tuning

ベースモデルの一部レイヤーを学習可能にする:

```python
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False
```

## テスト

```bash
pytest tests/
```

テストは `tests/lib/generate_dataset/` に配置。

## 注意事項

- データセットは `dataset/<name>/` 内にクラスごとのサブディレクトリで配置
- 訓練前に`config.json`を必ず作成すること
- データセット生成時、指定した出力先ディレクトリは削除される
- 非対応文字のログは `logs/` に日時付きファイルで出力される
