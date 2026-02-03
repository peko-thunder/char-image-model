# VGG16 文字画像分類モデル

## プロジェクト概要

VGG16ベースの転移学習を用いた文字画像分類モデル。

## ディレクトリ構成

```
char-image-model/
├── src/
│   ├── utils/
│   │   ├── config.py       # 設定クラス定義（DatasetConfig, TrainingConfig）
│   │   ├── dataset.py      # データセット読込
│   │   └── validate.py     # モデル検証（共通）
│   ├── vgg16/
│   │   └── train.py        # VGG16モデル訓練
│   ├── efficientnetb0/
│   │   └── train.py        # EfficientNetB0モデル訓練
│   └── efficientnetb7/
│       └── train.py        # EfficientNetB7モデル訓練
├── dataset/                # 訓練データ（クラスごとのサブディレクトリ）
└── models/                 # 保存されたモデルと設定
```

## 使い方

### 訓練

```bash
python -m src.vgg16.train
```

出力:
- `models/<output_dir>/model.keras` - 訓練済みモデル
- `models/<output_dir>/dataset_config.json` - DatasetConfig
- `models/<output_dir>/training_config.json` - TrainingConfig（epochs, dropout_rate）

### 検証

```bash
python -m src.utils.validate <model_dir> [-o output.csv]
```

引数:
- `model_dir`: モデルディレクトリのパス（`model.keras`と`dataset_config.json`を含む）
- `-o, --output`: 出力CSVのパス（省略時は`model_dir/validation_results.csv`）

例:
```bash
python -m src.utils.validate models/vgg16/default
python -m src.utils.validate models/efficientnetb0/exp1 -o results.csv
```

出力:
- `validation_results.csv` - 各画像の予測結果（画像パス、正解ラベル、予測ラベル、信頼度、結果）

## 主要コンポーネント

### 設定クラス (`src/utils/config.py`)

#### DatasetConfig
データセットの設定を管理するdataclass。

```python
@dataclass
class DatasetConfig:
    dataset_dir: str                    # データセットのパス
    image_size: tuple[int, int]         # 画像サイズ (H, W)
    batch_size: int                     # バッチサイズ
    validation_split: float | None      # 検証データの割合
    seed: Any | None                    # 乱数シード
```

#### TrainingConfig
トレーニングパラメータを管理するdataclass。

```python
@dataclass
class TrainingConfig:
    epochs: int                         # エポック数
    dropout_rate: float                 # ドロップアウト率
```

#### 保存・読込
```python
from src.utils.config import save_config, load_dataset_config, load_training_config

save_config(config, "path/to/config.json")
dataset_config = load_dataset_config("path/to/dataset_config.json")
training_config = load_training_config("path/to/training_config.json")
```

### モデルアーキテクチャ (`src/vgg16/train.py`)

- **ベースモデル**: VGG16（ImageNet事前学習済み、凍結）
- **データ拡張**: RandomZoom（訓練時のみ適用）
- **カスタムヘッド**: GlobalAveragePooling2D → Dropout(0.2) → Dense(softmax)

### 訓練設定

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

## 注意事項

- データセットは `dataset/<name>/` 内にクラスごとのサブディレクトリで配置
- 訓練時と検証時で同じ`DatasetConfig`を使用すること（`dataset_config.json`から読込推奨）
