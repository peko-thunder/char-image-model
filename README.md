# char-image-model

## Setup

### uv

```bash
uv python install 3.12
uv venv
uv pip install \
    --upgrade pip \
    --no-cache-dir \
    -r requirements.txt
```

### pyvenv

```bash
python -m venv sandbox
```

有効化

```bash
source sandbox/bin/activate
```

```bash
pip install \
    --upgrade pip \
    --no-cache-dir \
    -r requirements.txt
```

## Fonts

全てのフォントの著作権は提供元に属します。

| Name | Url | License | File |
| :-- | :-- | :-- | :-- |
| IPAmj明朝 | https://moji.or.jp/mojikiban/font/ | IPA Font License | fonts/ipamjm00601 | |
| NotoSansJP | https://fonts.google.com/noto/specimen/Noto+Sans+JP | OFL-1.1 | fonts/Noto_Sans_JP |
| NotoSerifJP | https://fonts.google.com/noto/specimen/Noto+Serif+JP | OFL-1.1 | fonts/Noto_Serif_JP |
