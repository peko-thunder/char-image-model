# CJK統合漢字とCJK互換漢字

## 概要

Unicodeには、見た目がほぼ同一でありながら異なるコードポイントを持つ漢字が存在します。

| 分類 | コード範囲 | 説明 |
|-----|-----------|------|
| CJK統合漢字 | U+4E00〜U+9FFF | 標準的な漢字（約20,000字） |
| CJK互換漢字 | U+F900〜U+FAFF | レガシー互換用（約500字） |

## なぜCJK互換漢字が存在するのか

### 歴史的背景

1. **レガシーエンコーディングとの互換性**
   - Shift_JIS（Windows-31J/cp932）には「IBM拡張文字」が含まれる
   - これらの文字はJIS X 0208標準とは微妙に異なる字形を持つ
   - Unicodeへのラウンドトリップ変換を保証するため、別コードポイントが割り当てられた

2. **字形の違い**
   - 主に点の位置、画の方向などの微細な差異
   - 例：「神」のしめすへん（ネ vs 示）

### Shift_JIS → Unicode変換時の動作

```
標準JIS漢字     → CJK統合漢字 (U+4E00〜)
IBM拡張漢字     → CJK互換漢字 (U+F900〜)
```

## 常用漢字リストに含まれるCJK互換漢字（54字）

以下の漢字は、Shift_JISエンコードのHTMLから抽出した際にCJK互換漢字として取得されます。

| 互換漢字 | コード | 統合漢字 | コード |
|:-------:|:------:|:-------:|:------:|
| 欄 | U+F91D | 欄 | U+6B04 |
| 廊 | U+F928 | 廊 | U+5ECA |
| 虜 | U+F936 | 虜 | U+865C |
| 殺 | U+F970 | 殺 | U+6BBA |
| 類 | U+F9D0 | 類 | U+985E |
| 侮 | U+FA30 | 侮 | U+4FAE |
| 僧 | U+FA31 | 僧 | U+50E7 |
| 免 | U+FA32 | 免 | U+514D |
| 勉 | U+FA33 | 勉 | U+52C9 |
| 勤 | U+FA34 | 勤 | U+52E4 |
| 卑 | U+FA35 | 卑 | U+5351 |
| 喝 | U+FA36 | 喝 | U+559D |
| 嘆 | U+FA37 | 嘆 | U+5606 |
| 器 | U+FA38 | 器 | U+5668 |
| 塀 | U+FA39 | 塀 | U+5840 |
| 墨 | U+FA3A | 墨 | U+58A8 |
| 層 | U+FA3B | 層 | U+5C64 |
| 悔 | U+FA3D | 悔 | U+6094 |
| 慨 | U+FA3E | 慨 | U+6168 |
| 憎 | U+FA3F | 憎 | U+618E |
| 懲 | U+FA40 | 懲 | U+61F2 |
| 敏 | U+FA41 | 敏 | U+654F |
| 既 | U+FA42 | 既 | U+65E2 |
| 暑 | U+FA43 | 暑 | U+6691 |
| 梅 | U+FA44 | 梅 | U+6885 |
| 海 | U+FA45 | 海 | U+6D77 |
| 漢 | U+FA47 | 漢 | U+6F22 |
| 煮 | U+FA48 | 煮 | U+716E |
| 碑 | U+FA4B | 碑 | U+7891 |
| 社 | U+FA4C | 社 | U+793E |
| 祉 | U+FA4D | 祉 | U+7949 |
| 祈 | U+FA4E | 祈 | U+7948 |
| 祖 | U+FA50 | 祖 | U+7956 |
| 祝 | U+FA51 | 祝 | U+795D |
| 禍 | U+FA52 | 禍 | U+798D |
| 穀 | U+FA54 | 穀 | U+7A40 |
| 突 | U+FA55 | 突 | U+7A81 |
| 節 | U+FA56 | 節 | U+7BC0 |
| 練 | U+FA57 | 練 | U+7DF4 |
| 繁 | U+FA59 | 繁 | U+7E41 |
| 署 | U+FA5A | 署 | U+7F72 |
| 者 | U+FA5B | 者 | U+8005 |
| 臭 | U+FA5C | 臭 | U+81ED |
| 著 | U+FA5F | 著 | U+8457 |
| 褐 | U+FA60 | 褐 | U+8910 |
| 視 | U+FA61 | 視 | U+8996 |
| 謁 | U+FA62 | 謁 | U+8B01 |
| 謹 | U+FA63 | 謹 | U+8B39 |
| 賓 | U+FA64 | 賓 | U+8CD3 |
| 贈 | U+FA65 | 贈 | U+8D08 |
| 逸 | U+FA67 | 逸 | U+9038 |
| 難 | U+FA68 | 難 | U+96E3 |
| 響 | U+FA69 | 響 | U+97FF |
| 頻 | U+FA6A | 頻 | U+983B |

## プログラムでの取り扱い

### 問題点

CJK互換漢字とCJK統合漢字は見た目が同じでも、**異なる文字として扱われます**：

```python
# 見た目は同じだが、異なる文字
compat = '\uFA43'  # 暑 (CJK互換漢字)
unified = '\u6691'  # 暑 (CJK統合漢字)

compat == unified  # False（一致しない！）
```

これにより以下の問題が発生します：

- 文字列比較で一致しない
- 検索でヒットしない
- ソート順が異なる
- データベースの重複チェックをすり抜ける

### 解決策：Unicode正規化（NFKC）

`unicodedata.normalize('NFKC', text)` を使用して、CJK互換漢字をCJK統合漢字に変換できます。

```python
import unicodedata

def normalize_kanji(text: str) -> str:
    """CJK互換漢字をCJK統合漢字に正規化する"""
    return unicodedata.normalize('NFKC', text)

# 使用例
compat = '\uFA43'  # 暑 (U+FA43)
normalized = normalize_kanji(compat)  # 暑 (U+6691)

print(f"変換前: U+{ord(compat):04X}")      # U+FA43
print(f"変換後: U+{ord(normalized):04X}")  # U+6691
```

### CJK互換漢字の判定

```python
def is_cjk_compatibility(char: str) -> bool:
    """CJK互換漢字かどうかを判定する"""
    code = ord(char)
    return 0xF900 <= code <= 0xFAFF

def get_kanji_info(char: str) -> dict:
    """漢字の情報を取得する"""
    code = ord(char)
    normalized = unicodedata.normalize('NFKC', char)
    norm_code = ord(normalized)

    return {
        'char': char,
        'code': f'U+{code:04X}',
        'is_compat': is_cjk_compatibility(char),
        'normalized': normalized,
        'normalized_code': f'U+{norm_code:04X}',
        'needs_normalization': char != normalized
    }
```

### 推奨される処理フロー

```
入力テキスト
    ↓
NFKC正規化（CJK互換漢字 → CJK統合漢字）
    ↓
検索・比較・保存
```

## 関連ファイル

`name_kanji.txt` は`https://kanji.jitenon.jp/cat/namae`の2999文字を抽出したテキストデータ

法務局の戸籍統一文字情報検索の文字数とも一致

https://houmukyoku.moj.go.jp/KOSEKIMOJIDB/M01.html

## 参考リンク

- [Unicode CJK Compatibility Ideographs](https://en.wikipedia.org/wiki/CJK_Compatibility_Ideographs)
- [Unicode正規化 - Wikipedia](https://ja.wikipedia.org/wiki/Unicode%E6%AD%A3%E8%A6%8F%E5%8C%96)
- [文化庁 常用漢字表](https://www.bunka.go.jp/kokugo_nihongo/sisaku/joho/joho/kijun/naikaku/kanji/)
