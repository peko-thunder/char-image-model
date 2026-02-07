## JIS X 0208コード表

> https://www.asahi-net.or.jp/~ax2s-kmtn/ref/jisx0208.html


## 子の名に使える漢字

常用漢字表と人名用漢字表に掲げられた漢字は，いずれも子の名に使用することができます。一覧表については，こちらをご覧ください。

> https://www.moj.go.jp/MINJI/minji86.html

### 常用漢字表（平成22年11月30日内閣告示）

> https://www.bunka.go.jp/kokugo_nihongo/sisaku/joho/joho/kijun/naikaku/pdf/joyokanjihyo_20101130.pdf

### 人名用漢字表

> https://www.moj.go.jp/content/001131003.pdf

## CJK互換漢字について

Shift_JISからUnicodeへの変換時に生成されるCJK互換漢字の詳細と、プログラムでの取り扱い方法について。

> [cjk_compatibility_kanji.md](./cjk_compatibility_kanji.md)

## 漢字辞典

https://kanji.jitenon.jp/

対象の漢字を取得するクエリ

```javascript
(function() {
    // 1. 指定したセレクタに一致する要素をすべて取得
    const elements = document.querySelectorAll('.parts_box li a');
    
    // 2. テキストを抽出し、改行で繋げた文字列にする（前後の空白は削除）
    const textList = Array.from(elements)
                          .map(el => el.textContent.trim())
                          .filter(text => text !== "") // 空文字を除外
                          .join('\n');

    // 3. コンソール用のcopy関数を使ってクリップボードへ送る
    if (textList) {
        copy(textList);
        console.log("以下の内容をクリップボードにコピーしました：\n" + textList);
    } else {
        console.error("対象の要素が見つかりませんでした。");
    }
})();
```

## その他参考サイト

### IC運転免許証に格納されたデータを紹介してみる

> https://qiita.com/ikazayim/items/2e9b8bdca96db6bf34cb#%E5%A4%96%E5%AD%97

### 運転免許証及び運転免許証作成システム等の仕様の改正について

> https://www.npa.go.jp/laws/notification/koutuu/menkyo/menkyo20240719_145.pdf

13ページ参照

詳細なフォントは不明だがゴシック系のサンプルが表示

> 注1 128バイトを超過する場合がある。
> 注2 「文字構成」: 外字を構成するドット数とする。
> (例) 50×50 : “50” 32×32 : “32”
> 注3 外字の走査方法は、以下のとおりとする。BitMapデータをMMR符号化すること。

### 人名記号外字 教育機関・公共団体版

外字に使われるフォントは有料フォントの可能性もある

> https://www.dynacw.co.jp/product/product_licensepackage_detail.aspx?sid=48
