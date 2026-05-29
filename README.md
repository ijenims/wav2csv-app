# WAV to CSV Converter

加速度計から取得したWAVファイルを、指定サンプリング周波数にリサンプリングし、
COMWAY アプリで読み込み可能なCSV形式に変換するWebアプリ。

ブラウザ内で完結（サーバへのデータ送信なし）。Amplify + S3 でホスティング。

---

## デプロイ

`dist/` フォルダの中身をそのままS3にアップロードする。

```
dist/
├── index.html
├── help.html
├── css/style.css
└── js/
    ├── main.js
    ├── wavParser.js
    ├── resampler.js
    ├── chart.js
    └── csvExport.js
```

## ローカル確認

ESモジュールを使用しているため `file://` では動作しない。ローカルサーバが必要。

```bash
cd dist
python -m http.server 8080
# → http://localhost:8080
```

## 機能

- WAV読み込み（PCM 8/16/24/32bit、IEEE float 32/64bit、マルチチャンネル）
- リサンプリング（100〜2000 Hz、Lanczos sinc補間 a=3）
- チャンネル別波形グラフ表示（Canvas直描き、200k点上限ダウンサンプル）
- COMWAY対応CSV出力（8列固定フォーマット）

## CSV出力仕様

| 列 | 内容 |
|----|------|
| 1 | `ags`（固定） |
| 2 | サンプル番号（0〜N-1） |
| 3〜(2+ch) | チャンネルデータ（小数6桁） |
| 残り | 0埋め（合計8列） |

ヘッダなし、UTF-8、LF改行。

## 技術スタック

- HTML / CSS / JavaScript（ESモジュール）
- 外部ライブラリ依存なし（グラフもCanvas直描き）
- WAVパース: DataView による手動バイナリ解析
- リサンプリング: Lanczos sinc補間 + 移動平均アンチエイリアシングフィルタ

## 旧バージョン

- `wav2csv.py` — Streamlit版（Python 3.10+、scipy/soundfile使用）
- `wav2csv.html` — 1ファイル完結版（アーカイブ）
