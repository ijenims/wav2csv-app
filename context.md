# wav2csv HTML移植セッションログ

**日付**: 2025-05-28（木）

---

## 背景

- 既存の `wav2csv.py`（Streamlit製）をHTML版に移植したい
- サーバレスで動かしたい（Amplify + S3）

## 決定事項

### 技術選定

| 項目 | Python版 | HTML版 |
|------|----------|--------|
| WAV読み込み | soundfile (`sf.read`) | WAVバイナリ手動パース（DataView） |
| リサンプリング | scipy `resample_poly` (Kaiser β=8.6) | Lanczos sinc補間 (a=3) + 移動平均LPF |
| グラフ | matplotlib | Chart.js (CDN) |
| CSV生成 | pandas → bytes | Blob + URL.createObjectURL |
| UI | Streamlit | 素のHTML/CSS/JS |

### リサンプリング方針

- OfflineAudioContext は sampleRate 下限が 3000 Hz のため使えない（目標Fs: 100〜2000 Hz）
- Lanczos sinc補間 (a=3) を採用。品質は実用十分（粗くはないが Kaiser polyphase ほど厳密ではない）
- ダウンサンプル時は移動平均フィルタでアンチエイリアシング

### CSV出力仕様（Python版踏襲）

- 8列固定、ヘッダなし
- 1列目: `"ags"` 固定
- 2列目: サンプル番号（0〜N-1）
- 3列目以降: チャンネルデータ（小数6桁）
- 残り列: 0埋め

## 成果物

### プロジェクト構成

```
wav2csv/
├── dist/                  ← デプロイ用（S3にアップする対象）
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── main.js        ← エントリポイント（イベントバインド + フロー制御）
│       ├── wavParser.js   ← WAVバイナリパース
│       ├── resampler.js   ← Lanczos sinc補間 + アンチエイリアシング
│       ├── chart.js       ← Chart.jsラッパー（ダウンサンプル描画）
│       └── csvExport.js   ← CSV生成 + ダウンロード + プレビュー
├── wav2csv.html           ← 旧1ファイル版（アーカイブ）
├── wav2csv.py             ← 元のStreamlit版
├── raw_data/
├── outputs/
├── requirements.txt
├── runtime.txt
└── README.md
```

### 対応WAVフォーマット

- PCM: 8 / 16 / 24 / 32 bit
- IEEE float: 32 / 64 bit
- マルチチャンネル対応

## 動作確認

- 4ch / 1280 Hz / 76800サンプルのWAVで動作確認済み
- リサンプル 1280 Hz → 1000 Hz 成功

## セッション2（2025-05-29）

### グラフ表示の改善

- **チャンネル別表示**: 1グラフに全ch重ねる方式 → チャンネルごとに個別グラフを縦並び
- **縦軸スケール統一**: 全チャンネルの最大絶対値で揃える

### Chart.js → Canvas直描きへの移行

Chart.jsでグラフ高さを制御しようとしたが、以下の問題で断念：
- `height` CSSを指定しても Chart.js が内部で上書きする
- `overflow: hidden` でも描画領域外のパディングが残る
- `aspectRatio` 指定だとY軸ラベルが消える等の副作用

**解決**: Chart.jsを完全に排除し、Canvas 2D APIで直描きに変更。
- 各チャンネル高さ80px固定（完全制御可能）
- 外部ライブラリ依存ゼロ
- Y軸ラベル（±max, 0）、チャンネルラベル、ゼロライン、薄い塗りつぶし

### UIテーマ変更

- 濃いボタン色 → 薄い背景+ボーダーのゴーストスタイル（控えめ）
- 成功メッセージ → 背景色なし、テキストのみ
- 全体的にグラフが主役になるようトーンを落とした
- CSSキャッシュ問題 → `?v=2` クエリパラメータで対応

### プログレスバー追加

- ファイル読み込み〜グラフ描画の各ステップで進捗表示
- サイドバーに配置（メインエリアを邪魔しない）
- `requestAnimationFrame` でUI更新の機会を確保

### help.html 追加

- アプリの目的（加速度計WAV → COMWAY対応CSV変換）
- 使い方、出力CSV仕様、注意事項、動作環境
- index.html からリンク追加

### file:// プロトコル問題

- ESモジュールは `file://` ではCORSで読み込めない
- `python -m http.server` 等のローカルサーバ経由で動作確認が必要
- 本番はAmplify+S3なので問題なし

### 現在のプロジェクト構成

```
wav2csv/
├── dist/                  ← デプロイ用（S3にアップする対象）
│   ├── index.html
│   ├── help.html
│   ├── css/style.css
│   └── js/
│       ├── main.js        ← エントリポイント（イベントバインド + フロー制御）
│       ├── wavParser.js   ← WAVバイナリパース
│       ├── resampler.js   ← Lanczos sinc補間 + アンチエイリアシング
│       ├── chart.js       ← Canvas直描きグラフ（Chart.js不使用）
│       └── csvExport.js   ← CSV生成 + ダウンロード + プレビュー
├── wav2csv.html           ← 旧1ファイル版（アーカイブ）
├── wav2csv.py             ← 元のStreamlit版
├── raw_data/
├── outputs/
├── requirements.txt
├── runtime.txt
└── README.md
```

## 残課題・今後の検討

- 大ファイル対応（Web Worker化で処理中のUI固まり防止）
- リサンプリング品質が不足した場合 → polyphase JS実装に差し替え
- UIの追加調整（必要に応じて壁打ち継続）
