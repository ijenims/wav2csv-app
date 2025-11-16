# streamlit_app.py
# WAV → （補間で）任意Fs（既定1000 Hz）にリサンプリングして CSV ダウンロードするアプリ
# 仕様（2025-11-09 確定）：
# - 出力はダウンロードのみ（サーバ保存はしない）
# - フロー：1) リサンプル → 2) CSVをダウンロード
# - グラフは全体で最大 200,000 点に自動ダウンサンプルして表示
# - 出力Fs入力：既定 1000 Hz、範囲 100–2000 Hz、ステップ 10 Hz
# - CSV：ヘッダあり(ch0, ch1, ...)、UTF-8、インデックスなし、小数6桁
# - Kaiser β=8.6 固定（非表示）
# - 全チャンネル出力

import io
import math
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf
from scipy.signal import resample_poly
import matplotlib.pyplot as plt

st.set_page_config(page_title="wav2csv", layout="wide")
st.title("WAV to CSV（polyphase, Kaiser β=8.6）")

MAX_PLOT_POINTS = 200_000  # 表示の負荷対策

# -------------------- ユーティリティ --------------------
def wav_to_df_and_fs(file_like: io.BytesIO) -> Tuple[pd.DataFrame, int]:
    """WAVを読み込み、(サンプル, チャンネル)のDataFrameとFsを返す。
    単chは(N,)→(N,1)にreshape。
    """
    data, fs = sf.read(file_like)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_ch = data.shape[1]
    df = pd.DataFrame(data, columns=[f"ch{i}" for i in range(n_ch)])
    return df, int(fs)


def resample_to_target(df: pd.DataFrame, fs_in: int, fs_out: int) -> pd.DataFrame:
    """非整数比OKのpolyphase法でリサンプリング。列=チャンネル、行=時間方向。
    Kaiser窓β=8.6（実務で無難）。
    """
    if fs_in == fs_out:
        return df.copy()
    up = int(fs_out)
    down = int(fs_in)
    g = math.gcd(up, down)
    up //= g
    down //= g
    y = resample_poly(df.values, up, down, axis=0, window=("kaiser", 8.6))
    return pd.DataFrame(y, columns=df.columns)


def downsample_df_for_plot(df: pd.DataFrame, limit: int = MAX_PLOT_POINTS) -> pd.DataFrame:
    """描画用に等間隔bin平均で間引く。全体点数が limit を超える場合のみ適用。"""
    n = len(df)
    if n <= limit:
        return df
    factor = math.ceil(n / limit)
    # 端数を処理するため末尾をパディングしてreshape→平均
    pad = (-n) % factor
    arr = df.values
    if pad:
        # 末尾値を複製してパディング（平均値に対する影響はごく小）
        last = arr[-1:, :]
        arr = np.vstack([arr, np.repeat(last, pad, axis=0)])
    arr = arr.reshape(-1, factor, arr.shape[1]).mean(axis=1)
    return pd.DataFrame(arr, columns=df.columns)


def plot_timeseries(df: pd.DataFrame, title: str = "", limit_points: int = MAX_PLOT_POINTS):
    """多chを縦に並べて1枚に描画（必要に応じてダウンサンプル）。"""
    df_plot = downsample_df_for_plot(df, limit=limit_points)
    n_ch = df_plot.shape[1]
    fig, axes = plt.subplots(n_ch, 1, figsize=(14, max(3, 2.2*n_ch)), sharex=True, sharey=False)
    if n_ch == 1:
        axes = [axes]
    for i, col in enumerate(df_plot.columns):
        axes[i].plot(df_plot[col])
        axes[i].set_ylabel(col)
        axes[i].grid(True, linewidth=0.3)
    axes[-1].set_xlabel("sample (downsampled for plot)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    st.pyplot(fig)


# -------------------- サイドバー --------------------
st.sidebar.header("操作")
uploaded = st.sidebar.file_uploader("WAVファイルをアップロード", type=["wav", "WAV"])  # ローカル or D&D

meta_placeholder = st.sidebar.empty()  # 2) メタ情報表示用

# 4) 変換後Fs入力（仕様：既定1000、範囲100–2000、刻み10）
fs_out = st.sidebar.number_input("変換後サンプリング周波数 Fs[Hz]", min_value=100, max_value=2000, value=1000, step=10)

# 5) 出力CSVファイル名（デフォルト=アップロード名+.csv）
default_csv_name = "output.csv"
if uploaded is not None and uploaded.name:
    base = pathlib.Path(uploaded.name).stem
    default_csv_name = f"{base}.csv"

csv_name = st.sidebar.text_input("CSVファイル名", value=default_csv_name)

# 200MB超の警告（処理は継続）
if uploaded is not None and hasattr(uploaded, "size") and uploaded.size is not None:
    size_mb = uploaded.size / (1024*1024)
    if size_mb > 200:
        st.sidebar.warning(f"アップロードサイズが大きいです（約 {size_mb:.1f} MB）。処理に時間がかかる場合があります。")

# -------------------- メイン --------------------
if uploaded is None:
    st.info("左のサイドバーからWAVをアップロードしてな。")
    st.stop()

# 読み込み
file_like = io.BytesIO(uploaded.getvalue())
df_raw, fs_in = wav_to_df_and_fs(file_like)
N = len(df_raw)
duration = N / fs_in if fs_in > 0 else 0.0

# 2) メタ表示
with meta_placeholder.container():
    st.sidebar.write(f"**チャンネル数**: {df_raw.shape[1]}")
    st.sidebar.write(f"**データ長[サンプル]**: {N}")
    st.sidebar.write(f"**サンプリング周波数 Fs[Hz]**: {fs_in}")
    st.sidebar.write(f"**推定時間[秒]**: {duration:.3f}")

# 3) グラフ（メインエリア）
st.subheader("元データ")
plot_timeseries(df_raw, title=f"raw @ {fs_in} Hz")

# セッション状態の初期化
for k, v in {
    "df_out": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 変換実行UI（2段階運用：リサンプル → ダウンロード）
st.subheader("リサンプリング")
col1, col2 = st.columns([1, 2])
with col1:
    do_resample = st.button("リサンプル（指定Fsへ）", type="primary")

# 1) リサンプルはボタン時のみ実行
if do_resample:
    df_out = resample_to_target(df_raw, fs_in=fs_in, fs_out=int(fs_out))
    st.session_state["df_out"] = df_out
    st.success(f"変換完了: {fs_in} Hz → {int(fs_out)} Hz / 形状: {df_out.shape}")

# 変換後のプレビュー＆グラフ（保持済みなら表示）
if st.session_state.get("df_out") is not None:
    df_out = st.session_state["df_out"]
    st.write("先頭5行プレビュー：")
    st.dataframe(df_out.head())

    st.subheader("変換後データ")
    plot_timeseries(df_out, title=f"resampled @ {int(fs_out)} Hz")

    # 2) ダウンロード（クライアントPCへ保存）
    # ▼▼ ここでCSVフォーマットを変換 ▼▼

    df_out = st.session_state["df_out"]

    # リサンプル後のサンプル数・チャンネル数
    N = len(df_out)
    ch = df_out.shape[1]

    # 8列の固定フォーマットを作る
    # col1="ags"、col2=0〜N-1、col3〜col(2+ch)=データ、残り0埋め
    data = np.zeros((N, 8), dtype=object)

    data[:, 0] = "ags"          # 1列目
    data[:, 1] = np.arange(N)   # 2列目（サンプル番号）

    # 3〜(2+ch)列 に データを詰める
    for i in range(ch):
        data[:, 2 + i] = df_out.iloc[:, i].values

    # DataFrame化（ヘッダなし）
    df_export = pd.DataFrame(data)

    # CSV バイトへ変換（ヘッダなし、indexなし）
    csv_bytes = df_export.to_csv(
        index=False,
        header=False,
        float_format="%.6f"
    ).encode("utf-8")

    # ▼ ダウンロードボタン
    st.download_button(
        label="CSVをダウンロード",
        data=csv_bytes,
        file_name=csv_name,
        mime="text/csv",
        use_container_width=True,
    )


# フッター
st.caption("© WAV→CSV / polyphase resampling (Kaiser β=8.6) / 多ch対応 / plot≤200kpts")
