/**
 * エントリポイント
 * UIイベントのバインドとフロー制御
 */

import { parseWav } from './wavParser.js';
import { resample } from './resampler.js';
import { plotData } from './chart.js';
import { generateCsv, downloadCsv, generatePreviewHtml } from './csvExport.js';

// ---- 状態 ----
let rawData = null;
let resampledData = null;

// ---- DOM参照 ----
const wavInput = document.getElementById('wav-input');
const metaInfo = document.getElementById('meta-info');
const sizeWarning = document.getElementById('size-warning');
const fsOutInput = document.getElementById('fs-out');
const csvNameInput = document.getElementById('csv-name');
const placeholder = document.getElementById('placeholder');
const rawSection = document.getElementById('raw-section');
const resampleSection = document.getElementById('resample-section');
const resultSection = document.getElementById('result-section');
const btnResample = document.getElementById('btn-resample');
const resampleStatus = document.getElementById('resample-status');
const previewTable = document.getElementById('preview-table');
const btnDownload = document.getElementById('btn-download');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const fileNameDisplay = document.getElementById('file-name-display');

// ---- プログレス制御 ----
function showProgress(percent, message) {
  progressContainer.style.display = 'block';
  progressBar.style.width = percent + '%';
  progressText.textContent = message;
}

function hideProgress() {
  progressContainer.style.display = 'none';
  progressBar.style.width = '0%';
  progressText.textContent = '';
}

/** UIを更新するために1フレーム待つ */
function nextFrame() {
  return new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

// ---- ファイル処理共通 ----
async function handleFile(file) {
  if (!file) return;

  // WAV拡張子チェック
  if (!file.name.toLowerCase().endsWith('.wav')) {
    alert('WAVファイルを選択してください。');
    return;
  }

  // ファイル名表示
  fileNameDisplay.textContent = '📄 ' + file.name;
  fileNameDisplay.style.display = 'block';
  btnFileSelect.textContent = file.name;

  // ファイル名からCSV名を設定
  const baseName = file.name.replace(/\.[^.]+$/, '');
  csvNameInput.value = baseName + '.csv';

  // サイズ警告
  const sizeMb = file.size / (1024 * 1024);
  if (sizeMb > 200) {
    sizeWarning.textContent = `アップロードサイズが大きいです（約 ${sizeMb.toFixed(1)} MB）。処理に時間がかかる場合があります。`;
    sizeWarning.style.display = 'block';
  } else {
    sizeWarning.style.display = 'none';
  }

  // プログレス開始
  showProgress(10, 'WAVファイル読み込み中...');
  await nextFrame();

  // WAV読み込み
  try {
    const arrayBuffer = await file.arrayBuffer();
    showProgress(30, 'WAVデータ解析中...');
    await nextFrame();
    rawData = parseWav(arrayBuffer);
  } catch (err) {
    hideProgress();
    alert('WAV読み込みエラー: ' + err.message);
    return;
  }

  // メタ情報表示
  const duration = rawData.length / rawData.fs;
  metaInfo.innerHTML = `
    <p><strong>チャンネル数</strong>: ${rawData.channels}</p>
    <p><strong>データ長[サンプル]</strong>: ${rawData.length}</p>
    <p><strong>サンプリング周波数 Fs[Hz]</strong>: ${rawData.fs}</p>
    <p><strong>推定時間[秒]</strong>: ${duration.toFixed(3)}</p>
  `;
  metaInfo.style.display = 'block';

  // UI表示切替
  placeholder.style.display = 'none';
  rawSection.style.display = 'block';
  resampleSection.style.display = 'block';
  resultSection.style.display = 'none';

  // 元データグラフ
  showProgress(60, 'グラフ描画中...');
  await nextFrame();
  plotData('charts-raw', rawData, `raw @ ${rawData.fs} Hz`);

  showProgress(100, '完了');
  await nextFrame();
  hideProgress();

  // リサンプル結果リセット
  resampledData = null;
}

// ---- WAVアップロード（ファイル選択） ----
const btnFileSelect = document.getElementById('btn-file-select');

btnFileSelect.addEventListener('click', () => {
  wavInput.click();
});

wavInput.addEventListener('change', (e) => {
  handleFile(e.target.files[0]);
});

// ---- ドラッグ&ドロップ ----
const dropZone = document.body;

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  placeholder.style.borderColor = '#86efac';
  placeholder.style.background = '#f0fdf4';
});

dropZone.addEventListener('dragleave', (e) => {
  e.preventDefault();
  placeholder.style.borderColor = '';
  placeholder.style.background = '';
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  placeholder.style.borderColor = '';
  placeholder.style.background = '';
  const file = e.dataTransfer.files[0];
  handleFile(file);
});

// ---- リサンプル実行 ----
btnResample.addEventListener('click', async () => {
  if (!rawData) return;

  btnResample.disabled = true;
  btnResample.textContent = '処理中...';
  resampleStatus.innerHTML = '';

  const fsOut = parseInt(fsOutInput.value, 10);

  showProgress(10, 'リサンプリング準備中...');
  await nextFrame();

  try {
    showProgress(30, `リサンプリング中 (${rawData.fs} → ${fsOut} Hz)...`);
    await nextFrame();
    resampledData = await resample(rawData, fsOut);

    showProgress(70, 'グラフ描画中...');
    await nextFrame();

    resampleStatus.innerHTML = `<div class="success-msg">変換完了: ${rawData.fs} Hz → ${fsOut} Hz / 形状: (${resampledData.length}, ${resampledData.channels})</div>`;

    // 変換後グラフ + プレビュー
    resultSection.style.display = 'block';
    plotData('charts-resampled', resampledData, `resampled @ ${fsOut} Hz`);
    previewTable.innerHTML = generatePreviewHtml(resampledData);

    showProgress(100, '完了');
    await nextFrame();
  } catch (err) {
    resampleStatus.innerHTML = `<div class="warning">エラー: ${err.message}</div>`;
  }

  hideProgress();
  btnResample.disabled = false;
  btnResample.textContent = 'リサンプル（指定Fsへ）';
});

// ---- CSVダウンロード ----
btnDownload.addEventListener('click', () => {
  if (!resampledData) return;
  const filename = csvNameInput.value || 'output.csv';
  const csv = generateCsv(resampledData);
  downloadCsv(csv, filename);
});
