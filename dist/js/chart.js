/**
 * グラフ描画モジュール
 * Canvas直描きによる時系列プロット
 * - チャンネルごとに個別canvas
 * - 縦軸スケールは全チャンネルの最大絶対値に統一
 * - 自動ダウンサンプル付き
 */

const MAX_PLOT_POINTS = 200_000;
const COLORS = ['#60a5fa', '#f87171', '#34d399', '#a78bfa', '#fbbf24', '#22d3ee', '#fb7185', '#94a3b8'];
const CHART_HEIGHT = 80;
const Y_LABEL_WIDTH = 40;

/**
 * 描画用にbin平均で間引く
 */
function downsampleForPlot(arr, limit) {
  const len = arr.length;
  if (len <= limit) return arr;
  const factor = Math.ceil(len / limit);
  const outLen = Math.ceil(len / factor);
  const out = new Float32Array(outLen);
  for (let i = 0, idx = 0; i < len; i += factor, idx++) {
    let sum = 0;
    const end = Math.min(i + factor, len);
    for (let j = i; j < end; j++) sum += arr[j];
    out[idx] = sum / (end - i);
  }
  return out;
}

/**
 * 全チャンネルの最大絶対値を求める
 */
function getGlobalMaxAbs(samples) {
  let maxAbs = 0;
  for (const ch of samples) {
    const len = ch.length;
    for (let i = 0; i < len; i++) {
      const v = ch[i];
      const abs = v < 0 ? -v : v;
      if (abs > maxAbs) maxAbs = abs;
    }
  }
  return maxAbs || 1;
}

/**
 * 1チャンネル分の波形を描画
 */
function drawChannel(canvas, plotData, yMax, color, label) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = CHART_HEIGHT;

  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.height = h + 'px';
  ctx.scale(dpr, dpr);

  const plotW = w - Y_LABEL_WIDTH;
  const plotH = h - 4; // 上下2pxマージン
  const plotTop = 2;
  const plotLeft = Y_LABEL_WIDTH;

  // 背景
  ctx.fillStyle = '#fafbfc';
  ctx.fillRect(plotLeft, 0, plotW, h);

  // ゼロライン
  const zeroY = plotTop + plotH / 2;
  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(plotLeft, zeroY);
  ctx.lineTo(w, zeroY);
  ctx.stroke();

  // 波形描画
  const len = plotData.length;
  if (len === 0) return;

  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();

  for (let i = 0; i < len; i++) {
    const x = plotLeft + (i / (len - 1)) * plotW;
    const normalized = plotData[i] / yMax; // -1 ~ 1
    const y = zeroY - normalized * (plotH / 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // 塗りつぶし（薄く）
  ctx.globalAlpha = 0.08;
  ctx.fillStyle = color;
  ctx.lineTo(plotLeft + plotW, zeroY);
  ctx.lineTo(plotLeft, zeroY);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;

  // Y軸ラベル
  ctx.fillStyle = '#9ca3af';
  ctx.font = '9px Inter, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText('+' + yMax.toFixed(2), plotLeft - 4, plotTop);
  ctx.textBaseline = 'bottom';
  ctx.fillText('-' + yMax.toFixed(2), plotLeft - 4, plotTop + plotH);
  ctx.textBaseline = 'middle';
  ctx.fillText('0', plotLeft - 4, zeroY);

  // チャンネルラベル
  ctx.fillStyle = color;
  ctx.font = 'bold 10px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(label, plotLeft + 4, plotTop + 2);
}

/**
 * チャンネルごとに個別グラフを描画
 * @param {string} containerId - コンテナdivのID
 * @param {object} data - AudioData
 * @param {string} title
 */
export function plotData(containerId, data, title) {
  const container = document.getElementById(containerId);
  const { samples, channels } = data;

  container.innerHTML = '';

  // タイトル
  const titleEl = document.createElement('div');
  titleEl.style.cssText = 'font-size:11px; color:#374151; font-weight:500; margin-bottom:4px; padding-left:' + Y_LABEL_WIDTH + 'px;';
  titleEl.textContent = title;
  container.appendChild(titleEl);

  const yMax = getGlobalMaxAbs(samples);
  const limitPerCh = Math.floor(MAX_PLOT_POINTS / channels);

  for (let ch = 0; ch < channels; ch++) {
    const canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.height = CHART_HEIGHT + 'px';
    canvas.style.display = 'block';
    container.appendChild(canvas);

    const plotArr = downsampleForPlot(samples[ch], limitPerCh);
    // 描画はcanvasがDOMに入ってからでないとclientWidthが取れない
    requestAnimationFrame(() => {
      drawChannel(canvas, plotArr, yMax, COLORS[ch % COLORS.length], `ch${ch}`);
    });
  }

  // X軸ラベル（最下段のみ）
  const xLabel = document.createElement('div');
  xLabel.style.cssText = 'font-size:9px; color:#9ca3af; text-align:center; padding-left:' + Y_LABEL_WIDTH + 'px; margin-top:2px;';
  xLabel.textContent = 'sample (downsampled for plot)';
  container.appendChild(xLabel);
}
