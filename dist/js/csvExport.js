/**
 * CSV生成 & ダウンロードモジュール
 * 出力仕様: 8列固定（ags, idx, ch0〜, 0埋め）、ヘッダなし、小数6桁
 */

/**
 * AudioDataからCSV文字列を生成
 * @param {import('./wavParser.js').AudioData} data
 * @returns {string}
 */
export function generateCsv(data) {
  const { samples, channels, length } = data;
  const lines = new Array(length);

  for (let i = 0; i < length; i++) {
    const row = ['ags', i];
    for (let ch = 0; ch < channels; ch++) {
      row.push(samples[ch][i].toFixed(6));
    }
    // 8列固定、残りは0埋め
    while (row.length < 8) row.push('0');
    lines[i] = row.join(',');
  }
  return lines.join('\n') + '\n';
}

/**
 * CSV文字列をBlobにしてダウンロード
 * @param {string} csvString
 * @param {string} filename
 */
export function downloadCsv(csvString, filename) {
  const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * プレビューテーブルのHTMLを生成
 * @param {import('./wavParser.js').AudioData} data
 * @returns {string} innerHTML用HTML文字列
 */
export function generatePreviewHtml(data) {
  const { samples, channels, length } = data;
  const rows = Math.min(5, length);

  let html = '<tr><th></th><th>idx</th>';
  for (let ch = 0; ch < channels; ch++) html += `<th>ch${ch}</th>`;
  for (let i = channels; i < 6; i++) html += '<th>-</th>';
  html += '</tr>';

  for (let i = 0; i < rows; i++) {
    html += `<tr><td>ags</td><td>${i}</td>`;
    for (let ch = 0; ch < channels; ch++) {
      html += `<td>${samples[ch][i].toFixed(6)}</td>`;
    }
    for (let j = channels; j < 6; j++) html += '<td>0</td>';
    html += '</tr>';
  }
  return html;
}
