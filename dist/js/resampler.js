/**
 * リサンプラー
 * Lanczos sinc補間 (a=3) + 移動平均アンチエイリアシングフィルタ
 */

/**
 * Lanczos sinc補間で1チャンネル分をリサンプリング
 * @param {Float32Array} input - 入力サンプル
 * @param {number} fsIn - 入力Fs
 * @param {number} fsOut - 出力Fs
 * @returns {Float32Array}
 */
function sincInterpolate(input, fsIn, fsOut) {
  const a = 3; // Lanczos窓の半径
  const ratio = fsIn / fsOut;
  const outLen = Math.round(input.length / ratio);
  const output = new Float32Array(outLen);

  for (let i = 0; i < outLen; i++) {
    const srcPos = i * ratio;
    const srcIdx = Math.floor(srcPos);
    let sum = 0;
    let weightSum = 0;

    const jStart = Math.max(0, srcIdx - a + 1);
    const jEnd = Math.min(input.length - 1, srcIdx + a);

    for (let j = jStart; j <= jEnd; j++) {
      const x = srcPos - j;
      let w;
      if (x === 0) {
        w = 1;
      } else if (Math.abs(x) >= a) {
        w = 0;
      } else {
        const px = Math.PI * x;
        w = (Math.sin(px) / px) * (Math.sin(px / a) / (px / a));
      }
      sum += input[j] * w;
      weightSum += w;
    }
    output[i] = weightSum !== 0 ? sum / weightSum : 0;
  }
  return output;
}

/**
 * 移動平均ローパスフィルタ（アンチエイリアシング用）
 * @param {Float32Array} input
 * @param {number} kernelSize - 奇数
 * @returns {Float32Array}
 */
function movingAverageFilter(input, kernelSize) {
  const out = new Float32Array(input.length);
  const half = Math.floor(kernelSize / 2);

  for (let i = 0; i < input.length; i++) {
    let sum = 0;
    let count = 0;
    const start = Math.max(0, i - half);
    const end = Math.min(input.length - 1, i + half);
    for (let j = start; j <= end; j++) {
      sum += input[j];
      count++;
    }
    out[i] = sum / count;
  }
  return out;
}

/**
 * マルチチャンネルリサンプリング
 * @param {import('./wavParser.js').AudioData} audioData
 * @param {number} targetFs
 * @returns {import('./wavParser.js').AudioData}
 */
export async function resample(audioData, targetFs) {
  const { samples, fs, channels } = audioData;

  // ダウンサンプル時はアンチエイリアシングフィルタを適用
  let filtered = samples;
  if (targetFs < fs) {
    const decimFactor = Math.floor(fs / targetFs);
    const kernelSize = Math.min(decimFactor * 2 + 1, 63);
    filtered = samples.map(ch => movingAverageFilter(ch, kernelSize));
  }

  // sinc補間でリサンプリング
  const result = filtered.map(ch => sincInterpolate(ch, fs, targetFs));
  const newLength = result[0].length;

  return { samples: result, fs: targetFs, channels, length: newLength };
}
