/**
 * WAVバイナリパーサー
 * PCM 8/16/24/32bit, IEEE float 32/64bit, マルチチャンネル対応
 */

/**
 * @typedef {Object} AudioData
 * @property {Float32Array[]} samples - チャンネルごとのサンプル配列
 * @property {number} fs - サンプリング周波数
 * @property {number} channels - チャンネル数
 * @property {number} length - サンプル数
 */

/**
 * ArrayBufferからWAVデータをパースする
 * @param {ArrayBuffer} arrayBuffer
 * @returns {AudioData}
 */
export function parseWav(arrayBuffer) {
  const view = new DataView(arrayBuffer);

  // RIFF ヘッダ確認
  const riff = readChunkId(view, 0);
  if (riff !== 'RIFF') throw new Error('RIFFヘッダが見つかりません');

  const wave = readChunkId(view, 8);
  if (wave !== 'WAVE') throw new Error('WAVEフォーマットではありません');

  // チャンク探索
  let offset = 12;
  let fmtFound = false;
  let dataOffset = 0;
  let dataSize = 0;
  let numChannels, sampleRate, bitsPerSample, audioFormat;

  while (offset < view.byteLength - 8) {
    const chunkId = readChunkId(view, offset);
    const chunkSize = view.getUint32(offset + 4, true);

    if (chunkId === 'fmt ') {
      audioFormat = view.getUint16(offset + 8, true);
      numChannels = view.getUint16(offset + 10, true);
      sampleRate = view.getUint32(offset + 12, true);
      bitsPerSample = view.getUint16(offset + 22, true);
      fmtFound = true;
    } else if (chunkId === 'data') {
      dataOffset = offset + 8;
      dataSize = chunkSize;
    }

    offset += 8 + chunkSize;
    if (offset % 2 !== 0) offset++; // パディング
  }

  if (!fmtFound) throw new Error('fmtチャンクが見つかりません');
  if (!dataOffset) throw new Error('dataチャンクが見つかりません');
  if (audioFormat !== 1 && audioFormat !== 3) {
    throw new Error(`未対応フォーマット (audioFormat=${audioFormat})`);
  }

  const bytesPerSample = bitsPerSample / 8;
  const numSamples = Math.floor(dataSize / (bytesPerSample * numChannels));

  // チャンネルごとにFloat32Arrayを作成
  const channels = [];
  for (let ch = 0; ch < numChannels; ch++) {
    channels.push(new Float32Array(numSamples));
  }

  let pos = dataOffset;
  for (let i = 0; i < numSamples; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      channels[ch][i] = readSample(view, pos, audioFormat, bitsPerSample);
      pos += bytesPerSample;
    }
  }

  return { samples: channels, fs: sampleRate, channels: numChannels, length: numSamples };
}

// ---- 内部ヘルパー ----

function readChunkId(view, offset) {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3)
  );
}

function readSample(view, pos, audioFormat, bitsPerSample) {
  if (audioFormat === 3) {
    // IEEE float
    if (bitsPerSample === 32) return view.getFloat32(pos, true);
    if (bitsPerSample === 64) return view.getFloat64(pos, true);
  }
  // PCM integer
  switch (bitsPerSample) {
    case 8:
      return (view.getUint8(pos) - 128) / 128;
    case 16:
      return view.getInt16(pos, true) / 32768;
    case 24: {
      let val = view.getUint8(pos) | (view.getUint8(pos + 1) << 8) | (view.getUint8(pos + 2) << 16);
      if (val & 0x800000) val |= ~0xFFFFFF;
      return val / 8388608;
    }
    case 32:
      return view.getInt32(pos, true) / 2147483648;
    default:
      return 0;
  }
}
