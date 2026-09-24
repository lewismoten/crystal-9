const FORMAT = "crystal-9-browser-packed-int4-v1";
const MAGIC = "C9B1";
const encoder = new TextEncoder();

function hex(bytes) {
  return [...new Uint8Array(bytes)].map(value => value.toString(16).padStart(2, "0")).join("");
}

export async function sha256Hex(bytes) {
  return hex(await crypto.subtle.digest("SHA-256", bytes));
}

function utf8(bytes) {
  return new TextDecoder().decode(bytes);
}

export async function parseBrowserArtifact(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 8 || utf8(bytes.slice(0, 4)) !== MAGIC) throw new Error("not a Crystal-9 browser artifact");
  const headerLength = view.getUint32(4, false);
  if (headerLength < 2 || headerLength > 128000 || 8 + headerLength > bytes.byteLength) throw new Error("invalid artifact header length");
  const header = JSON.parse(utf8(bytes.slice(8, 8 + headerLength)));
  if (header.format !== FORMAT) throw new Error("unsupported Crystal-9 browser artifact format");
  const payload = bytes.slice(8 + headerLength);
  if (await sha256Hex(payload) !== header.payload_sha256) throw new Error("artifact payload digest mismatch");
  for (const [name, record] of Object.entries(header.tensors)) {
    for (const field of ["scales", "packed"]) {
      const range = record[field];
      if (!range || !Number.isInteger(range.offset) || !Number.isInteger(range.length) || range.offset < 0 || range.length < 0 || range.offset + range.length > payload.byteLength) {
        throw new Error(`invalid ${field} range for ${name}`);
      }
    }
  }
  return { header, payload };
}

function signedNibble(bytes, index) {
  const nibble = (bytes[index >> 1] >> ((index & 1) * 4)) & 15;
  return nibble >= 8 ? nibble - 16 : nibble;
}

function makeTensor(record, payload) {
  const scaleBytes = payload.slice(record.scales.offset, record.scales.offset + record.scales.length);
  const packed = payload.slice(record.packed.offset, record.packed.offset + record.packed.length);
  if (record.scale_dtype !== "float32-le" || scaleBytes.byteLength % 4) throw new Error("unsupported scale storage");
  const scales = new DataView(scaleBytes.buffer, scaleBytes.byteOffset, scaleBytes.byteLength);
  const values = record.shape.reduce((total, item) => total * item, 1);
  if (values !== record.count || packed.byteLength * 2 < values) throw new Error("invalid tensor shape/count");
  const groupWidth = record.scheme === "row" ? record.shape[1] : record.scheme === "group" ? record.group_size : values;
  if (!Number.isInteger(groupWidth) || groupWidth < 1 || values % groupWidth || scales.byteLength / 4 !== values / groupWidth) throw new Error("invalid tensor quantization layout");
  return {
    shape: record.shape,
    at(index) {
      if (!Number.isInteger(index) || index < 0 || index >= values) throw new Error("tensor index outside bounds");
      return signedNibble(packed, index) * scales.getFloat32(Math.floor(index / groupWidth) * 4, true) / 7;
    },
  };
}

function linear(input, weight, bias) {
  const [rows, columns] = weight.shape;
  if (input.length !== columns || bias.shape.length !== 1 || bias.shape[0] !== rows) throw new Error("linear tensor shape mismatch");
  const output = new Float64Array(rows);
  for (let row = 0; row < rows; row++) {
    let total = bias.at(row);
    for (let column = 0; column < columns; column++) total += weight.at(row * columns + column) * input[column];
    output[row] = total;
  }
  return output;
}

function softmax(values) {
  const maximum = Math.max(...values);
  const exponentials = values.map(value => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map(value => value / total);
}

function layerNorm(values, weight, bias, epsilon) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const denominator = Math.sqrt(variance + epsilon);
  return Float64Array.from(values, (value, index) => ((value - mean) / denominator) * weight.at(index) + bias.at(index));
}

function top2(values) {
  return values.map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value || b.index - a.index).slice(0, 2);
}

function isLiveHistory(history, tokenizer) {
  if (typeof history !== "string" || history.length > tokenizer.max_history_moves || [...history].some(symbol => !tokenizer.input_symbols.includes(symbol))) return false;
  const board = Array(9).fill(".");
  const wins = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8]];
  const winner = () => wins.some(([a,b,c]) => board[a] !== "." && board[a] === board[b] && board[a] === board[c]);
  for (let turn = 0; turn < history.length; turn++) {
    if (winner()) return false;
    const position = "abcdefghi".indexOf(history[turn]);
    if (board[position] !== ".") return false;
    board[position] = turn % 2 ? "O" : "X";
  }
  return !winner() && board.includes(".");
}

export class Crystal9BrowserPolicy {
  constructor(header, payload) {
    this.header = header;
    this.architecture = header.architecture;
    this.tokenizer = header.tokenizer;
    this.tensors = Object.fromEntries(Object.entries(header.tensors).map(([name, record]) => [name, makeTensor(record, payload)]));
    const { vocab_size: vocabSize, hidden_size: hiddenSize, experts, heads } = this.architecture;
    if (vocabSize !== this.tokenizer.tokens.length || hiddenSize % heads || experts !== 9 || this.tensors["embedding.weight"].shape[1] !== hiddenSize) throw new Error("unsupported Crystal-9 browser architecture");
  }

  predict(history) {
    if (!isLiveHistory(history, this.tokenizer)) return this.tokenizer.invalid_output;
    const ids = [this.tokenizer.tokens.indexOf("<bos>"), ...[...history].map(symbol => this.tokenizer.tokens.indexOf(symbol))];
    while (ids.length < 9) ids.push(0);
    const width = this.architecture.hidden_size;
    const steps = ids.length;
    const last = history.length;
    const embedding = this.tensors["embedding.weight"], position = this.tensors["position.weight"];
    const hidden = ids.map((id, step) => Float64Array.from({ length: width }, (_, i) => embedding.at(id * width + i) + position.at(step * width + i)));
    const inputWeight = this.tensors["attention.in_proj_weight"], inputBias = this.tensors["attention.in_proj_bias"];
    const qkv = hidden.map(vector => linear(vector, inputWeight, inputBias));
    const heads = this.architecture.heads, headWidth = width / heads;
    const attended = new Float64Array(width);
    for (let head = 0; head < heads; head++) {
      const query = qkv[last].slice(head * headWidth, (head + 1) * headWidth);
      const scores = [];
      for (let step = 0; step <= last; step++) {
        let dot = 0;
        for (let i = 0; i < headWidth; i++) dot += query[i] * qkv[step][width + head * headWidth + i];
        scores.push(dot / Math.sqrt(headWidth));
      }
      const weights = softmax(scores);
      for (let step = 0; step <= last; step++) for (let i = 0; i < headWidth; i++) attended[head * headWidth + i] += weights[step] * qkv[step][2 * width + head * headWidth + i];
    }
    let state = linear(attended, this.tensors["attention.out_proj.weight"], this.tensors["attention.out_proj.bias"]);
    state = layerNorm(state, this.tensors["norm.weight"], this.tensors["norm.bias"], this.architecture.norm_eps);
    const routing = softmax([...linear(state, this.tensors["router.weight"], this.tensors["router.bias"])]);
    const selected = top2(routing);
    const routed = new Float64Array(width);
    for (const { index, value: routeWeight } of selected) {
      let expert = linear(state, this.tensors[`experts.${index}.0.weight`], this.tensors[`experts.${index}.0.bias`]);
      expert = Float64Array.from(expert, value => value / (1 + Math.exp(-value)));
      expert = linear(expert, this.tensors[`experts.${index}.2.weight`], this.tensors[`experts.${index}.2.bias`]);
      for (let i = 0; i < width; i++) routed[i] += routeWeight * expert[i];
    }
    for (let i = 0; i < width; i++) state[i] += routed[i];
    const logits = linear(state, this.tensors["output.weight"], this.tensors["output.bias"]);
    let choice = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[choice]) choice = i;
    return this.tokenizer.tokens[choice];
  }
}

export async function policyFromBrowserArtifact(bytes, expectedProvenance = null) {
  const { header, payload } = await parseBrowserArtifact(bytes);
  if (expectedProvenance) {
    if (header.source_artifact_sha256 !== expectedProvenance.source_artifact_sha256) throw new Error("parent artifact digest mismatch");
    if (header.integrity_sha256 !== expectedProvenance.packed_integrity_sha256) throw new Error("packed artifact integrity digest mismatch");
  }
  return new Crystal9BrowserPolicy(header, payload);
}
