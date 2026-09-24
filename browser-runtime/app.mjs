import { policyFromBrowserArtifact, sha256Hex } from "./runtime.mjs";

const history = document.getElementById("history");
const run = document.getElementById("run");
const output = document.getElementById("output");
const modelImage = document.getElementById("model-image");
const form = document.getElementById("form");

async function pixelsFromPng(image, length) {
  await image.decode();
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
  if (canvas.width * canvas.height * 3 < length) throw new Error("PNG transport is shorter than its declared payload");
  const rgb = new Uint8Array(canvas.width * canvas.height * 3);
  for (let source = 0, target = 0; source < rgba.length; source += 4) {
    rgb[target++] = rgba[source];
    rgb[target++] = rgba[source + 1];
    rgb[target++] = rgba[source + 2];
  }
  return rgb.slice(0, length);
}

async function load() {
  const manifest = await fetch("./manifest.json", { cache: "no-store" }).then(response => {
    if (!response.ok) throw new Error(`runtime manifest failed: ${response.status}`);
    return response.json();
  });
  output.textContent = "Reading model image…";
  modelImage.src = manifest.png + `?v=${Date.now()}`;
  modelImage.hidden = false;
  const payload = await pixelsFromPng(modelImage, manifest.transport_bytes);
  output.textContent = "Verifying model image…";
  if (await sha256Hex(payload) !== manifest.transport_sha256) throw new Error("PNG transport digest mismatch");
  return policyFromBrowserArtifact(payload, {
    source_artifact_sha256: manifest.source_artifact_sha256,
    packed_integrity_sha256: manifest.packed_integrity_sha256,
  });
}

let policy;
try {
  policy = await load();
  output.textContent = "Ready";
  run.disabled = false;
  history.focus();
} catch (error) {
  output.textContent = `Load rejected: ${error.message}`;
}

form.addEventListener("submit", event => {
  event.preventDefault();
  if (!policy) return;
  const value = history.value.trim().toLowerCase();
  history.value = value;
  try {
    output.textContent = policy.predict(value);
  } catch (error) {
    output.textContent = `Run rejected: ${error.message}`;
  }
});
