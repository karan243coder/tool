/* PixelFree — fully automatic AI image toolkit. Everything runs in the browser. */
import { env, AutoModel, AutoProcessor, RawImage, pipeline }
  from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';

env.allowLocalModels = false;
env.backends.onnx.wasm.proxy = false;

/* ---- console noise filter: RMBG-1.4 "custom" model class aur ONNX EP warnings
   harmless hain, in par asli errors chhup jaate hain. Inhe dabate hain.      ---- */
const NOISE = [
  'Unknown model class "custom"',
  "Model type for 'custom' not found",
  'assuming encoder-only architecture',
  'powerPreference option is currently ignored',
  'VerifyEachNodeIsAssignedToAnEp',
  'Rerunning with verbose output',
  'Feature extractor type "undefined"',
  'assuming ImageFeatureExtractor',
];
for (const fn of ['warn', 'error']) {
  const orig = console[fn].bind(console);
  console[fn] = (...a) => {
    const t = a.map(x => (typeof x === 'string' ? x : x?.message || '')).join(' ');
    if (NOISE.some(n => t.includes(n))) return;      // known-harmless, skip
    orig(...a);
    if (fn === 'error' && t) log('⚠ ' + t.slice(0, 200), 'warn');
  };
}
window.addEventListener('unhandledrejection', e =>
  log('✖ ' + (e.reason?.message || e.reason), 'err'));

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
let files = [], outputs = [], current = 0, running = false;

/* ============ UI helpers ============ */
function log(m, cls = '') {
  const l = $('#log');
  l.innerHTML += `<div class="${cls}">${new Date().toLocaleTimeString()} · ${m}</div>`;
  l.scrollTop = 1e9;
}
function setProg(p, txt) {
  $('#bar').style.width = Math.round(p * 100) + '%';
  $('#progTxt').textContent = txt || '';
  $('#prog').classList.toggle('hidden', p <= 0 || p >= 1);
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ============ tabs ============ */
$$('.tab').forEach(t => t.onclick = () => {
  $$('.tab').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  $$('.view').forEach(v => v.classList.add('hidden'));
  $('#view-' + t.dataset.tab).classList.remove('hidden');
  if (t.dataset.tab === 'clean') loadToCanvas(current);
});

/* ============ input ============ */
const drop = $('#drop'), input = $('#fileInput');
drop.onclick = () => input.click();
input.onchange = e => addFiles([...e.target.files]);
['dragenter', 'dragover'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('hot'); }));
['dragleave', 'drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('hot'); }));
drop.addEventListener('drop', ev => addFiles([...ev.dataTransfer.files].filter(f => f.type.startsWith('image/'))));
addEventListener('paste', ev => {
  const imgs = [...(ev.clipboardData?.files || [])].filter(f => f.type.startsWith('image/'));
  if (imgs.length) addFiles(imgs);
});

async function addFiles(fs) {
  if (!fs.length) return;
  for (const f of fs) files.push({ name: f.name || `pasted-${Date.now()}.png`, url: URL.createObjectURL(f), file: f });
  renderThumbs();
  log(`📥 ${fs.length} file(s) added · total ${files.length}`);
  if ($('#autoRun').checked && !running) { await sleep(250); runAuto(); }
}
function renderThumbs() {
  $('#thumbs').innerHTML = files.map((f, i) =>
    `<div class="card ${i === current ? 'sel' : ''}" data-i="${i}"><img src="${f.url}"><div>${f.name}</div></div>`).join('');
  $$('#thumbs .card').forEach(c => c.onclick = () => { current = +c.dataset.i; renderThumbs(); loadToCanvas(current); });
  $('#clearIn').classList.toggle('hidden', !files.length);
}
$('#clearIn').onclick = () => { files = []; renderThumbs(); };

function showResults() {
  $('#count').textContent = outputs.length ? `(${outputs.length})` : '';
  $('#results').innerHTML = outputs.map((o, i) =>
    `<div class="card"><img src="${o.url}"><div>${o.name}</div>
     <div class="sz">${(o.blob.size / 1024).toFixed(0)} KB</div>
     <a href="${o.url}" download="${o.name}">⬇ download</a>
     ${o.link ? `<a href="${o.link}" target="_blank">🔗 link</a>` : ''}
     ${o.debug ? `<a href="${o.debug}" target="_blank">🐞 mask</a>` : ''}</div>`).join('');
  $('#outwrap').classList.toggle('hidden', !outputs.length);
}
/* Bade images (12MP+) browser WASM heap crash karte hain. Kaam se pehle
   safe size par utaar dete hain; final output isi size par save hota hai. */
const MAX_PIXELS = (() => {
  const mem = navigator.deviceMemory || 4;          // GB, Chrome only
  const budget = mem >= 8 ? 8e6 : mem >= 4 ? 6e6 : 3e6;
  return budget;                                     // 8MP / 6MP / 3MP
})();
function capCanvas(img) {
  const px = img.width * img.height;
  if (px <= MAX_PIXELS) return img;
  const f = Math.sqrt(MAX_PIXELS / px);
  const w = Math.round(img.width * f), h = Math.round(img.height * f);
  const c = canvasOf(w, h);
  const x = c.getContext('2d'); x.imageSmoothingQuality = 'high';
  x.drawImage(img, 0, 0, w, h);
  log(`   ↓ ${img.width}x${img.height} → ${w}x${h} (memory safe)`, 'warn');
  return c;
}

const loadImg = src => new Promise((res, rej) => {
  const i = new Image(); i.crossOrigin = 'anonymous';
  i.onload = () => res(i); i.onerror = () => rej(new Error('image decode failed')); i.src = src;
});
function pushOut(name, blob, ext) {
  const n = name.replace(/\.[^.]+$/, '') + '.' + ext;
  const o = { name: n, blob, url: URL.createObjectURL(blob) };
  outputs.push(o); return o;
}
const canvasOf = (w, h) => Object.assign(document.createElement('canvas'), { width: w, height: h });
const toBlob = (c, t = 'image/png', q = 0.95) => new Promise(r => c.toBlob(r, t, q));

/* ============ AI: background removal (RMBG-1.4) ============ */
let model, processor, modelLoading;
async function initBg() {
  if (model) return;
  if (modelLoading) return modelLoading;
  modelLoading = (async () => {
    const device = (navigator.gpu ? 'webgpu' : 'wasm');
    log(`⬇ AI model load ho raha hai (RMBG-1.4, device=${device}) — pehli baar ~45MB, phir cache…`);
    setProg(0.05, 'AI model download…');
    // RMBG-1.4 ka official pattern: model_type 'custom' -> transformers.js base class
    // se construct karta hai. Console warnings normal hain, output sahi aata hai.
    const mkBg = d => AutoModel.from_pretrained('briaai/RMBG-1.4', {
      config: { model_type: 'custom' }, device: d, dtype: 'fp32',
      progress_callback: p => { if (p.status === 'progress' && p.total) setProg(0.05 + 0.4 * (p.loaded / p.total), `model ${(p.progress || 0).toFixed(0)}%`); }
    });
    try {
      model = await mkBg(device);
    } catch (e) {
      if (device === 'webgpu') { log('⚠ WebGPU fail — WASM par switch kar rahe hain', 'warn'); model = await mkBg('wasm'); }
      else throw e;
    }
    processor = await AutoProcessor.from_pretrained('briaai/RMBG-1.4', {
      config: {
        do_normalize: true, do_pad: false, do_rescale: true, do_resize: true,
        image_mean: [0.5, 0.5, 0.5], image_std: [1, 1, 1],
        resample: 2, rescale_factor: 1 / 255, size: { width: 1024, height: 1024 }
      }
    });
    log('✅ AI model ready (ab offline bhi chalega)', 'ok');
    setProg(0);
  })();
  return modelLoading;
}
/** returns {canvas, maskRaw} with alpha applied */
async function removeBg(url, bgColor) {
  await initBg();
  const image = await RawImage.fromURL(url);
  const { pixel_values } = await processor(image);
  const { output } = await model({ input: pixel_values });
  const mask = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(image.width, image.height);
  const c = canvasOf(image.width, image.height), x = c.getContext('2d');
  const tmp = canvasOf(image.width, image.height), tx = tmp.getContext('2d');
  tx.drawImage(image.toCanvas(), 0, 0);
  const d = tx.getImageData(0, 0, c.width, c.height);
  for (let i = 0; i < mask.data.length; i++) d.data[4 * i + 3] = mask.data[i];
  tx.putImageData(d, 0, 0);
  if (bgColor) { x.fillStyle = bgColor; x.fillRect(0, 0, c.width, c.height); }
  x.drawImage(tmp, 0, 0);
  return c;
}

/* ============ AI: PERSON-ONLY segmentation (human parsing) ============
   Sirf insaan rakhta hai — bed, mirror, chair, furniture, props SAB delete.
   Model: Xenova/segformer_b2_clothes (18 human-part classes).            */
const PERSON_PARTS = {
  core:   [2, 4, 5, 6, 7, 11, 12, 13, 14, 15],      // hair, clothes, face, limbs
  extras: [1, 3, 8, 9, 10, 17],                     // hat, sunglasses, belt, shoes, scarf
  bag:    [16],                                     // bag/purse
};
let parser, parserLoading;
async function initParser() {
  if (parser) return parser;
  if (parserLoading) return parserLoading;
  parserLoading = (async () => {
    const device = navigator.gpu ? 'webgpu' : 'wasm';
    log(`⬇ Person-parsing model load (segformer_b2_clothes, ${device}) — pehli baar ~25MB…`);
    setProg(0.05, 'person model download…');
    const mkP = d => pipeline('image-segmentation', 'Xenova/segformer_b2_clothes', {
      device: d, dtype: d === 'webgpu' ? 'fp32' : 'q8',
      progress_callback: p => { if (p.status === 'progress' && p.total) setProg(0.05 + 0.4 * (p.loaded / p.total), `person model ${(p.progress || 0).toFixed(0)}%`); }
    });
    try {
      parser = await mkP(device);
    } catch (e) {
      if (device === 'webgpu') { log('⚠ WebGPU fail — WASM par switch kar rahe hain', 'warn'); parser = await mkP('wasm'); }
      else throw e;
    }
    log('✅ Person model ready', 'ok');
    setProg(0);
    return parser;
  })();
  return parserLoading;
}

/** Build a person-only alpha mask (Uint8ClampedArray w*h). */
async function personAlpha(url, w, h, opts) {
  await initParser();
  const out = await parser(url);
  const keep = new Set([
    ...PERSON_PARTS.core,
    ...(opts.keepAccessories ? PERSON_PARTS.extras : []),
    ...(opts.keepBag ? PERSON_PARTS.bag : []),
  ]);
  const keepNames = new Set([...keep].map(i => PERSON_LABELS[i]));
  const alpha = new Uint8ClampedArray(w * h);
  let hit = 0;
  // precompute row/col lookup once per mask size (avoids w*h divisions per segment)
  let cachedW = -1, cachedH = -1, colMap = null, rowMap = null;
  for (const seg of out) {
    if (!keepNames.has(seg.label)) continue;
    hit++;
    const m = seg.mask;                     // RawImage, 1 channel
    const sw = m.width, sh = m.height;
    if (sw !== cachedW || sh !== cachedH) {
      colMap = new Int32Array(w); rowMap = new Int32Array(h);
      for (let x = 0; x < w; x++) colMap[x] = Math.min(sw - 1, (x * sw / w) | 0);
      for (let y = 0; y < h; y++) rowMap[y] = Math.min(sh - 1, (y * sh / h) | 0);
      cachedW = sw; cachedH = sh;
    }
    const md = m.data;
    for (let y = 0; y < h; y++) {
      const base = rowMap[y] * sw, orow = y * w;
      for (let x = 0; x < w; x++) {
        if (md[base + colMap[x]] > 127) alpha[orow + x] = 255;
      }
    }
    m.data = null;                          // release each mask asap
  }
  out.length = 0;
  return { alpha, hit };
}
const PERSON_LABELS = ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt',
  'Pants', 'Dress', 'Belt', 'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg',
  'Left-arm', 'Right-arm', 'Bag', 'Scarf'];

/** feather + despeckle the binary alpha so edges look natural */
async function refineAlpha(alpha, w, h, feather) {
  await cvReady();
  // NOTE: Array.from(alpha) 4K image par ~100M-element JS array banata tha ->
  // "Array buffer allocation failed". Ab seedha Mat buffer me copy karte hain.
  const m = new cv.Mat(h, w, cv.CV_8U);
  m.data.set(alpha);
  const scale = Math.max(1, Math.round(Math.min(w, h) / 500));
  // remove small islands (mirror reflections, stray blobs)
  const contours = new cv.MatVector(), hier = new cv.Mat();
  const bin = m.clone();
  cv.findContours(bin, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  let best = 0, areas = [];
  for (let i = 0; i < contours.size(); i++) areas.push(cv.contourArea(contours.get(i)));
  best = Math.max(0, ...areas);
  const cleaned = cv.Mat.zeros(h, w, cv.CV_8U);
  for (let i = 0; i < contours.size(); i++) {
    if (areas[i] > best * 0.06) cv.drawContours(cleaned, contours, i, new cv.Scalar(255), -1);
  }
  contours.delete(); hier.delete(); bin.delete();
  // close holes then feather
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3 * scale, 3 * scale));
  cv.morphologyEx(cleaned, cleaned, cv.MORPH_CLOSE, k);
  if (feather > 0) {
    const b = 2 * Math.round(feather * scale) + 1;
    cv.GaussianBlur(cleaned, cleaned, new cv.Size(b, b), 0);
  }
  const res = new Uint8ClampedArray(cleaned.data);   // copy before delete
  m.delete(); cleaned.delete(); k.delete();
  return res;
}

/** Full person cutout: everything except the human is deleted. */
/* ============ HYBRID PERSON CUTOUT ============
   Problem: segformer 512px par chalta hai -> patli sleeve, dupatta, dark
   fabric, baal ke kinare chhoot jaate hain (adhoora cutout).
   Solution: dono models ko jodo —
     • segformer  = SEMANTIC "insaan kahan hai" (bed/mirror reject karta hai)
     • RMBG-1.4   = PRECISE edge/matting (kapde ka har detail pakadta hai)
   Final alpha = RMBG ka detail, LEKIN sirf person ke region ke andar.
   Isse object bhi hat jaate hain AUR clothes bhi poore aate hain.        */

/** person region ko fulao taaki chhooti hui clothing bhi ander aa jaye */
async function dilateRegion(alpha, w, h, growPct) {
  await cvReady();
  const m = new cv.Mat(h, w, cv.CV_8U); m.data.set(alpha);
  const px = Math.max(3, Math.round(Math.min(w, h) * growPct / 100));
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(px | 1, px | 1));
  cv.dilate(m, m, k);
  // holes bhar do (kapdo ke beech ke gaps)
  const k2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(15, 15));
  cv.morphologyEx(m, m, cv.MORPH_CLOSE, k2);
  const out = new Uint8ClampedArray(m.data);
  m.delete(); k.delete(); k2.delete();
  return out;
}

/** RMBG se full-detail alpha nikalo (matting quality) */
async function rmbgAlpha(url, w, h) {
  await initBg();
  const image = await RawImage.fromURL(url);
  const { pixel_values } = await processor(image);
  const { output } = await model({ input: pixel_values });
  const m = await RawImage.fromTensor(output[0].mul(255).to('uint8')).resize(w, h);
  const a = new Uint8ClampedArray(w * h);
  a.set(m.data.subarray(0, w * h));
  return a;
}

/** biggest connected blob rakho (mirror reflection / doosre objects hatao) */
async function keepMainBlobs(alpha, w, h, minFrac) {
  await cvReady();
  const m = new cv.Mat(h, w, cv.CV_8U); m.data.set(alpha);
  const bin = new cv.Mat();
  cv.threshold(m, bin, 8, 255, cv.THRESH_BINARY);
  const labels = new cv.Mat(), stats = new cv.Mat(), cent = new cv.Mat();
  const n = cv.connectedComponentsWithStats(bin, labels, stats, cent);
  let maxA = 0;
  for (let i = 1; i < n; i++) maxA = Math.max(maxA, stats.intAt(i, cv.CC_STAT_AREA));
  const keep = new Uint8Array(n);
  for (let i = 1; i < n; i++) if (stats.intAt(i, cv.CC_STAT_AREA) >= maxA * minFrac) keep[i] = 1;
  const out = new Uint8ClampedArray(w * h);
  const ld = labels.data32S;
  for (let i = 0; i < w * h; i++) { const l = ld[i]; if (l && keep[l]) out[i] = alpha[i]; }
  m.delete(); bin.delete(); labels.delete(); stats.delete(); cent.delete();
  return out;
}

/** edge ko smooth + slight erode taaki background ka halo na rahe */
async function polishAlpha(alpha, w, h, feather, shrink) {
  await cvReady();
  const m = new cv.Mat(h, w, cv.CV_8U); m.data.set(alpha);
  if (shrink > 0) {
    const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(shrink * 2 + 1, shrink * 2 + 1));
    cv.erode(m, m, k); k.delete();
  }
  if (feather > 0) {
    const b = 2 * feather + 1;
    cv.GaussianBlur(m, m, new cv.Size(b, b), 0);
  }
  const out = new Uint8ClampedArray(m.data);
  m.delete();
  return out;
}

/* v5: BLOB-KEEP approach.
   Pehle intersect (RMBG ∩ region) karte the -> jo clothing segformer ne miss ki
   wo CROP ho jaati thi. Ab intersect NAHI karte:
     1. RMBG ka poora alpha lo (saara detail, kuch nahi kata)
     2. Usko connected blobs me todo
     3. Sirf wo blobs rakho jo person region ko TOUCH karte hain
   -> bed/mirror (alag blob) hat jaate hain, kapda (person se juda) poora bachta hai. */
async function blobsTouchingPerson(rAlpha, pAlpha, w, h, opts) {
  await cvReady();
  const m = new cv.Mat(h, w, cv.CV_8U); m.data.set(rAlpha);
  const bin = new cv.Mat();
  cv.threshold(m, bin, 10, 255, cv.THRESH_BINARY);
  // thin gaps (necklace, strap, dupatta) ko jodo taaki ek hi blob bane
  const kc = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(9, 9));
  cv.morphologyEx(bin, bin, cv.MORPH_CLOSE, kc); kc.delete();

  const labels = new cv.Mat(), stats = new cv.Mat(), cent = new cv.Mat();
  const n = cv.connectedComponentsWithStats(bin, labels, stats, cent);
  const ld = labels.data32S;

  // har blob me kitne person-pixels hain
  const pCount = new Int32Array(n), bCount = new Int32Array(n);
  for (let i = 0; i < w * h; i++) {
    const l = ld[i]; if (!l) continue;
    bCount[l]++; if (pAlpha[i] > 127) pCount[l]++;
  }
  let maxA = 0;
  for (let i = 1; i < n; i++) maxA = Math.max(maxA, bCount[i]);

  const keep = new Uint8Array(n);
  let kn = 0;
  for (let i = 1; i < n; i++) {
    const overlap = pCount[i] / Math.max(1, bCount[i]);
    // blob rakho agar: person ka thoda bhi hissa usme hai, ya
    // wo bahut chhota nahi hai aur person blob se juda hua hai
    if (pCount[i] > 30 || overlap > 0.02) { keep[i] = 1; kn++; }
    else if (opts.keepStray && bCount[i] > maxA * 0.25) { keep[i] = 1; kn++; }
  }
  const out = new Uint8ClampedArray(w * h);
  for (let i = 0; i < w * h; i++) { const l = ld[i]; if (l && keep[l]) out[i] = rAlpha[i]; }
  m.delete(); bin.delete(); labels.delete(); stats.delete(); cent.delete();
  return { alpha: out, blobs: n - 1, kept: kn };
}

async function personCutout(url, opts) {
  const img = capCanvas(await loadImg(url));
  const w = img.width, h = img.height;
  const feedUrl = img.tagName === 'CANVAS'
    ? await new Promise(r => img.toBlob(b => r(URL.createObjectURL(b)), 'image/jpeg', 0.95))
    : url;

  let finalAlpha, note = '';

  if (opts.mode === 'semantic') {
    const { alpha, hit } = await personAlpha(feedUrl, w, h, opts);
    finalAlpha = hit ? await refineAlpha(alpha, w, h, opts.feather)
                     : await rmbgAlpha(feedUrl, w, h);
    note = 'semantic';
  } else if (opts.mode === 'detail') {
    // pure RMBG — max detail, koi person filtering nahi
    finalAlpha = await rmbgAlpha(feedUrl, w, h);
    finalAlpha = await keepMainBlobs(finalAlpha, w, h, 0.12);
    note = 'detail-only';
  } else {
    // SMART (default): RMBG detail + person-touching blob filter
    const rAlpha = await rmbgAlpha(feedUrl, w, h);
    const { alpha: pRaw, hit } = await personAlpha(feedUrl, w, h, opts);
    if (!hit) {
      log('   ⚠ person parser blank — pure RMBG detail use kiya', 'warn');
      finalAlpha = await keepMainBlobs(rAlpha, w, h, 0.12);
      note = 'rmbg-fallback';
    } else {
      const region = opts.grow > 0 ? await dilateRegion(pRaw, w, h, opts.grow) : pRaw;
      const r = await blobsTouchingPerson(rAlpha, region, w, h, opts);
      finalAlpha = r.alpha;
      // safety: agar filter ne 60%+ kha liya, RMBG hi de do
      let kept = 0, tot = 0;
      for (let i = 0; i < w * h; i++) { if (rAlpha[i] > 10) tot++; if (finalAlpha[i] > 10) kept++; }
      const keepRatio = tot ? kept / tot : 1;
      if (keepRatio < 0.40) {
        log(`   ⚠ filter ne bahut kaata (${(keepRatio*100)|0}%) — RMBG full use kiya`, 'warn');
        finalAlpha = rAlpha;
        note = 'rescued';
      } else {
        note = `${r.kept}/${r.blobs} blobs · ${(keepRatio*100)|0}% detail`;
      }
    }
    finalAlpha = await polishAlpha(finalAlpha, w, h, opts.feather, opts.shrink);
  }
  log(`   🎭 mask: ${note}`);

  const c = canvasOf(w, h), x = c.getContext('2d');
  const tmp = canvasOf(w, h), tx = tmp.getContext('2d');
  tx.drawImage(img, 0, 0);
  const d = tx.getImageData(0, 0, w, h), dd = d.data;
  for (let i = 0; i < w * h; i++) dd[4 * i + 3] = finalAlpha[i];
  tx.putImageData(d, 0, 0);
  if (opts.bgColor) { x.fillStyle = opts.bgColor; x.fillRect(0, 0, w, h); }
  x.drawImage(tmp, 0, 0);

  if (opts.debug) {
    const dbg = canvasOf(w, h), g = dbg.getContext('2d');
    g.drawImage(img, 0, 0); g.globalAlpha = 0.55; g.fillStyle = '#ff0050';
    const md = g.getImageData(0, 0, w, h);
    for (let i = 0; i < w * h; i++) if (finalAlpha[i] < 128) { md.data[4*i]=255; md.data[4*i+1]=0; md.data[4*i+2]=80; }
    g.putImageData(md, 0, 0);
    window.__lastDebug = dbg.toDataURL('image/jpeg', 0.8);
  }
  if (feedUrl !== url) URL.revokeObjectURL(feedUrl);
  return { canvas: c, fallback: false };
}

/** Router: person-only mode ya normal subject mode */
async function cutout(url) {
  const bgColor = $('#bgSolid').checked ? $('#bgColor').value : null;
  if ($('#personOnly').checked) {
    const r = await personCutout(url, {
      bgColor,
      keepAccessories: $('#keepAcc').checked,
      keepBag: $('#keepBag').checked,
      feather: +$('#feather').value,
      grow: +($('#grow')?.value || 4),
      shrink: +($('#shrink')?.value || 0),
      mode: $('#cutMode')?.value || 'smart',
      keepStray: false,
      debug: $('#dbgMask')?.checked,
    });
    return r.canvas;
  }
  return removeBg(url, bgColor);
}

/* ============ OpenCV ============ */
let cvPromise;
function cvReady() {
  if (cvPromise) return cvPromise;
  cvPromise = new Promise((res, rej) => {
    const t0 = Date.now();
    const t = setInterval(() => {
      if (window.cv && cv.Mat && cv.inpaint) { clearInterval(t); res(); }
      else if (Date.now() - t0 > 60000) { clearInterval(t); rej(new Error('OpenCV load timeout')); }
    }, 150);
  });
  return cvPromise;
}

/* Auto-detect text + watermark regions.
   Combines: (a) morphological-gradient text detection,
             (b) low-contrast translucent-watermark detection (local variance + edge density). */
/* FAST auto text/watermark detection.
   Speed tricks: detection chhoti copy par (max 900px), single morphology width,
   ROI-limited inpaint (poori image par nahi), Telea only. */
const DET_MAX = 900;

/* ---------- SKIN / FACE PROTECTION ----------
   v5 me detector aankh/hoth/nathuni ko "text" samajh kar mita deta tha.
   Ab pehle skin + person region nikaalte hain aur usko mask se HATA dete hain.
   Result: chehre par kabhi inpaint nahi hoga.                              */
function skinMask(smallRGBA) {
  const ycrcb = new cv.Mat(), rgb = new cv.Mat();
  cv.cvtColor(smallRGBA, rgb, cv.COLOR_RGBA2RGB);
  cv.cvtColor(rgb, ycrcb, cv.COLOR_RGB2YCrCb);
  // YCrCb skin range — har skin tone (fair se dark tak) cover karta hai
  const lo = new cv.Mat(ycrcb.rows, ycrcb.cols, ycrcb.type(), [0, 133, 77, 0]);
  const hi = new cv.Mat(ycrcb.rows, ycrcb.cols, ycrcb.type(), [255, 180, 127, 255]);
  const sk = new cv.Mat();
  cv.inRange(ycrcb, lo, hi, sk);
  // HSV se doosra check (mila kar false-negative kam)
  const hsv = new cv.Mat(), sk2 = new cv.Mat();
  cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
  const lo2 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, 30, 60, 0]);
  const hi2 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [25, 170, 255, 255]);
  cv.inRange(hsv, lo2, hi2, sk2);
  cv.bitwise_or(sk, sk2, sk);
  // skin ke aas-paas ka area bhi protect (aankh, bhaunh, hoth skin ke andar hain)
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(21, 21));
  cv.morphologyEx(sk, sk, cv.MORPH_CLOSE, k);
  cv.dilate(sk, sk, k);
  ycrcb.delete(); rgb.delete(); lo.delete(); hi.delete();
  hsv.delete(); sk2.delete(); lo2.delete(); hi2.delete(); k.delete();
  return sk;
}

/* text hone ka sakht test: asli text me stroke-width consistent hota hai,
   bahut saare chhote components hote hain, aur colour flat hota hai.
   Aankh/face features ye test fail karte hain.                            */
function looksLikeText(bwRoi, grayRoi) {
  const r = { ok: false, why: '' };
  // 1. component count — text me kai letters hote hain
  const c = new cv.MatVector(), hi = new cv.Mat();
  const tmp = bwRoi.clone();
  cv.findContours(tmp, c, hi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  const n = c.size();
  let hs = [], ws = [];
  for (let i = 0; i < n; i++) {
    const b = cv.boundingRect(c.get(i));
    if (b.height > 2) { hs.push(b.height); ws.push(b.width); }
  }
  c.delete(); hi.delete(); tmp.delete();
  if (hs.length < 2) return r;                    // ek hi blob = text nahi
  // 2. letter heights similar hone chahiye (text baseline)
  const mean = hs.reduce((x, y) => x + y, 0) / hs.length;
  const varc = Math.sqrt(hs.reduce((x, y) => x + (y - mean) ** 2, 0) / hs.length) / mean;
  if (varc > 0.55) return r;                      // heights bikhre = natural image
  // 3. stroke width consistency (distance transform)
  const dist = new cv.Mat();
  cv.distanceTransform(bwRoi, dist, cv.DIST_L2, 3);
  const md = new cv.Mat(), sd = new cv.Mat();
  cv.meanStdDev(dist, md, sd, bwRoi);
  const m0 = md.doubleAt(0, 0), s0 = sd.doubleAt(0, 0);
  dist.delete(); md.delete(); sd.delete();
  if (m0 < 0.5 || s0 / m0 > 0.75) return r;       // stroke bikhra = text nahi
  r.ok = true; return r;
}

function autoMask(src, sens) {
  const t0 = performance.now();
  const f = Math.min(1, DET_MAX / Math.max(src.cols, src.rows));
  const small = new cv.Mat();
  if (f < 1) cv.resize(src, small, new cv.Size(Math.round(src.cols * f), Math.round(src.rows * f)), 0, 0, cv.INTER_AREA);
  else src.copyTo(small);

  const gray = new cv.Mat(), grad = new cv.Mat(), bw = new cv.Mat(), conn = new cv.Mat();
  cv.cvtColor(small, gray, cv.COLOR_RGBA2GRAY);
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
  cv.morphologyEx(gray, grad, cv.MORPH_GRADIENT, k);
  cv.threshold(grad, bw, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);

  // ---- protection zone banao ----
  const protect = $('#protectFace') ? $('#protectFace').checked : true;
  let prot = null;
  if (protect) {
    prot = skinMask(small);
    if (window.__personProt) {            // person mask bhi mila to use karo
      const pm = new cv.Mat(small.rows, small.cols, cv.CV_8U);
      pm.data.set(window.__personProt);
      cv.bitwise_or(prot, pm, prot); pm.delete();
    }
  }

  const smallMask = cv.Mat.zeros(small.rows, small.cols, cv.CV_8U);
  const area = small.rows * small.cols;
  const boxes = [];
  let rejFace = 0, rejText = 0;

  const k2 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(13 + sens, 3));
  cv.morphologyEx(bw, conn, cv.MORPH_CLOSE, k2);
  const contours = new cv.MatVector(), hier = new cv.Mat();
  cv.findContours(conn, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  for (let i = 0; i < contours.size(); i++) {
    const r = cv.boundingRect(contours.get(i));
    const ar = r.width / r.height, a = r.width * r.height;
    if (a < 60 || a > area * 0.30) continue;
    if (ar < 1.4 || ar > 40) continue;                 // text chaura hota hai
    if (r.height < 6 || r.height > small.rows * 0.22) continue;

    // --- FACE/SKIN GUARD ---
    if (prot) {
      const pr = prot.roi(r);
      const skinFrac = cv.countNonZero(pr) / a; pr.delete();
      if (skinFrac > 0.18) { rejFace++; continue; }    // chehre/skin par hai -> chhodo
    }

    const roi = bw.roi(r);
    const dense = cv.countNonZero(roi) / a;
    if (dense < 0.16 + (10 - sens) * 0.015 || dense > 0.92) { roi.delete(); continue; }

    // --- STRICT TEXT TEST ---
    const groi = gray.roi(r);
    const t = looksLikeText(roi, groi);
    roi.delete(); groi.delete();
    if (!t.ok) { rejText++; continue; }

    const p = 3;
    const rr = new cv.Rect(Math.max(0, r.x - p), Math.max(0, r.y - p),
      Math.min(small.cols - Math.max(0, r.x - p), r.width + 2 * p),
      Math.min(small.rows - Math.max(0, r.y - p), r.height + 2 * p));
    cv.rectangle(smallMask, new cv.Point(rr.x, rr.y), new cv.Point(rr.x + rr.width, rr.y + rr.height), new cv.Scalar(255), -1);
    boxes.push(rr);
  }
  contours.delete(); hier.delete(); k2.delete();

  // translucent watermark pass — sirf high sensitivity par, aur skin ke bahar
  if (sens >= 8) {
    const blur = new cv.Mat(), diff = new cv.Mat(), th2 = new cv.Mat(), d2 = new cv.Mat();
    cv.GaussianBlur(gray, blur, new cv.Size(0, 0), 3);
    cv.absdiff(gray, blur, diff);
    cv.threshold(diff, th2, 10, 255, cv.THRESH_BINARY);
    const k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
    cv.morphologyEx(th2, d2, cv.MORPH_CLOSE, k3);
    const c2 = new cv.MatVector(), h2 = new cv.Mat();
    cv.findContours(d2, c2, h2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    for (let i = 0; i < c2.size(); i++) {
      const r = cv.boundingRect(c2.get(i));
      const a = r.width * r.height;
      if (a < 400 || a > area * 0.15) continue;
      if (prot) { const pr = prot.roi(r); const sf = cv.countNonZero(pr) / a; pr.delete(); if (sf > 0.12) { rejFace++; continue; } }
      const roi = d2.roi(r); const dense = cv.countNonZero(roi) / a; roi.delete();
      if (dense > 0.30) {
        cv.rectangle(smallMask, new cv.Point(r.x, r.y), new cv.Point(r.x + r.width, r.y + r.height), new cv.Scalar(255), -1);
        boxes.push(r);
      }
    }
    blur.delete(); diff.delete(); th2.delete(); d2.delete(); c2.delete(); h2.delete(); k3.delete();
  }

  // final: protection zone ko mask se subtract karo (double safety)
  if (prot) {
    const inv = new cv.Mat();
    cv.bitwise_not(prot, inv);
    cv.bitwise_and(smallMask, inv, smallMask);
    inv.delete();
  }

  const kd = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
  cv.dilate(smallMask, smallMask, kd); kd.delete();

  const cap = +($('#cleanCap')?.value || 25) / 100;
  const frac = cv.countNonZero(smallMask) / area;
  if (frac > cap) {
    log(`   ⚠ detector ne ${(frac*100).toFixed(0)}% maanga (cap ${(cap*100)|0}%) — skip`, 'warn');
    smallMask.setTo(new cv.Scalar(0)); boxes.length = 0;
  }
  if (rejFace || rejText) log(`   🛡 rejected: ${rejFace} face/skin, ${rejText} not-text`);

  const mask = new cv.Mat();
  if (f < 1) cv.resize(smallMask, mask, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_NEAREST);
  else smallMask.copyTo(mask);

  const inv2 = f < 1 ? 1 / f : 1;
  const fullBoxes = boxes.map(b => ({
    x: Math.max(0, Math.floor(b.x * inv2) - 8), y: Math.max(0, Math.floor(b.y * inv2) - 8),
    w: Math.ceil(b.width * inv2) + 16, h: Math.ceil(b.height * inv2) + 16,
  })).map(b => ({ ...b, w: Math.min(b.w, src.cols - b.x), h: Math.min(b.h, src.rows - b.y) }))
    .filter(b => b.w > 1 && b.h > 1);

  small.delete(); gray.delete(); grad.delete(); bw.delete(); conn.delete(); smallMask.delete(); k.delete();
  if (prot) prot.delete();
  mask.__boxes = fullBoxes;
  mask.__ms = Math.round(performance.now() - t0);
  mask.__count = fullBoxes.length;
  return mask;
}

/** merge overlapping boxes so we inpaint fewer, bigger ROIs */
function mergeBoxes(boxes, W, H) {
  const out = [];
  for (const b of boxes) {
    let merged = false;
    for (const o of out) {
      const ix = Math.max(b.x, o.x), iy = Math.max(b.y, o.y);
      const ax = Math.min(b.x + b.w, o.x + o.w), ay = Math.min(b.y + b.h, o.y + o.h);
      if (ax > ix - 12 && ay > iy - 12) {
        const nx = Math.min(b.x, o.x), ny = Math.min(b.y, o.y);
        o.w = Math.min(W - nx, Math.max(b.x + b.w, o.x + o.w) - nx);
        o.h = Math.min(H - ny, Math.max(b.y + b.h, o.y + o.h) - ny);
        o.x = nx; o.y = ny; merged = true; break;
      }
    }
    if (!merged) out.push({ ...b });
  }
  return out;
}

function inpaintWith(src, mask, quality) {
  const rgb = new cv.Mat(); cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
  const radius = quality === 'best' ? 5 : 3;
  let boxes = mask.__boxes;
  if (!boxes || !boxes.length) {
    const r = cv.boundingRect(mask);
    if (!r.width) return rgb;
    boxes = [{ x: Math.max(0, r.x - 8), y: Math.max(0, r.y - 8),
               w: Math.min(src.cols, r.width + 16), h: Math.min(src.rows, r.height + 16) }];
  }
  boxes = mergeBoxes(boxes, src.cols, src.rows);
  const total = boxes.reduce((a, b) => a + b.w * b.h, 0);
  if (total > 0.6 * src.cols * src.rows || boxes.length > 60) {
    const out = new cv.Mat();
    cv.inpaint(rgb, mask, out, radius, cv.INPAINT_TELEA);
    rgb.delete(); return out;
  }
  for (const b of boxes) {
    const rect = new cv.Rect(b.x, b.y, b.w, b.h);
    const patch = rgb.roi(rect), mpatch = mask.roi(rect);
    if (cv.countNonZero(mpatch) === 0) { patch.delete(); mpatch.delete(); continue; }
    const pc = patch.clone(), mc2 = mpatch.clone(), res = new cv.Mat();
    cv.inpaint(pc, mc2, res, radius, cv.INPAINT_TELEA);
    if (quality === 'best') {
      const res2 = new cv.Mat(), blend = new cv.Mat();
      cv.inpaint(pc, mc2, res2, radius, cv.INPAINT_NS);
      cv.addWeighted(res, 0.5, res2, 0.5, 0, blend);
      blend.copyTo(patch); res2.delete(); blend.delete();
    } else res.copyTo(patch);
    pc.delete(); mc2.delete(); res.delete(); patch.delete(); mpatch.delete();
  }
  return rgb;
}

/* ================= OCR TEXT DETECTION (Tesseract.js) =================
   Guess-work band. Ab asli OCR se word-level boxes milte hain + confidence.
   Jo cheez OCR ko text nahi lagti, wo chhui hi nahi jaati. Aankh, chehra,
   pattern — kuch bhi galti se detect nahi hoga.                          */
let ocrWorker, ocrLoading;
async function initOCR() {
  if (ocrWorker) return ocrWorker;
  if (ocrLoading) return ocrLoading;
  ocrLoading = (async () => {
    if (!window.Tesseract) throw new Error('Tesseract.js load nahi hua — internet check karo');
    log('⬇ OCR engine load ho raha hai (~15MB, ek hi baar)…');
    setProg(0.1, 'OCR engine…');
    ocrWorker = await Tesseract.createWorker('eng', 1, {
      logger: m => { if (m.status === 'loading tesseract core' || m.status === 'loading language traineddata')
        setProg(0.1 + 0.5 * (m.progress || 0), m.status); },
    });
    log('✅ OCR ready (cached)', 'ok');
    setProg(0);
    return ocrWorker;
  })();
  return ocrLoading;
}

/** OCR se word boxes nikalo. Returns [{x,y,w,h,text,conf}] full-res coords me */
async function ocrBoxes(canvas, minConf) {
  await initOCR();
  // OCR chhoti image par bhi theek chalta hai; 1600px cap = speed
  const cap = 1600;
  const f = Math.min(1, cap / Math.max(canvas.width, canvas.height));
  let feed = canvas;
  if (f < 1) {
    feed = canvasOf(Math.round(canvas.width * f), Math.round(canvas.height * f));
    const fx = feed.getContext('2d');
    fx.imageSmoothingQuality = 'high';
    fx.drawImage(canvas, 0, 0, feed.width, feed.height);
  }
  const { data } = await ocrWorker.recognize(feed);
  const inv = f < 1 ? 1 / f : 1;
  const out = [];
  for (const w of (data.words || [])) {
    const t = (w.text || '').trim();
    if (!t) continue;
    if (w.confidence < minConf) continue;
    const b = w.bbox;
    out.push({
      x: Math.round(b.x0 * inv), y: Math.round(b.y0 * inv),
      w: Math.round((b.x1 - b.x0) * inv), h: Math.round((b.y1 - b.y0) * inv),
      text: t, conf: Math.round(w.confidence),
    });
  }
  return out;
}

/** Box ke andar sirf TEXT STROKE pixels ka mask (poora rectangle nahi).
    Isse text ke aas-paas ka asli photo bilkul safe rehta hai.            */
function strokeMaskInBox(src, b, pad) {
  const x = Math.max(0, b.x - pad), y = Math.max(0, b.y - pad);
  const w = Math.min(src.cols - x, b.w + 2 * pad), h = Math.min(src.rows - y, b.h + 2 * pad);
  if (w < 2 || h < 2) return null;
  const rect = new cv.Rect(x, y, w, h);
  const roi = src.roi(rect);
  const g = new cv.Mat(), bw = new cv.Mat();
  cv.cvtColor(roi, g, cv.COLOR_RGBA2GRAY);
  // Otsu dono direction me: text light-on-dark ya dark-on-light ho sakta hai
  const t1 = new cv.Mat(), t2 = new cv.Mat();
  cv.threshold(g, t1, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
  cv.threshold(g, t2, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU);
  // jisme kam pixels hain wahi text hai (text background se chhota hota hai)
  const n1 = cv.countNonZero(t1), n2 = cv.countNonZero(t2);
  (n1 <= n2 ? t1 : t2).copyTo(bw);
  // stroke ko thoda mota karo taaki anti-aliased kinare bhi cover ho
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
  cv.dilate(bw, bw, k);
  roi.delete(); g.delete(); t1.delete(); t2.delete(); k.delete();
  return { mat: bw, rect };
}

/** OCR-based mask: sirf text strokes, aur sirf un words par jo filter paas karein */
async function ocrMask(src, canvas, opts) {
  const t0 = performance.now();
  const boxes = await ocrBoxes(canvas, opts.minConf);
  const mask = cv.Mat.zeros(src.rows, src.cols, cv.CV_8U);
  const area = src.rows * src.cols;
  const used = [];
  let skipSkin = 0, skipSize = 0;

  let prot = null;
  if (opts.protectFace) {
    const small = new cv.Mat();
    const f = Math.min(1, 900 / Math.max(src.cols, src.rows));
    if (f < 1) cv.resize(src, small, new cv.Size(Math.round(src.cols*f), Math.round(src.rows*f)), 0,0, cv.INTER_AREA);
    else src.copyTo(small);
    const sm = skinMask(small);
    prot = new cv.Mat();
    cv.resize(sm, prot, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_NEAREST);
    sm.delete(); small.delete();
  }

  for (const b of boxes) {
    const ba = b.w * b.h;
    if (ba < 24 || ba > area * 0.4) { skipSize++; continue; }
    if (prot) {
      const r = new cv.Rect(Math.max(0,b.x), Math.max(0,b.y),
        Math.min(src.cols-Math.max(0,b.x), b.w), Math.min(src.rows-Math.max(0,b.y), b.h));
      if (r.width > 1 && r.height > 1) {
        const pr = prot.roi(r); const sf = cv.countNonZero(pr) / (r.width*r.height); pr.delete();
        if (sf > 0.25) { skipSkin++; continue; }
      }
    }
    const pad = Math.max(2, Math.round(b.h * 0.15));
    const sm = strokeMaskInBox(src, b, pad);
    if (!sm) continue;
    const dstRoi = mask.roi(sm.rect);
    cv.bitwise_or(dstRoi, sm.mat, dstRoi);
    dstRoi.delete(); sm.mat.delete();
    used.push({ x: sm.rect.x, y: sm.rect.y, w: sm.rect.width, h: sm.rect.height,
                text: b.text, conf: b.conf });
  }
  if (prot) {
    const inv = new cv.Mat(); cv.bitwise_not(prot, inv);
    cv.bitwise_and(mask, inv, mask); inv.delete(); prot.delete();
  }
  mask.__boxes = used;
  mask.__count = used.length;
  mask.__words = used.map(u => u.text);
  mask.__ms = Math.round(performance.now() - t0);
  mask.__skipped = { skin: skipSkin, size: skipSize };
  return mask;
}

/** PIXEL-EXACT compose: sirf mask>0 wale pixels replace, baaki byte-for-byte original.
    Ye guarantee hai ki photo ka koi aur pixel kabhi nahi badlega.          */
function composeExact(origCanvas, inpaintedMat, mask) {
  const w = origCanvas.width, h = origCanvas.height;
  const out = canvasOf(w, h), ox = out.getContext('2d');
  ox.drawImage(origCanvas, 0, 0);                    // 100% original
  const od = ox.getImageData(0, 0, w, h);

  const tmp = canvasOf(w, h);
  cv.imshow(tmp, inpaintedMat);
  const id = tmp.getContext('2d').getImageData(0, 0, w, h);

  const md = mask.data;                               // CV_8U, w*h
  let changed = 0;
  for (let i = 0; i < w * h; i++) {
    const m = md[i];
    if (m === 0) continue;                            // untouched
    changed++;
    const o = 4 * i;
    if (m === 255) {
      od.data[o] = id.data[o]; od.data[o+1] = id.data[o+1]; od.data[o+2] = id.data[o+2];
    } else {                                          // feathered edge -> blend
      const a = m / 255, ia = 1 - a;
      od.data[o]   = id.data[o]   * a + od.data[o]   * ia;
      od.data[o+1] = id.data[o+1] * a + od.data[o+1] * ia;
      od.data[o+2] = id.data[o+2] * a + od.data[o+2] * ia;
    }
  }
  ox.putImageData(od, 0, 0);
  out.__changed = changed;
  return out;
}

async function cleanImage(url, sens, brushMask, opts_preview) {
  await cvReady();
  const t0 = performance.now();
  const img = capCanvas(await loadImg(url));
  const c = canvasOf(img.width, img.height);
  c.getContext('2d').drawImage(img, 0, 0);
  const src = cv.imread(c);
  const quality = $('#cleanQ') ? $('#cleanQ').value : 'fast';
  const method = $('#detMethod') ? $('#detMethod').value : 'ocr';

  let mask;
  if (brushMask) {
    mask = brushMask(src);
  } else if (method === 'ocr') {
    mask = await ocrMask(src, c, {
      minConf: +($('#ocrConf')?.value || 55),
      protectFace: $('#protectFace') ? $('#protectFace').checked : true,
    });
  } else {
    mask = autoMask(src, sens);
  }

  // safety cap
  const covered = cv.countNonZero(mask);
  const cap = +($('#cleanCap')?.value || 25) / 100;
  if (covered / (src.rows * src.cols) > cap) {
    log(`   ⚠ mask ${(100*covered/(src.rows*src.cols)).toFixed(0)}% > cap — skip (kuch nahi badla)`, 'warn');
    mask.setTo(new cv.Scalar(0));
  }
  const finalCovered = cv.countNonZero(mask);

  // ---- preview ----
  if (opts_preview) {
    const pv = canvasOf(src.cols, src.rows);
    pv.getContext('2d').drawImage(c, 0, 0);
    const g = pv.getContext('2d');
    const im = g.getImageData(0, 0, src.cols, src.rows);
    const md = mask.data;
    for (let i = 0; i < src.cols * src.rows; i++) {
      if (md[i] > 0) { const o = 4*i;
        im.data[o] = 255; im.data[o+1] = 40; im.data[o+2] = 90; }
    }
    g.putImageData(im, 0, 0);
    // boxes bhi draw karo
    g.strokeStyle = '#3fb950'; g.lineWidth = Math.max(1, src.cols/500); g.font = `${Math.max(11, src.cols/70)}px monospace`;
    g.fillStyle = '#3fb950';
    for (const b of (mask.__boxes || [])) {
      g.strokeRect(b.x, b.y, b.w, b.h);
      if (b.text) g.fillText(`${b.text} ${b.conf||''}`, b.x, Math.max(10, b.y - 3));
    }
    const words = mask.__words || [];
    const pct = (100 * finalCovered / (src.rows * src.cols)).toFixed(2);
    const cnt = mask.__count || 0;
    const sk = mask.__skipped;
    src.delete(); mask.delete();
    return { canvas: pv, pct, covered: finalCovered, ms: Math.round(performance.now()-t0),
             preview: true, boxes: cnt, words, skipped: sk };
  }

  // ---- apply: pixel-exact ----
  let outCanvas = c, changed = 0;
  if (finalCovered > 0) {
    // feather mask edges thoda taaki seam na dikhe
    const soft = new cv.Mat();
    mask.copyTo(soft);
    cv.GaussianBlur(soft, soft, new cv.Size(3, 3), 0);
    const dst = inpaintWith(src, mask, quality);
    outCanvas = composeExact(c, dst, soft);
    changed = outCanvas.__changed || 0;
    dst.delete(); soft.delete();
  }
  const pct = (100 * finalCovered / (src.rows * src.cols)).toFixed(2);
  const boxes = mask.__count || 0;
  const words = mask.__words || [];
  const ms = Math.round(performance.now() - t0);
  src.delete(); mask.delete();
  return { canvas: outCanvas, pct, covered: finalCovered, ms, boxes, words, changed };
}

/* ============ convert / resize ============ */
async function convertCanvas(srcCanvasOrImg, fmt, q, maxw, whitebg) {
  let w = srcCanvasOrImg.width, h = srcCanvasOrImg.height;
  if (maxw && w > maxw) { h = Math.round(h * maxw / w); w = maxw; }
  const c = canvasOf(w, h), x = c.getContext('2d');
  x.imageSmoothingQuality = 'high';
  if (fmt === 'image/jpeg' || whitebg) { x.fillStyle = '#fff'; x.fillRect(0, 0, w, h); }
  x.drawImage(srcCanvasOrImg, 0, 0, w, h);
  return toBlob(c, fmt, q);
}
const extOf = f => ({ 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp' }[f] || 'png');


/* ============ VIDEO BACKGROUND REMOVAL ============
   Har frame par person cutout -> WebM (alpha) ya solid/greenscreen background.
   MediaRecorder + canvas stream. Fully automatic.                          */
let vidFile = null, vidAbort = false;

$('#vidInput').onchange = e => {
  const f = e.target.files[0]; if (!f) return;
  vidFile = f;
  const v = $('#srcVideo');
  v.src = URL.createObjectURL(f);
  $('#vidStage').classList.remove('hidden');
  v.onloadedmetadata = () => {
    log(`🎬 ${f.name} · ${v.videoWidth}x${v.videoHeight} · ${v.duration.toFixed(1)}s`);
    $('#vidMeta').textContent = `${v.videoWidth}×${v.videoHeight} · ${v.duration.toFixed(1)}s`;
  };
};
$('#vidDrop').onclick = () => $('#vidInput').click();
['dragenter','dragover'].forEach(e=>$('#vidDrop').addEventListener(e,ev=>{ev.preventDefault();$('#vidDrop').classList.add('hot');}));
['dragleave','drop'].forEach(e=>$('#vidDrop').addEventListener(e,ev=>{ev.preventDefault();$('#vidDrop').classList.remove('hot');}));
$('#vidDrop').addEventListener('drop', ev => {
  const f=[...ev.dataTransfer.files].find(x=>x.type.startsWith('video/'));
  if(f){ $('#vidInput').files=ev.dataTransfer.files; $('#vidInput').onchange({target:{files:[f]}}); }
});
$('#vidStop').onclick = () => { vidAbort = true; log('⏹ stop requested', 'warn'); };

function pickMime(alpha) {
  const opts = alpha
    ? ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
    : ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/mp4;codecs=h264', 'video/webm'];
  return opts.find(m => MediaRecorder.isTypeSupported(m)) || '';
}

$('#runVideo').onclick = async () => {
  if (!vidFile) return alert('Pehle video select karo');
  if (running) return;
  running = true; vidAbort = false;
  $$('.go').forEach(b => b.disabled = true); $('#vidStop').disabled = false;

  const v = $('#srcVideo');
  const bgMode = $('#vidBg').value;           // alpha | green | color | blur
  const bgCol  = $('#vidColor').value;
  const fps    = +$('#vidFps').value;
  const maxW   = +$('#vidW').value;
  const every  = +$('#vidSkip').value;        // mask reuse (speed)

  try {
    await cvReady();
    if ($('#personOnly').checked) {
      const cm = $('#cutMode')?.value || 'smart';
      if (cm !== 'detail') await initParser();
      if (cm !== 'semantic') await initBg();
    } else await initBg();

    const scale = maxW ? Math.min(1, maxW / v.videoWidth) : 1;
    const W = Math.round(v.videoWidth * scale) & ~1;
    const H = Math.round(v.videoHeight * scale) & ~1;
    const out = canvasOf(W, H), ox = out.getContext('2d');
    const grab = canvasOf(W, H), gx = grab.getContext('2d');
    $('#vidPreview').width = W; $('#vidPreview').height = H;
    const px = $('#vidPreview').getContext('2d');

    const alpha = bgMode === 'alpha';
    const mime = pickMime(alpha);
    log(`🎬 encoding ${W}x${H} @${fps}fps · ${mime || 'default'} · bg=${bgMode}`);

    const stream = out.captureStream(fps);
    // keep original audio if user wants
    let audioTrack = null;
    if ($('#vidAudio').checked && v.captureStream) {
      try { const s = v.captureStream(); audioTrack = s.getAudioTracks()[0]; if (audioTrack) stream.addTrack(audioTrack); } catch (e) {}
    }
    const chunks = [];
    const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: +$('#vidBr').value * 1e6 });
    rec.ondataavailable = e => e.data.size && chunks.push(e.data);
    const done = new Promise(r => rec.onstop = r);
    rec.start();

    const dur = v.duration;
    const step = 1 / fps;
    let lastAlpha = null, idx = 0, t0 = performance.now();

    v.pause(); v.muted = !$('#vidAudio').checked;

    for (let t = 0; t < dur && !vidAbort; t += step, idx++) {
      // seek precisely
      await new Promise(res => { v.onseeked = res; v.currentTime = Math.min(t, dur - 0.001); });
      gx.drawImage(v, 0, 0, W, H);

      // compute mask (reuse every N frames for speed)
      if (idx % every === 0 || !lastAlpha) {
        const url = grab.toDataURL('image/jpeg', 0.9);
        if ($('#personOnly').checked) {
          const cc = await personCutout(url, {
            bgColor: null,
            keepAccessories: $('#keepAcc').checked, keepBag: $('#keepBag').checked,
            feather: +$('#feather').value, grow: +($('#grow')?.value || 4),
            shrink: +($('#shrink')?.value || 0), mode: $('#cutMode')?.value || 'smart' });
          const dd = cc.canvas.getContext('2d').getImageData(0, 0, W, H);
          const a2 = new Uint8ClampedArray(W * H);
          for (let i = 0; i < W * H; i++) a2[i] = dd.data[4 * i + 3];
          lastAlpha = a2;
        } else {
          const c = await removeBg(url, null);
          const d = c.getContext('2d').getImageData(0, 0, W, H);
          const a = new Uint8ClampedArray(W * H);
          for (let i = 0; i < W * H; i++) a[i] = d.data[4 * i + 3];
          lastAlpha = a;
        }
      }

      // compose
      ox.clearRect(0, 0, W, H);
      if (bgMode === 'green') { ox.fillStyle = '#00b140'; ox.fillRect(0, 0, W, H); }
      else if (bgMode === 'color') { ox.fillStyle = bgCol; ox.fillRect(0, 0, W, H); }
      else if (bgMode === 'blur') {
        ox.filter = 'blur(14px)'; ox.drawImage(grab, 0, 0); ox.filter = 'none';
      }
      const fr = gx.getImageData(0, 0, W, H);
      if (lastAlpha) for (let i = 0; i < W * H; i++) fr.data[4 * i + 3] = lastAlpha[i];
      const tmp = canvasOf(W, H); tmp.getContext('2d').putImageData(fr, 0, 0);
      ox.drawImage(tmp, 0, 0);
      px.clearRect(0, 0, W, H); px.drawImage(out, 0, 0);

      const p = t / dur;
      const el = (performance.now() - t0) / 1000;
      const eta = p > 0.02 ? Math.round(el / p - el) : 0;
      setProg(p, `frame ${idx} · ${(100 * p).toFixed(0)}% · ETA ${eta}s`);
      await sleep(0);
    }

    rec.stop(); await done;
    if (audioTrack) audioTrack.stop();
    const blob = new Blob(chunks, { type: mime.split(';')[0] || 'video/webm' });
    const ext = blob.type.includes('mp4') ? 'mp4' : 'webm';
    const name = vidFile.name.replace(/\.[^.]+$/, '') + (alpha ? '-alpha' : '-nobg') + '.' + ext;
    const url = URL.createObjectURL(blob);
    $('#vidOut').innerHTML =
      `<video src="${url}" controls loop style="max-width:100%;border-radius:10px;background:#222"></video>
       <p><a class="go" href="${url}" download="${name}" style="display:inline-block;text-decoration:none">⬇ Download ${name} (${(blob.size/1048576).toFixed(1)} MB)</a></p>`;
    setProg(0);
    log(`✅ video ready — ${name} · ${(blob.size/1048576).toFixed(1)} MB · ${idx} frames`, 'ok');
    if (alpha) log('ℹ Alpha WebM: Chrome/Edge + video editors (Premiere/DaVinci/CapCut) me transparency dikhega. WhatsApp jaise apps alpha support nahi karte — waha green/colour mode use karo.', 'warn');
  } catch (e) {
    console.error(e); log('✖ video error: ' + e.message, 'err'); setProg(0);
  }
  $$('.go').forEach(b => b.disabled = false); $('#vidStop').disabled = true; running = false;
};

/* ============ upload (telegra.ph style) ============ */
async function tryFetch(url, opts, ms = 45000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), ms);
  try { return await fetch(url, { ...opts, signal: ctl.signal }); }
  finally { clearTimeout(t); }
}
async function upload(blob, name) {
  const hosts = [
    { n: 'tmpfiles.org', fn: async () => {
        const fd = new FormData(); fd.append('file', blob, name);
        const r = await tryFetch('https://tmpfiles.org/api/v1/upload', { method: 'POST', body: fd });
        const j = await r.json();
        return j?.data?.url?.replace('tmpfiles.org/', 'tmpfiles.org/dl/');
      } },
    { n: 'catbox.moe', fn: async () => {
        const fd = new FormData();
        fd.append('reqtype', 'fileupload'); fd.append('fileToUpload', blob, name);
        const r = await tryFetch('https://catbox.moe/user/api.php', { method: 'POST', body: fd });
        const t = (await r.text()).trim();
        return t.startsWith('http') ? t : null;
      } },
    { n: 'uguu.se', fn: async () => {
        const fd = new FormData(); fd.append('files[]', blob, name);
        const r = await tryFetch('https://uguu.se/upload?output=text', { method: 'POST', body: fd });
        const t = (await r.text()).trim();
        return t.startsWith('http') ? t : null;
      } },
    { n: 'litterbox', fn: async () => {
        const fd = new FormData();
        fd.append('reqtype', 'fileupload'); fd.append('time', '72h'); fd.append('fileToUpload', blob, name);
        const r = await tryFetch('https://litterbox.catbox.moe/resources/internals/api.php', { method: 'POST', body: fd });
        const t = (await r.text()).trim();
        return t.startsWith('http') ? t : null;
      } },
  ];
  for (const h of hosts) {
    try {
      const u = await h.fn();
      if (u) { log(`🔗 ${name} → ${h.n}`, 'ok'); return u; }
    } catch (e) { /* next host */ }
  }
  return null;
}

/* ============ THE AUTO PIPELINE ============ */
async function runAuto() {
  if (running) return;
  if (!files.length) return alert('Pehle images add karo');
  running = true;
  $$('.go').forEach(b => b.disabled = true);
  outputs = []; showResults();
  const doClean = $('#optClean').checked;
  const doBg = $('#optBg').checked;
  const doLink = $('#optLink').checked;
  const fmt = $('#fmt').value, q = $('#q').value / 100, maxw = +$('#maxw').value;
  const sens = +$('#sens').value;
  const bgColor = $('#bgSolid').checked ? $('#bgColor').value : null;

  log(`🚀 Auto pipeline start — ${files.length} file(s): ${[doClean && 'watermark/text remove', doBg && 'bg remove', 'convert→' + extOf(fmt), doLink && 'link'].filter(Boolean).join(' → ')}`);

  try {
    if (doBg) {
      await cvReady();
      if ($('#personOnly').checked) {
        const cm = $('#cutMode')?.value || 'smart';
        if (cm !== 'detail') await initParser();
        if (cm !== 'semantic') await initBg();
      } else await initBg();
    }
    if (doClean) { log('… OpenCV load'); await cvReady(); log('✅ OpenCV ready', 'ok'); }

    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      const base = i / files.length, step = 1 / files.length;
      setProg(base + step * 0.1, `${i + 1}/${files.length} · ${f.name}`);
      let workUrl = f.url, workCanvas = null;

      if (doClean) {
        log(`🧽 ${f.name} — OCR text detect + inpaint`);
        const r = await cleanImage(workUrl, sens);
        if (r.covered > 0) {
          workCanvas = r.canvas;
          workUrl = URL.createObjectURL(await toBlob(workCanvas, 'image/png'));
        } // warna original hi rakho — koi re-encode nahi, zero quality loss
        log(r.covered
          ? `   🔤 ${r.boxes} text region(s) hataye: ${(r.words||[]).slice(0,6).join(', ')}${(r.words||[]).length>6?'…':''} · ${r.pct}% pixels · ${r.ms}ms`
          : `   ✓ koi text nahi mila — image bilkul unchanged`, r.covered ? 'ok' : 'warn');
        setProg(base + step * 0.4, `${i + 1}/${files.length} · cleaned`);
      }
      if (doBg) {
        log(`✂️ ${f.name} — ${$('#personOnly').checked ? 'PERSON-ONLY cutout (bed/mirror/objects sab delete)' : 'AI background remove'}`);
        workCanvas = await cutout(workUrl);
        workUrl = URL.createObjectURL(await toBlob(workCanvas, 'image/png'));
        setProg(base + step * 0.7, `${i + 1}/${files.length} · bg removed`);
      }

      let outFmt = fmt;
      if (doBg && !bgColor && fmt === 'image/jpeg') { outFmt = 'image/png'; log('   ⚠ transparency ke liye PNG use kiya', 'warn'); }
      const srcEl = workCanvas || await loadImg(workUrl);
      const blob = await convertCanvas(srcEl, outFmt, q, maxw, false);
      const o = pushOut(f.name, blob, extOf(outFmt));
      if (window.__lastDebug) { o.debug = window.__lastDebug; window.__lastDebug = null; }
      showResults();
      log(`✅ ${o.name} · ${(blob.size / 1024).toFixed(0)} KB`, 'ok');

      if (doLink) {
        setProg(base + step * 0.9, `${i + 1}/${files.length} · uploading`);
        o.link = await upload(blob, o.name);
        if (!o.link) log(`   ✖ link fail (network/host blocked)`, 'warn');
        showResults();
      }
      // free intermediate blob URLs so heap na bhare
      if (workUrl !== f.url) URL.revokeObjectURL(workUrl);
      workCanvas = null;
      setProg(base + step, '');
      await sleep(0);
    }
    setProg(0);
    log(`🎉 Done — ${outputs.length} file(s) ready. ZIP se sab download kar lo.`, 'ok');
    $('#outwrap').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) {
    console.error(e);
    log('✖ Error: ' + e.message, 'err');
    setProg(0);
  }
  $$('.go').forEach(b => b.disabled = false);
  running = false;
}
$('#runAuto').onclick = runAuto;

/* live slider values */
[['grow','%'],['shrink',''],['feather',''],['ocrConf','']].forEach(([id,suf])=>{
  const el=$('#'+id), out=$('#'+id+'V');
  if(el&&out){ const u=()=>out.textContent=el.value+suf; el.oninput=u; u(); }
});

/* keep the two Person-Only toggles in sync */
const pA = $('#autoPerson'), pB = $('#personOnly');
pA.onchange = () => { pB.checked = pA.checked; };
pB.onchange = () => { pA.checked = pB.checked; };

/* ============ individual buttons (still fully automatic) ============ */
$('#runConvert').onclick = async () => {
  if (!files.length) return alert('Pehle images add karo');
  running = true; outputs = [];
  const fmt = $('#fmt').value, q = $('#q').value / 100, maxw = +$('#maxw').value, wb = $('#whitebg').checked;
  for (let i = 0; i < files.length; i++) {
    setProg((i + 0.5) / files.length, files[i].name);
    const img = await loadImg(files[i].url);
    const blob = await convertCanvas(img, fmt, q, maxw, wb);
    pushOut(files[i].name, blob, extOf(fmt)); showResults();
    log(`✅ ${files[i].name} → ${extOf(fmt)}`, 'ok');
  }
  setProg(0); running = false;
};

$('#runBg').onclick = async () => {
  if (!files.length) return alert('Pehle images add karo');
  running = true; $$('.go').forEach(b => b.disabled = true); outputs = [];
  try {
    await cvReady();
    if ($('#personOnly').checked) {
      const cm = $('#cutMode')?.value || 'smart';
      if (cm !== 'detail') await initParser();
      if (cm !== 'semantic') await initBg();
    } else await initBg();
    for (let i = 0; i < files.length; i++) {
      setProg((i + 0.5) / files.length, files[i].name);
      log(`✂️ ${files[i].name}`);
      const c = await cutout(files[i].url);
      pushOut(files[i].name + '-nobg', await toBlob(c, 'image/png'), 'png');
      showResults(); log(`✅ ${files[i].name}`, 'ok');
    }
  } catch (e) { log('✖ ' + e.message, 'err'); }
  setProg(0); $$('.go').forEach(b => b.disabled = false); running = false;
};

$('#runClean').onclick = async () => {
  if (!files.length) return alert('Pehle images add karo');
  running = true; $$('.go').forEach(b => b.disabled = true); outputs = [];
  try {
    await cvReady();
    const sens = +$('#sens').value, mode = $('#cleanMode').value;
    const list = mode === 'brush' ? [files[current]] : files;
    for (let i = 0; i < list.length; i++) {
      setProg((i + 0.5) / list.length, list[i].name);
      const bm = mode === 'brush' ? brushMaskFor : null;
      const r = await cleanImage(list[i].url, sens, bm);
      pushOut(list[i].name + '-clean', await toBlob(r.canvas, 'image/png'), 'png');
      showResults(); log(r.covered
        ? `✅ ${list[i].name} — ${r.boxes} region: ${(r.words||[]).slice(0,5).join(', ')} · ${r.pct}% pixels · ${r.ms}ms`
        : `✓ ${list[i].name} — koi text nahi mila, image unchanged`, r.covered ? 'ok' : 'warn');
    }
  } catch (e) { log('✖ ' + e.message, 'err'); }
  setProg(0); $$('.go').forEach(b => b.disabled = false); running = false;
};

$('#previewClean').onclick = async () => {
  if (!files.length) return alert('Pehle images add karo');
  log(`👁 Preview: ${files[current].name}`);
  $$('.go').forEach(b => b.disabled = true);
  try {
    await cvReady();
    const r = await cleanImage(files[current].url, +$('#sens').value, null, true);
    const c = $('#editCanvas'), mcv = $('#maskCanvas');
    const sc = Math.min(1, 880 / r.canvas.width);
    c.width = mcv.width = Math.round(r.canvas.width * sc);
    c.height = mcv.height = Math.round(r.canvas.height * sc);
    c.getContext('2d').drawImage(r.canvas, 0, 0, c.width, c.height);
    mcv.getContext('2d').clearRect(0, 0, mcv.width, mcv.height);
    if (r.covered) {
      log(`👁 Preview: ${r.boxes} text region mile — ${(r.words||[]).join(' | ')}`, 'ok');
      log(`   sirf ${r.pct}% pixels badlenge (laal = hatega, hara box = OCR word). Theek lage to Remove dabao.`);
    } else {
      log('👁 Preview: koi text detect nahi hua — image bilkul safe hai.', 'warn');
      log('   Watermark faint hai? OCR confidence ghatao (30-40), ya Brush mode use karo.', 'warn');
    }
    if (r.skipped && (r.skipped.skin || r.skipped.size))
      log(`   🛡 skipped: ${r.skipped.skin} skin/face par, ${r.skipped.size} size se bahar`);
  } catch (e) { log('✖ ' + e.message, 'err'); }
  $$('.go').forEach(b => b.disabled = false);
};

$('#runHost').onclick = async () => {
  const list = outputs.length ? outputs : files.map(f => ({ name: f.name, blob: f.file }));
  if (!list.length) return alert('Pehle images add karo');
  $$('.go').forEach(b => b.disabled = true); $('#links').innerHTML = '';
  for (let i = 0; i < list.length; i++) {
    setProg((i + 0.5) / list.length, list[i].name);
    const url = await upload(list[i].blob, list[i].name);
    list[i].link = url;
    $('#links').insertAdjacentHTML('beforeend', url
      ? `<p><b>${list[i].name}</b><br><a href="${url}" target="_blank">${url}</a>
         <button class="ghost cp" data-u="${url}">copy</button></p>`
      : `<p><b>${list[i].name}</b>: ✖ upload failed</p>`);
  }
  $$('.cp').forEach(b => b.onclick = () => { navigator.clipboard.writeText(b.dataset.u); b.textContent = 'copied ✓'; });
  setProg(0); showResults(); $$('.go').forEach(b => b.disabled = false);
};

/* ============ brush (optional manual override) ============ */
const ec = $('#editCanvas'), mc = $('#maskCanvas');
let painting = false;
async function loadToCanvas(i) {
  if (!files[i]) return;
  const img = await loadImg(files[i].url);
  const scale = Math.min(1, 880 / img.width);
  ec.width = mc.width = Math.round(img.width * scale);
  ec.height = mc.height = Math.round(img.height * scale);
  ec.getContext('2d').drawImage(img, 0, 0, ec.width, ec.height);
  mc.getContext('2d').clearRect(0, 0, mc.width, mc.height);
}
function pos(e) {
  const r = mc.getBoundingClientRect(), t = e.touches ? e.touches[0] : e;
  return { x: (t.clientX - r.left) * mc.width / r.width, y: (t.clientY - r.top) * mc.height / r.height };
}
function paint(e) {
  if (!painting) return; e.preventDefault();
  const { x, y } = pos(e), g = mc.getContext('2d');
  g.fillStyle = 'rgba(255,0,80,.6)'; g.beginPath();
  g.arc(x, y, +$('#brush').value / 2, 0, 7); g.fill();
}
mc.addEventListener('mousedown', e => { painting = true; paint(e); });
mc.addEventListener('mousemove', paint);
mc.addEventListener('touchstart', e => { painting = true; paint(e); }, { passive: false });
mc.addEventListener('touchmove', paint, { passive: false });
addEventListener('mouseup', () => painting = false);
addEventListener('touchend', () => painting = false);
$('#clearMask').onclick = () => mc.getContext('2d').clearRect(0, 0, mc.width, mc.height);
function brushMaskFor(src) {
  const m = mc.getContext('2d').getImageData(0, 0, mc.width, mc.height);
  const a = new cv.Mat(mc.height, mc.width, cv.CV_8U);
  for (let i = 0; i < mc.width * mc.height; i++) a.data[i] = m.data[4 * i + 3] > 10 ? 255 : 0;
  const big = new cv.Mat();
  cv.resize(a, big, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_NEAREST);
  a.delete(); return big;
}

/* ============ ZIP ============ */
$('#dlZip').onclick = async () => {
  if (!outputs.length) return alert('Koi result nahi');
  const zip = new JSZip();
  outputs.forEach(o => zip.file(o.name, o.blob));
  if (outputs.some(o => o.link)) zip.file('links.txt', outputs.map(o => `${o.name}\t${o.link || '-'}`).join('\n'));
  const b = await zip.generateAsync({ type: 'blob' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = 'pixelfree.zip'; a.click();
};
$('#copyAll').onclick = () => {
  const txt = outputs.filter(o => o.link).map(o => o.link).join('\n');
  if (!txt) return alert('Koi link nahi — pehle "Get share links" on karke run karo');
  navigator.clipboard.writeText(txt); $('#copyAll').textContent = 'copied ✓';
  setTimeout(() => $('#copyAll').textContent = '🔗 Copy all links', 1500);
};

addEventListener('beforeunload', () => { try { ocrWorker?.terminate(); } catch(e){} });

/* warm up in background so first real run is fast */
addEventListener('load', () => { cvReady().then(() => log('✅ OpenCV engine ready', 'ok')).catch(() => {}); });
log('👋 Ready. Images drop karo — sab kuch automatically ho jayega.');
log('🏷 build v7.0-ocr · memory-safe · MAX_PIXELS=' + (MAX_PIXELS/1e6).toFixed(0) + 'MP · deviceMemory=' + (navigator.deviceMemory||'?') + 'GB', 'ok');
