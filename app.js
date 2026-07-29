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
     ${o.link ? `<a href="${o.link}" target="_blank">🔗 link</a>` : ''}</div>`).join('');
  $('#outwrap').classList.toggle('hidden', !outputs.length);
}
/* Bade images (12MP+) browser WASM heap crash karte hain. Kaam se pehle
   safe size par utaar dete hain; final output isi size par save hota hai. */
const MAX_PIXELS = 12e6;   // ~12MP (4240x2830)
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
  for (const seg of out) {
    if (!keepNames.has(seg.label)) continue;
    hit++;
    const m = seg.mask;                     // RawImage, 1 channel
    const sw = m.width, sh = m.height;
    for (let y = 0; y < h; y++) {
      const sy = Math.min(sh - 1, (y * sh / h) | 0);
      for (let x = 0; x < w; x++) {
        const sx = Math.min(sw - 1, (x * sw / w) | 0);
        const v = m.data[sy * sw + sx];
        if (v > 127) alpha[y * w + x] = 255;
      }
    }
  }
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
async function personCutout(url, opts) {
  const img = capCanvas(await loadImg(url));
  const w = img.width, h = img.height;
  // capped canvas se hi model ko feed karo taaki sizes match rahein
  const feedUrl = img.tagName === 'CANVAS'
    ? await new Promise(r => img.toBlob(b => r(URL.createObjectURL(b)), 'image/jpeg', 0.95))
    : url;
  let { alpha, hit } = await personAlpha(feedUrl, w, h, opts);
  if (!hit) {
    log('   ⚠ koi person nahi mila — normal AI cutout par fallback', 'warn');
    return { canvas: await removeBg(url, opts.bgColor), fallback: true };
  }
  alpha = await refineAlpha(alpha, w, h, opts.feather);
  const c = canvasOf(w, h), x = c.getContext('2d');
  const tmp = canvasOf(w, h), tx = tmp.getContext('2d');
  tx.drawImage(img, 0, 0);
  const d = tx.getImageData(0, 0, w, h);
  for (let i = 0; i < w * h; i++) d.data[4 * i + 3] = alpha[i];
  tx.putImageData(d, 0, 0);
  if (opts.bgColor) { x.fillStyle = opts.bgColor; x.fillRect(0, 0, w, h); }
  x.drawImage(tmp, 0, 0);
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
function autoMask(src, sens) {
  const t0 = performance.now();
  // --- work on a downscaled copy for detection ---
  const f = Math.min(1, DET_MAX / Math.max(src.cols, src.rows));
  const small = new cv.Mat();
  if (f < 1) cv.resize(src, small, new cv.Size(Math.round(src.cols * f), Math.round(src.rows * f)), 0, 0, cv.INTER_AREA);
  else src.copyTo(small);

  const gray = new cv.Mat(), grad = new cv.Mat(), bw = new cv.Mat(), conn = new cv.Mat();
  cv.cvtColor(small, gray, cv.COLOR_RGBA2GRAY);
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
  cv.morphologyEx(gray, grad, cv.MORPH_GRADIENT, k);
  cv.threshold(grad, bw, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);

  const smallMask = cv.Mat.zeros(small.rows, small.cols, cv.CV_8U);
  const area = small.rows * small.cols;
  const boxes = [];

  // single-pass horizontal text grouping (pehle 2 passes the -> 2x fast)
  const k2 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(13 + sens, 3));
  cv.morphologyEx(bw, conn, cv.MORPH_CLOSE, k2);
  const contours = new cv.MatVector(), hier = new cv.Mat();
  cv.findContours(conn, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  for (let i = 0; i < contours.size(); i++) {
    const r = cv.boundingRect(contours.get(i));
    const ar = r.width / r.height, a = r.width * r.height;
    if (a < 50 || a > area * 0.45) continue;
    const roi = bw.roi(r); const dense = cv.countNonZero(roi) / a; roi.delete();
    if (ar > 1.0 && ar < 40 && r.height > 5 && r.height < small.rows * 0.35 &&
        dense > 0.14 + (10 - sens) * 0.02) {
      const p = 3;
      const rr = new cv.Rect(Math.max(0, r.x - p), Math.max(0, r.y - p),
        Math.min(small.cols - Math.max(0, r.x - p), r.width + 2 * p),
        Math.min(small.rows - Math.max(0, r.y - p), r.height + 2 * p));
      cv.rectangle(smallMask, new cv.Point(rr.x, rr.y), new cv.Point(rr.x + rr.width, rr.y + rr.height), new cv.Scalar(255), -1);
      boxes.push(rr);
    }
  }
  contours.delete(); hier.delete(); k2.delete();

  // translucent watermark pass (only when sensitivity high)
  if (sens >= 6) {
    const blur = new cv.Mat(), diff = new cv.Mat(), th2 = new cv.Mat(), d2 = new cv.Mat();
    cv.GaussianBlur(gray, blur, new cv.Size(0, 0), 3);
    cv.absdiff(gray, blur, diff);
    cv.threshold(diff, th2, 6 + (10 - sens), 255, cv.THRESH_BINARY);
    const k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
    cv.morphologyEx(th2, d2, cv.MORPH_CLOSE, k3);
    const c2 = new cv.MatVector(), h2 = new cv.Mat();
    cv.findContours(d2, c2, h2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    for (let i = 0; i < c2.size(); i++) {
      const r = cv.boundingRect(c2.get(i));
      const a = r.width * r.height;
      if (a < 250 || a > area * 0.25) continue;
      const roi = d2.roi(r); const dense = cv.countNonZero(roi) / a; roi.delete();
      if (dense > 0.25) {
        cv.rectangle(smallMask, new cv.Point(r.x, r.y), new cv.Point(r.x + r.width, r.y + r.height), new cv.Scalar(255), -1);
        boxes.push(r);
      }
    }
    blur.delete(); diff.delete(); th2.delete(); d2.delete(); c2.delete(); h2.delete(); k3.delete();
  }

  const kd = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
  cv.dilate(smallMask, smallMask, kd); kd.delete();

  /* SAFETY: agar detector image ka bahut bada hissa maang raha hai to wo
     galat detect kar raha hai (texture/pattern ko text samajh liya).
     Aise me mask drop kar dete hain — aadhi photo inpaint karne se behtar
     hai kuch na karna. User sensitivity kam karke ya brush se kar sakta hai. */
  const cap = +($('#cleanCap')?.value || 25) / 100;
  const frac = cv.countNonZero(smallMask) / area;
  if (frac > cap) {
    log(`   ⚠ detector ne ${(frac*100).toFixed(0)}% area maanga (cap ${(cap*100)|0}%) — skip. Sensitivity kam karo ya brush mode use karo.`, 'warn');
    smallMask.setTo(new cv.Scalar(0));
    boxes.length = 0;
  }

  // --- upscale mask back to full res ---
  const mask = new cv.Mat();
  if (f < 1) cv.resize(smallMask, mask, new cv.Size(src.cols, src.rows), 0, 0, cv.INTER_NEAREST);
  else smallMask.copyTo(mask);

  // scale boxes back for ROI inpainting
  const inv = f < 1 ? 1 / f : 1;
  const fullBoxes = boxes.map(b => ({
    x: Math.max(0, Math.floor(b.x * inv) - 8),
    y: Math.max(0, Math.floor(b.y * inv) - 8),
    w: Math.ceil(b.width * inv) + 16,
    h: Math.ceil(b.height * inv) + 16,
  })).map(b => ({ ...b, w: Math.min(b.w, src.cols - b.x), h: Math.min(b.h, src.rows - b.y) }))
    .filter(b => b.w > 1 && b.h > 1);

  small.delete(); gray.delete(); grad.delete(); bw.delete(); conn.delete(); smallMask.delete(); k.delete();
  mask.__boxes = fullBoxes;
  mask.__ms = Math.round(performance.now() - t0);
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

/** FAST inpaint: sirf mask wale ROI patches par chalta hai, poori image par nahi. */
function inpaintWith(src, mask, quality) {
  const rgb = new cv.Mat(); cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
  const radius = quality === 'best' ? 5 : 3;
  let boxes = mask.__boxes;

  // no box info (brush mode) -> bounding box of whole mask
  if (!boxes || !boxes.length) {
    const nz = new cv.Mat();
    cv.findNonZero(mask, nz);
    if (nz.rows === 0) { nz.delete(); return rgb; }
    const r = cv.boundingRect(mask);
    boxes = [{ x: Math.max(0, r.x - 8), y: Math.max(0, r.y - 8),
               w: Math.min(src.cols, r.width + 16), h: Math.min(src.rows, r.height + 16) }];
    nz.delete();
  }
  boxes = mergeBoxes(boxes, src.cols, src.rows);

  // agar mask poori image ka bada hissa hai to ek hi full pass sasta padta hai
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
    } else {
      res.copyTo(patch);
    }
    pc.delete(); mc2.delete(); res.delete(); patch.delete(); mpatch.delete();
  }
  return rgb;
}

async function cleanImage(url, sens, brushMask) {
  await cvReady();
  const t0 = performance.now();
  const img = capCanvas(await loadImg(url));
  const c = canvasOf(img.width, img.height);
  c.getContext('2d').drawImage(img, 0, 0);
  const src = cv.imread(c);
  const quality = $('#cleanQ') ? $('#cleanQ').value : 'fast';
  const mask = brushMask ? brushMask(src) : autoMask(src, sens);
  const covered = cv.countNonZero(mask);
  if (covered > 0) {
    const dst = inpaintWith(src, mask, quality);
    cv.imshow(c, dst); dst.delete();
  }
  const pct = (100 * covered / (src.rows * src.cols)).toFixed(1);
  const ms = Math.round(performance.now() - t0);
  src.delete(); mask.delete();
  return { canvas: c, pct, covered, ms };
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
    await ($('#personOnly').checked ? initParser() : initBg());

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
          const r = await personAlpha(url, W, H, {
            keepAccessories: $('#keepAcc').checked, keepBag: $('#keepBag').checked });
          lastAlpha = r.hit ? await refineAlpha(r.alpha, W, H, +$('#feather').value) : lastAlpha;
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
    if (doBg) { await cvReady(); await ($('#personOnly').checked ? initParser() : initBg()); }
    if (doClean) { log('… OpenCV load'); await cvReady(); log('✅ OpenCV ready', 'ok'); }

    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      const base = i / files.length, step = 1 / files.length;
      setProg(base + step * 0.1, `${i + 1}/${files.length} · ${f.name}`);
      let workUrl = f.url, workCanvas = null;

      if (doClean) {
        log(`🧽 ${f.name} — text/watermark detect + inpaint`);
        const r = await cleanImage(workUrl, sens);
        workCanvas = r.canvas;
        workUrl = URL.createObjectURL(await toBlob(workCanvas, 'image/png'));
        log(`   removed ${r.pct}% area · ${r.ms}ms`, r.covered ? 'ok' : 'warn');
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
    await ($('#personOnly').checked ? initParser() : initBg());
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
      showResults(); log(`✅ ${list[i].name} — ${r.pct}% inpainted · ${r.ms}ms`, 'ok');
    }
  } catch (e) { log('✖ ' + e.message, 'err'); }
  setProg(0); $$('.go').forEach(b => b.disabled = false); running = false;
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

/* warm up in background so first real run is fast */
addEventListener('load', () => { cvReady().then(() => log('✅ OpenCV engine ready', 'ok')).catch(() => {}); });
log('👋 Ready. Images drop karo — sab kuch automatically ho jayega.');
