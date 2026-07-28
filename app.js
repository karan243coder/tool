/* PixelFree — fully automatic AI image toolkit. Everything runs in the browser. */
import { env, AutoModel, AutoProcessor, RawImage }
  from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';

env.allowLocalModels = false;
env.backends.onnx.wasm.proxy = false;

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
    model = await AutoModel.from_pretrained('briaai/RMBG-1.4', {
      config: { model_type: 'custom' }, device, dtype: 'fp32',
      progress_callback: p => { if (p.status === 'progress' && p.total) setProg(0.05 + 0.4 * (p.loaded / p.total), `model ${(p.progress || 0).toFixed(0)}%`); }
    });
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
function autoMask(src, sens) {
  const gray = new cv.Mat(), grad = new cv.Mat(), bw = new cv.Mat(), conn = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
  cv.morphologyEx(gray, grad, cv.MORPH_GRADIENT, k);
  cv.threshold(grad, bw, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);

  const mask = cv.Mat.zeros(src.rows, src.cols, cv.CV_8U);
  const area = src.rows * src.cols;
  const scale = Math.max(1, Math.round(Math.min(src.cols, src.rows) / 400));

  // pass 1: horizontal text lines
  const widths = [Math.round((9 + sens) * scale), Math.round((17 + sens * 2) * scale)];
  for (const w of widths) {
    const k2 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(w, Math.max(3, 3 * scale)));
    cv.morphologyEx(bw, conn, cv.MORPH_CLOSE, k2);
    const contours = new cv.MatVector(), hier = new cv.Mat();
    cv.findContours(conn, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    for (let i = 0; i < contours.size(); i++) {
      const r = cv.boundingRect(contours.get(i));
      const ar = r.width / r.height, a = r.width * r.height;
      const roi = bw.roi(r); const dense = cv.countNonZero(roi) / a; roi.delete();
      const ok = a > 60 * scale * scale && a < area * 0.45 &&
        ar > 1.0 && ar < 40 &&
        r.height > 6 && r.height < src.rows * 0.35 &&
        dense > 0.14 + (10 - sens) * 0.02;
      if (ok) {
        const pad = Math.round(3 * scale);
        cv.rectangle(mask,
          new cv.Point(Math.max(0, r.x - pad), Math.max(0, r.y - pad)),
          new cv.Point(Math.min(src.cols, r.x + r.width + pad), Math.min(src.rows, r.y + r.height + pad)),
          new cv.Scalar(255), -1);
      }
    }
    contours.delete(); hier.delete(); k2.delete();
  }

  // pass 2: translucent/diagonal watermark — high edge density blobs anywhere
  if (sens >= 5) {
    const blur = new cv.Mat(), diff = new cv.Mat(), th2 = new cv.Mat(), d2 = new cv.Mat();
    cv.GaussianBlur(gray, blur, new cv.Size(0, 0), 3 * scale);
    cv.absdiff(gray, blur, diff);
    cv.threshold(diff, th2, 6 + (10 - sens), 255, cv.THRESH_BINARY);
    const k3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5 * scale, 5 * scale));
    cv.morphologyEx(th2, d2, cv.MORPH_CLOSE, k3);
    const c2 = new cv.MatVector(), h2 = new cv.Mat();
    cv.findContours(d2, c2, h2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    for (let i = 0; i < c2.size(); i++) {
      const r = cv.boundingRect(c2.get(i));
      const a = r.width * r.height;
      const roi = d2.roi(r); const dense = cv.countNonZero(roi) / a; roi.delete();
      if (a > 300 * scale * scale && a < area * 0.25 && dense > 0.25) {
        cv.rectangle(mask, new cv.Point(r.x, r.y), new cv.Point(r.x + r.width, r.y + r.height), new cv.Scalar(255), -1);
      }
    }
    blur.delete(); diff.delete(); th2.delete(); d2.delete(); c2.delete(); h2.delete(); k3.delete();
  }

  const kd = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3 * scale, 3 * scale));
  cv.dilate(mask, mask, kd); kd.delete();
  gray.delete(); grad.delete(); bw.delete(); conn.delete(); k.delete();
  return mask;
}

/** inpaint using given mask; two-pass (Telea then NS blend) for smoother fill */
function inpaintWith(src, mask) {
  const rgb = new cv.Mat(); cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
  const a = new cv.Mat(), b = new cv.Mat(), out = new cv.Mat();
  const radius = Math.max(3, Math.round(Math.min(src.cols, src.rows) / 200));
  cv.inpaint(rgb, mask, a, radius, cv.INPAINT_TELEA);
  cv.inpaint(rgb, mask, b, radius, cv.INPAINT_NS);
  cv.addWeighted(a, 0.5, b, 0.5, 0, out);
  rgb.delete(); a.delete(); b.delete();
  return out;
}

async function cleanImage(url, sens, brushMask) {
  await cvReady();
  const img = await loadImg(url);
  const c = canvasOf(img.width, img.height);
  c.getContext('2d').drawImage(img, 0, 0);
  const src = cv.imread(c);
  const mask = brushMask ? brushMask(src) : autoMask(src, sens);
  const covered = cv.countNonZero(mask);
  let outCanvas = c;
  if (covered > 0) {
    const dst = inpaintWith(src, mask);
    cv.imshow(c, dst); dst.delete();
  }
  const pct = (100 * covered / (src.rows * src.cols)).toFixed(1);
  src.delete(); mask.delete();
  return { canvas: outCanvas, pct, covered };
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
    if (doBg) await initBg();
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
        log(`   removed ${r.pct}% area`, r.covered ? 'ok' : 'warn');
        setProg(base + step * 0.4, `${i + 1}/${files.length} · cleaned`);
      }
      if (doBg) {
        log(`✂️ ${f.name} — AI background remove`);
        workCanvas = await removeBg(workUrl, bgColor);
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
      setProg(base + step, '');
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
    await initBg();
    const bgColor = $('#bgSolid').checked ? $('#bgColor').value : null;
    for (let i = 0; i < files.length; i++) {
      setProg((i + 0.5) / files.length, files[i].name);
      log(`✂️ ${files[i].name}`);
      const c = await removeBg(files[i].url, bgColor);
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
      showResults(); log(`✅ ${list[i].name} — ${r.pct}% inpainted`, 'ok');
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
