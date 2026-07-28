# PixelFree — Fully Automatic AI Image Toolkit (100% free)

Sab kuch **browser me** chalta hai. Koi server, koi API key, koi login, koi limit.

## ⚡ Auto Pipeline (main feature)
Image drop karo → **bas.** Kuch click karne ki zarurat nahi.
Ye steps automatically order me chalte hain:

1. 🧽 **Watermark / text remove** — auto detect + AI inpaint
2. ✂️ **Background remove** — AI (RMBG-1.4) → transparent PNG
3. 🔄 **Format convert** — PNG / JPG / WEBP + resize + quality
4. 🔗 **Share link** — telegra.ph style anonymous URL (optional)
5. ⬇ **ZIP** me sab download / sab links copy

Batch: jitni bhi files daalo, sab par same pipeline chalega. Manual kuch nahi.
"Auto-start jaise hi file drop ho" checkbox on hai → literally zero click.

## Features detail
| Tool | Tech | Auto? |
|---|---|---|
| BG remove | RMBG-1.4 (Transformers.js, WebGPU) | Fully auto |
| Watermark/text remove | OpenCV.js: morphological-gradient text detect + local-variance watermark detect → Telea + Navier–Stokes inpaint blend | Fully auto (brush optional) |
| Convert | Canvas encoder | Fully auto |
| File→Link | tmpfiles → catbox → uguu → litterbox fallback chain | Fully auto |

## Chalao
```bash
cd aitools
python3 -m http.server 8000
```
Browser: http://localhost:8000

> `file://` se mat kholna — ES modules + AI model fetch block ho jayenge. Local server zaroori hai.

## Free hosting par live karo
- **GitHub Pages** — repo banao, ye 3 files push karo, Settings → Pages → main branch. Free + permanent.
- **Netlify / Vercel / Cloudflare Pages** — folder drag-drop. Free.

Kyunki poori app client-side hai, hosting hamesha free rahegi — koi backend bill nahi.

## Notes
- Pehli baar BG remove chalane par ~45 MB AI model download hoga, phir cache — uske baad **offline** bhi chalega.
- Chrome / Edge me WebGPU se sabse fast (2-4x).
- Watermark removal ki sensitivity slider (Watermark tab) se tune kar sakte ho: zyada = aggressive.
- File→Link ke liye internet chahiye. Ek host block ho to agla apne aap try hota hai.
