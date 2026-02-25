/**
 * MaskCanvas — three-layer canvas stack for mask drawing.
 *
 * Layers (bottom -> top):
 *   1. canvas-image  — displays the source image (read-only)
 *   2. canvas-mask   — stores the mask the user paints (semi-transparent red overlay)
 *   3. canvas-cursor  — lightweight layer that renders the brush cursor
 */

class MaskCanvas {
  static MAX_DIM = 1920;

  constructor(imageCanvas, maskCanvas, cursorCanvas) {
    this.imageCanvas = imageCanvas;
    this.maskCanvas = maskCanvas;
    this.cursorCanvas = cursorCanvas;

    this.imageCtx = imageCanvas.getContext("2d");
    this.maskCtx = maskCanvas.getContext("2d");
    this.cursorCtx = cursorCanvas.getContext("2d");

    this.brushSize = 20;
    this.tool = "brush"; // "brush" | "eraser"
    this.painting = false;
    this.lastPoint = null;

    this._bindEvents();
    window.addEventListener("resize", () => this._updateWrapperSize());
  }

  /* ---- public API ---- */

  loadImage(img) {
    const natW = img.naturalWidth;
    const natH = img.naturalHeight;

    const ratio = Math.min(1, MaskCanvas.MAX_DIM / Math.max(natW, natH));
    const w = Math.round(natW * ratio);
    const h = Math.round(natH * ratio);

    [this.imageCanvas, this.maskCanvas, this.cursorCanvas].forEach((c) => {
      c.width = w;
      c.height = h;
    });

    this.imageCtx.drawImage(img, 0, 0, w, h);
    this.maskCtx.clearRect(0, 0, w, h);
    this._updateWrapperSize();
  }

  /** Restore a previously-saved mask data URL onto the mask layer. */
  loadMask(dataUrl) {
    const img = new Image();
    img.onload = () => {
      this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
      this.maskCtx.drawImage(img, 0, 0, this.maskCanvas.width, this.maskCanvas.height);
    };
    img.src = dataUrl;
  }

  /** Return raw mask canvas as data URL (red paint, for internal round-trip). */
  saveMaskRaw() {
    return this.maskCanvas.toDataURL("image/png");
  }

  /**
   * Export the mask as a white-on-black PNG suitable for training.
   * Internally we paint red for visibility; this converts to binary.
   */
  exportMask() {
    const w = this.maskCanvas.width;
    const h = this.maskCanvas.height;
    const src = this.maskCtx.getImageData(0, 0, w, h);

    const out = document.createElement("canvas");
    out.width = w;
    out.height = h;
    const ctx = out.getContext("2d");
    const dst = ctx.createImageData(w, h);

    for (let i = 0; i < src.data.length; i += 4) {
      const alpha = src.data[i + 3];
      const val = alpha > 0 ? 255 : 0;
      dst.data[i] = val;
      dst.data[i + 1] = val;
      dst.data[i + 2] = val;
      dst.data[i + 3] = 255;
    }
    ctx.putImageData(dst, 0, 0);
    return out.toDataURL("image/png");
  }

  clearMask() {
    this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
  }

  setTool(tool) {
    this.tool = tool;
  }

  setBrushSize(size) {
    this.brushSize = size;
  }

  setOpacity(pct) {
    this.maskCanvas.style.opacity = pct / 100;
  }

  /* ---- internals ---- */

  _updateWrapperSize() {
    const wrapper = this.imageCanvas.parentElement;
    if (!wrapper || !this.imageCanvas.width) return;

    const w = this.imageCanvas.width;
    const h = this.imageCanvas.height;
    const availW = wrapper.parentElement.clientWidth;
    const availH = window.innerHeight * 0.82;

    const scale = Math.min(availW / w, availH / h, 1);
    wrapper.style.width = Math.floor(w * scale) + "px";
    wrapper.style.height = Math.floor(h * scale) + "px";
  }

  _bindEvents() {
    const c = this.cursorCanvas;
    c.addEventListener("pointerdown", (e) => this._onDown(e));
    c.addEventListener("pointermove", (e) => this._onMove(e));
    c.addEventListener("pointerup", () => this._onUp());
    c.addEventListener("pointerleave", () => this._onUp());
  }

  _pos(e) {
    const rect = this.cursorCanvas.getBoundingClientRect();
    const scaleX = this.cursorCanvas.width / rect.width;
    const scaleY = this.cursorCanvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  _onDown(e) {
    this.painting = true;
    this.lastPoint = this._pos(e);
    this._draw(this.lastPoint, this.lastPoint);
  }

  _onMove(e) {
    const p = this._pos(e);
    this._drawCursor(p);
    if (!this.painting) return;
    this._draw(this.lastPoint, p);
    this.lastPoint = p;
  }

  _onUp() {
    this.painting = false;
    this.lastPoint = null;
  }

  _draw(from, to) {
    const ctx = this.maskCtx;
    ctx.save();
    ctx.lineWidth = this.brushSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (this.tool === "eraser") {
      ctx.globalCompositeOperation = "destination-out";
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "rgba(255, 60, 60, 0.9)";
    }

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    ctx.restore();
  }

  _drawCursor(p) {
    const ctx = this.cursorCtx;
    const w = this.cursorCanvas.width;
    const h = this.cursorCanvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.beginPath();
    ctx.arc(p.x, p.y, this.brushSize / 2, 0, Math.PI * 2);
    ctx.strokeStyle = this.tool === "eraser" ? "#ff4444" : "#00e0ff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

window.MaskCanvas = MaskCanvas;
