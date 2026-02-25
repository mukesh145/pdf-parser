/**
 * app.js — wires the upload form, image navigation, and submit button
 * to the MaskCanvas drawing engine and the backend API.
 */

(function () {
  /* ---- state ---- */
  let imageList = [];   // filenames from the server
  let currentIdx = 0;
  let masks = {};       // filename → dataURL (saved per-image)
  let editor = null;    // MaskCanvas instance

  /* ---- DOM refs ---- */
  const uploadSection  = document.getElementById("upload-section");
  const editorSection  = document.getElementById("editor-section");
  const dropZone       = document.getElementById("drop-zone");
  const fileInput      = document.getElementById("file-input");
  const btnBrush       = document.getElementById("btn-brush");
  const btnEraser      = document.getElementById("btn-eraser");
  const brushRange     = document.getElementById("brush-size");
  const brushLabel     = document.getElementById("brush-size-label");
  const opacityRange   = document.getElementById("mask-opacity");
  const opacityLabel   = document.getElementById("mask-opacity-label");
  const btnClear       = document.getElementById("btn-clear");
  const btnPrev        = document.getElementById("btn-prev");
  const btnNext        = document.getElementById("btn-next");
  const counter        = document.getElementById("image-counter");
  const btnSubmit      = document.getElementById("btn-submit");
  const toast          = document.getElementById("toast");

  /* ---- init ---- */
  editor = new MaskCanvas(
    document.getElementById("canvas-image"),
    document.getElementById("canvas-mask"),
    document.getElementById("canvas-cursor")
  );

  /* ---- upload handling ---- */
  dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    uploadFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener("change", () => uploadFiles(fileInput.files));

  async function uploadFiles(fileListObj) {
    const hasZip = Array.from(fileListObj).some(
      (f) => f.name.toLowerCase().endsWith(".zip")
    );

    if (hasZip) {
      await fetch("/api/clear", { method: "POST" });
      imageList = [];
      masks = {};
    }

    const fd = new FormData();
    for (const f of fileListObj) fd.append("files", f);
    showToast("Uploading…");
    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      showToast(`Uploaded ${data.uploaded.length} image(s)`);
      await refreshImageList();
    } catch (err) {
      showToast("Upload failed: " + err.message, true);
    }
  }

  async function refreshImageList() {
    const res = await fetch("/api/images");
    const data = await res.json();
    imageList = data.images;
    if (imageList.length === 0) return;
    currentIdx = 0;
    uploadSection.classList.add("hidden");
    editorSection.classList.remove("hidden");
    loadCurrentImage();
  }

  /* ---- image navigation ---- */
  function loadCurrentImage() {
    const filename = imageList[currentIdx];
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      editor.loadImage(img);
      if (masks[filename]) editor.loadMask(masks[filename]);
      updateNav();
    };
    img.src = `/api/images/${encodeURIComponent(filename)}`;
  }

  function saveCurrent() {
    if (imageList.length === 0) return;
    masks[imageList[currentIdx]] = editor.saveMaskRaw();
  }

  function updateNav() {
    counter.textContent = `${currentIdx + 1} / ${imageList.length}`;
    btnPrev.disabled = currentIdx === 0;
    btnNext.disabled = currentIdx === imageList.length - 1;
  }

  btnPrev.addEventListener("click", () => { saveCurrent(); currentIdx--; loadCurrentImage(); });
  btnNext.addEventListener("click", () => { saveCurrent(); currentIdx++; loadCurrentImage(); });

  /* ---- toolbar ---- */
  btnBrush.addEventListener("click", () => {
    editor.setTool("brush");
    btnBrush.classList.add("active");
    btnEraser.classList.remove("active");
  });
  btnEraser.addEventListener("click", () => {
    editor.setTool("eraser");
    btnEraser.classList.add("active");
    btnBrush.classList.remove("active");
  });
  brushRange.addEventListener("input", () => {
    editor.setBrushSize(+brushRange.value);
    brushLabel.textContent = brushRange.value;
  });
  opacityRange.addEventListener("input", () => {
    editor.setOpacity(+opacityRange.value);
    opacityLabel.textContent = opacityRange.value + "%";
  });
  btnClear.addEventListener("click", () => editor.clearMask());

  /* ---- submit ---- */
  btnSubmit.addEventListener("click", async () => {
    saveCurrent();

    const payload = imageList
      .filter((f) => masks[f])
      .map((f) => ({ image_filename: f, mask_data_url: masks[f] }));

    if (payload.length === 0) {
      showToast("Draw at least one mask before submitting.", true);
      return;
    }

    showToast("Submitting masks…");
    try {
      const res = await fetch("/api/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ masks: payload }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      showToast(`Queued training job ${data.job_id} (${data.pairs} pairs)`);
    } catch (err) {
      showToast("Submit failed: " + err.message, true);
    }
  });

  /* ---- toast helper ---- */
  function showToast(msg, isError) {
    toast.textContent = msg;
    toast.className = "toast" + (isError ? " error" : "");
    toast.classList.remove("hidden");
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => toast.classList.add("hidden"), 4000);
  }
})();
