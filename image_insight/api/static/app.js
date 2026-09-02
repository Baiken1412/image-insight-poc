// ---------- shared icons ----------
const ICON_CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"><path d="m5 12 4.5 4.5L19 7"/></svg>';
const ICON_WARN = '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"><path d="M12 4 3.5 19h17L12 4Z"/><path d="M12 9v4m0 3h.01"/></svg>';

// ---------- escaping ----------
function esc(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

// ---------- clock ----------
function updateClock() {
  const el = document.getElementById("clock");
  if (!el) return;
  const now = new Date();
  el.textContent =
    now.toLocaleTimeString("zh-CN", { hour12: false }) +
    "  " +
    now.toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" });
}
updateClock();
setInterval(updateClock, 1000);

// ---------- toast ----------
let toastTimer = null;
function showToast(message, kind = "ok") {
  const toast = document.getElementById("toast");
  const icon = document.getElementById("toast-icon");
  const text = document.getElementById("toast-text");
  text.textContent = message;
  icon.textContent = kind === "error" ? "!" : "✓";
  toast.classList.toggle("error", kind === "error");
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3000);
}

// ---------- lightbox (click a thumbnail for a full-size view) ----------
// Delegated on document.body since thumbnails are re-created on every
// render — binding directly to them would need re-wiring each time.
(function setupLightbox() {
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightbox-img");
  function open(src) {
    lightboxImg.src = src;
    lightbox.hidden = false;
  }
  function close() {
    lightbox.hidden = true;
    lightboxImg.src = "";
  }
  document.body.addEventListener("click", (e) => {
    const target = e.target.closest(".result-image img, .preview-item img");
    if (target) open(target.src);
  });
  lightbox.addEventListener("click", close);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lightbox.hidden) close();
  });
})();

// ---------- file selection (single shared upload area for both modes) ----------
// Both modes analyze the SAME selected photos — switching mode only changes
// which endpoint they get posted to and how the result is rendered, not
// what's selected. Keeps its own File[] array so a single thumbnail can be
// removed without clearing the whole batch (<input>.files is read-only, so
// removal works by rebuilding it through a DataTransfer).
function wireUpload(inputId, zoneId, previewsId, promptId) {
  const input = document.getElementById(inputId);
  const zone = document.getElementById(zoneId);
  const prompt = document.getElementById(promptId);
  const previews = document.getElementById(previewsId);
  let files = [];

  function syncInputFiles() {
    const dt = new DataTransfer();
    files.forEach((f) => dt.items.add(f));
    input.files = dt.files;
  }

  function render() {
    if (files.length === 0) {
      previews.innerHTML = "";
      previews.classList.remove("visible");
      prompt.style.display = "";
      return;
    }
    prompt.style.display = "none";
    previews.classList.add("visible");
    previews.innerHTML = files
      .map(
        (f, i) => `
        <div class="preview-item">
          <img src="${URL.createObjectURL(f)}" alt="${esc(f.name)}" title="${esc(f.name)}" />
          <button type="button" class="preview-remove" data-index="${i}" aria-label="移除这张照片">×</button>
        </div>`
      )
      .join("");
    previews.querySelectorAll(".preview-remove").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        files.splice(Number(btn.dataset.index), 1);
        syncInputFiles();
        render();
      });
    });
  }

  function addFiles(newFiles) {
    const images = Array.from(newFiles).filter((f) => f.type.startsWith("image/"));
    if (!images.length) return;
    files = files.concat(images);
    syncInputFiles();
    render();
    showToast(`已添加 ${images.length} 张照片`);
  }

  input.addEventListener("change", () => addFiles(input.files));
  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add("dragging");
    })
  );
  ["dragleave", "dragend"].forEach((evt) => zone.addEventListener(evt, () => zone.classList.remove("dragging")));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragging");
    if (e.dataTransfer && e.dataTransfer.files) addFiles(e.dataTransfer.files);
  });

  return {
    getFiles: () => files.slice(),
    clear: () => {
      files = [];
      syncInputFiles();
      render();
    },
  };
}
const upload = wireUpload("upload-files", "upload-zone", "file-previews", "upload-prompt");

function buildImageMap(files) {
  const map = {};
  files.forEach((f) => {
    map[f.name] = URL.createObjectURL(f);
  });
  return map;
}

// ---------- mode switching ----------
const MODE_META = {
  fast: {
    title: "开放词汇识别 · 秒级反馈",
    desc: '物品名称由模型自由描述，自动归类到 12 大类 / 84 细类；把握不足时标记"待核实"，不强行给出确定类别。',
    scope: ["名称", "数量", "分类"],
    buttonIdle: "开始快速识别",
    buttonBusy: "快速识别中…",
  },
  advanced: {
    title: "结构化理解 · 深度解析",
    desc: '输出品牌、型号、颜色、可见文字、外观特征等详细信息，并结合本地 OCR 交叉校验；不确定字段明确标注"疑似"或"无法确定"，绝不编造。',
    scope: ["品牌型号", "OCR文字", "外观特征", "逐项复核"],
    buttonIdle: "开始高级识别",
    buttonBusy: "高级解析中…",
  },
};

const state = { mode: "fast" };

function selectMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode-button").forEach((btn) => {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });
  const meta = MODE_META[mode];
  document.getElementById("mode-desc-title").textContent = meta.title;
  document.getElementById("mode-desc-text").textContent = meta.desc;
  document.getElementById("mode-scope").innerHTML = meta.scope
    .map((s) => `<span class="scope-pill">${esc(s)}</span>`)
    .join("");
  document.getElementById("analyze-button-text").textContent = meta.buttonIdle;
  document.getElementById("result").innerHTML = "";
}
document.querySelectorAll(".mode-button").forEach((btn) => btn.addEventListener("click", () => selectMode(btn.dataset.mode)));

// ---------- confidence / review tags ----------
// 特别需求补充.md §1.3: confidence is for ranking/uncertainty only and must
// NOT be shown to the end user as a percentage — so this renders a
// qualitative tier (高/中/低), never the raw number.
function confidenceTag(confidence) {
  if (confidence == null) return "";
  const tier = confidence >= 0.7 ? "高" : confidence >= 0.35 ? "中" : "低";
  return `<span class="confidence-tag">把握 ${tier}</span>`;
}

function reviewTags({ forced, sensitive, pending }) {
  let out = "";
  if (forced) out += '<span class="review-tag sensitive">强制核实</span>';
  else if (sensitive) out += '<span class="review-tag sensitive">敏感类别</span>';
  if (pending) out += '<span class="review-tag">待核实</span>';
  return out;
}

// ---------- field status dot (read-only — reflects the model's own
// confirmed/suspected/unknown status, not a user-editable toggle) ----------
function fieldMarkup(label, field) {
  if (!field) return "";
  const status = field.status || "unknown";
  const dotChar = status === "confirmed" ? "✓" : status === "suspected" ? "?" : "·";
  const titleMap = { confirmed: "已确认", suspected: "疑似，建议人工确认", unknown: "无法确定" };
  const value = field.value == null ? "—" : esc(field.value);
  return `
    <div class="field">
      <label>${esc(label)}</label>
      <div class="field-value">
        <span>${value}</span>
        <span class="field-confirm ${status}" title="${titleMap[status] || ""}">${dotChar}</span>
      </div>
    </div>`;
}

// ---------- truncated-generation banner ----------
// The model's raw response was cut off (hit max_new_tokens) before its JSON
// closed; image_insight.vlm.json_repair salvages whatever complete items
// finished generating, but the photo may genuinely have had more items than
// are shown — this must never look like a normal, complete result.
function truncatedBanner(truncated) {
  return truncated
    ? '<div class="review-banner">✂ 模型生成内容被截断，本结果可能遗漏了照片中的部分物品，请人工核对数量是否完整。</div>'
    : "";
}

// ---------- per-item detail blocks ----------
function itemBlockFast(item) {
  const pending = !!item.pending_review;
  const tags = reviewTags({ sensitive: item.is_sensitive_category, pending });
  const categoryLine = pending
    ? "待核实"
    : `${esc(item.category)}${item.subcategory ? " / " + esc(item.subcategory) : ""}`;
  return `
    <div class="item-block">
      <div class="result-title">
        <h3>${esc(item.name)} <span class="quantity">× ${esc(item.count)}</span></h3>
        ${tags}
        ${!pending ? confidenceTag(item.confidence) : ""}
      </div>
      <p class="quick-category">业务分类　<strong>${categoryLine}</strong></p>
    </div>`;
}

function itemBlockAdvanced(item) {
  const pending = !!item.category_pending_review;
  const forced = !!item.is_forced_manual_review;
  const categoryValue = item.category && item.category.value ? item.category.value : "未分类";
  const subcategoryValue = item.subcategory && item.subcategory.value ? item.subcategory.value : null;
  const nameLine = pending ? "待核实" : `${esc(categoryValue)}${subcategoryValue ? " / " + esc(subcategoryValue) : ""}`;
  const tags = reviewTags({ forced, pending });
  const chips = (item.visible_text || []).map((t) => `<span class="ocr-chip">${esc(t)}</span>`).join("");
  const uncertainty = item.needs_manual_review
    ? `<div class="uncertainty">${ICON_WARN}<span>存在不确定字段，建议人工复核后再采信本次结果</span></div>`
    : "";
  return `
    <div class="item-block">
      <div class="result-title">
        <h3>${nameLine} <span class="quantity">× ${esc(item.count)}</span></h3>
        ${tags}
        ${!pending ? confidenceTag(item.category_confidence) : ""}
      </div>
      <div class="field-grid">
        ${fieldMarkup("品牌", item.brand)}
        ${fieldMarkup("型号", item.model)}
        ${fieldMarkup("颜色", item.color)}
        ${fieldMarkup("特征", item.condition)}
      </div>
      ${chips ? `<div class="ocr-row">${chips}</div>` : ""}
      ${item.appearance_notes ? `<p class="appearance">${esc(item.appearance_notes)}</p>` : ""}
      ${uncertainty}
    </div>`;
}

// ---------- one goods_id group -> one result-card ----------
function groupPhotosMarkup(group, imageMap) {
  const urls = (group.source_files || []).map((name) => imageMap[name]).filter(Boolean);
  const shown = urls.slice(0, 4);
  const badge = urls.length > 1 ? `<span class="image-index">${urls.length} 张</span>` : "";
  return `<div class="result-image">${shown.map((u) => `<img src="${u}" alt="" />`).join("")}${badge}</div>`;
}

function resultCard(group, mode, imageMap, index) {
  const result = group.result || {};
  let inner;
  let isSensitive = false;
  let isPending = false;

  if (result.parse_error) {
    inner = `<div class="error-banner">✕ 模型输出无法解析，已转人工复核。<br>详情：${esc(result.parse_error)}</div>`;
  } else {
    const items = result.items || [];
    const banner = truncatedBanner(result.truncated);
    isPending = !!result.truncated;
    if (items.length === 0) {
      inner = banner + '<p class="empty-note">未识别到任何物品</p>';
    } else {
      const ocrTexts = result.ocr_text || [];
      const ocrBlock = ocrTexts.length
        ? `<div class="ocr-row">${ocrTexts.map((t) => `<span class="ocr-chip">${esc(t)}</span>`).join("")}</div>`
        : "";
      const blocks = items.map((it) => (mode === "fast" ? itemBlockFast(it) : itemBlockAdvanced(it))).join("");
      inner = banner + blocks + ocrBlock;
      isSensitive = items.some((it) => (mode === "fast" ? it.is_sensitive_category : it.is_forced_manual_review));
      isPending = isPending || items.some((it) => (mode === "fast" ? it.pending_review : it.category_pending_review));
    }
  }

  const cardClass = isSensitive ? " sensitive" : isPending ? " pending" : "";
  const files = group.source_files || [];
  const filesLine =
    files.length > 1 ? `合并自 ${files.length} 张照片：${files.map(esc).join("、")}` : esc(files[0] || group.label);

  return `
    <article class="result-card${cardClass}" style="animation-delay:${index * 0.06}s">
      ${groupPhotosMarkup(group, imageMap)}
      <div class="result-detail">
        <div class="group-meta"><span class="group-label">${esc(group.label)}</span><span class="group-files">${filesLine}</span></div>
        ${inner}
      </div>
    </article>`;
}

// ---------- summary band ----------
function renderSummaryBand(tiles) {
  return `<div class="summary-band">
    <div class="summary-main"><div class="summary-icon">${ICON_CHECK}</div><div><strong>${esc(
    tiles.mainTitle
  )}</strong><span>${esc(tiles.mainSub)}</span></div></div>
    ${tiles.stats
      .map(([label, value]) => `<div class="summary-stat"><strong class="accent">${esc(value)}</strong><span>${esc(label)}</span></div>`)
      .join("")}
  </div>`;
}

function formatElapsed(ms) {
  return (ms / 1000).toFixed(1) + "s";
}

function summaryTiles(groups, mode, elapsedMs) {
  const items = groups.flatMap((g) => (g.result && g.result.items) || []);
  const totalQty = items.reduce((sum, it) => sum + (it.count || 0), 0);
  const pendingKey = mode === "fast" ? "pending_review" : "category_pending_review";
  const pendingCount = items.filter((it) => it[pendingKey]).length;
  const sensitiveCount = items.filter((it) =>
    mode === "fast" ? it.is_sensitive_category : it.is_forced_manual_review
  ).length;

  let mainSub = mode === "fast" ? "快速视觉识别 · 建议人工复核后入库" : "高级结构化解析 · 建议人工复核后入库";
  const flags = [];
  if (sensitiveCount) flags.push(`${sensitiveCount} 项敏感/强制核实`);
  if (pendingCount) flags.push(`${pendingCount} 项待核实`);
  if (flags.length) mainSub = flags.join(" · ");

  return {
    mainTitle: "识别完成 · 已生成结构化建议",
    mainSub,
    stats: [
      ["物品总数", totalQty],
      ["识别条目", items.length],
      ["处理用时", formatElapsed(elapsedMs)],
    ],
  };
}

function renderResults(data, mode, imageMap, elapsedMs) {
  const container = document.getElementById("result");
  const groups = data.groups || [];
  if (groups.length === 0) {
    container.innerHTML = '<p class="empty-note">未识别到任何物品，或本次结果需要人工核实。</p>';
    return;
  }
  const summary = renderSummaryBand(summaryTiles(groups, mode, elapsedMs));
  const cards = groups.map((g, i) => resultCard(g, mode, imageMap, i)).join("");
  container.innerHTML = summary + `<div class="results-stack">${cards}</div>`;
}

// ---------- loading view ----------
// There is one HTTP request per batch (not one per stage), so this cycles
// through the pipeline's real stages on a timer rather than reacting to
// actual server events; the progress bar creeps toward ~90% on an
// asymptotic curve (real request duration is unknown in advance) and simply
// gets replaced by the real result markup on completion, never claiming
// 100% before the response actually arrives.
const FAST_STAGES = ["正在校验图像质量…", "正在识别物品类型…", "正在匹配细类…", "正在生成结果…"];
const ADVANCED_STAGES = [
  "正在校验图像质量…",
  "正在分析物品外观…",
  "正在提取品牌与型号…",
  "正在做 OCR 文字校验…",
  "正在标记不确定字段…",
];

function renderLoadingView(mode) {
  const stages = mode === "fast" ? FAST_STAGES : ADVANCED_STAGES;
  const title = mode === "fast" ? "正在进行快速视觉识别" : "正在进行高级结构化解析";
  document.getElementById("result").innerHTML = `
    <div class="loading-view">
      <div class="scanner">
        <div class="scanner-orbit"></div>
        <div class="loading-title">${title}</div>
        <div class="loading-step" id="loading-step">${stages[0]}</div>
        <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>
      </div>
    </div>`;
  let i = 0;
  const stepTimer = setInterval(() => {
    i = (i + 1) % stages.length;
    const el = document.getElementById("loading-step");
    if (el) el.textContent = stages[i];
  }, 1400);
  const start = performance.now();
  const progressTimer = setInterval(() => {
    const fill = document.getElementById("progress-fill");
    if (!fill) return;
    const t = (performance.now() - start) / 1000;
    const pct = 90 * (1 - Math.exp(-t / 12));
    fill.style.width = pct.toFixed(1) + "%";
  }, 200);
  return () => {
    clearInterval(stepTimer);
    clearInterval(progressTimer);
  };
}

// ---------- submit ----------
async function handleSubmit(event) {
  event.preventDefault();
  const files = upload.getFiles();
  if (!files.length) {
    showToast("请先选择照片", "error");
    return;
  }
  const mode = state.mode;
  const url = mode === "fast" ? "/api/analyze/fast" : "/api/analyze/advanced";
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));
  const imageMap = buildImageMap(files);

  const button = document.getElementById("analyze-button");
  const buttonText = document.getElementById("analyze-button-text");
  button.disabled = true;
  buttonText.textContent = MODE_META[mode].buttonBusy;
  const stopLoading = renderLoadingView(mode);
  const start = performance.now();
  try {
    const response = await fetch(url, { method: "POST", body: formData });
    const body = await response.json();
    const elapsed = performance.now() - start;
    stopLoading();
    if (!response.ok) {
      document.getElementById("result").innerHTML = `<div class="error-banner">✕ 请求失败：${esc(
        body.detail || response.status
      )}</div>`;
      showToast("识别失败：" + (body.detail || response.status), "error");
      return;
    }
    renderResults(body, mode, imageMap, elapsed);
    showToast("识别完成，已生成结构化结果并自动保存");
  } catch (err) {
    stopLoading();
    document.getElementById("result").innerHTML = `<div class="error-banner">✕ 网络或程序错误：${esc(String(err))}</div>`;
    showToast("网络或程序错误", "error");
  } finally {
    button.disabled = false;
    buttonText.textContent = MODE_META[mode].buttonIdle;
  }
}
document.getElementById("upload-form").addEventListener("submit", handleSubmit);

document.getElementById("clear-button").addEventListener("click", () => {
  document.getElementById("result").innerHTML = "";
});
