const form = document.getElementById('job-form');
const startBtn = document.getElementById('start-btn');
const barFill = document.getElementById('bar-fill');
const progressText = document.getElementById('progress-text');
const statusMessage = document.getElementById('status-message');
const statusDetailBox = document.getElementById('status-detail-box');
const statusDetail = document.getElementById('status-detail');
const activeChunks = document.getElementById('active-chunks');
const resultCard = document.getElementById('result-card');
const summary = document.getElementById('summary');
const tableBody = document.querySelector('#cards-table tbody');
const searchInput = document.getElementById('search');
const onlyIssues = document.getElementById('only-issues');
const addRowBtn = document.getElementById('add-row');
const exportCsvBtn = document.getElementById('export-csv');
const exportJsonBtn = document.getElementById('export-json');

const sourceInput = form.querySelector('input[name="source"]');
const rangesInput = form.querySelector('input[name="ranges"]');
const stepInput = form.querySelector('input[name="step"]');
const overlapInput = form.querySelector('input[name="overlap"]');

const previewCard = document.getElementById('preview-card');
const previewToggleBtn = document.getElementById('preview-toggle');
const previewError = document.getElementById('preview-error');
const previewFrameWrap = document.getElementById('preview-frame-wrap');
const previewCanvas = document.getElementById('pdf-preview-canvas');
const previewImage = document.getElementById('image-preview');
const previewEmpty = document.getElementById('preview-empty');
const pagePrevBtn = document.getElementById('page-prev');
const pageNextBtn = document.getElementById('page-next');
const pageIndicator = document.getElementById('page-indicator');
const chunkSummary = document.getElementById('chunk-summary');
const chunkList = document.getElementById('chunk-list');

const PREVIEW_COLLAPSE_KEY = 'docs2anki.previewCollapsed';
const PREVIEW_EMPTY_TEXT = 'ファイルを選択するとここにプレビューが表示されます';
const IMAGE_PREVIEW_MIME_PREFIX = 'image/';

let pollTimer = null;
let currentJobId = '';
let cards = [];

let previewMode = 'none'; // none | pdf | image
let previewPdfDoc = null;
let previewPageCount = 0;
let previewChunks = [];
let selectedChunkIndex = -1;
let previewPage = 1;
let previewDebounceTimer = null;
let previewResizeTimer = null;
let previewRenderToken = 0;
let previewLoadingToken = 0;
let previewRenderTask = null;
let previewImageEntries = [];
let pdfPreviewReady = false;

function setStatus(text) {
  statusMessage.textContent = text || '';
}

function setStatusDetail(text) {
  const message = String(text || '').trim();
  if (!message) {
    statusDetailBox.hidden = true;
    statusDetail.textContent = '';
    return;
  }
  statusDetailBox.hidden = false;
  statusDetail.textContent = message;
}

function formatList(items, limit = 12) {
  const values = Array.isArray(items) ? items.map((v) => String(v || '').trim()).filter(Boolean) : [];
  if (values.length === 0) {
    return '';
  }
  const head = values.slice(0, limit).map((v) => `- ${v}`).join('\n');
  if (values.length <= limit) {
    return head;
  }
  return `${head}\n- ... ほか ${values.length - limit} 件`;
}

function buildJobDetail(data = {}) {
  const hasFailed = Array.isArray(data.failedChunks) && data.failedChunks.length > 0;
  const hasWarnings = Array.isArray(data.warnings) && data.warnings.length > 0;
  const hasError = Boolean(String(data.error || '').trim());
  if (!hasFailed && !hasWarnings && !hasError) {
    return '';
  }

  const lines = [];
  if (data.jobId || currentJobId) {
    lines.push(`job: ${data.jobId || currentJobId}`);
  }
  if (data.status) {
    lines.push(`status: ${data.status}`);
  }
  if (Array.isArray(data.failedChunks) && data.failedChunks.length > 0) {
    lines.push(`failed chunks: ${data.failedChunks.join(', ')}`);
  }
  if (data.error) {
    lines.push('');
    lines.push('error:');
    lines.push(String(data.error).trim());
  }
  if (Array.isArray(data.warnings) && data.warnings.length > 0) {
    lines.push('');
    lines.push('warnings:');
    lines.push(formatList(data.warnings, 14));
  }
  return lines.join('\n').trim();
}

function setProgress(completed, total) {
  const safeTotal = total > 0 ? total : 1;
  const percent = Math.min(100, Math.round((completed / safeTotal) * 100));
  barFill.style.width = `${percent}%`;
  progressText.textContent = `${completed}/${total} (${percent}%)`;
}

function renderActive(labels = []) {
  activeChunks.innerHTML = '';
  labels.slice(0, 6).forEach((label) => {
    const span = document.createElement('span');
    span.textContent = `処理中: ${label}`;
    activeChunks.appendChild(span);
  });
  if (labels.length > 6) {
    const span = document.createElement('span');
    span.textContent = `ほか ${labels.length - 6} 件`;
    activeChunks.appendChild(span);
  }
}

function escapeHTML(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function normalizeCard(card = {}) {
  return {
    page: String(card.page ?? ''),
    question: String(card.question ?? ''),
    answer: String(card.answer ?? ''),
    confidence: Number.isFinite(Number(card.confidence)) ? Number(card.confidence) : 0,
    issue: Array.isArray(card.issue) ? card.issue.map((x) => String(x)).filter(Boolean) : [],
  };
}

function parseRangesExpression(expression) {
  const parts = String(expression || '').split(',').map((v) => v.trim()).filter(Boolean);
  if (parts.length === 0) {
    throw new Error('ページ範囲を指定してください');
  }

  const ranges = [];
  for (const part of parts) {
    if (part.includes('-')) {
      const [left, right] = part.split('-', 2).map((v) => v.trim());
      const start = Number.parseInt(left, 10);
      const end = Number.parseInt(right, 10);
      if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < 1 || end < start) {
        throw new Error(`範囲指定が不正です: ${part}`);
      }
      ranges.push({ start, end });
      continue;
    }

    const page = Number.parseInt(part, 10);
    if (!Number.isInteger(page) || page < 1) {
      throw new Error(`ページ指定が不正です: ${part}`);
    }
    ranges.push({ start: page, end: page });
  }

  ranges.sort((a, b) => {
    if (a.start === b.start) {
      return a.end - b.end;
    }
    return a.start - b.start;
  });

  const merged = [];
  for (const current of ranges) {
    const last = merged[merged.length - 1];
    if (!last) {
      merged.push({ ...current });
      continue;
    }
    if (current.start <= last.end + 1) {
      if (current.end > last.end) {
        last.end = current.end;
      }
      continue;
    }
    merged.push({ ...current });
  }

  return merged;
}

function parseStepOverlap() {
  const step = Number.parseInt(stepInput.value || '1', 10);
  const overlap = Number.parseInt(overlapInput.value || '0', 10);

  if (!Number.isInteger(step) || step < 1) {
    throw new Error('step は1以上で指定してください');
  }
  if (!Number.isInteger(overlap) || overlap < 0 || overlap >= step) {
    throw new Error('overlap は0以上かつ step 未満で指定してください');
  }

  return { step, overlap };
}

function computeChunks(ranges, step, overlap) {
  const chunks = [];
  const stride = step - overlap;

  ranges.forEach((range) => {
    let page = range.start;
    while (page <= range.end) {
      const end = Math.min(page + step - 1, range.end);
      chunks.push({ start: page, end });
      if (end === range.end) {
        break;
      }
      page += stride;
    }
  });

  return chunks;
}

function normalizePreviewMIME(value) {
  return String(value || '').trim().toLowerCase().split(';', 1)[0];
}

function inferMimeFromName(name) {
  const lowered = String(name || '').trim().toLowerCase();
  if (lowered.endsWith('.pdf')) return 'application/pdf';
  if (lowered.endsWith('.png')) return 'image/png';
  if (lowered.endsWith('.jpg') || lowered.endsWith('.jpeg')) return 'image/jpeg';
  if (lowered.endsWith('.webp')) return 'image/webp';
  if (lowered.endsWith('.gif')) return 'image/gif';
  if (lowered.endsWith('.bmp')) return 'image/bmp';
  if (lowered.endsWith('.tif') || lowered.endsWith('.tiff')) return 'image/tiff';
  return '';
}

function getSourceKind(file) {
  const mime = normalizePreviewMIME(file?.type || '');
  const fallback = inferMimeFromName(file?.name || '');
  const resolved = mime || fallback;
  if (resolved === 'application/pdf') return 'pdf';
  if (resolved.startsWith(IMAGE_PREVIEW_MIME_PREFIX)) return 'image';
  return 'unknown';
}

function setPreviewError(message) {
  const text = String(message || '').trim();
  if (text === '') {
    previewError.hidden = true;
    previewError.textContent = '';
    return;
  }
  previewError.hidden = false;
  previewError.textContent = text;
}

function setPreviewCollapsed(collapsed) {
  previewCard.classList.toggle('collapsed', Boolean(collapsed));
  previewToggleBtn.setAttribute('aria-expanded', String(!collapsed));
  try {
    localStorage.setItem(PREVIEW_COLLAPSE_KEY, collapsed ? '1' : '0');
  } catch (_) {
    // ignore
  }
}

function setupPreviewCollapsedState() {
  let collapsed = false;
  try {
    collapsed = localStorage.getItem(PREVIEW_COLLAPSE_KEY) === '1';
  } catch (_) {
    collapsed = false;
  }
  setPreviewCollapsed(collapsed);
}

function getSelectedChunk() {
  if (selectedChunkIndex < 0 || selectedChunkIndex >= previewChunks.length) {
    return null;
  }
  return previewChunks[selectedChunkIndex];
}

function updatePreviewIndicator() {
  const chunk = getSelectedChunk();
  if (previewMode === 'image') {
    if (previewPageCount <= 0) {
      pageIndicator.textContent = 'image -';
      return;
    }
    if (!chunk) {
      pageIndicator.textContent = `image ${previewPage}/${previewPageCount}`;
      return;
    }
    pageIndicator.textContent = `image ${previewPage}/${previewPageCount} | chunk ${chunk.start}-${chunk.end}`;
    return;
  }
  if (!previewPdfDoc || previewMode !== 'pdf') {
    pageIndicator.textContent = 'page -';
    return;
  }
  if (!chunk) {
    pageIndicator.textContent = `page ${previewPage}/${previewPageCount}`;
    return;
  }
  pageIndicator.textContent = `page ${previewPage}/${previewPageCount} | chunk ${chunk.start}-${chunk.end}`;
}

function updatePreviewNavButtons() {
  if (previewMode === 'image') {
    if (previewPageCount <= 0) {
      pagePrevBtn.disabled = true;
      pageNextBtn.disabled = true;
      return;
    }
    const chunk = getSelectedChunk();
    let minPage = 1;
    let maxPage = previewPageCount;
    if (chunk) {
      minPage = Math.max(1, chunk.start);
      maxPage = Math.min(previewPageCount, chunk.end);
    }
    if (minPage > maxPage) {
      pagePrevBtn.disabled = true;
      pageNextBtn.disabled = true;
      return;
    }
    pagePrevBtn.disabled = previewPage <= minPage;
    pageNextBtn.disabled = previewPage >= maxPage;
    return;
  }
  if (!previewPdfDoc || previewMode !== 'pdf') {
    pagePrevBtn.disabled = true;
    pageNextBtn.disabled = true;
    return;
  }

  const chunk = getSelectedChunk();
  let minPage = 1;
  let maxPage = previewPageCount;

  if (chunk) {
    minPage = Math.max(1, chunk.start);
    maxPage = Math.min(previewPageCount, chunk.end);
  }

  if (minPage > maxPage) {
    pagePrevBtn.disabled = true;
    pageNextBtn.disabled = true;
    return;
  }

  pagePrevBtn.disabled = previewPage <= minPage;
  pageNextBtn.disabled = previewPage >= maxPage;
}

function renderChunkList() {
  chunkList.innerHTML = '';
  chunkSummary.textContent = `${previewChunks.length}件`;

  if (previewChunks.length === 0) {
    const empty = document.createElement('span');
    empty.className = 'preview-empty-inline';
    empty.textContent = previewMode === 'pdf'
      ? '範囲とstepの設定後にチャンクを表示します'
      : 'ファイルを選択するとチャンクを表示します';
    chunkList.appendChild(empty);
    return;
  }

  previewChunks.forEach((chunk, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `chunk-btn${index === selectedChunkIndex ? ' active' : ''}`;
    button.textContent = `${chunk.start}-${chunk.end}`;
    button.title = `対象ページ ${chunk.start}-${chunk.end}`;
    button.addEventListener('click', () => {
      selectedChunkIndex = index;
      previewPage = chunk.start;
      renderChunkList();
      void renderPreviewPage();
    });
    chunkList.appendChild(button);
  });
}

function clearPreviewCanvas() {
  const context = previewCanvas.getContext('2d');
  if (context) {
    context.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  }
  previewCanvas.width = 0;
  previewCanvas.height = 0;
  previewCanvas.style.width = '0px';
  previewCanvas.style.height = '0px';
}

function clearPreviewImage() {
  previewImageEntries.forEach((entry) => {
    if (entry?.url) {
      URL.revokeObjectURL(entry.url);
    }
  });
  previewImageEntries = [];
  previewImage.hidden = true;
  previewImage.removeAttribute('src');
}

function showPdfPreviewSurface() {
  previewCanvas.hidden = false;
  previewImage.hidden = true;
}

function showImagePreviewSurface() {
  previewCanvas.hidden = true;
  previewImage.hidden = false;
}

async function renderImagePreviewPage() {
  if (previewMode !== 'image' || previewPageCount <= 0) {
    updatePreviewIndicator();
    updatePreviewNavButtons();
    return;
  }

  const chunk = getSelectedChunk();
  let minPage = 1;
  let maxPage = previewPageCount;
  if (chunk) {
    minPage = Math.max(1, chunk.start);
    maxPage = Math.min(previewPageCount, chunk.end);
    if (minPage > maxPage) {
      setPreviewError(`チャンク ${chunk.start}-${chunk.end} は画像枚数(${previewPageCount})の範囲外です`);
      previewFrameWrap.classList.remove('has-file');
      previewEmpty.hidden = false;
      previewEmpty.textContent = '対象画像がありません';
      updatePreviewIndicator();
      updatePreviewNavButtons();
      return;
    }
  }

  previewPage = Math.max(minPage, Math.min(maxPage, previewPage));
  const entry = previewImageEntries[previewPage - 1];
  if (!entry) {
    setPreviewError(`画像ページ ${previewPage} の取得に失敗しました`);
    updatePreviewIndicator();
    updatePreviewNavButtons();
    return;
  }

  showImagePreviewSurface();
  previewImage.src = entry.url;
  previewFrameWrap.classList.add('has-file');
  previewEmpty.hidden = true;
  setPreviewError('');
  updatePreviewIndicator();
  updatePreviewNavButtons();
}

async function renderPreviewPage() {
  const token = ++previewRenderToken;

  if (previewMode === 'image') {
    await renderImagePreviewPage();
    return;
  }

  if (!previewPdfDoc || previewMode !== 'pdf') {
    clearPreviewCanvas();
    updatePreviewIndicator();
    updatePreviewNavButtons();
    return;
  }

  const chunk = getSelectedChunk();
  let minPage = 1;
  let maxPage = previewPageCount;

  if (chunk) {
    minPage = Math.max(1, chunk.start);
    maxPage = Math.min(previewPageCount, chunk.end);
    if (minPage > maxPage) {
      setPreviewError(`チャンク ${chunk.start}-${chunk.end} はPDF総ページ数(${previewPageCount})の範囲外です`);
      clearPreviewCanvas();
      updatePreviewIndicator();
      updatePreviewNavButtons();
      return;
    }
  }

  previewPage = Math.max(minPage, Math.min(maxPage, previewPage));

  try {
    const page = await previewPdfDoc.getPage(previewPage);
    if (token !== previewRenderToken) {
      return;
    }

    const natural = page.getViewport({ scale: 1 });
    const availableWidth = Math.max(260, previewFrameWrap.clientWidth - 22);
    const scale = availableWidth / natural.width;
    const viewport = page.getViewport({ scale });

    const pixelRatio = window.devicePixelRatio || 1;
    previewCanvas.width = Math.floor(viewport.width * pixelRatio);
    previewCanvas.height = Math.floor(viewport.height * pixelRatio);
    previewCanvas.style.width = `${Math.floor(viewport.width)}px`;
    previewCanvas.style.height = `${Math.floor(viewport.height)}px`;

    const context = previewCanvas.getContext('2d', { alpha: false });
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, viewport.width, viewport.height);

    if (previewRenderTask && typeof previewRenderTask.cancel === 'function') {
      try {
        previewRenderTask.cancel();
      } catch (_) {
        // ignore
      }
    }

    previewRenderTask = page.render({
      canvasContext: context,
      viewport,
    });

    await previewRenderTask.promise;
    if (token !== previewRenderToken) {
      return;
    }

    showPdfPreviewSurface();
    previewFrameWrap.classList.add('has-file');
    previewEmpty.hidden = true;
    setPreviewError('');
  } catch (error) {
    if (error && error.name === 'RenderingCancelledException') {
      return;
    }
    setPreviewError(`ページ描画に失敗しました: ${error.message || error}`);
  }

  updatePreviewIndicator();
  updatePreviewNavButtons();
}

function refreshChunkPreview() {
  if (previewMode === 'image') {
    if (previewPageCount <= 0) {
      previewChunks = [];
      selectedChunkIndex = -1;
      previewPage = 1;
      renderChunkList();
      updatePreviewIndicator();
      updatePreviewNavButtons();
      return;
    }
    try {
      const ranges = parseRangesExpression(rangesInput.value);
      const { step, overlap } = parseStepOverlap();
      const nextChunks = computeChunks(ranges, step, overlap);
      if (nextChunks.length === 0) {
        throw new Error('対象ページがありません');
      }

      const clamped = [];
      let adjustedCount = 0;
      nextChunks.forEach((chunk) => {
        if (chunk.start > previewPageCount) {
          adjustedCount += 1;
          return;
        }
        const adjusted = {
          start: Math.max(1, chunk.start),
          end: Math.min(previewPageCount, chunk.end),
        };
        if (adjusted.end !== chunk.end) {
          adjustedCount += 1;
        }
        if (adjusted.start <= adjusted.end) {
          clamped.push(adjusted);
        }
      });
      if (clamped.length === 0) {
        throw new Error(`指定範囲が画像枚数(${previewPageCount})の範囲外です`);
      }

      const previous = previewChunks[selectedChunkIndex];
      previewChunks = clamped;
      if (previous) {
        const matched = previewChunks.findIndex((chunk) => chunk.start === previous.start && chunk.end === previous.end);
        selectedChunkIndex = matched >= 0 ? matched : 0;
      } else {
        selectedChunkIndex = 0;
      }

      const selected = getSelectedChunk();
      if (selected) {
        if (previewPage < selected.start || previewPage > selected.end) {
          previewPage = selected.start;
        }
      } else {
        previewPage = 1;
      }

      renderChunkList();
      void renderPreviewPage();
      if (adjustedCount > 0) {
        setPreviewError(`指定範囲の一部を画像枚数(${previewPageCount})に合わせて調整しました`);
      } else {
        setPreviewError('');
      }
    } catch (error) {
      previewChunks = [];
      selectedChunkIndex = -1;
      previewPage = 1;
      renderChunkList();
      updatePreviewIndicator();
      updatePreviewNavButtons();
      setPreviewError(error.message || 'プレビュー計算に失敗しました');
    }
    return;
  }

  if (!previewPdfDoc || previewMode !== 'pdf') {
    previewChunks = [];
    selectedChunkIndex = -1;
    previewPage = 1;
    renderChunkList();
    clearPreviewCanvas();
    updatePreviewIndicator();
    updatePreviewNavButtons();
    return;
  }

  try {
    const ranges = parseRangesExpression(rangesInput.value);
    const { step, overlap } = parseStepOverlap();
    const nextChunks = computeChunks(ranges, step, overlap);

    if (nextChunks.length === 0) {
      throw new Error('対象ページがありません');
    }

    const previous = previewChunks[selectedChunkIndex];
    previewChunks = nextChunks;

    if (previous) {
      const matched = previewChunks.findIndex((chunk) => chunk.start === previous.start && chunk.end === previous.end);
      selectedChunkIndex = matched >= 0 ? matched : 0;
    } else {
      selectedChunkIndex = 0;
    }

    const selected = getSelectedChunk();
    if (selected) {
      if (previewPage < selected.start || previewPage > selected.end) {
        previewPage = selected.start;
      }
    } else {
      previewPage = 1;
    }

    setPreviewError('');
    renderChunkList();
    void renderPreviewPage();
  } catch (error) {
    previewChunks = [];
    selectedChunkIndex = -1;
    previewPage = 1;
    renderChunkList();
    clearPreviewCanvas();
    updatePreviewIndicator();
    updatePreviewNavButtons();
    setPreviewError(error.message || 'プレビュー計算に失敗しました');
  }
}

function schedulePreviewRefresh() {
  if (previewDebounceTimer) {
    clearTimeout(previewDebounceTimer);
  }
  previewDebounceTimer = setTimeout(() => {
    previewDebounceTimer = null;
    refreshChunkPreview();
  }, 160);
}

function isPdfJsReady() {
  return Boolean(window.pdfjsLib && typeof window.pdfjsLib.getDocument === 'function');
}

function configurePdfJs() {
  if (!isPdfJsReady()) {
    return false;
  }
  window.pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  return true;
}

function loadImagePreviewElement(url) {
  return new Promise((resolve, reject) => {
    const onLoad = () => {
      previewImage.removeEventListener('load', onLoad);
      previewImage.removeEventListener('error', onError);
      resolve();
    };
    const onError = () => {
      previewImage.removeEventListener('load', onLoad);
      previewImage.removeEventListener('error', onError);
      reject(new Error('画像の読み込みに失敗しました'));
    };
    previewImage.addEventListener('load', onLoad);
    previewImage.addEventListener('error', onError);
    previewImage.src = url;
  });
}

function resetPreviewFile() {
  previewLoadingToken += 1;
  previewRenderToken += 1;

  if (previewRenderTask && typeof previewRenderTask.cancel === 'function') {
    try {
      previewRenderTask.cancel();
    } catch (_) {
      // ignore
    }
  }
  previewRenderTask = null;

  if (previewPdfDoc && typeof previewPdfDoc.destroy === 'function') {
    previewPdfDoc.destroy().catch(() => {
      // ignore
    });
  }

  previewMode = 'none';
  previewPdfDoc = null;
  previewPageCount = 0;
  previewChunks = [];
  selectedChunkIndex = -1;
  previewPage = 1;

  clearPreviewImage();
  showPdfPreviewSurface();
  clearPreviewCanvas();
  previewFrameWrap.classList.remove('has-file');
  previewEmpty.hidden = false;
  previewEmpty.textContent = PREVIEW_EMPTY_TEXT;

  renderChunkList();
  updatePreviewIndicator();
  updatePreviewNavButtons();
  setPreviewError('');
}

async function handlePreviewFileChange() {
  const files = Array.from(sourceInput.files || []);
  if (files.length === 0) {
    resetPreviewFile();
    return;
  }

  const sourceKinds = files.map((file) => getSourceKind(file));
  if (sourceKinds.some((kind) => kind === 'unknown')) {
    resetPreviewFile();
    setPreviewError('対応形式は PDF または画像(PNG/JPEG/WEBP/GIF/BMP/TIFF)です。');
    return;
  }
  const uniqueKinds = [...new Set(sourceKinds)];
  if (uniqueKinds.length !== 1) {
    resetPreviewFile();
    setPreviewError('PDFと画像を同時に選択できません。どちらか一方のみ選択してください。');
    return;
  }
  const sourceKind = uniqueKinds[0];
  if (sourceKind === 'pdf' && files.length > 1) {
    resetPreviewFile();
    setPreviewError('PDFは1ファイルのみ選択してください。');
    return;
  }

  const loadingToken = ++previewLoadingToken;
  previewRenderToken += 1;
  previewMode = sourceKind;

  if (previewRenderTask && typeof previewRenderTask.cancel === 'function') {
    try {
      previewRenderTask.cancel();
    } catch (_) {
      // ignore
    }
  }
  previewRenderTask = null;
  clearPreviewImage();
  clearPreviewCanvas();
  previewFrameWrap.classList.remove('has-file');
  previewEmpty.hidden = false;
  setPreviewError('');

  if (sourceKind === 'image') {
    if (previewPdfDoc && typeof previewPdfDoc.destroy === 'function') {
      previewPdfDoc.destroy().catch(() => {
        // ignore
      });
    }
    previewPdfDoc = null;
    previewImageEntries = files.map((file, index) => ({
      file,
      page: index + 1,
      url: URL.createObjectURL(file),
    }));
    previewPageCount = previewImageEntries.length;
    previewChunks = [];
    selectedChunkIndex = -1;
    previewPage = 1;

    previewEmpty.textContent = `${previewPageCount}枚の画像を読み込み中...`;
    renderChunkList();
    updatePreviewIndicator();
    updatePreviewNavButtons();

    try {
      const first = previewImageEntries[0];
      if (!first) {
        throw new Error('画像が見つかりません');
      }
      await loadImagePreviewElement(first.url);
      if (loadingToken !== previewLoadingToken) {
        return;
      }
      showImagePreviewSurface();
      previewFrameWrap.classList.add('has-file');
      previewEmpty.hidden = true;
      previewEmpty.textContent = PREVIEW_EMPTY_TEXT;
      setPreviewError('');
      refreshChunkPreview();
    } catch (error) {
      if (loadingToken !== previewLoadingToken) {
        return;
      }
      resetPreviewFile();
      setPreviewError(`画像読み込みに失敗しました: ${error.message || error}`);
    }
    return;
  }

  if (!pdfPreviewReady) {
    resetPreviewFile();
    setPreviewError('PDFプレビューライブラリの読み込みに失敗しました。ネットワーク接続を確認してください。');
    return;
  }

  previewPdfDoc = null;
  previewPageCount = 0;
  previewChunks = [];
  selectedChunkIndex = -1;
  previewPage = 1;
  renderChunkList();
  showPdfPreviewSurface();
  previewEmpty.textContent = 'PDFを読み込み中...';
  updatePreviewIndicator();
  updatePreviewNavButtons();

  try {
    const bytes = new Uint8Array(await files[0].arrayBuffer());
    if (loadingToken !== previewLoadingToken) {
      return;
    }

    const loadingTask = window.pdfjsLib.getDocument({ data: bytes });
    const doc = await loadingTask.promise;
    if (loadingToken !== previewLoadingToken) {
      if (typeof doc.destroy === 'function') {
        doc.destroy().catch(() => {
          // ignore
        });
      }
      return;
    }

    if (previewPdfDoc && typeof previewPdfDoc.destroy === 'function') {
      await previewPdfDoc.destroy();
    }

    previewPdfDoc = doc;
    previewPageCount = Number(doc.numPages || 0);
    previewMode = 'pdf';

    previewFrameWrap.classList.add('has-file');
    previewEmpty.hidden = true;
    previewEmpty.textContent = PREVIEW_EMPTY_TEXT;
    setPreviewError('');

    refreshChunkPreview();
  } catch (error) {
    if (loadingToken !== previewLoadingToken) {
      return;
    }
    resetPreviewFile();
    setPreviewError(`PDF読み込みに失敗しました: ${error.message || error}`);
  }
}

function movePreviewPage(delta) {
  if (previewMode !== 'pdf' && previewMode !== 'image') {
    return;
  }
  if (previewMode === 'pdf' && !previewPdfDoc) {
    return;
  }
  if (previewMode === 'image' && previewPageCount <= 0) {
    return;
  }
  if (previewChunks.length === 0) {
    return;
  }

  const current = getSelectedChunk();
  if (!current) {
    return;
  }

  const minPage = Math.max(1, current.start);
  const maxPage = Math.min(previewPageCount, current.end);
  if (minPage > maxPage) {
    return;
  }

  let candidate = previewPage + delta;

  if (candidate < minPage) {
    if (selectedChunkIndex === 0) {
      candidate = minPage;
    } else {
      selectedChunkIndex -= 1;
      const prevChunk = getSelectedChunk();
      previewPage = Math.min(previewPageCount, Math.max(1, prevChunk.end));
      renderChunkList();
      void renderPreviewPage();
      return;
    }
  }

  if (candidate > maxPage) {
    if (selectedChunkIndex === previewChunks.length - 1) {
      candidate = maxPage;
    } else {
      selectedChunkIndex += 1;
      const nextChunk = getSelectedChunk();
      previewPage = Math.max(1, nextChunk.start);
      renderChunkList();
      void renderPreviewPage();
      return;
    }
  }

  previewPage = candidate;
  void renderPreviewPage();
}

function renderTable() {
  const keyword = searchInput.value.trim().toLowerCase();
  const issueOnly = onlyIssues.checked;

  tableBody.innerHTML = '';

  const visible = cards.filter((card) => {
    const hasIssue = card.issue.length > 0;
    if (issueOnly && !hasIssue) {
      return false;
    }
    if (!keyword) {
      return true;
    }
    const merged = `${card.page} ${card.question} ${card.answer} ${card.issue.join(' ')}`.toLowerCase();
    return merged.includes(keyword);
  });

  visible.forEach((card, index) => {
    const tr = document.createElement('tr');
    if (card.issue.length > 0) {
      tr.classList.add('issue-row');
    }

    tr.innerHTML = `
      <td>${index + 1}</td>
      <td><input type="text" data-field="page" value="${escapeHTML(card.page)}" /></td>
      <td><textarea data-field="question">${escapeHTML(card.question)}</textarea></td>
      <td><textarea data-field="answer">${escapeHTML(card.answer)}</textarea></td>
      <td><input type="number" data-field="confidence" min="0" max="1" step="0.01" value="${Number(card.confidence).toFixed(2)}" /></td>
      <td>
        <input type="text" data-field="issue" value="${escapeHTML(card.issue.join(', '))}" />
        <div>${card.issue.map((label) => `<span class="issue-badge">${escapeHTML(label)}</span>`).join('')}</div>
      </td>
      <td><button type="button" class="delete-btn">削除</button></td>
    `;

    tr.querySelectorAll('[data-field]').forEach((input) => {
      input.addEventListener('input', () => {
        const field = input.dataset.field;
        if (field === 'confidence') {
          card.confidence = Math.max(0, Math.min(1, Number(input.value || 0)));
        } else if (field === 'issue') {
          card.issue = String(input.value)
            .split(',')
            .map((x) => x.trim())
            .filter(Boolean);
        } else {
          card[field] = input.value;
        }
        updateSummary();
      });
    });

    tr.querySelector('.delete-btn').addEventListener('click', () => {
      const idx = cards.indexOf(card);
      if (idx >= 0) {
        cards.splice(idx, 1);
      }
      renderTable();
      updateSummary();
    });

    tableBody.appendChild(tr);
  });
}

function updateSummary(extra = '') {
  const issueCount = cards.filter((c) => c.issue.length > 0).length;
  summary.textContent = `${cards.length}件 / issueあり ${issueCount}件${extra ? ` / ${extra}` : ''}`;
}

function csvEscape(value) {
  const text = String(value ?? '');
  if (/[";\n\r]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function exportCSV() {
  const lines = cards.map((card) => `${csvEscape(card.question)};${csvEscape(card.answer)}`);
  const blob = new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
  downloadBlob(blob, 'cards.csv');
}

function exportJSON() {
  const blob = new Blob([JSON.stringify(cards, null, 2)], { type: 'application/json;charset=utf-8' });
  downloadBlob(blob, 'cards.json');
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function stopPolling() {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

async function pollJob(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    const data = await res.json();

    if (!res.ok) {
      const code = data?.error?.code ? `[${data.error.code}] ` : '';
      throw new Error(`${code}${data?.error?.message || 'ジョブ取得に失敗しました'}`);
    }

    const total = data.totalChunks || 0;
    const completed = data.completedChunks || 0;
    setProgress(completed, total);
    renderActive(Array.isArray(data.activeChunks) ? data.activeChunks : []);

    if (data.status === 'queued') {
      setStatus('ジョブをキューに登録しました...');
      const detail = buildJobDetail(data);
      setStatusDetail(detail);
      pollTimer = setTimeout(() => pollJob(jobId), 1000);
      return;
    }

    if (data.status === 'running') {
      setStatus('Geminiでカードを生成中です...');
      setStatusDetail(buildJobDetail(data));
      pollTimer = setTimeout(() => pollJob(jobId), 1200);
      return;
    }

    if (data.status === 'failed') {
      stopPolling();
      startBtn.disabled = false;
      setStatus('失敗しました。詳細を確認してください。');
      setStatusDetail(buildJobDetail(data) || String(data.error || 'unknown error'));
      return;
    }

    if (data.status === 'completed') {
      stopPolling();
      startBtn.disabled = false;
      setStatus('完了しました。テーブルで内容を修正できます。');
      setStatusDetail(buildJobDetail(data));
      cards = Array.isArray(data.cards) ? data.cards.map(normalizeCard) : [];
      resultCard.hidden = false;
      renderTable();
      const warning = Array.isArray(data.warnings) && data.warnings.length > 0
        ? `警告 ${data.warnings.length}件（失敗チャンク: ${(data.failedChunks || []).join(', ') || 'なし'}）`
        : '';
      updateSummary(warning);
      return;
    }

    pollTimer = setTimeout(() => pollJob(jobId), 1200);
  } catch (err) {
    stopPolling();
    startBtn.disabled = false;
    setStatus(`エラー: ${err.message}`);
    setStatusDetail(String(err.message || err));
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  stopPolling();
  currentJobId = '';
  cards = [];
  resultCard.hidden = true;

  const formData = new FormData(form);
  startBtn.disabled = true;
  setStatus('ジョブ作成中...');
  setStatusDetail('');
  setProgress(0, 1);
  renderActive([]);

  try {
    const res = await fetch('/api/jobs', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      const code = data?.error?.code ? `[${data.error.code}] ` : '';
      throw new Error(`${code}${data?.error?.message || 'ジョブ作成に失敗しました'}`);
    }

    currentJobId = data.jobId;
    setStatus(`ジョブ開始: ${currentJobId}`);
    pollJob(currentJobId);
  } catch (err) {
    startBtn.disabled = false;
    setStatus(`エラー: ${err.message}`);
    setStatusDetail(String(err.message || err));
  }
});

searchInput.addEventListener('input', renderTable);
onlyIssues.addEventListener('change', renderTable);

addRowBtn.addEventListener('click', () => {
  cards.push({ page: '', question: '', answer: '', confidence: 1, issue: [] });
  renderTable();
  updateSummary();
});

exportCsvBtn.addEventListener('click', exportCSV);
exportJsonBtn.addEventListener('click', exportJSON);

sourceInput.addEventListener('change', () => {
  void handlePreviewFileChange();
});

[rangesInput, stepInput, overlapInput].forEach((input) => {
  input.addEventListener('input', schedulePreviewRefresh);
  input.addEventListener('change', schedulePreviewRefresh);
});

previewToggleBtn.addEventListener('click', () => {
  const collapsed = previewCard.classList.contains('collapsed');
  setPreviewCollapsed(!collapsed);
});

pagePrevBtn.addEventListener('click', () => movePreviewPage(-1));
pageNextBtn.addEventListener('click', () => movePreviewPage(1));

window.addEventListener('resize', () => {
  if (previewResizeTimer) {
    clearTimeout(previewResizeTimer);
  }
  previewResizeTimer = setTimeout(() => {
    previewResizeTimer = null;
    if (previewMode === 'pdf' && previewPdfDoc) {
      void renderPreviewPage();
    }
  }, 140);
});

window.addEventListener('beforeunload', () => {
  resetPreviewFile();
});

setProgress(0, 1);
updateSummary();
setupPreviewCollapsedState();
resetPreviewFile();

pdfPreviewReady = configurePdfJs();
