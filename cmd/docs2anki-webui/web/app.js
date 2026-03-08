const appUtils = window.docs2ankiAppUtils;
if (!appUtils) {
  throw new Error('docs2ankiAppUtils failed to load');
}

const cardsModule = window.docs2ankiCards;
if (!cardsModule) {
  throw new Error('docs2ankiCards failed to load');
}

const jobsModule = window.docs2ankiJobs;
if (!jobsModule) {
  throw new Error('docs2ankiJobs failed to load');
}

const {
  computeChunks,
  inferMimeFromName,
  normalizePreviewMIME,
  parseRangesExpression,
  rangeExpressionForCount,
} = appUtils;

const { createCardsController } = cardsModule;
const { createJobsController } = jobsModule;

const form = document.getElementById('job-form');
const startBtn = document.getElementById('start-btn');
const barFill = document.getElementById('bar-fill');
const progressText = document.getElementById('progress-text');
const statusMessage = document.getElementById('status-message');
const statusDetailBox = document.getElementById('status-detail-box');
const statusDetail = document.getElementById('status-detail');
const activeChunks = document.getElementById('active-chunks');
const stopBtn = document.getElementById('stop-btn');
const chunkActionRow = document.getElementById('chunk-action-row');
const chunkRetryBtn = document.getElementById('chunk-retry-btn');
const chunkSkipBtn = document.getElementById('chunk-skip-btn');
const geminiConsoleBox = document.getElementById('gemini-console-box');
const geminiConsole = document.getElementById('gemini-console');
const geminiConsoleCount = document.getElementById('gemini-console-count');
const resultCard = document.getElementById('result-card');
const summary = document.getElementById('summary');
const tableBody = document.querySelector('#cards-table tbody');
const searchInput = document.getElementById('search');
const onlyIssues = document.getElementById('only-issues');
const addRowBtn = document.getElementById('add-row');
const exportEscapeInput = document.getElementById('export-escape');
const exportCsvBtn = document.getElementById('export-csv');
const exportJsonBtn = document.getElementById('export-json');

const sourceInput = form.querySelector('input[name="source"]');
const rangesInput = form.querySelector('input[name="ranges"]');
const stepInput = form.querySelector('input[name="step"]');
const overlapInput = form.querySelector('input[name="overlap"]');
const helpButtons = Array.from(form.querySelectorAll('.help-btn'));
const providerInputs = Array.from(form.querySelectorAll('input[name="provider"]'));
const geminiPanel = document.getElementById('provider-panel-gemini');
const openaiPanel = document.getElementById('provider-panel-openai');
const openaiModelInput = form.querySelector('input[name="openaiModel"]');
const openaiDetailOriginalRow = document.getElementById('openai-detail-original-row');
const openaiDetailOriginalInput = form.querySelector('input[name="openaiImageDetailOriginal"]');

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
const POLL_ACTIVE_MS = 550;
const POLL_IDLE_MS = 900;
const settingHelpTexts = Object.freeze({
  ranges: 'ページ範囲: 処理対象のページ(または画像番号)です。例: 1-12, 5, 8-10',
  step: 'Step: ページ範囲のページを分割した、1チャンクに含めるページ数です。例: step=3 なら 1-3, 4-6, 7-9 ...を1チャンク(1まとまり)として選択中のAPIで処理します。',
  overlap: 'Overlap: チャンク同士で重複させるページ数です。0以上かつ Step 未満で指定します。例: step=3, overlap=1 なら 1-3, 3-5, 5-7 ...',
  minConfidence: 'Min Confidence: この値未満のカードに low_confidence を付けます。値は 0.0 から 1.0 です。',
  delayMs: 'Delay(ms): 各APIリクエスト前に待つ時間(ミリ秒)です。レート制限が厳しいときに増やします。0以上で指定します。',
  thinkingBudget: 'Thinking Budget: Geminiの思考予算です。-1 は動的、0以上は予算指定です。モデルやAPIバージョンによって実際の挙動は異なる場合があります。',
  reasoningEffort: 'Reasoning Effort: OpenAIの推論量です。low / medium / high / extra high から選びます。高いほど複数箇所の照合や構造理解に有利ですが、遅く高コストになりやすいです。',
  openAIDetailOriginal: 'detail="original": OpenAI gpt-5.4系の画像入力で解像度を上げる設定です。小さい文字・手書き・低品質スキャンに有利ですが、トークン消費は増えます。PDF入力には適用されません。',
});

let latestLogSeq = 0;

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

function setStopButton(visible, disabled = false) {
  stopBtn.hidden = !visible;
  stopBtn.disabled = !visible || disabled;
}

function setChunkActionControls(visible, disabled = false) {
  chunkActionRow.hidden = !visible;
  chunkRetryBtn.disabled = !visible || disabled;
  chunkSkipBtn.disabled = !visible || disabled;
}

function resetGeminiConsole() {
  latestLogSeq = 0;
  geminiConsole.textContent = '';
  geminiConsoleCount.textContent = '0行';
}

function formatGeminiLogLine(entry = {}) {
  const time = String(entry.time || '--:--:--');
  const chunk = String(entry.chunk || '').trim();
  const message = String(entry.message || '').trim();
  if (!message) {
    return '';
  }
  if (chunk) {
    return `[${time}] [${chunk}] ${message}`;
  }
  return `[${time}] ${message}`;
}

function syncGeminiConsole(logs = []) {
  const entries = Array.isArray(logs) ? logs : [];
  if (entries.length === 0) {
    if (latestLogSeq !== 0 || geminiConsole.textContent !== '') {
      geminiConsole.textContent = '';
      latestLogSeq = 0;
    }
    geminiConsoleCount.textContent = '0行';
    return;
  }

  const firstSeq = Number(entries[0]?.seq || 0);
  const lastSeq = Number(entries[entries.length - 1]?.seq || 0);
  const autoScroll = (geminiConsole.scrollTop + geminiConsole.clientHeight) >= (geminiConsole.scrollHeight - 20);

  if (latestLogSeq === 0 || firstSeq > latestLogSeq + 1 || lastSeq < latestLogSeq) {
    geminiConsole.textContent = '';
    latestLogSeq = 0;
  }

  const nextLines = [];
  entries.forEach((entry) => {
    const seq = Number(entry?.seq || 0);
    if (!Number.isFinite(seq) || seq <= latestLogSeq) {
      return;
    }
    const line = formatGeminiLogLine(entry);
    if (!line) {
      return;
    }
    nextLines.push(line);
  });

  if (nextLines.length > 0) {
    const prefix = geminiConsole.textContent ? '\n' : '';
    geminiConsole.textContent += `${prefix}${nextLines.join('\n')}`;
    latestLogSeq = lastSeq;
  }

  geminiConsoleCount.textContent = `${entries.length}行`;
  if (autoScroll) {
    geminiConsole.scrollTop = geminiConsole.scrollHeight;
  }
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

function getSelectedProvider() {
  const selected = providerInputs.find((input) => input.checked);
  return selected?.value === 'openai' ? 'openai' : 'gemini';
}

function isGPT54Model(value) {
  return String(value || '').trim().toLowerCase().startsWith('gpt-5.4');
}

function setSectionVisibility(section, visible) {
  if (!section) {
    return;
  }

  section.hidden = !visible;
  section.setAttribute('aria-hidden', visible ? 'false' : 'true');

  section.querySelectorAll('input, select, textarea, button').forEach((control) => {
    if (!visible) {
      if (!control.disabled) {
        control.disabled = true;
        control.dataset.hiddenDisabled = 'true';
      }
      return;
    }

    if (control.dataset.hiddenDisabled === 'true') {
      control.disabled = false;
      delete control.dataset.hiddenDisabled;
    }
  });
}

function syncProviderFields() {
  const provider = getSelectedProvider();
  setSectionVisibility(geminiPanel, provider === 'gemini');
  setSectionVisibility(openaiPanel, provider === 'openai');

  const showOriginal = provider === 'openai' && isGPT54Model(openaiModelInput?.value);
  setSectionVisibility(openaiDetailOriginalRow, showOriginal);
  if (!showOriginal && openaiDetailOriginalInput) {
    openaiDetailOriginalInput.checked = false;
  }
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
    rangesInput.value = rangeExpressionForCount(previewPageCount);
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
    rangesInput.value = rangeExpressionForCount(previewPageCount);
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

function handleSourceInputChange() {
  void handlePreviewFileChange();
}

function handleHelpButtonClick(event) {
  event.preventDefault();
  event.stopPropagation();
  const helpBtn = event.currentTarget;
  const key = String(helpBtn.dataset.helpKey || '').trim();
  const text = settingHelpTexts[key];
  if (!text) {
    return;
  }
  window.alert(text);
}

function handlePreviewToggleClick() {
  const collapsed = previewCard.classList.contains('collapsed');
  setPreviewCollapsed(!collapsed);
}

function handleWindowResize() {
  if (previewResizeTimer) {
    clearTimeout(previewResizeTimer);
  }
  previewResizeTimer = setTimeout(() => {
    previewResizeTimer = null;
    if (previewMode === 'pdf' && previewPdfDoc) {
      void renderPreviewPage();
    }
  }, 140);
}

function handleWindowBeforeUnload() {
  resetPreviewFile();
}

const cardsController = createCardsController({
  resultCard,
  summary,
  tableBody,
  searchInput,
  onlyIssues,
  addRowBtn,
  exportEscapeInput,
  exportCsvBtn,
  exportJsonBtn,
});

const jobsController = createJobsController({
  form,
  startBtn,
  stopBtn,
  chunkRetryBtn,
  chunkSkipBtn,
  pollActiveMs: POLL_ACTIVE_MS,
  pollIdleMs: POLL_IDLE_MS,
  setStatus,
  setStatusDetail,
  setStopButton,
  setChunkActionControls,
  setProgress,
  renderActive,
  resetGeminiConsole,
  syncGeminiConsole,
  resetResults: () => cardsController.reset(),
  setResults: (cards, extra) => cardsController.setCards(cards, extra),
});

function bindPreviewEvents() {
  sourceInput.addEventListener('change', handleSourceInputChange);
  helpButtons.forEach((button) => {
    button.addEventListener('click', handleHelpButtonClick);
  });
  providerInputs.forEach((input) => {
    input.addEventListener('change', syncProviderFields);
  });
  if (openaiModelInput) {
    openaiModelInput.addEventListener('input', syncProviderFields);
    openaiModelInput.addEventListener('change', syncProviderFields);
  }

  [rangesInput, stepInput, overlapInput].forEach((input) => {
    input.addEventListener('input', schedulePreviewRefresh);
    input.addEventListener('change', schedulePreviewRefresh);
  });

  previewToggleBtn.addEventListener('click', handlePreviewToggleClick);
  pagePrevBtn.addEventListener('click', () => movePreviewPage(-1));
  pageNextBtn.addEventListener('click', () => movePreviewPage(1));
}

function bindWindowEvents() {
  window.addEventListener('resize', handleWindowResize);
  window.addEventListener('beforeunload', handleWindowBeforeUnload);
}

function bindEventListeners() {
  jobsController.bindEvents();
  cardsController.bindEvents();
  bindPreviewEvents();
  bindWindowEvents();
}

function initializeUi() {
  setProgress(0, 1);
  cardsController.reset();
  syncProviderFields();
  setupPreviewCollapsedState();
  resetPreviewFile();
  jobsController.initializeUi();
}

function initApp() {
  bindEventListeners();
  initializeUi();
  pdfPreviewReady = configurePdfJs();
}

initApp();
