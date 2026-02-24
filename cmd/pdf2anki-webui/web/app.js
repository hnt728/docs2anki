const form = document.getElementById('job-form');
const startBtn = document.getElementById('start-btn');
const barFill = document.getElementById('bar-fill');
const progressText = document.getElementById('progress-text');
const statusMessage = document.getElementById('status-message');
const activeChunks = document.getElementById('active-chunks');
const resultCard = document.getElementById('result-card');
const summary = document.getElementById('summary');
const tableBody = document.querySelector('#cards-table tbody');
const searchInput = document.getElementById('search');
const onlyIssues = document.getElementById('only-issues');
const addRowBtn = document.getElementById('add-row');
const exportCsvBtn = document.getElementById('export-csv');
const exportJsonBtn = document.getElementById('export-json');

let pollTimer = null;
let currentJobId = '';
let cards = [];

function setStatus(text) {
  statusMessage.textContent = text || '';
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
  if (/[\";\n\r]/.test(text)) {
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
      throw new Error(data?.error?.message || 'ジョブ取得に失敗しました');
    }

    const total = data.totalChunks || 0;
    const completed = data.completedChunks || 0;
    setProgress(completed, total);
    renderActive(Array.isArray(data.activeChunks) ? data.activeChunks : []);

    if (data.status === 'queued') {
      setStatus('ジョブをキューに登録しました...');
      pollTimer = setTimeout(() => pollJob(jobId), 1000);
      return;
    }

    if (data.status === 'running') {
      setStatus('Geminiでカードを生成中です...');
      pollTimer = setTimeout(() => pollJob(jobId), 1200);
      return;
    }

    if (data.status === 'failed') {
      stopPolling();
      startBtn.disabled = false;
      setStatus(`失敗: ${data.error || 'unknown error'}`);
      return;
    }

    if (data.status === 'completed') {
      stopPolling();
      startBtn.disabled = false;
      setStatus('完了しました。テーブルで内容を修正できます。');
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
  setProgress(0, 1);
  renderActive([]);

  try {
    const res = await fetch('/api/jobs', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data?.error?.message || 'ジョブ作成に失敗しました');
    }

    currentJobId = data.jobId;
    setStatus(`ジョブ開始: ${currentJobId}`);
    pollJob(currentJobId);
  } catch (err) {
    startBtn.disabled = false;
    setStatus(`エラー: ${err.message}`);
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

setProgress(0, 1);
updateSummary();
