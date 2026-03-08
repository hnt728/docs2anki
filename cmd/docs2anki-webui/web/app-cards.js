(function initDocs2AnkiCards(globalScope) {
  const appUtils = globalScope.docs2ankiAppUtils;
  if (!appUtils) {
    throw new Error('docs2ankiAppUtils failed to load');
  }

  const {
    csvEscape,
    escapeHTML,
  } = appUtils;

  function createCardsController({
    resultCard,
    summary,
    tableBody,
    searchInput,
    onlyIssues,
    addRowBtn,
    exportEscapeInput,
    exportCsvBtn,
    exportJsonBtn,
  }) {
    let cards = [];

    function updateSummary(extra = '') {
      const issueCount = cards.filter((card) => card.issue.length > 0).length;
      summary.textContent = `${cards.length}件 / issueあり ${issueCount}件${extra ? ` / ${extra}` : ''}`;
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

    function downloadBlob(blob, filename) {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    }

    function shouldEscapeExports() {
      return !exportEscapeInput || exportEscapeInput.checked;
    }

    function serializeExportJSONValue(value, indentLevel = 0, escapeStrings = true) {
      if (Array.isArray(value)) {
        if (value.length === 0) {
          return '[]';
        }
        const indent = '  '.repeat(indentLevel);
        const childIndent = '  '.repeat(indentLevel + 1);
        return `[\n${value.map((item) => `${childIndent}${serializeExportJSONValue(item, indentLevel + 1, escapeStrings)}`).join(',\n')}\n${indent}]`;
      }

      if (value && typeof value === 'object') {
        const entries = Object.entries(value);
        if (entries.length === 0) {
          return '{}';
        }
        const indent = '  '.repeat(indentLevel);
        const childIndent = '  '.repeat(indentLevel + 1);
        return `{\n${entries.map(([key, item]) => `${childIndent}${JSON.stringify(key)}: ${serializeExportJSONValue(item, indentLevel + 1, escapeStrings)}`).join(',\n')}\n${indent}}`;
      }

      if (typeof value === 'string') {
        return escapeStrings ? JSON.stringify(value) : `"${String(value)}"`;
      }

      if (typeof value === 'number') {
        return Number.isFinite(value) ? String(value) : 'null';
      }

      if (typeof value === 'boolean') {
        return value ? 'true' : 'false';
      }

      if (value == null) {
        return 'null';
      }

      return escapeStrings ? JSON.stringify(value) : `"${String(value)}"`;
    }

    function exportCSV() {
      const escapeStrings = shouldEscapeExports();
      const lines = cards.map((card) => {
        const question = escapeStrings ? csvEscape(card.question) : String(card.question ?? '');
        const answer = escapeStrings ? csvEscape(card.answer) : String(card.answer ?? '');
        return `${question};${answer}`;
      });
      const blob = new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
      downloadBlob(blob, 'cards.csv');
    }

    function exportJSON() {
      const body = serializeExportJSONValue(cards, 0, shouldEscapeExports());
      const blob = new Blob([body], { type: 'application/json;charset=utf-8' });
      downloadBlob(blob, 'cards.json');
    }

    function handleAddRowClick() {
      cards.push({ page: '', question: '', answer: '', confidence: 1, issue: [] });
      renderTable();
      updateSummary();
    }

    function bindEvents() {
      searchInput.addEventListener('input', renderTable);
      onlyIssues.addEventListener('change', renderTable);
      addRowBtn.addEventListener('click', handleAddRowClick);
      exportCsvBtn.addEventListener('click', exportCSV);
      exportJsonBtn.addEventListener('click', exportJSON);
    }

    function initialize() {
      updateSummary();
    }

    function reset() {
      cards = [];
      resultCard.hidden = true;
      tableBody.innerHTML = '';
      updateSummary();
    }

    function setCards(nextCards, extra = '') {
      cards = Array.isArray(nextCards) ? nextCards : [];
      resultCard.hidden = false;
      renderTable();
      updateSummary(extra);
    }

    return Object.freeze({
      bindEvents,
      initialize,
      reset,
      setCards,
    });
  }

  globalScope.docs2ankiCards = Object.freeze({
    createCardsController,
  });
}(window));
