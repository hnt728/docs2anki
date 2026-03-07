(function initDocs2AnkiAppUtils(globalScope) {
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

  function rangeExpressionForCount(count) {
    const total = Number.parseInt(String(count || 0), 10);
    if (!Number.isInteger(total) || total < 1) {
      return '';
    }
    if (total === 1) {
      return '1';
    }
    return `1-${total}`;
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

  function csvEscape(value) {
    const text = String(value ?? '');
    if (/[";\n\r]/.test(text)) {
      return `"${text.replaceAll('"', '""')}"`;
    }
    return text;
  }

  globalScope.docs2ankiAppUtils = Object.freeze({
    computeChunks,
    csvEscape,
    escapeHTML,
    formatList,
    inferMimeFromName,
    normalizeCard,
    normalizePreviewMIME,
    parseRangesExpression,
    rangeExpressionForCount,
  });
}(window));
