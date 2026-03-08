(function initDocs2AnkiJobs(globalScope) {
  const appUtils = globalScope.docs2ankiAppUtils;
  if (!appUtils) {
    throw new Error('docs2ankiAppUtils failed to load');
  }

  const {
    formatList,
    normalizeCard,
  } = appUtils;

  function createJobsController({
    form,
    startBtn,
    stopBtn,
    chunkRetryBtn,
    chunkSkipBtn,
    pollActiveMs,
    pollIdleMs,
    setStatus,
    setStatusDetail,
    setStopButton,
    setChunkActionControls,
    setProgress,
    renderActive,
    resetGeminiConsole,
    syncGeminiConsole,
    resetResults,
    setResults,
  }) {
    let pollTimer = null;
    let currentJobId = '';

    function providerName(data = {}) {
      return String(data?.config?.provider || '').trim().toLowerCase() === 'openai' ? 'OpenAI' : 'Gemini';
    }

    function stopPolling() {
      if (pollTimer) {
        clearTimeout(pollTimer);
        pollTimer = null;
      }
    }

    function schedulePoll(jobId, delayMs) {
      pollTimer = setTimeout(() => {
        void pollJob(jobId);
      }, delayMs);
    }

    function buildJobDetail(data = {}) {
      const hasFailed = Array.isArray(data.failedChunks) && data.failedChunks.length > 0;
      const hasWarnings = Array.isArray(data.warnings) && data.warnings.length > 0;
      const hasError = Boolean(String(data.error || '').trim());
      const hasPending = Boolean(String(data.pendingChunk || '').trim() || String(data.pendingError || '').trim());
      if (!hasFailed && !hasWarnings && !hasError && !hasPending) {
        return '';
      }

      const lines = [];
      if (data.jobId || currentJobId) {
        lines.push(`job: ${data.jobId || currentJobId}`);
      }
      if (data.status) {
        lines.push(`status: ${data.status}`);
      }
      if (data?.config?.provider) {
        lines.push(`provider: ${providerName(data)}`);
      }
      if (data?.config?.model) {
        lines.push(`model: ${String(data.config.model)}`);
      }
      if (Array.isArray(data.failedChunks) && data.failedChunks.length > 0) {
        lines.push(`failed chunks: ${data.failedChunks.join(', ')}`);
      }
      if (data.pendingChunk || data.pendingError) {
        lines.push('');
        lines.push(`pending chunk: ${String(data.pendingChunk || '-')}`);
        lines.push('pending error:');
        lines.push(String(data.pendingError || '-').trim());
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

    async function submitChunkAction(action) {
      if (!currentJobId) {
        return;
      }
      const normalized = String(action || '').trim().toLowerCase();
      if (normalized !== 'retry' && normalized !== 'skip') {
        return;
      }

      setChunkActionControls(true, true);
      setStatus(`操作を送信中: ${normalized}`);

      try {
        const res = await fetch(`/api/jobs/${currentJobId}/action`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: normalized }),
        });
        const data = await res.json();
        const snapshot = data?.job || null;
        if (!res.ok && !snapshot) {
          const code = data?.error?.code ? `[${data.error.code}] ` : '';
          throw new Error(`${code}${data?.error?.message || '操作に失敗しました'}`);
        }
        const accepted = Boolean(data?.accepted);
        if (snapshot) {
          syncGeminiConsole(snapshot.logs);
          setProgress(snapshot.completedChunks || 0, snapshot.totalChunks || 0);
          renderActive(Array.isArray(snapshot.activeChunks) ? snapshot.activeChunks : []);
        }
        if (!accepted) {
          setStatus(`操作は受け付けられませんでした: ${normalized}`);
          setChunkActionControls(snapshot?.status === 'paused', false);
          schedulePoll(currentJobId, 150);
          return;
        }
        setStatus(`操作を受け付けました: ${normalized}`);
        schedulePoll(currentJobId, 100);
      } catch (err) {
        setChunkActionControls(true, false);
        setStatus(`操作エラー: ${err.message}`);
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
        syncGeminiConsole(data.logs);

        if (data.status === 'queued') {
          setStatus('ジョブをキューに登録しました...');
          setStatusDetail(buildJobDetail(data));
          setStopButton(true, false);
          setChunkActionControls(false, true);
          schedulePoll(jobId, pollIdleMs);
          return;
        }

        if (data.status === 'running') {
          setStatus(`${providerName(data)}でカードを生成中です...`);
          setStatusDetail(buildJobDetail(data));
          setStopButton(true, false);
          setChunkActionControls(false, true);
          schedulePoll(jobId, pollActiveMs);
          return;
        }

        if (data.status === 'paused') {
          setStatus('チャンク処理エラーで一時停止中です。リトライまたはスキップを選択してください。');
          setStatusDetail(buildJobDetail(data));
          setStopButton(true, false);
          setChunkActionControls(true, false);
          schedulePoll(jobId, pollIdleMs);
          return;
        }

        if (data.status === 'stopping') {
          setStatus('停止処理中です...');
          setStatusDetail(buildJobDetail(data));
          setStopButton(true, true);
          setChunkActionControls(false, true);
          schedulePoll(jobId, pollActiveMs);
          return;
        }

        if (data.status === 'stopped') {
          stopPolling();
          startBtn.disabled = false;
          setStopButton(false, true);
          setChunkActionControls(false, true);
          setStatus('停止しました。');
          setStatusDetail(buildJobDetail(data));
          return;
        }

        if (data.status === 'failed') {
          stopPolling();
          startBtn.disabled = false;
          setStopButton(false, true);
          setChunkActionControls(false, true);
          setStatus('失敗しました。詳細を確認してください。');
          setStatusDetail(buildJobDetail(data) || String(data.error || 'unknown error'));
          return;
        }

        if (data.status === 'completed') {
          stopPolling();
          startBtn.disabled = false;
          setStopButton(false, true);
          setChunkActionControls(false, true);
          setStatus('完了しました。テーブルで内容を修正できます。');
          setStatusDetail(buildJobDetail(data));
          const nextCards = Array.isArray(data.cards) ? data.cards.map(normalizeCard) : [];
          const warning = Array.isArray(data.warnings) && data.warnings.length > 0
            ? `警告 ${data.warnings.length}件（失敗チャンク: ${(data.failedChunks || []).join(', ') || 'なし'}）`
            : '';
          setResults(nextCards, warning);
          return;
        }

        schedulePoll(jobId, pollIdleMs);
      } catch (err) {
        stopPolling();
        startBtn.disabled = false;
        setStopButton(false, true);
        setChunkActionControls(false, true);
        setStatus(`エラー: ${err.message}`);
        setStatusDetail(String(err.message || err));
      }
    }

    async function handleFormSubmit(event) {
      event.preventDefault();

      stopPolling();
      currentJobId = '';
      resetResults();
      resetGeminiConsole();

      const formData = new FormData(form);
      startBtn.disabled = true;
      setStopButton(false, true);
      setChunkActionControls(false, true);
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
        setStopButton(true, false);
        setChunkActionControls(false, true);
        void pollJob(currentJobId);
      } catch (err) {
        startBtn.disabled = false;
        setStopButton(false, true);
        setChunkActionControls(false, true);
        setStatus(`エラー: ${err.message}`);
        setStatusDetail(String(err.message || err));
      }
    }

    async function handleStopButtonClick() {
      if (!currentJobId || stopBtn.disabled) {
        return;
      }

      stopBtn.disabled = true;
      setStatus('停止要求を送信中...');

      try {
        const res = await fetch(`/api/jobs/${currentJobId}/stop`, { method: 'POST' });
        const data = await res.json();
        const snapshot = data?.job || null;
        if (!res.ok && !snapshot) {
          const code = data?.error?.code ? `[${data.error.code}] ` : '';
          throw new Error(`${code}${data?.error?.message || '停止要求に失敗しました'}`);
        }
        if (snapshot) {
          syncGeminiConsole(snapshot.logs);
          setProgress(snapshot.completedChunks || 0, snapshot.totalChunks || 0);
          renderActive(Array.isArray(snapshot.activeChunks) ? snapshot.activeChunks : []);
        }
        setStatus('停止要求を送信しました。停止まで少し待ってください。');
        setStatusDetail(snapshot ? buildJobDetail(snapshot) : '');
        setChunkActionControls(false, true);
        schedulePoll(currentJobId, 120);
      } catch (err) {
        stopBtn.disabled = false;
        setStatus(`停止要求エラー: ${err.message}`);
      }
    }

    function handleChunkRetryClick() {
      void submitChunkAction('retry');
    }

    function handleChunkSkipClick() {
      void submitChunkAction('skip');
    }

    function bindEvents() {
      form.addEventListener('submit', handleFormSubmit);
      stopBtn.addEventListener('click', handleStopButtonClick);
      chunkRetryBtn.addEventListener('click', handleChunkRetryClick);
      chunkSkipBtn.addEventListener('click', handleChunkSkipClick);
    }

    function initializeUi() {
      setStopButton(false, true);
      setChunkActionControls(false, true);
      resetGeminiConsole();
    }

    return Object.freeze({
      bindEvents,
      initializeUi,
    });
  }

  globalScope.docs2ankiJobs = Object.freeze({
    createJobsController,
  });
}(window));
