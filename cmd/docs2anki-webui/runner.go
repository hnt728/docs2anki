package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"google.golang.org/genai"
)

func runJob(ctx context.Context, job *job, sources []uploadedSource, opts processOptions) error {
	chunks, prepWarnings, cleanup, err := buildChunkTasks(sources, opts)
	if err != nil {
		return err
	}
	defer cleanup()
	if len(chunks) == 0 {
		return fmt.Errorf("対象ページがありません")
	}

	job.markRunning(len(chunks))
	job.appendLogf("system", "処理開始: chunks=%d", len(chunks))
	for _, msg := range prepWarnings {
		job.appendWarning(msg)
		job.appendLogf("system", "warning: %s", msg)
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  opts.APIKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return fmt.Errorf("Geminiクライアント作成に失敗: %w", err)
	}

	cardsByChunk := make([][]Card, len(chunks))
	failedChunks := make([]string, 0)
	warnings := append([]string{}, prepWarnings...)
	failedCount := 0

	for _, task := range chunks {
		if ctx.Err() != nil && job.isStopRequested() {
			return context.Canceled
		}

		label := taskLabel(task)
		job.markChunkStart(label)
		job.appendLog(label, "開始")
		for {
			cards, runErr := runChunk(ctx, client, opts, task, func(message string) {
				job.appendLog(label, message)
			})
			if runErr == nil {
				job.appendLogf(label, "完了: cards=%d", len(cards))
				cardsByChunk[task.Index] = cards
				break
			}

			if errors.Is(runErr, context.Canceled) && job.isStopRequested() {
				return context.Canceled
			}

			job.appendLogf(label, "エラー: %v", runErr)
			job.pauseForChunkIssue(label, runErr)
			decision, err := job.waitChunkDecision(ctx)
			if err != nil {
				if errors.Is(err, context.Canceled) && job.isStopRequested() {
					return context.Canceled
				}
				return err
			}
			job.resumeFromChunkIssue()

			if decision == chunkDecisionRetry {
				job.appendLog(label, "リトライします。")
				continue
			}

			failedCount++
			failedChunks = append(failedChunks, label)
			warning := fmt.Sprintf("%s をスキップ: %v", label, runErr)
			warnings = append(warnings, warning)
			job.appendWarning(warning)
			job.appendLog(label, "スキップしました。")
			break
		}
		job.markChunkDone(label)
	}

	if ctx.Err() != nil && job.isStopRequested() {
		return context.Canceled
	}

	if failedCount == len(chunks) {
		detail := summarizeMessages(warnings, 12)
		if detail == "" {
			return fmt.Errorf("全チャンクの処理に失敗しました")
		}
		return fmt.Errorf("全チャンクの処理に失敗しました。\n%s", detail)
	}

	allCards := make([]Card, 0)
	for _, batch := range cardsByChunk {
		allCards = append(allCards, batch...)
	}

	job.markCompleted(allCards, failedChunks, warnings)
	job.appendLogf("system", "完了: cards=%d failedChunks=%d", len(allCards), len(failedChunks))
	return nil
}

func waitFileReady(ctx context.Context, client *genai.Client, fileName string) (*genai.File, error) {
	deadline := time.Now().Add(5 * time.Minute)
	for {
		file, err := client.Files.Get(ctx, fileName, nil)
		if err != nil {
			return nil, fmt.Errorf("アップロード済みファイルの状態確認に失敗: %w", err)
		}
		switch file.State {
		case genai.FileStateActive, genai.FileStateUnspecified:
			return file, nil
		case genai.FileStateFailed:
			return nil, fmt.Errorf("Gemini側でファイル処理に失敗しました: %s", formatFileStatus(file.Error))
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("アップロード済みファイルの準備待ちがタイムアウトしました")
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(1200 * time.Millisecond):
		}
	}
}

type uploadedChunkFile struct {
	URI      string
	MIMEType string
	Page     int
}

func runChunk(ctx context.Context, client *genai.Client, opts processOptions, task chunkTask, onLog func(string)) ([]Card, error) {
	if len(task.Assets) == 0 {
		return nil, fmt.Errorf("チャンクにファイルが含まれていません")
	}
	if onLog == nil {
		onLog = func(string) {}
	}

	uploadedFiles := make([]uploadedChunkFile, 0, len(task.Assets))
	uploadedNames := make([]string, 0, len(task.Assets))
	defer func() {
		for _, name := range uploadedNames {
			_, _ = client.Files.Delete(context.Background(), name, nil)
		}
	}()
	onLog(fmt.Sprintf("アップロード開始: files=%d", len(task.Assets)))

	for _, asset := range task.Assets {
		onLog(fmt.Sprintf("アップロード中: page=%d file=%s", asset.Page, asset.DisplayName))
		uploaded, err := client.Files.UploadFromPath(ctx, asset.Path, &genai.UploadFileConfig{
			MIMEType:    asset.MIMEType,
			DisplayName: uploadDisplayNameForAsset(task, asset),
		})
		if err != nil {
			return nil, fmt.Errorf("チャンクファイルアップロード失敗(%s): %w", taskLabel(task), err)
		}
		uploadedNames = append(uploadedNames, uploaded.Name)

		uploaded, err = waitFileReady(ctx, client, uploaded.Name)
		if err != nil {
			return nil, fmt.Errorf("チャンクファイル準備待ち失敗(%s): %w", taskLabel(task), err)
		}
		onLog(fmt.Sprintf("アップロード完了: page=%d", asset.Page))
		uploadedFiles = append(uploadedFiles, uploadedChunkFile{
			URI:      uploaded.URI,
			MIMEType: asset.MIMEType,
			Page:     asset.Page,
		})
	}

	if opts.DelayMS > 0 {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(time.Duration(opts.DelayMS) * time.Millisecond):
		}
	}

	onLog("Gemini呼び出し")
	cards, err := runChunkOnce(ctx, client, uploadedFiles, opts, task, onLog)
	if err != nil {
		return nil, fmt.Errorf("チャンク処理失敗(%s, size=%s): %w", taskLabel(task), formatBytes(task.FileSize), err)
	}
	onLog(fmt.Sprintf("JSON解析完了: cards=%d", len(cards)))
	return cards, nil
}

func runChunkOnce(ctx context.Context, client *genai.Client, files []uploadedChunkFile, opts processOptions, task chunkTask, onLog func(string)) ([]Card, error) {
	prompt := buildPrompt(opts.FrontPrompt, opts.BackPrompt, task)
	parts := make([]*genai.Part, 0, len(files)*2+1)
	for _, file := range files {
		if task.Kind == sourceKindImage {
			parts = append(parts, &genai.Part{Text: fmt.Sprintf("以下は page %d の画像です。", file.Page)})
		}
		parts = append(parts, &genai.Part{
			FileData: &genai.FileData{
				FileURI:  file.URI,
				MIMEType: file.MIMEType,
			},
		})
	}
	parts = append(parts, &genai.Part{Text: prompt})

	contents := []*genai.Content{{
		Role:  "user",
		Parts: parts,
	}}

	cfg := &genai.GenerateContentConfig{
		ResponseMIMEType:   "application/json",
		ResponseJsonSchema: buildSchema(),
		MediaResolution:    genai.MediaResolutionHigh,
	}
	if opts.ThinkingBudget >= -1 {
		budget := int32(opts.ThinkingBudget)
		cfg.ThinkingConfig = &genai.ThinkingConfig{ThinkingBudget: &budget}
	}

	collector := newStreamLogCollector(func(line string) {
		if onLog != nil {
			onLog("Gemini> " + line)
		}
	})

	var fullText strings.Builder
	var previousText string
	for resp, err := range client.Models.GenerateContentStream(ctx, opts.Model, contents, cfg) {
		if err != nil {
			return nil, err
		}
		if resp == nil {
			continue
		}
		piece := resp.Text()
		if piece == "" {
			continue
		}
		delta := piece
		if previousText != "" && strings.HasPrefix(piece, previousText) {
			delta = piece[len(previousText):]
		}
		previousText = piece
		if delta == "" {
			continue
		}
		fullText.WriteString(delta)
		collector.Push(delta)
	}
	collector.Flush()

	text := strings.TrimSpace(fullText.String())
	if text == "" {
		if onLog != nil {
			onLog("Gemini応答が空でした。")
		}
		return nil, nil
	}
	if onLog != nil {
		onLog(fmt.Sprintf("Gemini応答受信: %d bytes", len(text)))
	}

	var raw []map[string]any
	if err := json.Unmarshal([]byte(text), &raw); err != nil {
		return nil, fmt.Errorf("JSON解析に失敗: %w", err)
	}

	cards := make([]Card, 0, len(raw))
	for _, item := range raw {
		card, err := normalizeCard(item, opts.MinConfidence)
		if err != nil {
			continue
		}
		cards = append(cards, card)
	}
	return cards, nil
}

type streamLogCollector struct {
	pending string
	emit    func(string)
}

func newStreamLogCollector(emit func(string)) *streamLogCollector {
	return &streamLogCollector{
		pending: "",
		emit:    emit,
	}
}

func (c *streamLogCollector) Push(fragment string) {
	if c == nil {
		return
	}
	text := strings.ReplaceAll(fragment, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	c.pending += text

	for {
		idx := strings.IndexByte(c.pending, '\n')
		if idx < 0 {
			break
		}
		c.emitLine(c.pending[:idx])
		c.pending = c.pending[idx+1:]
	}

	for len(c.pending) > streamLogChunkBytes {
		c.emitLine(c.pending[:streamLogChunkBytes])
		c.pending = c.pending[streamLogChunkBytes:]
	}
}

func (c *streamLogCollector) Flush() {
	if c == nil {
		return
	}
	c.emitLine(c.pending)
	c.pending = ""
}

func (c *streamLogCollector) emitLine(line string) {
	if c == nil || c.emit == nil {
		return
	}
	msg := strings.TrimSpace(line)
	if msg == "" {
		return
	}
	c.emit(msg)
}

func normalizeCard(item map[string]any, minConfidence float64) (Card, error) {
	page := asString(item["page"])
	question := asString(item["question"])
	answer := asString(item["answer"])
	confidence := asFloat(item["confidence"])
	issues := normalizeIssues(item["issue"])

	if confidence < minConfidence {
		if !contains(issues, "low_confidence") {
			issues = append(issues, "low_confidence")
		}
	}

	return Card{
		Page:       strings.TrimSpace(page),
		Question:   strings.TrimSpace(question),
		Answer:     strings.TrimSpace(answer),
		Confidence: confidence,
		Issue:      issues,
	}, nil
}

func normalizeIssues(v any) []string {
	arr, ok := v.([]any)
	if !ok {
		return []string{}
	}
	seen := make(map[string]struct{})
	issues := make([]string, 0, len(arr))
	for _, it := range arr {
		s := strings.TrimSpace(asString(it))
		if s == "" {
			continue
		}
		if _, exists := issueSet[s]; !exists {
			continue
		}
		if _, exists := seen[s]; exists {
			continue
		}
		seen[s] = struct{}{}
		issues = append(issues, s)
	}
	return issues
}

func buildPrompt(frontPrompt, backPrompt string, task chunkTask) string {
	if task.Kind == sourceKindPDF {
		return strings.TrimSpace(fmt.Sprintf(`あなたは与えられたPDFチャンクからAnki向け一問一答カードを作成します。

このPDFは事前に対象ページだけを分割したチャンクです。
このPDFに含まれる内容のみを使い、推測で補完しないでください。

要件:
- question(front): %s
- answer(back): %s
- 応答は配列(JSON)のみ
- 各要素のキーは page, question, answer, confidence, issue
- page はPDF内の印字ページ番号（読めない場合は空文字）
- question/answer は必要なら null 可
- confidence は 0.0〜1.0
- issue は次のみ: %s
- チャンク内にQ/A化できる内容が無ければ [] を返す`,
			frontPrompt,
			backPrompt,
			strings.Join(issueCatalog, ", "),
		))
	}

	return strings.TrimSpace(fmt.Sprintf(`あなたは与えられた画像チャンクからAnki向け一問一答カードを作成します。

このチャンクには page %s の画像が順番に含まれます。
各画像に付いたページ番号を使い、page には必ず対応する画像番号(1始まり)を入れてください。
画像に含まれる内容のみを使い、推測で補完しないでください。

要件:
- question(front): %s
- answer(back): %s
- 応答は配列(JSON)のみ
- 各要素のキーは page, question, answer, confidence, issue
- page は画像番号(例: "3")
- question/answer は必要なら null 可
- confidence は 0.0〜1.0
- issue は次のみ: %s
- 画像内にQ/A化できる内容が無ければ [] を返す`,
		task.Range.Label(),
		frontPrompt,
		backPrompt,
		strings.Join(issueCatalog, ", "),
	))
}

func buildSchema() map[string]any {
	return map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"page": map[string]any{
					"type":        "string",
					"description": "PDFのページ番号または画像番号",
				},
				"question": map[string]any{
					"type":        []string{"string", "null"},
					"description": "Front(表)",
				},
				"answer": map[string]any{
					"type":        []string{"string", "null"},
					"description": "Back(裏)",
				},
				"confidence": map[string]any{
					"type":    "number",
					"minimum": 0,
					"maximum": 1,
				},
				"issue": map[string]any{
					"type": "array",
					"items": map[string]any{
						"type": "string",
						"enum": issueCatalog,
					},
				},
			},
			"required":             []string{"page", "question", "answer", "confidence", "issue"},
			"additionalProperties": false,
		},
	}
}
