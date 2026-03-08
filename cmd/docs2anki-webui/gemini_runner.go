package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"google.golang.org/genai"
)

func init() {
	registerChunkRunnerFactory(providerGemini, newGeminiChunkRunner)
}

func newGeminiChunkRunner(ctx context.Context, opts processOptions, _ []chunkTask) (chunkRunner, []string, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  opts.APIKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("Geminiクライアント作成に失敗: %w", err)
	}
	return func(ctx context.Context, opts processOptions, task chunkTask, onLog func(string)) ([]Card, error) {
		return runGeminiChunk(ctx, client, opts, task, onLog)
	}, nil, nil
}

func waitGeminiFileReady(ctx context.Context, client *genai.Client, fileName string) (*genai.File, error) {
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

type geminiUploadedChunkFile struct {
	URI      string
	MIMEType string
	Page     int
}

func runGeminiChunk(ctx context.Context, client *genai.Client, opts processOptions, task chunkTask, onLog func(string)) ([]Card, error) {
	if len(task.Assets) == 0 {
		return nil, fmt.Errorf("チャンクにファイルが含まれていません")
	}
	if onLog == nil {
		onLog = func(string) {}
	}

	uploadedFiles := make([]geminiUploadedChunkFile, 0, len(task.Assets))
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

		uploaded, err = waitGeminiFileReady(ctx, client, uploaded.Name)
		if err != nil {
			return nil, fmt.Errorf("チャンクファイル準備待ち失敗(%s): %w", taskLabel(task), err)
		}
		onLog(fmt.Sprintf("アップロード完了: page=%d", asset.Page))
		uploadedFiles = append(uploadedFiles, geminiUploadedChunkFile{
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
	cards, err := runGeminiChunkOnce(ctx, client, uploadedFiles, opts, task, onLog)
	if err != nil {
		return nil, fmt.Errorf("チャンク処理失敗(%s, size=%s): %w", taskLabel(task), formatBytes(task.FileSize), err)
	}
	onLog(fmt.Sprintf("JSON解析完了: cards=%d", len(cards)))
	return cards, nil
}

func runGeminiChunkOnce(ctx context.Context, client *genai.Client, files []geminiUploadedChunkFile, opts processOptions, task chunkTask, onLog func(string)) ([]Card, error) {
	prompt := buildGeminiPrompt(opts.FrontPrompt, opts.BackPrompt, task)
	parts := make([]*genai.Part, 0, len(files)*2+1)
	for _, file := range files {
		if task.Kind == sourceKindImage {
			parts = append(parts, &genai.Part{Text: fmt.Sprintf("以下は入力画像番号 %d の画像です。", file.Page)})
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
		ResponseJsonSchema: buildGeminiSchema(),
		MediaResolution:    genai.MediaResolutionHigh,
	}
	if opts.ThinkingBudget >= -1 {
		budget := int32(opts.ThinkingBudget)
		cfg.ThinkingConfig = &genai.ThinkingConfig{ThinkingBudget: &budget}
	}

	collector := newGeminiStreamLogCollector(func(line string) {
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

type geminiStreamLogCollector struct {
	pending string
	emit    func(string)
}

func newGeminiStreamLogCollector(emit func(string)) *geminiStreamLogCollector {
	return &geminiStreamLogCollector{
		pending: "",
		emit:    emit,
	}
}

func (c *geminiStreamLogCollector) Push(fragment string) {
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

func (c *geminiStreamLogCollector) Flush() {
	if c == nil {
		return
	}
	c.emitLine(c.pending)
	c.pending = ""
}

func (c *geminiStreamLogCollector) emitLine(line string) {
	if c == nil || c.emit == nil {
		return
	}
	msg := strings.TrimSpace(line)
	if msg == "" {
		return
	}
	c.emit(msg)
}

func buildGeminiPrompt(frontPrompt, backPrompt string, task chunkTask) string {
	if task.Kind == sourceKindPDF {
		return strings.TrimSpace(fmt.Sprintf(`あなたは与えられたPDFチャンクからAnki向け一問一答カードを作成します。

このPDFは事前に対象ページだけを分割したチャンクです。
このPDFに含まれる内容のみを使い、推測で補完しないでください。

要件:
- question(front): %s
- answer(back): %s
- 応答は配列(JSON)のみ
- 各要素のキーは page, question, answer, confidence, issue
- %s
- question/answer は必要なら null 可
- confidence は 0.0〜1.0
%s
- チャンク内にQ/A化できる内容が無ければ [] を返す`,
			frontPrompt,
			backPrompt,
			pagePromptInstruction(task.Kind),
			buildIssuePromptText(),
		))
	}

	return strings.TrimSpace(fmt.Sprintf(`あなたは与えられた画像チャンクからAnki向け一問一答カードを作成します。

このチャンクには入力画像番号 %s の画像が順番に含まれます。
入力画像番号は管理用です。出力の page には使わず、画像内の印字ページ番号を使ってください。
画像に含まれる内容のみを使い、推測で補完しないでください。

要件:
- question(front): %s
- answer(back): %s
- 応答は配列(JSON)のみ
- 各要素のキーは page, question, answer, confidence, issue
- %s
- question/answer は必要なら null 可
- confidence は 0.0〜1.0
%s
- 画像内にQ/A化できる内容が無ければ [] を返す`,
		task.Range.Label(),
		frontPrompt,
		backPrompt,
		pagePromptInstruction(task.Kind),
		buildIssuePromptText(),
	))
}

func buildGeminiSchema() map[string]any {
	return map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"page": map[string]any{
					"type":        "string",
					"description": "PDFまたは画像内に印字されたページ番号。読めない場合は空文字。",
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
