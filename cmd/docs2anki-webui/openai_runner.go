package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

func init() {
	registerChunkRunnerFactory(providerOpenAI, newOpenAIChunkRunner)
}

func newOpenAIChunkRunner(_ context.Context, opts processOptions, tasks []chunkTask) (chunkRunner, []string, error) {
	client := newOpenAIClient(opts.APIKey)
	return func(ctx context.Context, opts processOptions, task chunkTask, onLog func(string)) ([]Card, error) {
		return runOpenAIChunk(ctx, client, opts, task, onLog)
	}, buildOpenAIWarnings(opts, tasks), nil
}

func newOpenAIClient(apiKey string) openai.Client {
	return openai.NewClient(option.WithAPIKey(apiKey))
}

func buildOpenAIWarnings(opts processOptions, tasks []chunkTask) []string {
	if !opts.OpenAIImageDetailOriginal || len(tasks) == 0 || tasks[0].Kind != sourceKindPDF {
		return nil
	}
	return []string{`OpenAIの detail="original" は画像入力でのみ有効です。PDFチャンクは input_file として送信するため、このジョブでは適用されません。`}
}

func runOpenAIChunk(ctx context.Context, client openai.Client, opts processOptions, task chunkTask, onLog func(string)) ([]Card, error) {
	if len(task.Assets) == 0 {
		return nil, fmt.Errorf("チャンクにファイルが含まれていません")
	}
	if onLog == nil {
		onLog = func(string) {}
	}

	if opts.DelayMS > 0 {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(time.Duration(opts.DelayMS) * time.Millisecond):
		}
	}

	onLog("OpenAI呼び出し")
	cards, err := runOpenAIChunkOnce(ctx, client, opts, task, onLog)
	if err != nil {
		return nil, fmt.Errorf("チャンク処理失敗(%s, size=%s): %w", taskLabel(task), formatBytes(task.FileSize), err)
	}
	onLog(fmt.Sprintf("JSON解析完了: cards=%d", len(cards)))
	return cards, nil
}

func runOpenAIChunkOnce(ctx context.Context, client openai.Client, opts processOptions, task chunkTask, onLog func(string)) ([]Card, error) {
	req, err := buildOpenAIRequest(opts, task)
	if err != nil {
		return nil, err
	}

	resp, err := client.Responses.New(ctx, req)
	if err != nil {
		return nil, err
	}

	text := strings.TrimSpace(resp.OutputText())
	if text == "" {
		onLog("OpenAI応答が空でした。")
		return nil, nil
	}
	onLog(fmt.Sprintf("OpenAI応答受信: %d bytes", len(text)))

	raw, err := parseOpenAICards(text)
	if err != nil {
		return nil, err
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

func buildOpenAIRequest(opts processOptions, task chunkTask) (responses.ResponseNewParams, error) {
	input, err := buildOpenAIInput(task, opts)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}

	format := responses.ResponseFormatTextConfigParamOfJSONSchema("anki_cards", buildOpenAISchema())
	if schema := format.OfJSONSchema; schema != nil {
		schema.Strict = openai.Bool(true)
	}

	req := responses.ResponseNewParams{
		Model: shared.ResponsesModel(opts.Model),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: input,
		},
		Text: responses.ResponseTextConfigParam{
			Format: format,
		},
	}

	if effort := strings.TrimSpace(opts.OpenAIReasoningEffort); effort != "" {
		req.Reasoning = shared.ReasoningParam{
			Effort: shared.ReasoningEffort(strings.ToLower(effort)),
		}
	}

	return req, nil
}

func buildOpenAIInput(task chunkTask, opts processOptions) (responses.ResponseInputParam, error) {
	content := make(responses.ResponseInputMessageContentListParam, 0, len(task.Assets)*2+1)

	switch task.Kind {
	case sourceKindPDF:
		for _, asset := range task.Assets {
			fileData, err := base64File(asset.Path)
			if err != nil {
				return nil, fmt.Errorf("PDFチャンクの読込に失敗(%s): %w", taskLabel(task), err)
			}
			content = append(content, responses.ResponseInputContentUnionParam{
				OfInputFile: &responses.ResponseInputFileParam{
					Filename: openai.String(asset.DisplayName),
					FileData: openai.String(fileData),
					Detail:   responses.ResponseInputFileDetailHigh,
				},
			})
		}
	case sourceKindImage:
		detail := responses.ResponseInputImageDetailHigh
		if opts.OpenAIImageDetailOriginal && isGPT54Model(opts.Model) {
			detail = responses.ResponseInputImageDetailOriginal
		}
		for _, asset := range task.Assets {
			imageURL, err := dataURLFromFile(asset.Path, asset.MIMEType)
			if err != nil {
				return nil, fmt.Errorf("画像チャンクの読込に失敗(%s, page=%d): %w", taskLabel(task), asset.Page, err)
			}
			content = append(content, responses.ResponseInputContentParamOfInputText(fmt.Sprintf("以下は入力画像番号 %d の画像です。", asset.Page)))
			content = append(content, responses.ResponseInputContentUnionParam{
				OfInputImage: &responses.ResponseInputImageParam{
					ImageURL: openai.String(imageURL),
					Detail:   detail,
				},
			})
		}
	default:
		return nil, fmt.Errorf("未対応の入力形式です: %s", task.Kind)
	}

	content = append(content, responses.ResponseInputContentParamOfInputText(buildOpenAIPrompt(opts.FrontPrompt, opts.BackPrompt, task)))

	return responses.ResponseInputParam{
		responses.ResponseInputItemParamOfMessage(content, responses.EasyInputMessageRoleUser),
	}, nil
}

func buildOpenAICardSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"page": map[string]any{
				"type":        "string",
				"description": "PDFまたは画像内に印字されたページ番号。読めない場合は空文字。",
			},
			"question": map[string]any{
				"type": "string",
			},
			"answer": map[string]any{
				"type": "string",
			},
			"confidence": map[string]any{
				"type": "number",
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
	}
}

func buildOpenAICardsSchema() map[string]any {
	return map[string]any{
		"type":  "array",
		"items": buildOpenAICardSchema(),
	}
}

func buildOpenAISchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"cards": buildOpenAICardsSchema(),
		},
		"required":             []string{"cards"},
		"additionalProperties": false,
	}
}

func buildOpenAIPrompt(frontPrompt, backPrompt string, task chunkTask) string {
	if task.Kind == sourceKindPDF {
		return strings.TrimSpace(fmt.Sprintf(`あなたは与えられたPDFチャンクからAnki向け一問一答カードを作成します。

このPDFは事前に対象ページだけを分割したチャンクです。
このPDFに含まれる内容のみを使い、推測で補完しないでください。

要件:
- question(front): %s
- answer(back): %s
- 最上位は JSON object
- 最上位キーは cards のみ
- cards はカード配列
- 各カードのキーは page, question, answer, confidence, issue
- %s
- question/answer は文字列。作れない場合は空文字
- confidence は 0.0〜1.0
%s
- カードが0件なら {"cards": []}
- 配列単体は返さない`,
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
- 最上位は JSON object
- 最上位キーは cards のみ
- cards はカード配列
- 各カードのキーは page, question, answer, confidence, issue
- %s
- question/answer は文字列。作れない場合は空文字
- confidence は 0.0〜1.0
%s
- カードが0件なら {"cards": []}
- 配列単体は返さない`,
		task.Range.Label(),
		frontPrompt,
		backPrompt,
		pagePromptInstruction(task.Kind),
		buildIssuePromptText(),
	))
}

func parseOpenAICards(text string) ([]map[string]any, error) {
	trimmed := strings.TrimSpace(text)
	switch {
	case strings.HasPrefix(trimmed, "{"):
		var payload struct {
			Cards []map[string]any `json:"cards"`
		}
		if err := json.Unmarshal([]byte(trimmed), &payload); err != nil {
			return nil, fmt.Errorf("JSON解析に失敗: %w", err)
		}
		if payload.Cards == nil {
			return nil, fmt.Errorf("JSON解析に失敗: cards フィールドが見つかりません")
		}
		return payload.Cards, nil
	case strings.HasPrefix(trimmed, "["):
		var raw []map[string]any
		if err := json.Unmarshal([]byte(trimmed), &raw); err != nil {
			return nil, fmt.Errorf("JSON解析に失敗: %w", err)
		}
		return raw, nil
	default:
		return nil, fmt.Errorf("JSON解析に失敗: 想定外のJSON形式です")
	}
}

func base64File(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(data), nil
}

func dataURLFromFile(path string, mimeType string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	resolvedMIME := strings.TrimSpace(mimeType)
	if resolvedMIME == "" {
		resolvedMIME = detectContentTypeFromFile(path)
	}
	if resolvedMIME == "" {
		resolvedMIME = "application/octet-stream"
	}
	return fmt.Sprintf("data:%s;base64,%s", resolvedMIME, base64.StdEncoding.EncodeToString(data)), nil
}
