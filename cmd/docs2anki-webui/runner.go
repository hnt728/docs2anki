package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
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
	warnings := append([]string{}, prepWarnings...)
	for _, msg := range prepWarnings {
		job.appendWarning(msg)
		job.appendLogf("system", "warning: %s", msg)
	}

	cardsByChunk := make([][]Card, len(chunks))
	failedChunks := make([]string, 0)
	failedCount := 0

	runChunkWithProvider, providerWarnings, err := newChunkRunner(ctx, opts, chunks)
	if err != nil {
		return err
	}
	warnings = append(warnings, providerWarnings...)
	for _, msg := range providerWarnings {
		job.appendWarning(msg)
		job.appendLogf("system", "warning: %s", msg)
	}

	for _, task := range chunks {
		if ctx.Err() != nil && job.isStopRequested() {
			return context.Canceled
		}

		label := taskLabel(task)
		job.markChunkStart(label)
		job.appendLog(label, "開始")
		for {
			cards, runErr := runChunkWithProvider(ctx, opts, task, func(message string) {
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

type chunkRunner func(context.Context, processOptions, chunkTask, func(string)) ([]Card, error)

type chunkRunnerFactory func(context.Context, processOptions, []chunkTask) (chunkRunner, []string, error)

var chunkRunnerFactories = map[providerKind]chunkRunnerFactory{}

func registerChunkRunnerFactory(provider providerKind, factory chunkRunnerFactory) {
	if factory == nil {
		panic(fmt.Sprintf("chunk runner factory is nil: %s", provider))
	}
	if _, exists := chunkRunnerFactories[provider]; exists {
		panic(fmt.Sprintf("chunk runner factory already registered: %s", provider))
	}
	chunkRunnerFactories[provider] = factory
}

func newChunkRunner(ctx context.Context, opts processOptions, tasks []chunkTask) (chunkRunner, []string, error) {
	factory, exists := chunkRunnerFactories[opts.Provider]
	if !exists {
		return nil, nil, fmt.Errorf("未対応のAPIプロバイダです: %s", opts.Provider)
	}
	return factory(ctx, opts, tasks)
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
		if alias, exists := legacyIssueAliases[s]; exists {
			s = alias
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

func buildIssuePromptText() string {
	var b strings.Builder
	b.WriteString("- issue は次のみ。必要なものだけを選び、問題がなければ空配列:\n")
	for _, def := range issueDefinitions {
		fmt.Fprintf(&b, "  - %s: %s\n", def.Label, def.Description)
	}
	return strings.TrimSpace(b.String())
}

func buildSourceNotationPromptText() string {
	return strings.TrimSpace(`- front/back の指示で要約・抽象化・言い換えを明示的に求められている場合、またはカードとして成立させるための最小限の整形が必要な場合を除き、原則として原文の表記をそのまま使用する。
- 特に、固有名詞・専門用語・略語・数値・単位・記号・式は原文の表記を維持し、不要な言い換えや表記変更をしない。`)
}

func pagePromptInstruction(kind sourceKind) string {
	switch kind {
	case sourceKindImage:
		return "page は画像内に印字されたページ番号。読めない場合は空文字"
	case sourceKindPDF:
		return "page はPDF内に印字されたページ番号。読めない場合は空文字"
	default:
		return "page はPDF/画像内に印字されたページ番号。読めない場合は空文字"
	}
}
