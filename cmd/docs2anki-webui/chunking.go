package main

import (
	"fmt"
	"os"
	"path/filepath"

	pdfapi "github.com/pdfcpu/pdfcpu/pkg/api"
)

func buildChunkTasks(sources []uploadedSource, opts processOptions) ([]chunkTask, []string, func(), error) {
	if len(sources) == 0 {
		return nil, nil, nil, fmt.Errorf("アップロードファイルがありません")
	}
	kind := sources[0].Kind
	for _, src := range sources[1:] {
		if src.Kind != kind {
			return nil, nil, nil, fmt.Errorf("PDFと画像は同時にアップロードできません")
		}
	}

	ranges, err := parseRanges(opts.Ranges)
	if err != nil {
		return nil, nil, nil, err
	}
	spans, err := iterChunks(ranges, opts.Step, opts.Overlap)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(spans) == 0 {
		return nil, nil, nil, fmt.Errorf("対象ページがありません")
	}

	if kind == sourceKindPDF {
		if len(sources) != 1 {
			return nil, nil, nil, fmt.Errorf("PDFは1ファイルのみアップロードできます")
		}
		return prepareChunkTasks(sources[0].Path, spans)
	}

	validSpans, spanWarnings := clampSpansToCount(spans, len(sources), "画像枚数")
	if len(validSpans) == 0 {
		return nil, nil, nil, fmt.Errorf("指定範囲が画像枚数(%d)の範囲外です", len(sources))
	}

	tasks := make([]chunkTask, 0, len(validSpans))
	for _, span := range validSpans {
		assets := make([]chunkAsset, 0, span.End-span.Start+1)
		totalSize := int64(0)
		for page := span.Start; page <= span.End; page++ {
			src := sources[page-1]
			totalSize += src.Size
			assets = append(assets, chunkAsset{
				Path:        src.Path,
				DisplayName: src.DisplayName,
				MIMEType:    src.MIMEType,
				Page:        page,
			})
		}
		tasks = append(tasks, chunkTask{
			Range:    span,
			Assets:   assets,
			FileSize: totalSize,
			Kind:     sourceKindImage,
		})
	}
	for i := range tasks {
		tasks[i].Index = i
	}
	return tasks, spanWarnings, func() {}, nil
}

func taskLabel(task chunkTask) string {
	if task.Kind == sourceKindPDF {
		return task.Range.Label()
	}
	if task.Range.Start == task.Range.End {
		return fmt.Sprintf("img-%d", task.Range.Start)
	}
	return fmt.Sprintf("img-%s", task.Range.Label())
}

func prepareChunkTasks(pdfPath string, spans []pageRange) ([]chunkTask, []string, func(), error) {
	warnings := make([]string, 0)

	tmpDir, err := os.MkdirTemp("", "docs2anki-webui-chunks-*")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("チャンク作業ディレクトリ作成に失敗: %w", err)
	}
	cleanup := func() {
		_ = os.RemoveAll(tmpDir)
	}

	sourcePath := pdfPath
	optimizedSourcePath := filepath.Join(tmpDir, optimizedSourceFileName)
	if err := pdfapi.OptimizeFile(pdfPath, optimizedSourcePath, nil); err != nil {
		warnings = append(warnings, fmt.Sprintf("元PDFの最適化に失敗したため元ファイルを使用します: %v", err))
	} else {
		sourcePath = optimizedSourcePath
	}

	pageCount, err := pdfapi.PageCountFile(sourcePath)
	if err != nil && sourcePath != pdfPath {
		warnings = append(warnings, fmt.Sprintf("最適化PDFのページ数取得に失敗したため元PDFにフォールバックします: %v", err))
		sourcePath = pdfPath
		pageCount, err = pdfapi.PageCountFile(sourcePath)
	}
	if err != nil {
		cleanup()
		return nil, nil, nil, fmt.Errorf("PDFページ数の取得に失敗: %w", err)
	}
	if pageCount < 1 {
		cleanup()
		return nil, nil, nil, fmt.Errorf("PDFページ数が不正です")
	}

	validSpans, spanWarnings := clampSpansToCount(spans, pageCount, "PDFページ数")
	warnings = append(warnings, spanWarnings...)
	if len(validSpans) == 0 {
		cleanup()
		return nil, nil, nil, fmt.Errorf("指定範囲がPDFページ数(%d)の範囲外です", pageCount)
	}

	preparer := &chunkPreparer{
		sourcePath: sourcePath,
		tmpDir:     tmpDir,
		maxBytes:   defaultMaxChunkPDFBytes,
		warnings:   &warnings,
	}

	tasks := make([]chunkTask, 0, len(validSpans))
	for _, span := range validSpans {
		built, buildErr := preparer.build(span)
		if buildErr != nil {
			cleanup()
			return nil, nil, nil, buildErr
		}
		tasks = append(tasks, built...)
	}
	for i := range tasks {
		tasks[i].Index = i
	}
	return tasks, warnings, cleanup, nil
}

type chunkPreparer struct {
	sourcePath string
	tmpDir     string
	maxBytes   int64
	warnings   *[]string
	seq        int
}

func (p *chunkPreparer) build(span pageRange) ([]chunkTask, error) {
	chunkPath, chunkSize, err := p.renderChunk(span)
	if err != nil {
		return nil, err
	}

	if chunkSize > p.maxBytes {
		*p.warnings = append(*p.warnings,
			fmt.Sprintf("チャンク %s のPDFサイズ %s は目安 %s を超えています。Gemini API 側でファイルサイズエラーになる可能性があります",
				span.Label(),
				formatBytes(chunkSize),
				formatBytes(p.maxBytes),
			),
		)
	}

	return []chunkTask{{
		Range: span,
		Assets: []chunkAsset{{
			Path:        chunkPath,
			DisplayName: fmt.Sprintf("chunk-%s.pdf", span.Label()),
			MIMEType:    "application/pdf",
			Page:        span.Start,
		}},
		FileSize: chunkSize,
		Kind:     sourceKindPDF,
	}}, nil
}

func (p *chunkPreparer) renderChunk(span pageRange) (string, int64, error) {
	p.seq++
	base := fmt.Sprintf("chunk-%05d-%d-%d", p.seq, span.Start, span.End)
	trimmedPath := filepath.Join(p.tmpDir, base+".trim.pdf")
	finalPath := filepath.Join(p.tmpDir, base+".pdf")
	selectedPages := []string{span.Label()}

	if err := pdfapi.TrimFile(p.sourcePath, trimmedPath, selectedPages, nil); err != nil {
		return "", 0, fmt.Errorf("チャンク %s の抽出に失敗: %w", span.Label(), err)
	}

	if err := pdfapi.OptimizeFile(trimmedPath, finalPath, nil); err != nil {
		*p.warnings = append(*p.warnings, fmt.Sprintf("チャンク %s の最適化に失敗したため未最適化PDFを使用します: %v", span.Label(), err))
		_ = os.Remove(finalPath)
		if renameErr := os.Rename(trimmedPath, finalPath); renameErr != nil {
			return "", 0, fmt.Errorf("チャンク %s の保存に失敗: %w", span.Label(), renameErr)
		}
	} else {
		_ = os.Remove(trimmedPath)
	}

	info, err := os.Stat(finalPath)
	if err != nil {
		return "", 0, fmt.Errorf("チャンク %s のファイル情報取得に失敗: %w", span.Label(), err)
	}
	if info.Size() <= 0 {
		return "", 0, fmt.Errorf("チャンク %s が空のPDFとして生成されました", span.Label())
	}
	return finalPath, info.Size(), nil
}

func clampSpansToCount(spans []pageRange, pageCount int, scopeLabel string) ([]pageRange, []string) {
	warnings := make([]string, 0)
	out := make([]pageRange, 0, len(spans))
	for _, span := range spans {
		if span.Start > pageCount {
			warnings = append(warnings, fmt.Sprintf("%s は%s(%d)外のためスキップしました", span.Label(), scopeLabel, pageCount))
			continue
		}
		adjusted := span
		if adjusted.End > pageCount {
			adjusted.End = pageCount
			warnings = append(warnings, fmt.Sprintf("%s は%s(%d)に合わせて %s に調整しました", span.Label(), scopeLabel, pageCount, adjusted.Label()))
		}
		if adjusted.Start < 1 {
			adjusted.Start = 1
		}
		if adjusted.Start > adjusted.End {
			warnings = append(warnings, fmt.Sprintf("%s は有効ページが無いためスキップしました", span.Label()))
			continue
		}
		out = append(out, adjusted)
	}
	return out, warnings
}
