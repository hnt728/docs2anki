package main

import (
	"context"
	"embed"
	"fmt"
	"sync"
	"time"
)

const (
	defaultGeminiModel      = "gemini-3-flash-preview"
	defaultOpenAIModel      = "gpt-5.4"
	defaultOpenAIReasoning  = "medium"
	defaultMaxChunkPDFBytes = int64(19 * 1024 * 1024) // warning threshold
	optimizedSourceFileName = "source.optimized.pdf"
	maxJobLogEntries        = 900
	maxJobLogMessageBytes   = 700
	streamLogChunkBytes     = 220
)

type sourceKind string

const (
	sourceKindPDF   sourceKind = "pdf"
	sourceKindImage sourceKind = "image"
)

type providerKind string

const (
	providerGemini providerKind = "gemini"
	providerOpenAI providerKind = "openai"
)

type issueDefinition struct {
	Label       string
	Description string
}

func buildIssueCatalog(defs []issueDefinition) []string {
	labels := make([]string, 0, len(defs))
	for _, def := range defs {
		labels = append(labels, def.Label)
	}
	return labels
}

func buildIssueSet(defs []issueDefinition) map[string]struct{} {
	set := make(map[string]struct{}, len(defs))
	for _, def := range defs {
		set[def.Label] = struct{}{}
	}
	return set
}

var (
	//go:embed web/*
	staticAssets embed.FS

	issueDefinitions = []issueDefinition{
		{
			Label:       "page_number_missing",
			Description: "PDF/画像内に印字ページ番号があるはずだが、見えない、隠れている、反射している、荒い等の理由で読み取れない",
		},
		{
			Label:       "page_split",
			Description: "文章や表が途中で切れて読めない、またはチャンク境界で文脈が途切れて内容が不完全",
		},
		{
			Label:       "unreadable_content",
			Description: "レイアウト、画質、反射、欠け等の理由で本文や表の内容を読み取れない",
		},
		{
			Label:       "non_qa_content",
			Description: "Q&Aにするのに十分な情報がない、またはQ&A化に向かない内容",
		},
		{
			Label:       "other",
			Description: "上記に当てはまらない問題",
		},
	}
	issueCatalog = buildIssueCatalog(issueDefinitions)
	issueSet     = func() map[string]struct{} {
		set := buildIssueSet(issueDefinitions)
		set["low_confidence"] = struct{}{}
		return set
	}()
	legacyIssueAliases = map[string]string{
		"insufficient_context": "non_qa_content",
	}
	allowedSourceMIMEs = map[string]sourceKind{
		"application/pdf": sourceKindPDF,
		"image/png":       sourceKindImage,
		"image/jpeg":      sourceKindImage,
		"image/webp":      sourceKindImage,
		"image/gif":       sourceKindImage,
		"image/bmp":       sourceKindImage,
		"image/tiff":      sourceKindImage,
	}
	sourceExtToMIME = map[string]string{
		".pdf":  "application/pdf",
		".png":  "image/png",
		".jpg":  "image/jpeg",
		".jpeg": "image/jpeg",
		".webp": "image/webp",
		".gif":  "image/gif",
		".bmp":  "image/bmp",
		".tif":  "image/tiff",
		".tiff": "image/tiff",
	}
)

type app struct {
	jobs           *jobStore
	maxUploadBytes int64
}

type processOptions struct {
	Provider                  providerKind
	APIKey                    string
	Model                     string
	Ranges                    string
	Step                      int
	Overlap                   int
	FrontPrompt               string
	BackPrompt                string
	MinConfidence             float64
	DelayMS                   int
	ThinkingBudget            int
	OpenAIReasoningEffort     string
	OpenAIImageDetailOriginal bool
}

type uploadedSource struct {
	Path        string
	DisplayName string
	MIMEType    string
	Kind        sourceKind
	Size        int64
}

type jobConfigSummary struct {
	Provider                  string  `json:"provider"`
	SourceType                string  `json:"sourceType"`
	Model                     string  `json:"model"`
	Ranges                    string  `json:"ranges"`
	Step                      int     `json:"step"`
	Overlap                   int     `json:"overlap"`
	FrontPrompt               string  `json:"frontPrompt"`
	BackPrompt                string  `json:"backPrompt"`
	MinConfidence             float64 `json:"minConfidence"`
	DelayMS                   int     `json:"delayMs"`
	ThinkingBudget            int     `json:"thinkingBudget"`
	OpenAIReasoningEffort     string  `json:"openaiReasoningEffort,omitempty"`
	OpenAIImageDetailOriginal bool    `json:"openaiImageDetailOriginal,omitempty"`
}

type Card struct {
	Page       string   `json:"page"`
	Question   string   `json:"question"`
	Answer     string   `json:"answer"`
	Confidence float64  `json:"confidence"`
	Issue      []string `json:"issue"`
}

type pageRange struct {
	Start int
	End   int
}

func (r pageRange) Label() string {
	return fmt.Sprintf("%d-%d", r.Start, r.End)
}

type chunkTask struct {
	Index    int
	Range    pageRange
	Assets   []chunkAsset
	FileSize int64
	Kind     sourceKind
}

type chunkAsset struct {
	Path        string
	DisplayName string
	MIMEType    string
	Page        int
}

type jobStore struct {
	mu   sync.RWMutex
	jobs map[string]*job
}

type chunkDecision string

const (
	chunkDecisionRetry chunkDecision = "retry"
	chunkDecisionSkip  chunkDecision = "skip"
)

type jobActionRequest struct {
	Action string `json:"action"`
}

type jobLogEntry struct {
	Seq     int64  `json:"seq"`
	Time    string `json:"time"`
	Chunk   string `json:"chunk,omitempty"`
	Message string `json:"message"`
}

type job struct {
	mu sync.RWMutex

	ID        string           `json:"id"`
	Status    string           `json:"status"`
	Config    jobConfigSummary `json:"config"`
	CreatedAt time.Time        `json:"createdAt"`
	UpdatedAt time.Time        `json:"updatedAt"`

	TotalChunks     int      `json:"totalChunks"`
	CompletedChunks int      `json:"completedChunks"`
	ActiveChunks    []string `json:"activeChunks"`
	FailedChunks    []string `json:"failedChunks,omitempty"`

	IssueCount int           `json:"issueCount"`
	Warnings   []string      `json:"warnings,omitempty"`
	Error      string        `json:"error,omitempty"`
	Cards      []Card        `json:"cards,omitempty"`
	Logs       []jobLogEntry `json:"logs,omitempty"`

	StopRequested bool   `json:"stopRequested,omitempty"`
	PendingChunk  string `json:"pendingChunk,omitempty"`
	PendingError  string `json:"pendingError,omitempty"`

	activeSet map[string]struct{}
	cancelFn  context.CancelFunc
	nextLogID int64
	decisionQ chan chunkDecision
}
