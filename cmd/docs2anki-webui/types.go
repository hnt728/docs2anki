package main

import (
	"context"
	"embed"
	"fmt"
	"sync"
	"time"
)

const (
	defaultModel            = "gemini-3-flash-preview"
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

var (
	//go:embed web/*
	staticAssets embed.FS

	issueCatalog = []string{
		"page_number_missing",
		"page_split",
		"insufficient_context",
		"unreadable_content",
		"non_qa_content",
		"other",
	}
	issueSet = map[string]struct{}{
		"page_number_missing":  {},
		"page_split":           {},
		"insufficient_context": {},
		"unreadable_content":   {},
		"non_qa_content":       {},
		"other":                {},
		"low_confidence":       {},
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
	APIKey         string
	Model          string
	Ranges         string
	Step           int
	Overlap        int
	FrontPrompt    string
	BackPrompt     string
	MinConfidence  float64
	DelayMS        int
	ThinkingBudget int
}

type uploadedSource struct {
	Path        string
	DisplayName string
	MIMEType    string
	Kind        sourceKind
	Size        int64
}

type jobConfigSummary struct {
	SourceType     string  `json:"sourceType"`
	Model          string  `json:"model"`
	Ranges         string  `json:"ranges"`
	Step           int     `json:"step"`
	Overlap        int     `json:"overlap"`
	FrontPrompt    string  `json:"frontPrompt"`
	BackPrompt     string  `json:"backPrompt"`
	MinConfidence  float64 `json:"minConfidence"`
	DelayMS        int     `json:"delayMs"`
	ThinkingBudget int     `json:"thinkingBudget"`
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
