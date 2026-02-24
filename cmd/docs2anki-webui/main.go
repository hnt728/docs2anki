package main

import (
	"context"
	crand "crypto/rand"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	pdfapi "github.com/pdfcpu/pdfcpu/pkg/api"
	"google.golang.org/genai"
)

const (
	defaultModel            = "gemini-3-flash-preview"
	defaultMaxChunkPDFBytes = int64(19 * 1024 * 1024) // warning threshold
	optimizedSourceFileName = "source.optimized.pdf"
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
	Concurrency    int
	DelayMS        int
	Retries        int
	ThinkingBudget int
}

type jobConfigSummary struct {
	Model          string  `json:"model"`
	Ranges         string  `json:"ranges"`
	Step           int     `json:"step"`
	Overlap        int     `json:"overlap"`
	FrontPrompt    string  `json:"frontPrompt"`
	BackPrompt     string  `json:"backPrompt"`
	MinConfidence  float64 `json:"minConfidence"`
	Concurrency    int     `json:"concurrency"`
	DelayMS        int     `json:"delayMs"`
	Retries        int     `json:"retries"`
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
	FilePath string
	FileSize int64
}

type chunkResult struct {
	Index int
	Range pageRange
	Cards []Card
	Err   error
}

type jobStore struct {
	mu   sync.RWMutex
	jobs map[string]*job
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

	IssueCount int      `json:"issueCount"`
	Warnings   []string `json:"warnings,omitempty"`
	Error      string   `json:"error,omitempty"`
	Cards      []Card   `json:"cards,omitempty"`

	activeSet map[string]struct{}
}

func newJobStore() *jobStore {
	return &jobStore{jobs: make(map[string]*job)}
}

func (s *jobStore) create(cfg jobConfigSummary) *job {
	j := &job{
		ID:        randomID(),
		Status:    "queued",
		Config:    cfg,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		activeSet: make(map[string]struct{}),
	}
	s.mu.Lock()
	s.jobs[j.ID] = j
	s.mu.Unlock()
	return j
}

func (s *jobStore) get(id string) (*job, bool) {
	s.mu.RLock()
	j, ok := s.jobs[id]
	s.mu.RUnlock()
	return j, ok
}

func (j *job) markRunning(totalChunks int) {
	j.mu.Lock()
	j.Status = "running"
	j.TotalChunks = totalChunks
	j.UpdatedAt = time.Now()
	j.mu.Unlock()
}

func (j *job) markChunkStart(label string) {
	j.mu.Lock()
	j.activeSet[label] = struct{}{}
	j.ActiveChunks = sortedActive(j.activeSet)
	j.UpdatedAt = time.Now()
	j.mu.Unlock()
}

func (j *job) markChunkDone(label string) {
	j.mu.Lock()
	delete(j.activeSet, label)
	j.ActiveChunks = sortedActive(j.activeSet)
	j.CompletedChunks++
	j.UpdatedAt = time.Now()
	j.mu.Unlock()
}

func (j *job) appendWarning(message string) {
	msg := strings.TrimSpace(message)
	if msg == "" {
		return
	}
	j.mu.Lock()
	j.Warnings = append(j.Warnings, msg)
	j.UpdatedAt = time.Now()
	j.mu.Unlock()
}

func (j *job) markCompleted(cards []Card, failedChunks []string, warnings []string) {
	issueCount := 0
	for _, card := range cards {
		if len(card.Issue) > 0 {
			issueCount++
		}
	}

	j.mu.Lock()
	j.Status = "completed"
	j.Cards = cards
	j.IssueCount = issueCount
	j.FailedChunks = append([]string{}, failedChunks...)
	j.Warnings = append([]string{}, warnings...)
	j.ActiveChunks = nil
	j.activeSet = make(map[string]struct{})
	j.UpdatedAt = time.Now()
	j.mu.Unlock()
}

func (j *job) markFailed(err error) {
	j.mu.Lock()
	j.Status = "failed"
	j.Error = err.Error()
	j.ActiveChunks = nil
	j.activeSet = make(map[string]struct{})
	j.UpdatedAt = time.Now()
	j.mu.Unlock()
}

func (j *job) snapshot() job {
	j.mu.RLock()
	defer j.mu.RUnlock()
	cp := *j
	cp.ActiveChunks = append([]string{}, j.ActiveChunks...)
	cp.FailedChunks = append([]string{}, j.FailedChunks...)
	cp.Warnings = append([]string{}, j.Warnings...)
	cp.Cards = append([]Card{}, j.Cards...)
	cp.activeSet = nil
	cp.mu = sync.RWMutex{}
	return cp
}

func sortedActive(m map[string]struct{}) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func randomID() string {
	buf := make([]byte, 12)
	if _, err := crand.Read(buf); err != nil {
		return fmt.Sprintf("job-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(buf)
}

func main() {
	var (
		addr        = flag.String("addr", ":8080", "HTTP listen address")
		maxUploadMB = flag.Int64("max-upload-mb", 300, "maximum upload file size in MB")
	)
	flag.Parse()

	maxUploadBytes := *maxUploadMB * 1024 * 1024
	application := &app{
		jobs:           newJobStore(),
		maxUploadBytes: maxUploadBytes,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", application.handleIndex)
	mux.Handle("/static/", application.handleStatic())
	mux.HandleFunc("/api/jobs", application.handleCreateJob)
	mux.HandleFunc("/api/jobs/", application.handleGetJob)
	mux.HandleFunc("/api/health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	srv := &http.Server{
		Addr:              *addr,
		Handler:           loggingMiddleware(mux),
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("docs2anki listening on http://localhost%s", *addr)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func (a *app) handleStatic() http.Handler {
	sub, err := fs.Sub(staticAssets, "web")
	if err != nil {
		panic(err)
	}
	return http.StripPrefix("/static/", http.FileServer(http.FS(sub)))
}

func (a *app) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET only")
		return
	}
	data, err := staticAssets.ReadFile("web/index.html")
	if err != nil {
		writeError(w, http.StatusInternalServerError, "internal", err.Error())
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(data)
}

func (a *app) handleCreateJob(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST only")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, a.maxUploadBytes)
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_form", "multipartフォームの解析に失敗しました")
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}

	opts, err := parseProcessOptions(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_options", err.Error())
		return
	}

	file, header, err := r.FormFile("pdf")
	if err != nil {
		writeError(w, http.StatusBadRequest, "missing_pdf", "PDFファイルを指定してください")
		return
	}
	defer file.Close()

	tmpPath, err := persistUploadedPDF(file, header.Filename)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "save_failed", err.Error())
		return
	}

	currentJob := a.jobs.create(jobConfigSummary{
		Model:          opts.Model,
		Ranges:         opts.Ranges,
		Step:           opts.Step,
		Overlap:        opts.Overlap,
		FrontPrompt:    opts.FrontPrompt,
		BackPrompt:     opts.BackPrompt,
		MinConfidence:  opts.MinConfidence,
		Concurrency:    opts.Concurrency,
		DelayMS:        opts.DelayMS,
		Retries:        opts.Retries,
		ThinkingBudget: opts.ThinkingBudget,
	})

	go func(j *job, pdfPath string, displayName string, options processOptions) {
		defer os.Remove(pdfPath)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Hour)
		defer cancel()

		if err := runJob(ctx, j, pdfPath, displayName, options); err != nil {
			j.markFailed(err)
			log.Printf("job %s failed: %v", j.ID, err)
		}
	}(currentJob, tmpPath, header.Filename, opts)

	writeJSON(w, http.StatusAccepted, map[string]string{"jobId": currentJob.ID})
}

func (a *app) handleGetJob(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET only")
		return
	}
	id := strings.TrimPrefix(r.URL.Path, "/api/jobs/")
	if id == "" || strings.Contains(id, "/") {
		writeError(w, http.StatusNotFound, "not_found", "job not found")
		return
	}
	job, ok := a.jobs.get(id)
	if !ok {
		writeError(w, http.StatusNotFound, "not_found", "job not found")
		return
	}
	writeJSON(w, http.StatusOK, job.snapshot())
}

func runJob(ctx context.Context, job *job, pdfPath string, displayName string, opts processOptions) error {
	ranges, err := parseRanges(opts.Ranges)
	if err != nil {
		return err
	}
	spans, err := iterChunks(ranges, opts.Step, opts.Overlap)
	if err != nil {
		return err
	}
	if len(spans) == 0 {
		return fmt.Errorf("対象ページがありません")
	}

	chunks, prepWarnings, cleanup, err := prepareChunkTasks(pdfPath, spans)
	if err != nil {
		return err
	}
	defer cleanup()
	if len(chunks) == 0 {
		return fmt.Errorf("対象ページがありません")
	}

	job.markRunning(len(chunks))
	for _, msg := range prepWarnings {
		job.appendWarning(msg)
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  opts.APIKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return fmt.Errorf("Geminiクライアント作成に失敗: %w", err)
	}

	workerCount := opts.Concurrency
	if workerCount > len(chunks) {
		workerCount = len(chunks)
	}
	if workerCount < 1 {
		workerCount = 1
	}

	tasks := make(chan chunkTask)
	results := make(chan chunkResult, len(chunks))
	var workers sync.WaitGroup

	for i := 0; i < workerCount; i++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for task := range tasks {
				label := task.Range.Label()
				job.markChunkStart(label)
				cards, runErr := runChunkWithRetry(ctx, client, displayName, opts, task)
				job.markChunkDone(label)
				results <- chunkResult{
					Index: task.Index,
					Range: task.Range,
					Cards: cards,
					Err:   runErr,
				}
			}
		}()
	}

	go func() {
		for _, task := range chunks {
			tasks <- task
		}
		close(tasks)
		workers.Wait()
		close(results)
	}()

	cardsByChunk := make([][]Card, len(chunks))
	failedChunks := make([]string, 0)
	warnings := append([]string{}, prepWarnings...)
	failedCount := 0

	for res := range results {
		if res.Err != nil {
			failedCount++
			failedChunks = append(failedChunks, res.Range.Label())
			warning := fmt.Sprintf("%s の処理に失敗: %v", res.Range.Label(), res.Err)
			warnings = append(warnings, warning)
			job.appendWarning(warning)
			continue
		}
		cardsByChunk[res.Index] = res.Cards
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
			return nil, fmt.Errorf("Gemini側でPDF処理に失敗しました: %s", formatFileStatus(file.Error))
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

func runChunkWithRetry(ctx context.Context, client *genai.Client, displayName string, opts processOptions, task chunkTask) ([]Card, error) {
	uploaded, err := client.Files.UploadFromPath(ctx, task.FilePath, &genai.UploadFileConfig{
		MIMEType:    "application/pdf",
		DisplayName: chunkDisplayName(displayName, task.Range),
	})
	if err != nil {
		return nil, fmt.Errorf("チャンクPDFアップロード失敗(%s): %w", task.Range.Label(), err)
	}
	defer func() {
		_, _ = client.Files.Delete(context.Background(), uploaded.Name, nil)
	}()

	uploaded, err = waitFileReady(ctx, client, uploaded.Name)
	if err != nil {
		return nil, fmt.Errorf("チャンクPDF準備待ち失敗(%s): %w", task.Range.Label(), err)
	}

	totalAttempts := opts.Retries + 1
	var lastErr error
	for attempt := 0; attempt <= opts.Retries; attempt++ {
		if opts.DelayMS > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(opts.DelayMS) * time.Millisecond):
			}
		}

		cards, runErr := runChunkOnce(ctx, client, uploaded.URI, opts)
		if runErr == nil {
			return cards, nil
		}
		lastErr = runErr
		if attempt == opts.Retries {
			break
		}
		base := 0.5 * math.Pow(2, float64(attempt))
		jitter := rand.Float64() * 0.3
		wait := time.Duration((base+jitter)*1000) * time.Millisecond
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(wait):
		}
	}
	if lastErr == nil {
		lastErr = errors.New("unknown chunk processing error")
	}
	return nil, fmt.Errorf("チャンク処理失敗(%s, size=%s, attempts=%d): %w", task.Range.Label(), formatBytes(task.FileSize), totalAttempts, lastErr)
}

func runChunkOnce(ctx context.Context, client *genai.Client, fileURI string, opts processOptions) ([]Card, error) {
	prompt := buildPrompt(opts.FrontPrompt, opts.BackPrompt)
	contents := []*genai.Content{{
		Role: "user",
		Parts: []*genai.Part{
			{
				FileData: &genai.FileData{
					FileURI:  fileURI,
					MIMEType: "application/pdf",
				},
			},
			{Text: prompt},
		},
	}}

	cfg := &genai.GenerateContentConfig{
		ResponseMIMEType:   "application/json",
		ResponseJsonSchema: buildSchema(),
	}
	if opts.ThinkingBudget >= -1 {
		budget := int32(opts.ThinkingBudget)
		cfg.ThinkingConfig = &genai.ThinkingConfig{ThinkingBudget: &budget}
	}

	resp, err := client.Models.GenerateContent(ctx, opts.Model, contents, cfg)
	if err != nil {
		return nil, err
	}
	text := strings.TrimSpace(resp.Text())
	if text == "" {
		return nil, nil
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

func buildPrompt(frontPrompt, backPrompt string) string {
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

func buildSchema() map[string]any {
	return map[string]any{
		"type": "array",
		"items": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"page": map[string]any{
					"type":        "string",
					"description": "PDFに写っているページ番号。無ければ空文字",
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

func parseProcessOptions(r *http.Request) (processOptions, error) {
	opts := processOptions{
		APIKey:         strings.TrimSpace(r.FormValue("apiKey")),
		Model:          strings.TrimSpace(r.FormValue("model")),
		Ranges:         strings.TrimSpace(r.FormValue("ranges")),
		FrontPrompt:    strings.TrimSpace(r.FormValue("frontPrompt")),
		BackPrompt:     strings.TrimSpace(r.FormValue("backPrompt")),
		ThinkingBudget: 0,
	}

	if opts.Model == "" {
		opts.Model = defaultModel
	}
	if opts.Ranges == "" {
		return opts, fmt.Errorf("ページ範囲を指定してください")
	}
	if opts.FrontPrompt == "" {
		opts.FrontPrompt = "本文の要点から短い質問を作る"
	}
	if opts.BackPrompt == "" {
		opts.BackPrompt = "質問に対する簡潔な答えを1-3文で"
	}
	if opts.APIKey == "" {
		opts.APIKey = strings.TrimSpace(os.Getenv("GOOGLE_API_KEY"))
	}
	if opts.APIKey == "" {
		opts.APIKey = strings.TrimSpace(os.Getenv("GEMINI_API_KEY"))
	}
	if opts.APIKey == "" {
		return opts, fmt.Errorf("Gemini APIキーが未設定です（フォームまたは環境変数）")
	}

	step, err := parseIntFormValue(r.FormValue("step"), 1)
	if err != nil || step < 1 {
		return opts, fmt.Errorf("step は1以上で指定してください")
	}
	overlap, err := parseIntFormValue(r.FormValue("overlap"), 0)
	if err != nil || overlap < 0 || overlap >= step {
		return opts, fmt.Errorf("overlap は0以上かつ step 未満で指定してください")
	}
	concurrency, err := parseIntFormValue(r.FormValue("concurrency"), 1)
	if err != nil || concurrency < 1 {
		return opts, fmt.Errorf("concurrency は1以上で指定してください")
	}
	delayMS, err := parseIntFormValue(r.FormValue("delayMs"), 0)
	if err != nil || delayMS < 0 {
		return opts, fmt.Errorf("delayMs は0以上で指定してください")
	}
	retries, err := parseIntFormValue(r.FormValue("retries"), 2)
	if err != nil || retries < 0 {
		return opts, fmt.Errorf("retries は0以上で指定してください")
	}
	budget, err := parseIntFormValue(r.FormValue("thinkingBudget"), 0)
	if err != nil || budget < -1 {
		return opts, fmt.Errorf("thinkingBudget は -1 以上で指定してください")
	}
	minConfidence, err := parseFloatFormValue(r.FormValue("minConfidence"), 0.7)
	if err != nil {
		return opts, fmt.Errorf("minConfidence の形式が不正です")
	}
	if minConfidence < 0 {
		minConfidence = 0
	}
	if minConfidence > 1 {
		minConfidence = 1
	}

	opts.Step = step
	opts.Overlap = overlap
	opts.Concurrency = concurrency
	opts.DelayMS = delayMS
	opts.Retries = retries
	opts.ThinkingBudget = budget
	opts.MinConfidence = minConfidence
	return opts, nil
}

func parseIntFormValue(value string, defaultValue int) (int, error) {
	if strings.TrimSpace(value) == "" {
		return defaultValue, nil
	}
	return strconv.Atoi(strings.TrimSpace(value))
}

func parseFloatFormValue(value string, defaultValue float64) (float64, error) {
	if strings.TrimSpace(value) == "" {
		return defaultValue, nil
	}
	return strconv.ParseFloat(strings.TrimSpace(value), 64)
}

func parseRanges(expr string) ([]pageRange, error) {
	parts := strings.Split(expr, ",")
	ranges := make([]pageRange, 0, len(parts))

	for _, raw := range parts {
		part := strings.TrimSpace(raw)
		if part == "" {
			continue
		}
		if strings.Contains(part, "-") {
			sides := strings.SplitN(part, "-", 2)
			if len(sides) != 2 {
				return nil, fmt.Errorf("範囲形式が不正です: %s", part)
			}
			start, err := strconv.Atoi(strings.TrimSpace(sides[0]))
			if err != nil {
				return nil, fmt.Errorf("範囲形式が不正です: %s", part)
			}
			end, err := strconv.Atoi(strings.TrimSpace(sides[1]))
			if err != nil {
				return nil, fmt.Errorf("範囲形式が不正です: %s", part)
			}
			if start < 1 || end < 1 || end < start {
				return nil, fmt.Errorf("範囲指定が不正です: %s", part)
			}
			ranges = append(ranges, pageRange{Start: start, End: end})
			continue
		}

		page, err := strconv.Atoi(part)
		if err != nil || page < 1 {
			return nil, fmt.Errorf("ページ指定が不正です: %s", part)
		}
		ranges = append(ranges, pageRange{Start: page, End: page})
	}

	if len(ranges) == 0 {
		return nil, fmt.Errorf("ページ範囲が空です")
	}

	sort.Slice(ranges, func(i, j int) bool {
		if ranges[i].Start == ranges[j].Start {
			return ranges[i].End < ranges[j].End
		}
		return ranges[i].Start < ranges[j].Start
	})

	merged := make([]pageRange, 0, len(ranges))
	for _, current := range ranges {
		if len(merged) == 0 {
			merged = append(merged, current)
			continue
		}
		last := &merged[len(merged)-1]
		if current.Start <= last.End+1 {
			if current.End > last.End {
				last.End = current.End
			}
			continue
		}
		merged = append(merged, current)
	}
	return merged, nil
}

func iterChunks(ranges []pageRange, step int, overlap int) ([]pageRange, error) {
	if step <= 0 {
		return nil, fmt.Errorf("step は1以上で指定してください")
	}
	if overlap < 0 || overlap >= step {
		return nil, fmt.Errorf("overlap は0以上かつ step 未満で指定してください")
	}

	stride := step - overlap
	out := make([]pageRange, 0)
	for _, r := range ranges {
		start := r.Start
		for start <= r.End {
			end := start + step - 1
			if end > r.End {
				end = r.End
			}
			out = append(out, pageRange{Start: start, End: end})
			if end == r.End {
				break
			}
			start += stride
		}
	}
	return out, nil
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

	validSpans, spanWarnings := clampSpansToPageCount(spans, pageCount)
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
		Range:    span,
		FilePath: chunkPath,
		FileSize: chunkSize,
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

func clampSpansToPageCount(spans []pageRange, pageCount int) ([]pageRange, []string) {
	warnings := make([]string, 0)
	out := make([]pageRange, 0, len(spans))
	for _, span := range spans {
		if span.Start > pageCount {
			warnings = append(warnings, fmt.Sprintf("%s はPDFページ数(%d)外のためスキップしました", span.Label(), pageCount))
			continue
		}
		adjusted := span
		if adjusted.End > pageCount {
			adjusted.End = pageCount
			warnings = append(warnings, fmt.Sprintf("%s はPDFページ数(%d)に合わせて %s に調整しました", span.Label(), pageCount, adjusted.Label()))
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

func chunkDisplayName(displayName string, span pageRange) string {
	base := strings.TrimSpace(displayName)
	if base == "" {
		base = "uploaded"
	}
	ext := strings.ToLower(filepath.Ext(base))
	if ext == ".pdf" {
		base = strings.TrimSuffix(base, filepath.Ext(base))
	}
	return safeDisplayName(fmt.Sprintf("%s-%s.pdf", base, span.Label()))
}

func formatBytes(size int64) string {
	if size < 1024 {
		return fmt.Sprintf("%dB", size)
	}
	if size < 1024*1024 {
		return fmt.Sprintf("%.1fKB", float64(size)/1024.0)
	}
	return fmt.Sprintf("%.2fMB", float64(size)/(1024.0*1024.0))
}

func summarizeMessages(messages []string, limit int) string {
	if len(messages) == 0 {
		return ""
	}
	if limit < 1 {
		limit = len(messages)
	}
	var b strings.Builder
	upper := len(messages)
	if upper > limit {
		upper = limit
	}
	for i := 0; i < upper; i++ {
		b.WriteString("- ")
		b.WriteString(strings.TrimSpace(messages[i]))
		if i != upper-1 {
			b.WriteString("\n")
		}
	}
	if len(messages) > upper {
		b.WriteString("\n")
		b.WriteString(fmt.Sprintf("- ... ほか %d 件", len(messages)-upper))
	}
	return b.String()
}

func formatFileStatus(status *genai.FileStatus) string {
	if status == nil {
		return "unknown file error"
	}
	parts := make([]string, 0, 3)
	if status.Code != nil {
		parts = append(parts, fmt.Sprintf("code=%d", *status.Code))
	}
	if msg := strings.TrimSpace(status.Message); msg != "" {
		parts = append(parts, msg)
	}
	if len(status.Details) > 0 {
		raw, err := json.Marshal(status.Details)
		if err == nil {
			parts = append(parts, fmt.Sprintf("details=%s", string(raw)))
		}
	}
	if len(parts) == 0 {
		return "unknown file error"
	}
	return strings.Join(parts, " | ")
}

func persistUploadedPDF(src io.Reader, filename string) (string, error) {
	ext := strings.ToLower(filepath.Ext(filename))
	if ext == "" {
		ext = ".pdf"
	}
	tmp, err := os.CreateTemp("", "docs2anki-webui-*."+strings.TrimPrefix(ext, "."))
	if err != nil {
		return "", fmt.Errorf("一時ファイル作成に失敗: %w", err)
	}
	defer tmp.Close()

	if _, err := io.Copy(tmp, src); err != nil {
		return "", fmt.Errorf("アップロードファイル保存に失敗: %w", err)
	}
	return tmp.Name(), nil
}

func safeDisplayName(name string) string {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return "uploaded.pdf"
	}
	if len(trimmed) > 180 {
		trimmed = trimmed[:180]
	}
	return trimmed
}

func asString(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	case float64:
		if math.Mod(t, 1) == 0 {
			return strconv.FormatInt(int64(t), 10)
		}
		return strconv.FormatFloat(t, 'f', -1, 64)
	case json.Number:
		return t.String()
	default:
		return fmt.Sprintf("%v", t)
	}
}

func asFloat(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case float32:
		return float64(t)
	case int:
		return float64(t)
	case int64:
		return float64(t)
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(t), 64)
		if err != nil {
			return 0
		}
		return f
	default:
		return 0
	}
}

func contains(list []string, target string) bool {
	for _, v := range list {
		if v == target {
			return true
		}
	}
	return false
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(data)
}

func writeError(w http.ResponseWriter, status int, code string, message string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]string{
			"code":    code,
			"message": message,
		},
	})
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}
