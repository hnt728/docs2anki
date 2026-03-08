package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io/fs"
	"log"
	"net/http"
	"strings"
	"time"
)

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
	mux.HandleFunc("/api/jobs/", application.handleJobRequest)
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

	headers, err := parseUploadedSourceHeaders(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, "missing_file", "PDFまたは画像ファイルを指定してください")
		return
	}
	sources, err := persistUploadedSources(headers)
	if err != nil {
		writeError(w, http.StatusBadRequest, "unsupported_file", err.Error())
		return
	}
	opts, err := parseProcessOptions(r)
	if err != nil {
		cleanupUploadedSources(sources)
		writeError(w, http.StatusBadRequest, "invalid_options", err.Error())
		return
	}
	sourceType := "unknown"
	if len(sources) > 0 {
		sourceType = string(sources[0].Kind)
	}

	currentJob := a.jobs.create(jobConfigSummary{
		Provider:                  string(opts.Provider),
		SourceType:                sourceType,
		Model:                     opts.Model,
		Ranges:                    opts.Ranges,
		Step:                      opts.Step,
		Overlap:                   opts.Overlap,
		FrontPrompt:               opts.FrontPrompt,
		BackPrompt:                opts.BackPrompt,
		MinConfidence:             opts.MinConfidence,
		DelayMS:                   opts.DelayMS,
		ThinkingBudget:            opts.ThinkingBudget,
		OpenAIReasoningEffort:     opts.OpenAIReasoningEffort,
		OpenAIImageDetailOriginal: opts.OpenAIImageDetailOriginal,
	})

	go func(j *job, uploaded []uploadedSource, options processOptions) {
		defer cleanupUploadedSources(uploaded)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Hour)
		j.setCancel(cancel)
		defer func() {
			cancel()
			j.clearCancel()
		}()

		j.appendLogf("system", "ジョブ開始: provider=%s model=%s", providerDisplayName(options.Provider), options.Model)

		if err := runJob(ctx, j, uploaded, options); err != nil {
			if errors.Is(err, context.Canceled) && j.isStopRequested() {
				j.markStopped()
				j.appendLog("system", "停止しました。")
				log.Printf("job %s stopped", j.ID)
				return
			}
			j.markFailed(err)
			j.appendLogf("system", "ジョブ失敗: %v", err)
			log.Printf("job %s failed: %v", j.ID, err)
		}
	}(currentJob, sources, opts)

	writeJSON(w, http.StatusAccepted, map[string]string{"jobId": currentJob.ID})
}

func parseJobRequestPath(path string) (id string, action string, ok bool) {
	trimmed := strings.TrimPrefix(path, "/api/jobs/")
	if trimmed == "" {
		return "", "", false
	}
	parts := strings.Split(trimmed, "/")
	if len(parts) == 1 && strings.TrimSpace(parts[0]) != "" {
		return parts[0], "", true
	}
	if len(parts) == 2 && strings.TrimSpace(parts[0]) != "" && parts[1] == "stop" {
		return parts[0], "stop", true
	}
	if len(parts) == 2 && strings.TrimSpace(parts[0]) != "" && parts[1] == "action" {
		return parts[0], "action", true
	}
	return "", "", false
}

func (a *app) handleJobRequest(w http.ResponseWriter, r *http.Request) {
	id, action, ok := parseJobRequestPath(r.URL.Path)
	if !ok {
		writeError(w, http.StatusNotFound, "not_found", "job not found")
		return
	}

	currentJob, exists := a.jobs.get(id)
	if !exists {
		writeError(w, http.StatusNotFound, "not_found", "job not found")
		return
	}

	if action == "stop" {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST only")
			return
		}
		accepted := currentJob.requestStop()
		if accepted {
			currentJob.appendLog("system", "停止要求を受け付けました。")
		}
		statusCode := http.StatusAccepted
		if !accepted {
			statusCode = http.StatusConflict
		}
		writeJSON(w, statusCode, map[string]any{
			"accepted": accepted,
			"job":      currentJob.snapshot(),
		})
		return
	}

	if action == "action" {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "POST only")
			return
		}

		var payload jobActionRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_action", "action payload is invalid")
			return
		}

		decision := chunkDecision(strings.ToLower(strings.TrimSpace(payload.Action)))
		if decision != chunkDecisionRetry && decision != chunkDecisionSkip {
			writeError(w, http.StatusBadRequest, "invalid_action", "action must be retry or skip")
			return
		}
		accepted := currentJob.submitChunkDecision(decision)
		if accepted {
			currentJob.appendLogf("system", "ユーザー操作: %s", decision)
		}
		statusCode := http.StatusAccepted
		if !accepted {
			statusCode = http.StatusConflict
		}
		writeJSON(w, statusCode, map[string]any{
			"accepted": accepted,
			"job":      currentJob.snapshot(),
		})
		return
	}

	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "GET only")
		return
	}
	writeJSON(w, http.StatusOK, currentJob.snapshot())
}
