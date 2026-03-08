package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"google.golang.org/genai"
)

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

func providerDisplayName(provider providerKind) string {
	switch provider {
	case providerOpenAI:
		return "OpenAI"
	default:
		return "Gemini"
	}
}

func isGPT54Model(model string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(model)), "gpt-5.4")
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
