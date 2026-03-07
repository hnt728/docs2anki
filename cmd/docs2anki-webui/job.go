package main

import (
	"context"
	crand "crypto/rand"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

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
		decisionQ: make(chan chunkDecision, 1),
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
	if j.StopRequested {
		j.Status = "stopping"
	} else {
		j.Status = "running"
	}
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
	j.StopRequested = false
	j.PendingChunk = ""
	j.PendingError = ""
	j.activeSet = make(map[string]struct{})
	j.UpdatedAt = time.Now()
	j.mu.Unlock()
}

func (j *job) markStopped() {
	j.mu.Lock()
	j.Status = "stopped"
	j.Error = ""
	j.ActiveChunks = nil
	j.PendingChunk = ""
	j.PendingError = ""
	j.activeSet = make(map[string]struct{})
	j.UpdatedAt = time.Now()
	j.mu.Unlock()
}

func (j *job) markFailed(err error) {
	j.mu.Lock()
	j.Status = "failed"
	j.Error = err.Error()
	j.ActiveChunks = nil
	j.PendingChunk = ""
	j.PendingError = ""
	j.activeSet = make(map[string]struct{})
	j.UpdatedAt = time.Now()
	j.mu.Unlock()
}

func (j *job) setCancel(cancel context.CancelFunc) {
	var shouldCancel bool

	j.mu.Lock()
	j.cancelFn = cancel
	if j.StopRequested && cancel != nil {
		shouldCancel = true
	}
	j.mu.Unlock()

	if shouldCancel {
		cancel()
	}
}

func (j *job) clearCancel() {
	j.mu.Lock()
	j.cancelFn = nil
	j.mu.Unlock()
}

func (j *job) isStopRequested() bool {
	j.mu.RLock()
	defer j.mu.RUnlock()
	return j.StopRequested
}

func (j *job) requestStop() bool {
	var cancel context.CancelFunc

	j.mu.Lock()
	switch j.Status {
	case "completed", "failed", "stopped":
		j.mu.Unlock()
		return false
	}
	j.StopRequested = true
	if j.Status == "queued" || j.Status == "running" || j.Status == "paused" {
		j.Status = "stopping"
	}
	cancel = j.cancelFn
	j.UpdatedAt = time.Now()
	j.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	return true
}

func (j *job) pauseForChunkIssue(label string, err error) {
	message := strings.TrimSpace(err.Error())
	if message == "" {
		message = "unknown error"
	}
	j.clearDecisionQueue()

	j.mu.Lock()
	if j.StopRequested {
		j.Status = "stopping"
	} else {
		j.Status = "paused"
	}
	j.PendingChunk = strings.TrimSpace(label)
	j.PendingError = message
	j.UpdatedAt = time.Now()
	j.mu.Unlock()
}

func (j *job) resumeFromChunkIssue() {
	j.mu.Lock()
	j.PendingChunk = ""
	j.PendingError = ""
	if j.StopRequested {
		j.Status = "stopping"
	} else {
		j.Status = "running"
	}
	j.UpdatedAt = time.Now()
	j.mu.Unlock()
	j.clearDecisionQueue()
}

func (j *job) submitChunkDecision(decision chunkDecision) bool {
	if decision != chunkDecisionRetry && decision != chunkDecisionSkip {
		return false
	}

	j.mu.RLock()
	paused := j.Status == "paused"
	pendingChunk := strings.TrimSpace(j.PendingChunk)
	j.mu.RUnlock()
	if !paused || pendingChunk == "" {
		return false
	}

	select {
	case j.decisionQ <- decision:
		return true
	default:
		select {
		case <-j.decisionQ:
		default:
		}
		select {
		case j.decisionQ <- decision:
			return true
		default:
			return false
		}
	}
}

func (j *job) waitChunkDecision(ctx context.Context) (chunkDecision, error) {
	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case decision := <-j.decisionQ:
			if decision == chunkDecisionRetry || decision == chunkDecisionSkip {
				return decision, nil
			}
		}
	}
}

func (j *job) clearDecisionQueue() {
	for {
		select {
		case <-j.decisionQ:
		default:
			return
		}
	}
}

func (j *job) appendLog(chunk string, message string) {
	msg := strings.TrimSpace(strings.ReplaceAll(message, "\r", ""))
	if msg == "" {
		return
	}
	if len(msg) > maxJobLogMessageBytes {
		msg = strings.TrimSpace(msg[:maxJobLogMessageBytes]) + " ..."
	}
	entryTime := time.Now()

	j.mu.Lock()
	j.nextLogID++
	j.Logs = append(j.Logs, jobLogEntry{
		Seq:     j.nextLogID,
		Time:    entryTime.Format("15:04:05"),
		Chunk:   strings.TrimSpace(chunk),
		Message: msg,
	})
	if len(j.Logs) > maxJobLogEntries {
		j.Logs = append([]jobLogEntry{}, j.Logs[len(j.Logs)-maxJobLogEntries:]...)
	}
	j.UpdatedAt = entryTime
	j.mu.Unlock()
}

func (j *job) appendLogf(chunk string, format string, args ...any) {
	j.appendLog(chunk, fmt.Sprintf(format, args...))
}

func (j *job) snapshot() job {
	j.mu.RLock()
	defer j.mu.RUnlock()
	cp := *j
	cp.ActiveChunks = append([]string{}, j.ActiveChunks...)
	cp.FailedChunks = append([]string{}, j.FailedChunks...)
	cp.Warnings = append([]string{}, j.Warnings...)
	cp.Cards = append([]Card{}, j.Cards...)
	cp.Logs = append([]jobLogEntry{}, j.Logs...)
	cp.activeSet = nil
	cp.cancelFn = nil
	cp.nextLogID = 0
	cp.decisionQ = nil
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
