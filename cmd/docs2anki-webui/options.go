package main

import (
	"fmt"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
)

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

	if opts.Ranges == "" {
		return opts, fmt.Errorf("ページ範囲を指定してください")
	}
	step, err := parseIntFormValue(r.FormValue("step"), 1)
	if err != nil || step < 1 {
		return opts, fmt.Errorf("step は1以上で指定してください")
	}
	overlap, err := parseIntFormValue(r.FormValue("overlap"), 0)
	if err != nil || overlap < 0 || overlap >= step {
		return opts, fmt.Errorf("overlap は0以上かつ step 未満で指定してください")
	}
	opts.Step = step
	opts.Overlap = overlap

	delayMS, err := parseIntFormValue(r.FormValue("delayMs"), 0)
	if err != nil || delayMS < 0 {
		return opts, fmt.Errorf("delayMs は0以上で指定してください")
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

	opts.DelayMS = delayMS
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
