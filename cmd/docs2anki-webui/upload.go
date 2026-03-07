package main

import (
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

func parseUploadedSourceHeaders(r *http.Request) ([]*multipart.FileHeader, error) {
	if r.MultipartForm == nil {
		return nil, http.ErrMissingFile
	}
	if files := r.MultipartForm.File["source"]; len(files) > 0 {
		return files, nil
	}
	if files := r.MultipartForm.File["pdf"]; len(files) > 0 {
		return files, nil
	}
	return nil, http.ErrMissingFile
}

func persistUploadedSources(headers []*multipart.FileHeader) ([]uploadedSource, error) {
	if len(headers) == 0 {
		return nil, http.ErrMissingFile
	}
	sources := make([]uploadedSource, 0, len(headers))
	cleanup := func() {
		cleanupUploadedSources(sources)
		sources = nil
	}

	for _, header := range headers {
		file, err := header.Open()
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("アップロードファイルの読み込みに失敗: %w", err)
		}
		tmpPath, err := persistUploadedFile(file, header.Filename)
		_ = file.Close()
		if err != nil {
			cleanup()
			return nil, err
		}
		source, err := buildUploadedSource(tmpPath, header)
		if err != nil {
			_ = os.Remove(tmpPath)
			cleanup()
			return nil, err
		}
		sources = append(sources, source)
	}

	baseKind := sources[0].Kind
	for _, src := range sources[1:] {
		if src.Kind != baseKind {
			cleanup()
			return nil, fmt.Errorf("PDFと画像は同時にアップロードできません")
		}
	}
	if baseKind == sourceKindPDF && len(sources) > 1 {
		cleanup()
		return nil, fmt.Errorf("PDFは1ファイルのみアップロードできます")
	}
	return sources, nil
}

func cleanupUploadedSources(sources []uploadedSource) {
	for _, src := range sources {
		if strings.TrimSpace(src.Path) == "" {
			continue
		}
		_ = os.Remove(src.Path)
	}
}

func buildUploadedSource(path string, header *multipart.FileHeader) (uploadedSource, error) {
	filename := ""
	contentType := ""
	if header != nil {
		filename = strings.TrimSpace(header.Filename)
		contentType = header.Header.Get("Content-Type")
	}
	mimeType, kind, err := detectUploadedSourceType(path, filename, contentType)
	if err != nil {
		return uploadedSource{}, err
	}
	displayName := safeDisplayName(filename)
	if displayName == "uploaded" {
		if ext := extensionForMIME(mimeType); ext != "" {
			displayName += ext
		}
	}
	info, err := os.Stat(path)
	if err != nil {
		return uploadedSource{}, fmt.Errorf("アップロードファイル情報の取得に失敗: %w", err)
	}
	if info.Size() <= 0 {
		return uploadedSource{}, fmt.Errorf("アップロードファイルが空です")
	}
	return uploadedSource{
		Path:        path,
		DisplayName: displayName,
		MIMEType:    mimeType,
		Kind:        kind,
		Size:        info.Size(),
	}, nil
}

func detectUploadedSourceType(path string, filename string, headerContentType string) (string, sourceKind, error) {
	candidates := make([]string, 0, 3)
	if sniffed := detectContentTypeFromFile(path); sniffed != "" {
		candidates = append(candidates, sniffed)
	}
	if fromHeader := normalizeMIME(headerContentType); fromHeader != "" {
		candidates = append(candidates, fromHeader)
	}
	if fromExt, ok := sourceExtToMIME[strings.ToLower(filepath.Ext(filename))]; ok {
		candidates = append(candidates, fromExt)
	}

	seen := make(map[string]struct{}, len(candidates))
	for _, candidate := range candidates {
		candidate = normalizeMIME(candidate)
		if candidate == "" {
			continue
		}
		if _, exists := seen[candidate]; exists {
			continue
		}
		seen[candidate] = struct{}{}
		if kind, ok := allowedSourceMIMEs[candidate]; ok {
			return candidate, kind, nil
		}
	}

	return "", "", fmt.Errorf("対応形式は PDF または画像(PNG/JPEG/WEBP/GIF/BMP/TIFF)です")
}

func detectContentTypeFromFile(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()

	buf := make([]byte, 512)
	n, err := f.Read(buf)
	if err != nil && !errors.Is(err, io.EOF) {
		return ""
	}
	if n <= 0 {
		return ""
	}
	return normalizeMIME(http.DetectContentType(buf[:n]))
}

func normalizeMIME(value string) string {
	trimmed := strings.ToLower(strings.TrimSpace(value))
	if trimmed == "" {
		return ""
	}
	mediaType, _, err := mime.ParseMediaType(trimmed)
	if err != nil {
		return trimmed
	}
	return strings.ToLower(strings.TrimSpace(mediaType))
}

func extensionForMIME(mimeType string) string {
	switch normalizeMIME(mimeType) {
	case "application/pdf":
		return ".pdf"
	case "image/png":
		return ".png"
	case "image/jpeg":
		return ".jpg"
	case "image/webp":
		return ".webp"
	case "image/gif":
		return ".gif"
	case "image/bmp":
		return ".bmp"
	case "image/tiff":
		return ".tiff"
	default:
		return ""
	}
}

func persistUploadedFile(src io.Reader, filename string) (string, error) {
	ext := strings.ToLower(filepath.Ext(filename))
	if ext == "" {
		ext = ".bin"
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
		return "uploaded"
	}
	if len(trimmed) > 180 {
		trimmed = trimmed[:180]
	}
	return trimmed
}

func uploadDisplayNameForAsset(task chunkTask, asset chunkAsset) string {
	name := safeDisplayName(strings.TrimSpace(asset.DisplayName))
	if name == "uploaded" {
		if suffix := extensionForMIME(asset.MIMEType); suffix != "" {
			name += suffix
		}
	}

	if task.Kind == sourceKindPDF {
		base := strings.TrimSuffix(name, filepath.Ext(name))
		return safeDisplayName(fmt.Sprintf("%s-%s.pdf", base, task.Range.Label()))
	}
	if task.Range.Start == task.Range.End {
		return name
	}

	ext := filepath.Ext(name)
	base := strings.TrimSuffix(name, ext)
	if ext == "" {
		ext = extensionForMIME(asset.MIMEType)
	}
	return safeDisplayName(fmt.Sprintf("%s-p%d%s", base, asset.Page, ext))
}
