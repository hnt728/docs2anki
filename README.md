# pdf2anki Web UI (Go)

Python非依存で動く、Gemini API公式Go SDKベースの `pdf2anki` Web UIです。

- ライトモード固定
- スマホ/PCレスポンシブ
- PDF -> カード生成
- ページ範囲/step/overlap連動のPDFプレビュー（折りたたみ可）
- UI上でカード内容の確認・修正
- CSV（Anki用 `Front;Back`）エクスポート

## 1. 前提

- Go 1.26 以上
- Gemini APIキー
  - 環境変数: `GOOGLE_API_KEY` または `GEMINI_API_KEY`
  - もしくは画面のフォーム入力

## 2. 起動

```bash
cd webui
go mod tidy
go run ./cmd/pdf2anki-webui
```

起動後: `http://localhost:8080`

## 3. 配布用バイナリ作成

```bash
cd webui
go build -trimpath -ldflags "-s -w" -o dist/pdf2anki-webui ./cmd/pdf2anki-webui
```

## 4. クロスプラットフォームビルド

```bash
cd webui
mkdir -p dist

GOOS=windows GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o dist/pdf2anki-webui-windows-amd64.exe ./cmd/pdf2anki-webui
GOOS=windows GOARCH=arm64 go build -trimpath -ldflags "-s -w" -o dist/pdf2anki-webui-windows-arm64.exe ./cmd/pdf2anki-webui
GOOS=darwin  GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o dist/pdf2anki-webui-darwin-amd64 ./cmd/pdf2anki-webui
GOOS=darwin  GOARCH=arm64 go build -trimpath -ldflags "-s -w" -o dist/pdf2anki-webui-darwin-arm64 ./cmd/pdf2anki-webui
GOOS=linux   GOARCH=amd64 go build -trimpath -ldflags "-s -w" -o dist/pdf2anki-webui-linux-amd64 ./cmd/pdf2anki-webui
GOOS=linux   GOARCH=arm64 go build -trimpath -ldflags "-s -w" -o dist/pdf2anki-webui-linux-arm64 ./cmd/pdf2anki-webui
```

## 5. 仕様メモ

- ページ範囲指定・`step/overlap` 分割はPython版に合わせたロジックで実装
- PDFはGemini Files APIにアップロードして処理
- 処理後カードはブラウザ内で編集し、編集後データをCSV/JSONでエクスポート可能
- プレビュー描画は `pdf.js`（CDN読込）を使用
