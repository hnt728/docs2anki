# docs2anki

English README: [README.md](./README.md)

`docs2anki` は、PDFページまたは画像をGeminiで一問一答カード化し、Anki向けにエクスポートできるツールです。

## 主な機能

- 1つのPDFまたは複数画像をアップロードしてカードを生成
- `ranges` / `step` / `overlap` でチャンク分割を調整
- 実行前にPDF/画像ともページとチャンク境界をプレビュー
- 進捗、警告、失敗チャンクをUI上で確認
- 生成後のカードをブラウザ上で編集
- `Front;Back`（ヘッダなし）の `cards.csv` をエクスポート
- `page` / `question` / `answer` / `confidence` / `issue` を含む `cards.json` をエクスポート

## GitHub Releases のバイナリを使う

Releaseバイナリを使う場合、Goは不要です。

1. GitHub ReleasesからOS/アーキテクチャに合ったバイナリをダウンロード
2. バイナリを実行
3. `http://localhost:8080` を開く
4. Gemini APIキーをフォームに入力、または `GOOGLE_API_KEY` / `GEMINI_API_KEY` を設定

## ソースから起動

```bash
cd docs2anki
go mod tidy
go run ./cmd/docs2anki-webui
```

ソース実行時の前提:

- Go `1.26` 以上
- Gemini APIキー（`GOOGLE_API_KEY` または `GEMINI_API_KEY`、またはフォーム入力）

## ビルド

```bash
cd docs2anki
go build -trimpath -ldflags "-s -w" -o dist/docs2anki-webui ./cmd/docs2anki-webui
```

## クロスプラットフォームビルド

補助スクリプトを使う場合:

```bash
cd docs2anki
./build-cross.sh
```

または `GOOS` / `GOARCH` を指定して手動でビルドできます。

## サーバーフラグ

- `-addr`（デフォルト: `:8080`）: HTTP待受アドレス
- `-max-upload-mb`（デフォルト: `300`）: アップロード上限サイズ

## 補足

- UI文言は現在日本語です。
- PDFプレビューはCDN経由の `pdf.js` を使うため、PDFプレビュー表示にはネットワーク接続が必要です。
- 画像入力では各画像が1ページとして扱われ、`ranges` / `step` / `overlap` でチャンク分割されます。
- 処理は非同期ジョブとして実行されます（`/api/jobs`, `/api/jobs/{jobId}`）。
