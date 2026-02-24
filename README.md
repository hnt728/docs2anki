# docs2anki

日本語はこちら: [README.ja.md](./README.ja.md)

`docs2anki` converts selected PDF pages into Q&A flashcards with Gemini and lets you review and export the results for Anki.

## Features

- Upload a PDF and generate cards from selected page ranges
- Configure chunking with `ranges`, `step`, and `overlap`
- Preview pages and chunk boundaries before processing
- Track job progress, warnings, and failed chunks
- Review and edit generated cards directly in the browser
- Export `cards.csv` in `Front;Back` format (no header)
- Export `cards.json` with `page`, `question`, `answer`, `confidence`, and `issue`

## Use prebuilt binary (GitHub Releases)

Go is not required when you use release binaries.

1. Download the binary for your OS/architecture from GitHub Releases.
2. Run it.
3. Open `http://localhost:8080`.
4. Enter your Gemini API key in the form, or set `GOOGLE_API_KEY` / `GEMINI_API_KEY`.

## Build from source

```bash
cd webui
go mod tidy
go run ./cmd/docs2anki-webui
```

Source build requirements:

- Go `1.26` or later
- Gemini API key (`GOOGLE_API_KEY` or `GEMINI_API_KEY`, or form input)

## Build a binary

```bash
cd webui
go build -trimpath -ldflags "-s -w" -o dist/docs2anki-webui ./cmd/docs2anki-webui
```

## Cross-platform builds

Use the helper script:

```bash
cd webui
./build-cross.sh
```

Or build manually with `GOOS`/`GOARCH`.

## Server flags

- `-addr` (default: `:8080`): HTTP listen address
- `-max-upload-mb` (default: `300`): max upload size

## Notes

- The UI text is currently in Japanese.
- PDF preview uses `pdf.js` from a CDN, so preview rendering requires network access.
- Processing runs as async jobs (`/api/jobs` and `/api/jobs/{jobId}`).
