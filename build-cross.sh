#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p dist

build() {
  local os="$1"
  local arch="$2"
  local out="$3"
  echo "building ${out}"
  GOOS="${os}" GOARCH="${arch}" go build -trimpath -ldflags "-s -w" -o "${out}" ./cmd/pdf2anki-webui
}

build windows amd64 dist/pdf2anki-webui-windows-amd64.exe
build windows arm64 dist/pdf2anki-webui-windows-arm64.exe
build darwin amd64 dist/pdf2anki-webui-darwin-amd64
build darwin arm64 dist/pdf2anki-webui-darwin-arm64
build linux amd64 dist/pdf2anki-webui-linux-amd64
build linux arm64 dist/pdf2anki-webui-linux-arm64

echo "done"
