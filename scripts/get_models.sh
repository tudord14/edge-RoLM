#!/usr/bin/env bash
# Download GGUF model files from a GitHub Release
# Usage:
#   bash scripts/get_models.sh [OUTDIR]
# Environment overrides:
#   REPO=tudord14/edge-RoLM   # owner/repo
#   TAG=v1.0                  # release tag

set -euo pipefail

REPO="${REPO:-tudord14/edge-RoLM}"
TAG="${TAG:-v1.0}"
OUTDIR="${1:-models}"

# ---- list your model filenames here ----
FILES=(
  "Model-125M-f16.gguf"
  "Model-125M-Q5_K_M.gguf"
  "Model-350M-f16.gguf"
  "Model-350M-Q5_K_M.gguf"
)
# ----------------------------------------

mkdir -p "$OUTDIR"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

download() {
  local fname="$1"
  local url="https://github.com/${REPO}/releases/download/${TAG}/${fname}"
  local outpath="${OUTDIR}/${fname}"

  echo "-> ${fname}"
  if have_cmd curl; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$outpath" "$url"
  elif have_cmd wget; then
    wget -O "$outpath" "$url"
  else
    echo "Error: need curl or wget installed." >&2
    exit 1
  fi
}

echo "Repo: $REPO"
echo "Tag : $TAG"
echo "Out : $OUTDIR"
echo "Files:"
printf '  - %s\n' "${FILES[@]}"
echo

for f in "${FILES[@]}"; do
  download "$f"
done

echo
echo "All done. Saved to: $OUTDIR"
ls -lh "$OUTDIR"
