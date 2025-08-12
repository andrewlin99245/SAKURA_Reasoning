set -euo pipefail

PROJECT_NAME="SAKURA_Reasoning"
HOME_PROJ="$HOME/$PROJECT_NAME"
WORK_PROJ="/work/$USER/$PROJECT_NAME"
WORK_VENV="$WORK_PROJ/.venv"

echo "==> 0) Ensure project exists at $HOME_PROJ"
test -d "$HOME_PROJ"

echo "==> 1) Ensure uv is available (install to ~/.local/bin if missing)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Make uv available for future logins too
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

echo "==> 2) Prepare /work project dir and uv cache dir"
mkdir -p "$WORK_PROJ"
export UV_CACHE_DIR="/work/$USER/.uv-cache"
mkdir -p "$UV_CACHE_DIR"

echo "==> 3) Move/Copy lock and metadata to /work"
# Prefer copy; keep originals in repo. (Change to 'mv' if you insist.)
if [ -f "$HOME_PROJ/uv.lock" ]; then
  cp -f "$HOME_PROJ/uv.lock" "$WORK_PROJ/uv.lock"
else
  echo "ERROR: $HOME_PROJ/uv.lock not found. Aborting." >&2
  exit 1
fi

if [ -f "$HOME_PROJ/pyproject.toml" ]; then
  cp -f "$HOME_PROJ/pyproject.toml" "$WORK_PROJ/pyproject.toml"
else
  echo "ERROR: $HOME_PROJ/pyproject.toml not found (needed by uv). Aborting." >&2
  exit 1
fi

echo "==> 4) Create venv in /work and sync from lock"
uv venv "$WORK_VENV"
cd "$WORK_PROJ"
uv sync --frozen

echo "==> 5) Symlink .venv back into the home project"
rm -rf "$HOME_PROJ/.venv" 2>/dev/null || true
ln -s "$WORK_VENV" "$HOME_PROJ/.venv"

echo "==> 6) Quick verification"
source "$HOME_PROJ/.venv/bin/activate"
python -c "import sys; print('prefix:', sys.prefix)"
which uv || true
which python
uv pip list | head -n 20 || true

echo "All set. Your project now uses the venv in: $WORK_VENV"
