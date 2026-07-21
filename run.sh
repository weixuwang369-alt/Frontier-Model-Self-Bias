#!/usr/bin/env bash
# One-command launcher for the SelfBias project.
#
#   ./run.sh            # sync deps, then open the dashboard (default)
#   ./run.sh demo       # run the full pipeline offline on the mock provider ($0)
#   ./run.sh estimate   # dry-run cost estimate for the example config (no API calls)
#   ./run.sh check      # test keys for the example config (one tiny call per model)
#   ./run.sh status     # show status of the mock run
#   ./run.sh test       # run the test suite
#
# Finds `uv` even when it isn't on your PATH, and installs deps on first run.

set -euo pipefail
cd "$(dirname "$0")"

# Make sure uv is reachable (it may be installed but not on PATH).
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/Library/Python/3.14/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' not found. Install it with:  python3 -m pip install --user uv" >&2
  exit 1
fi

echo "› syncing dependencies…"
uv sync --quiet

cmd="${1:-dashboard}"
case "$cmd" in
  dashboard|ui|"")
    echo "› launching dashboard at http://localhost:8501  (Ctrl+C to stop)"
    exec uv run streamlit run dashboard/app.py
    ;;
  demo|run)
    exec uv run selfbias run config/experiment.mock.yaml --yes
    ;;
  estimate)
    exec uv run selfbias estimate config/experiment.example.yaml
    ;;
  check)
    exec uv run selfbias check config/experiment.example.yaml
    ;;
  status)
    exec uv run selfbias status config/experiment.mock.yaml
    ;;
  analyze)
    exec uv run selfbias analyze config/experiment.mock.yaml
    ;;
  test)
    exec uv run pytest
    ;;
  *)
    echo "usage: ./run.sh [dashboard|demo|estimate|check|status|analyze|test]" >&2
    exit 1
    ;;
esac
