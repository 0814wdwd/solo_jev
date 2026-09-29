#!/usr/bin/env bash
# Set up jeff, the MIT-licensed self-hosted stand-in for Jev, so the probes can
# run with no API key.
#
# WHERE TO RUN THIS: not on the 210 code machine. /home/ubuntu/lyz/资源总览.md
# says that box is for code and infra only, and its disk is already full. Run it
# on xtra3090 or the DCU node, then point the probes at it with
#   --base-url http://<host>:8000
#
# Needs ~2 GB free for the GLiFormer-large-v1 weights (400M params) plus deps.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party/jeff}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/models/gliformer-large-v1}"

echo "== jeff repo: $REPO_DIR"
[ -d "$REPO_DIR" ] || git clone --depth 1 https://github.com/logan-markewich/jeff.git "$REPO_DIR"

command -v uv >/dev/null || { echo "== installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

cd "$REPO_DIR"
echo "== uv sync (installs torch + typesafe-sdk; several GB)"
uv sync --extra dev

if [ ! -d "$MODEL_DIR" ]; then
  echo "== downloading knowledgator/gliformer-large-v1 (~1.6 GB)"
  # Behind the GFW use the mirror: export HF_ENDPOINT=https://hf-mirror.com
  uv run hf download knowledgator/gliformer-large-v1 --local-dir "$MODEL_DIR"
fi

cat <<'NOTE'

== start the server (choose ONE isolate mode per run; see run_jeff_ablation.sh)

  JEFF_API_KEYS=devkey \
  JEFF_MODEL=models/gliformer-large-v1 \
  JEFF_ISOLATE=none \
  JEFF_MAX_QUESTIONS=512 \
  JEFF_MAX_STATE_CHARS=200000 \
  uv run jeff

JEFF_MAX_QUESTIONS defaults to 64 and JEFF_MAX_STATE_CHARS to 20000, both far
below Jev's documented 64k/32k token budgets. Raise them as above or jeff will
reject exactly the large blocks this project is about.

Knobs that matter here:
  JEFF_ISOLATE=none|nouls|all   whether questions share one encoder pass over
                                the state. `none` = maximum state amortization,
                                `all` = each question re-encodes the state.
                                This is the state-sharing ablation.
  JEFF_STATE_FORMAT=kv|json|values   jeff's own state rendering
  JEFF_MAX_BATCH / JEFF_MAX_WAIT_MS  server-side request batching
  JEFF_QUANT=fp32|int8, JEFF_THREADS  CPU arm (JEFF_BACKEND=onnx)
  JEFF_TEMPERATURE=3.2          probability calibration scaling

Then, from the solo_jev root:
  python3 probes/probe_parallel.py --base-url http://localhost:8000 --max-q 256
NOTE
