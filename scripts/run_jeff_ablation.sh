#!/usr/bin/env bash
# The state-amortization ablation: same state, same questions, three isolate
# modes. Restarts jeff for each mode because JEFF_ISOLATE is read at startup.
#
# What it measures: if questions sharing one encoder pass (isolate=none) is much
# cheaper/faster than each question re-encoding the state (isolate=all), then
# packing many questions against one shared state is the mechanism that replaces
# SOLO's prefix-cache reuse. jeff is not Jev, so this establishes the shape of
# the effect, not Jev's numbers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JEFF_DIR="$ROOT/third_party/jeff"
OUT_DIR="${OUT_DIR:-$ROOT/results/jeff_ablation}"
PORT="${PORT:-8000}"
MAX_Q="${MAX_Q:-128}"
mkdir -p "$OUT_DIR"

for mode in none nouls all; do
  echo "== JEFF_ISOLATE=$mode"
  ( cd "$JEFF_DIR" && \
    JEFF_API_KEYS=devkey \
    JEFF_MODEL="${JEFF_MODEL:-models/gliformer-large-v1}" \
    JEFF_ISOLATE="$mode" \
    JEFF_MAX_QUESTIONS=512 \
    JEFF_MAX_STATE_CHARS=200000 \
    JEFF_PORT="$PORT" \
    uv run jeff > "$OUT_DIR/server_$mode.log" 2>&1 & echo $! > "$OUT_DIR/pid_$mode" )

  for _ in $(seq 60); do
    curl -sf "http://localhost:$PORT/v1/models" -H 'Authorization: Bearer devkey' >/dev/null && break
    sleep 2
  done

  TYPESAFE_API_KEY=devkey python3 "$ROOT/probes/probe_parallel.py" \
    --base-url "http://localhost:$PORT" --max-q "$MAX_Q" --reps 3 \
    --out "$OUT_DIR/parallel_$mode.json" | tee "$OUT_DIR/parallel_$mode.txt"

  TYPESAFE_API_KEY=devkey python3 "$ROOT/probes/probe_tokenize.py" \
    --base-url "http://localhost:$PORT" --cols 20 \
    --out "$OUT_DIR/tokenize_$mode.json" | tee "$OUT_DIR/tokenize_$mode.txt"

  kill "$(cat "$OUT_DIR/pid_$mode")" 2>/dev/null || true
  wait 2>/dev/null || true
done

echo "== done; results in $OUT_DIR"
