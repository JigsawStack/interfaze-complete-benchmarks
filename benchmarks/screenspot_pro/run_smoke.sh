#!/usr/bin/env bash
# Smoke-test all 8 models on 50 ScreenSpot-Pro samples each, sequentially.
# Sequential warms the shared HF cache volume on the first model, then each
# subsequent model reuses cached images.
#
# Usage: ./benchmarks/screenspot_pro/run_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

models=(
  "interfaze interfaze-beta"
  "gemini    gemini-3-flash-preview"
  "gemini    gemini-2.5-pro"
  "openai    gpt-5.4-mini"
  "anthropic claude-sonnet-4-6"
  "openai    gpt-5.4"
  "openai    gpt-5.5"
  "gemini    gemini-3.1-pro-preview"
)

for line in "${models[@]}"; do
  read -r provider model <<<"$line"
  echo
  echo "==================================================================="
  echo "smoke: $provider / $model  (limit=50, reasoning=off, temp=0)"
  echo "==================================================================="
  uv run modal run benchmarks/screenspot_pro/modal_app.py::run \
    --provider "$provider" --model "$model" --limit 50
done
