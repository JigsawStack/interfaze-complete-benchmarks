#!/usr/bin/env bash
# Smoke-test all 8 models on 50 MMMU-Pro samples each, both settings,
# sequentially. Sequential warms the shared HF cache volume on the first
# run, then each subsequent run reuses cached dataset bytes.
#
# Usage: ./benchmarks/mmmu_pro/run_smoke.sh
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

settings=(standard vision)

for setting in "${settings[@]}"; do
  for line in "${models[@]}"; do
    read -r provider model <<<"$line"
    echo
    echo "==================================================================="
    echo "smoke: $provider / $model  setting=$setting  (limit=50, reasoning=off)"
    echo "==================================================================="
    uv run modal run benchmarks/mmmu_pro/modal_app.py::run \
      --provider "$provider" --model "$model" --setting "$setting" --limit 50
  done
done
