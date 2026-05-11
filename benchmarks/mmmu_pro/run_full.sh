#!/usr/bin/env bash
# Launch full MMMU-Pro runs (1730 samples per setting) for all 8 models,
# both settings, as detached Modal jobs that survive Ctrl-C. Use the
# `check` entrypoint to monitor progress without disturbing the runners.
#
# Usage: ./benchmarks/mmmu_pro/run_full.sh
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
    echo "launching: $provider / $model  setting=$setting"
    uv run modal run --detach benchmarks/mmmu_pro/modal_app.py::run \
      --provider "$provider" --model "$model" --setting "$setting" &
  done
done

wait
echo "all runs launched detached. Check progress with:"
echo "  uv run modal run benchmarks/mmmu_pro/modal_app.py::check --provider <p> --model <m> --setting <s>"
