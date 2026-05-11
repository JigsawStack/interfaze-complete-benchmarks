# interfaze-complete-benchmarks

Runner scripts for the public benchmarks Interfaze is evaluated on. Each
benchmark lives in its own directory under `benchmarks/`.

## Setup

```bash
uv sync
uv run playwright install chromium   # required for olmOCR equation rendering
```

Create a `.env` in the repo root with whichever provider keys you plan to use:

```
INTERFAZE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_KEY=...
OPENROUTER_API_KEY=...
JIGSAWSTACK_API_KEY=...   # only for benchmarks/obj_detection/ob_det_api.py
```

Every runner accepts `--limit N` for a smoke test and `--evaluate-only` /
`--predict-only` to split prediction and scoring. Re-running a benchmark
resumes from its checkpoint file — already-completed samples are skipped.

---

## OCRBench v2

Links: [paper](https://arxiv.org/abs/2501.00321) · [repo](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench_v2)

```bash
# Interfaze
uv run -m benchmarks.ocrbench_v2.ocrbench_v2

# Per-provider runners
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_openai
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_openai_mini
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_anthropic
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_gemini
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_gemini_pro_31
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_grok
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_kimi          # via OpenRouter

# Text-spotting EN subset only
uv run -m benchmarks.ocrbench_v2.ocrbench_v2_text_spotting_en

# Evaluate without re-running predictions
uv run -m benchmarks.ocrbench_v2.ocrbench_v2 --evaluate-only
```

---

## olmOCR-Bench

Links: [repo](https://github.com/allenai/olmocr/tree/main/olmocr/bench) · [dataset](https://huggingface.co/datasets/allenai/olmOCR-bench)

```bash
# Interfaze
uv run -m benchmarks.olmocr.olmocr_bench

# Per-provider runners
uv run -m benchmarks.olmocr.olmocr_bench_openai_mini
uv run -m benchmarks.olmocr.olmocr_bench_gemini_pro_31
uv run -m benchmarks.olmocr.olmocr_bench_grok

# Useful flags
uv run -m benchmarks.olmocr.olmocr_bench --sample           # tiny sample dataset
uv run -m benchmarks.olmocr.olmocr_bench --generate-only    # predictions only
uv run -m benchmarks.olmocr.olmocr_bench --skip-generation  # evaluation only
```

---

## RefCOCO (Object Detection)

Links: [RefCOCO/RefCOCO+ paper](https://arxiv.org/abs/1608.00272) · [RefCOCOg paper](https://arxiv.org/abs/1511.02283) · [dataset](https://huggingface.co/datasets/lmms-lab/RefCOCO)

Metric: Acc@IoU=0.5 on the referring-expression bounding box.

```bash
# Interfaze (RefCOCO val by default)
uv run -m benchmarks.obj_detection.refcoco
uv run -m benchmarks.obj_detection.refcoco --split testA
uv run -m benchmarks.obj_detection.refcoco --dataset lmms-lab/RefCOCO+ --split testB

# Any provider via the multi runner
uv run -m benchmarks.obj_detection.refcoco_multi --provider openai    --model gpt-5.4
uv run -m benchmarks.obj_detection.refcoco_multi --provider anthropic --model claude-sonnet-4-6
uv run -m benchmarks.obj_detection.refcoco_multi --provider gemini    --model gemini-3-flash-preview

# JigsawStack object_detection API (instead of a VLM)
uv run -m benchmarks.obj_detection.ob_det_api

# Evaluate only
uv run -m benchmarks.obj_detection.refcoco --evaluate-only
```

---

## VoxPopuli-Cleaned-AA (ASR)

Links: [dataset](https://huggingface.co/datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA)

Metric: WER with Whisper-style text normalization.

```bash
# Interfaze
uv run -m benchmarks.asr.voxpopuli_aa

# Other providers (audio-capable)
uv run -m benchmarks.asr.voxpopuli_aa_multi --provider gemini    --model gemini-3-flash-preview
uv run -m benchmarks.asr.voxpopuli_aa_multi --provider openai    --model gpt-4o-audio-preview
uv run -m benchmarks.asr.voxpopuli_aa_multi --provider anthropic --model claude-sonnet-4-6

# Evaluate only
uv run -m benchmarks.asr.voxpopuli_aa --evaluate-only
```

---

## MMMLU (Multilingual MMLU)

Links: [dataset](https://huggingface.co/datasets/openai/MMMLU)

14 languages, exact-match accuracy macro-averaged across languages.

```bash
# Interfaze
uv run -m benchmarks.mmmlu.mmmlu
uv run -m benchmarks.mmmlu.mmmlu --languages DE_DE FR_FR     # subset of languages

# Any provider
uv run -m benchmarks.mmmlu.mmmlu_multi --provider openai    --model gpt-5.4-mini
uv run -m benchmarks.mmmlu.mmmlu_multi --provider gemini    --model gemini-3.1-pro-preview
uv run -m benchmarks.mmmlu.mmmlu_multi --provider anthropic --model claude-sonnet-4-6
uv run -m benchmarks.mmmlu.mmmlu_multi --provider interfaze --model interfaze-beta

# Evaluate only
uv run -m benchmarks.mmmlu.mmmlu --evaluate-only
```

---

## MMMU-Pro

Links: [paper](https://arxiv.org/abs/2409.02813) · [dataset](https://huggingface.co/datasets/MMMU/MMMU_Pro)

Two settings: `standard` (text + inline images) and `vision` (rendered question image).

```bash
# Any provider, standard or vision
uv run -m benchmarks.mmmu_pro.mmmu_pro_multi --provider gemini    --model gemini-3.1-pro-preview --setting standard
uv run -m benchmarks.mmmu_pro.mmmu_pro_multi --provider gemini    --model gemini-3.1-pro-preview --setting vision
uv run -m benchmarks.mmmu_pro.mmmu_pro_multi --provider openai    --model gpt-5.5               --setting standard
uv run -m benchmarks.mmmu_pro.mmmu_pro_multi --provider anthropic --model claude-sonnet-4-6     --setting vision
uv run -m benchmarks.mmmu_pro.mmmu_pro_multi --provider interfaze --model interfaze-beta        --setting standard

# Run on Modal instead of locally
bash benchmarks/mmmu_pro/run_full.sh
bash benchmarks/mmmu_pro/run_smoke.sh
```

---

## GPQA Diamond

Links: [paper](https://arxiv.org/abs/2311.12022) · [dataset](https://huggingface.co/datasets/Idavidrein/gpqa) (config: `gpqa_diamond`)

```bash
# OpenAI
uv run -m benchmarks.gpqa.gpqa_openai
uv run -m benchmarks.gpqa.gpqa_openai --model gpt-5.4-mini

# Gemini
uv run -m benchmarks.gpqa.gpqa_gemini
uv run -m benchmarks.gpqa.gpqa_gemini --model gemini-3.1-pro-preview

# Any model via OpenRouter (Grok, Kimi, Anthropic, etc.)
uv run -m benchmarks.gpqa.gpqa_openrouter --model x-ai/grok-4.3 --thinking on
uv run -m benchmarks.gpqa.gpqa_openrouter --model moonshotai/kimi-k2.6

# Evaluate only
uv run -m benchmarks.gpqa.gpqa_openai --evaluate-only
```

---

## Spider 2.0-Lite (SQLite subset, N=135)

Links: [repo](https://github.com/xlang-ai/Spider2) · [paper](https://arxiv.org/abs/2411.07763)

Text-to-SQL with execution-accuracy scoring against per-example SQLite databases.

```bash
# One-time setup: clone Spider2 + download SQLite databases
uv run -m benchmarks.spider2_lite.fetch_data

# Run
uv run -m benchmarks.spider2_lite.spider2_lite
uv run -m benchmarks.spider2_lite.spider2_lite --predict-only
uv run -m benchmarks.spider2_lite.spider2_lite --evaluate-only
```
