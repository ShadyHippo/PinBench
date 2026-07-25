# PinBench: Pinyin-to-Character Conversion Benchmark

*An instructions-and-parsing intelligence benchmark disguised as a translation benchmark.*

- **Pin** as in **Pinyin** — the input the benchmark feeds models
- **Pin** as in **to pin** — the model must pin each syllable to the correct character, pin the output format, and pin the right breakdown, all while following strict formatting rules

A benchmark for evaluating language models on **structured Pinyin-to-Character conversion** — testing whether models can follow a strict output format while using context (including English clues) to infer the intended Chinese when pinyin is slightly wrong.

## Quick Start

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install openai pyyaml

# Run quick mock test (no API keys needed)
python3 run_benchmark.py --mock --runs 1

# Run against a model (set OPENCODE_API_KEY first)
export OPENCODE_API_KEY="your-key"
python3 run_benchmark.py -c nemotron_config.yaml
```

### Running a Multi-Run Benchmark

```bash
# 5 runs per model, aggregated with stats
python3 multi_run.py --configs nemotron_config.yaml deepseek_free_config.yaml --runs 5 --output results/my_bench

# Re-aggregate existing runs (e.g. after grader changes)
python3 multi_run.py --aggregate results/my_bench
```

| Flag | What it does |
|------|-------------|
| `-c config.yaml` | Config file to use |
| `-r N` | Override runs per model |
| `-w N` | Override max workers (default: 3) |
| `--filter-test 1 2 3` | Only run specific test IDs |
| `--mock` | Run mock tests (no API key needed) |
| `--no-progress` | Suppress per-test progress output |

## System Prompt

The system prompt lives in `system_prompt.txt` (NOT in `test_cases.json`). It casts the model as an **Expert Mandarin Chinese Tutor** helping a beginner student:

- Use English context to disambiguate pinyin, but **never translate English words**
- **Group compound words** as single entries (明天, not 明 + 天)
- View the phrase **holistically** — correct misspelled or incorrect pinyin using context
- Output format: Phonetic Sandbox → Table (one row for the whole phrase) → Word-by-word breakdown

The prompt is designed to be stable — we change tests and grading to match it, not the other way around.

## Test Cases

**46 tests** (v1.3) organized into sections:

### Clean Tests (26 tests)
Standard Pinyin-to-Character conversion with correct pinyin:
- **Basic commands** — "Sue told Sara qu xi shou ba" → 去洗手吧
- **English name filtering** — "John told Mary ni hao ma" → 你好吗
- **Mixed English/Pinyin** — "Grandma said wo yao chi fan le" → 我要吃饭了
- **Grammar particles** — ba (吧), ma (吗), ne (呢), le (了), guo (过), de (的)
- **Complex structures** — bei (被, passive), rang (让, causative), bi (比, comparison)
- **Third-person ta** — 他/she/it ambiguity resolution
- **Pure Pinyin** — "wo ai ni" → 我爱你
- **And more** — measure words, complements, questions, time expressions

### Corrected Pinyin / Dirty Tests (10 pairs = 20 tests)
Each dirty test has one deliberate pinyin error that a human Chinese learner would understand. Each has a clean counterpart:

| Test | Prompt (dirty) | Expected | Confusion | Difficulty |
|------|----------------|----------|-----------|------------|
| 33 | chi zoa fan | 吃早饭 | transposition | Easy |
| 35 | shie shie ni | 谢谢你 | invalid pinyin (shie→xie) | Easy |
| 45 | zhir dao le | 知道了 | extra 'r' (invalid) | Easy |
| 46 | zao shong hao | 早上好 | ong/ang | Medium |
| 27 | shou xi lai | 收起来 | x/q | Medium |
| 31 | zao xiang hao | 早上好 | x/sh | Medium |
| 37 | chu shang xue | 去上学 | c/q | Hard |
| 29 | chu shui jiao | 去睡觉 | c/q | Hard |
| 39 | qu shang que | 去上学 | q/x | Hard |
| 43 | shi wan le (doing dishes) | 洗完了 | context-dependent | Hardest |

### Compound Words
The prompt requires standard Chinese compounds to be output as single entries in the word-by-word section:
- 明天 NOT 明 + 天
- 我们 NOT 我 + 们
- 什么 NOT 什 + 么
- 知道 NOT 知 + 道
- 12 compounds total across the test suite

## Grading System

Grading checks 5 criteria on the model's output. **Two are required** for a test to pass:

| Criterion | Weight | Required | What it checks |
|-----------|--------|----------|----------------|
| `phonetic_sandbox` | 0.10 | — | `[Phonetic Sandbox]` section exists |
| `table_exists` | 0.10 | — | Table with `Chinese \| Pinyin \| English` headers |
| `table_content` | 0.40 | **✓** | Table row has correct Chinese, pinyin, and English |
| `literal_section` | 0.10 | — | Word-by-word breakdown section exists |
| `literal_words` | 0.30 | **✓** | Each expected word has Chinese + pinyin + keyword match |

**Pass** = `table_content` AND `literal_words` both pass.
**Score** = weighted sum of all 5 criteria (capped at 1.0).

The grader supports:
- **Compound word fallback** — accepts compound entry (明天) OR individual characters (明 + 天)
- **English fuzzy matching** — Jaccard similarity + bidirectional substring for English column
- **Pronoun normalization** — 她/牠 → 他, 妳/您 → 你

## Configuration

Create a config file to run models via the OpenCode Zen API (or any OpenAI-compatible endpoint):

```yaml
name: "my_benchmark"
test_file: "test_cases.json"

providers:
  - type: "openai_compatible"
    model: "nemotron-3-ultra-free"
    base_url: "https://opencode.ai/zen/v1"
    api_key: "${OPENCODE_API_KEY}"   # resolves from env var
    temperature: 0.0
    max_tokens: 4096
    timeout: 120

runs_per_model: 1
max_workers: 3
output_dir: "results"
save_raw_responses: true
save_results_json: true
```

### Provider Types

| Type | Description |
|------|-------------|
| `openai_compatible` | Any OpenAI-compatible API (OpenCode, vLLM, llama.cpp, etc.) |
| `vllm` | vLLM server |
| `mock` | Test without API (for development) |

API keys and base URLs can be set in the config directly or via environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`). The config supports `${VAR_NAME}` substitution for env vars.

## Output

### Single run: `results/<timestamp>/`
| File | Description |
|------|-------------|
| `summary.json` | Aggregated stats by model, provider, category, test |
| `results.json` | Detailed grading for each response |
| `results.csv` | Tabular data for analysis |
| `raw_responses.json` | Full model outputs + grading per test |
| `config.yaml` | Run configuration snapshot |

### Multi-run: `results/<name>/`
```
results/my_bench/
├── _index.json                   # Run index
├── aggregated.json               # Cross-run aggregation + test data + system prompt
├── aggregated_<timestamp>.json   # Timestamped version for history
├── aggregated.txt                # Text summary
├── nemotron/run_001/             # Per-run results (same structure as single run)
│   ├── raw_responses.json
│   ├── summary.json
│   └── run.log
├── nemotron/run_002/
└── ...
```

### Regrading old runs
If the grader or test data changes, regrade old raw outputs without re-running models:
```bash
python3 regrade.py results/v1.2_regraded results/v1.2_regraded
```

## Architecture

```
PinBench/
├── system_prompt.txt       # The tutor system prompt (source of truth)
├── test_cases.json         # 46 tests with expected table + literal entries
├── grader.py               # Grading: 5 criteria, compound fallback, fuzzy matching
├── providers.py            # Model provider: openai_compatible, vllm, mock
├── runner.py               # TestCase, BenchmarkConfig, RunConfig dataclasses
├── run_benchmark.py        # CLI entry point (single run)
├── multi_run.py            # Orchestrator (N runs, aggregation, archival)
├── regrade.py              # Regrade old raw outputs against current grader
├── plan.md                 # Project plan and current results
└── *_config.yaml           # Per-model config files
```

## Requirements

- Python 3.10+
- `openai` (for API providers)
- `pyyaml` (for config files)

No GPU required — all evaluation is done via API calls.

## Current Results (v2.0 — Tutor Prompt)

| Model | Runs | Pass Rate | Avg Score | Notes |
|---|---|---|---|---|
| Nemotron 3 Ultra Free | 5 | **81.9%** | 91.0% | Best overall |
| DeepSeek V4 Flash Free | 5 | **76.1%** | 88.1% | Close second |
| Ling 3.0 Flash Free | — | N/A | N/A | Free tier exhausted |
| Laguna S 2.1 Free | — | N/A | N/A | Free tier exhausted |
| North Mini Code Free | — | 0% | 0% | Unusable for this task |

See `plan.md` for full breakdown and comparison with earlier prompt versions.
