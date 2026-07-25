# PinBench: Pinyin-to-Character Conversion Benchmark

*An instructions-and-parsing intelligence benchmark disguised as a translation benchmark.*

- **Pin** as in **Pinyin** — the input the benchmark feeds models
- **Pin** as in **to pin** — the model must pin each syllable to the correct character, pin the output format, and pin the right breakdown, all while following strict formatting rules

A benchmark framework for testing small language models (15B and below) on their ability to follow a strict Pinyin-to-Character conversion system prompt. The task requires models to:

1. **Extract only Pinyin syllables** from input (ignoring ALL English words like names, "said", "told", etc.)
2. **Map each syllable to its most literal Chinese character** (no omissions, no corrections unless absolutely necessary)
3. **Output in a strict format** with a table and tutor's notes

**What this benchmark actually tests:**

You are not just testing: *"Can the model convert pinyin to Chinese?"*

You are testing: *"Can the model infer the user's intended Chinese while respecting the user's input?"*

This is a harder problem. A model that maps every pinyin syllable literally will fail on intentionally slightly-off pinyin. A model that over-corrects will change things that were already right. A good model balances both — it uses phrase-level context to resolve ambiguity but doesn't rewrite valid input.

## Quick Start

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run quick mock test (no API keys needed)
python3 run_benchmark.py --mock --runs 1 --workers 2

# Run with Featherless.ai (set FEATHERLESS_API_KEY first)
export FEATHERLESS_API_KEY="your-key"
python3 run_benchmark.py -c benchmark_config.yaml

# Run with Together.ai
export TOGETHER_API_KEY="your-key"
python3 run_benchmark.py -c benchmark_config.yaml

# Run with any OpenAI-compatible API
# (e.g. OpenCode, vLLM, llama.cpp, etc.)
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.opencode.ai/v1"
python3 run_benchmark.py -c benchmark_config.yaml
```

> **Tip:** You can also pass API keys and base URLs directly in the config file
> (see [Provider Configuration](#provider-configuration) below).

### Running a Benchmark

Create a config file (or copy one of the existing `*_config.yaml` files) and run:

```bash
python run_benchmark.py -c your_config.yaml --verbose > results.txt
```

The `--verbose` flag prints the full prompt and model response for each test — useful for debugging or inspecting outputs. Redirecting to a file with `>` lets you review results without scrolling.

Common options:

| Flag | What it does |
|------|-------------|
| `-c config.yaml` | Config file to use |
| `--verbose` | Print full prompt + response per test |
| `-r N` | Override runs per model (config default: 3) |
| `-w N` | Override max workers (config default: 1) |
| `--filter-cat category` | Only run tests in a category |
| `--filter-test 1 2 3` | Only run specific test IDs |
| `--mock` | Run mock tests (no API key needed) |

## Configuration

Copy and edit `benchmark_config.yaml`:

```yaml
name: "my_benchmark_run"
test_file: "test_cases.json"

providers:
  - type: "featherless"
    model: "meta-llama/Llama-3.2-3B-Instruct"
    temperature: 0.0
    max_tokens: 2048
    timeout: 120
  
  - type: "ollama"
    model: "qwen2.5:3b-instruct"
    temperature: 0.0

runs_per_model: 3
max_workers: 4
output_dir: "results"
```

### Provider Types

| Type | Description | Required Env Var |
|------|-------------|------------------|
| `featherless` | Featherless.ai API | `FEATHERLESS_API_KEY` |
| `together` | Together.ai API | `TOGETHER_API_KEY` |
| `openai` | OpenAI API | `OPENAI_API_KEY` |
| `openai_compatible` | Any OpenAI-compatible API (OpenCode, vLLM, etc.) | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |
| `ollama` | Local Ollama server | (none) |
| `local` | OpenAI-compatible local server (llama.cpp, vLLM) | (none) |
| `vllm` | vLLM server | (none) |
| `llamacpp` | llama.cpp server | (none) |
| `hf` / `huggingface` | HuggingFace Transformers (local) | (none) |
| `mock` | Test without API | (none) |

### Provider Configuration

You can set API keys and base URLs in the config file instead of environment variables:

```yaml
providers:
  # OpenAI-compatible API (e.g. OpenCode)
  - type: "openai_compatible"
    model: "opencode/deepseek-v4-flash"
    api_key: "your-key"          # Optional; falls back to OPENAI_API_KEY
    base_url: "https://api.opencode.ai/v1"  # Optional; falls back to OPENAI_BASE_URL
    temperature: 0.0
    max_tokens: 2048
    timeout: 120

  # Local vLLM server
  - type: "vllm"
    model: "Qwen/Qwen2.5-3B-Instruct"
    base_url: "http://localhost:8000/v1"
    api_key: "EMPTY"

  # Local llama.cpp server
  - type: "llamacpp"
    model: "qwen2.5-3b-instruct"
    base_url: "http://localhost:8080"
```

## Test Cases

The benchmark includes 25 test cases covering:

- **Basic commands** - "Sue told Sara qu xi shou ba" → "去洗手吧"
- **English name filtering** - "John told Mary ni hao ma" → "你好嗎"
- **Mixed English/Pinyin** - "Grandma said wo yao chi fan le" → "我要吃飯了"
- **Grammar particles** - ba, ma, ne, le, guo, de, etc.
- **Complex structures** - bei (passive), rang (causative), bi (comparison), etc.
- **Pure Pinyin** - "wo ai ni" → "我愛你"
- **Quoted Pinyin** - 'He said "wo yao qu"' → "我要去"

Each test case specifies exact regex patterns for grading.

## Grading System

Uses **regex-based grading** on the model's output:

| Criterion | Weight | Required |
|-----------|--------|----------|
| `literal_section` | 0.25 | ✓ |
| `table_chinese` | 0.20 | ✓ |
| `table_pinyin` | 0.20 | ✓ |
| `table_english` | 0.15 | ✓ |
| `grammar_vocab` | 0.10 | ✓ |
| `natural_alt` | 0.10 | ✓ |

**Pass = All required sections match their regex patterns**

## Output

Results are saved to `results/<timestamp>/`:

| File | Description |
|------|-------------|
| `summary.json` | Aggregated stats by model, provider, category, test |
| `results.json` | Detailed grading for each response |
| `results.csv` | Tabular data for analysis (pandas, Excel, etc.) |
| `raw_responses.json` | Full model outputs for debugging |
| `config.yaml` | Run configuration snapshot |

## Example Output

```
============================================================
BENCHMARK SUMMARY
============================================================
Total tests:     300
Passed:          247 (82.3%)
Failed:          53
Avg Score:       85.20%
Avg Latency:     1234ms
Avg Tokens:      567

By Model:
  meta-llama/Llama-3.2-3B-Instruct  : 50/50 (100.0%) score=98.00%
  Qwen/Qwen2.5-3B-Instruct          : 50/50 (100.0%) score=96.00%
  google/gemma-2-2b-it              : 45/50 (90.0%)  score=82.00%
  microsoft/Phi-3.5-mini-instruct   : 42/50 (84.0%)  score=78.00%

By Category:
  basic_command            : 100.0%
  english_names            : 95.0%
  negative_command         : 90.0%
  ba_particle              : 80.0%
  guo_particle             : 75.0%
  passive_bei              : 60.0%
```

## Extending Test Cases

Edit `test_cases.json` to add more tests. Each test needs:

```json
{
  "id": 26,
  "title": "New test name",
  "category": "category_name",
  "prompt": "User input with English + pinyin",
  "expected_literal_chars": "预期字符",
  "expected_pinyin_table": "预期字符",
  "expected_pinyin_tone_marked": "yù qí zì fú",
  "expected_english_literal": "expected literal translation",
  "expected_grammar_vocab": "explanation of grammar",
  "expected_natural_alternative": "natural Chinese alternative",
  "regex_patterns": {
    "literal_section": "Literal Translation:\\s*预期字符",
    "table_chinese": "\\|\\s*预期字符\\s*\\|",
    "table_pinyin": "\\|\\s*yù\\s*qí\\s*zì\\s*fú\\s*\\|",
    "table_english": "\\|\\s*expected\\s*literal\\s*translation\\s*\\|",
    "grammar_vocab": "Grammar/Vocab:.*yù.*expected.*qí.*literal",
    "natural_alt": "Natural Alternative:.*natural Chinese"
  }
}
```

> **Security note:** Don't commit config files containing real API keys.
> The `.gitignore` already excludes `*_config.yaml` and `.env` files.
> Use environment variables or a separate untracked config file instead.

## Running Specific Tests

```bash
# Only basic command tests
python3 run_benchmark.py -c benchmark_config.yaml --filter-cat basic_command

# Only specific test IDs
python3 run_benchmark.py -c benchmark_config.yaml --filter-test 1 2 3

# Fewer runs for quick iteration
python3 run_benchmark.py -c benchmark_config.yaml --runs 1
```

## Compare Previous Runs

```bash
python3 run_benchmark.py --compare results/20240115_120000 results/20240116_120000
```

## Architecture

```
PinBench/
├── test_cases.json       # Test suite with regex patterns
├── test_cases.py         # Test case dataclasses
├── grader.py             # Regex-based grading + aggregation
├── providers.py          # Model provider abstractions
├── runner.py             # Benchmark runner (config-based)
├── run_benchmark.py      # CLI entry point
├── benchmark_config.yaml # Example configuration
└── requirements.txt      # Dependencies
```

## Requirements

- Python 3.10+
- `openai` (for Featherless/Together/OpenAI APIs)
- `requests` (for Ollama/local servers)
- `pyyaml` (for config files)

Optional for local models:
- `torch`, `transformers`, `accelerate` (for HF Transformers)

## Why This Benchmark?

Small models (<15B) often fail at:
- **Following strict output formats** (tables, sections)
- **Ignoring English words** in mixed input
- **Mapping every pinyin syllable** (not omitting particles)
- **Producing literal character-by-character** translations

This benchmark isolates these failure modes with precise regex grading, enabling rapid iteration on model selection for Chinese tutoring applications.