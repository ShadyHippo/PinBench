# PinBench — Pinyin-to-Character Benchmark

## 1. Objective

Build a benchmark that tests small language models on two distinct skills:
1. **Format compliance** — produce a specific structured output (table + literal breakdown)
2. **Contextual pinyin correction** — infer the intended Chinese character when pinyin is slightly wrong, exactly like a human Chinese learner would

The goal is to identify models useful for Chinese learners, and eventually use the benchmark data to train smaller specialized models.

## 2. Test Design Principles

### 2.1 Each dirty test tests exactly one confusable pair
- A "dirty" test has deliberately wrong pinyin that a human learner would still understand
- Each test has exactly one phonetic confusion (e.g., x/sh, c/q, q/x, ong/ang)
- No multi-typo tests — isolate what the model can and cannot handle

### 2.2 Three difficulty tiers for dirty pinyin
| Tier | Example | Model behavior |
|------|---------|----------------|
| Invalid pinyin (shie→xie) | shie shie ni → 谢谢你 | Easier: models correct to valid pinyin |
| Valid-but-wrong (xiang→想 not 上) | zao xiang hao → 早上好 | Harder: literal mapping gives wrong chars |
| Context-dependent (sh→x with dish context) | shi wan le + dishes context → 洗完了 | Hardest: requires using English context |

### 2.3 Clean counterparts for every dirty test
- Each dirty test has a clean version with correct pinyin
- Isolates whether failures are from pinyin correction or general format issues

### 2.4 System prompt constraints
- "IGNORE ALL ENGLISH WORDS" — models must skip English names/verbs entirely
- Structured output format enforced (table + word-by-word breakdown)
- No examples of wrong pinyin in the prompt
- **Compound words MUST be grouped** — compounds like 明天, 我们, 什么 are a single entry, NOT split into individual characters
- Template says `[Chinese word or compound]` not `[Character]` to reinforce this

## 3. Architecture

```
pinbench/
├── test_cases.json          # 46 tests: prompts, expected tables, literal breakdowns
├── grader.py                # Grading logic: compares model output against expected
│   └── 6 criteria: table_chinese, grammar_vocab, table_pinyin, 
│                   table_english, literal_section, natural_alt
├── runner.py                # TestCase, BenchmarkConfig, RunConfig data classes
├── providers.py             # Model providers: vllm, openai_compatible
│   └── ModelProvider interface: generate(system_prompt, user_prompt) -> ModelResponse
├── run_benchmark.py         # CLI entry point: runs benchmark, retries API errors
│   └── env var resolution, eager raw response saving, _save_results
├── multi_run.py             # Orchestrator: runs N iterations per config, aggregates
│   └── timestamped output, prompt archival in aggregated results
├── plan.md                  # This file
├── *_config.yaml            # Model configs (per-model)
└── results/                 # Output directories
    └── multi_<ts>/          # Multi-run aggregations
        ├── <model>/run_<n>/ # Per-run results
        │   ├── raw_responses.json
        │   ├── summary.json
        │   ├── results.json
        │   └── run.log
        ├── aggregated_<ts>.json  # Cross-run aggregation + test data
        ├── aggregated_<ts>.txt
        └── _index.json
```

## 4. Implementation Phases

### Phase 1: Core ✓
- [x] Write 46 test cases (v1.3) with clean/dirty pairs, compound words merged
- [x] Implement grader with 6 criteria
- [x] Provider factory for vllm + openai_compatible
- [x] Config loader with `${VAR}` env substitution
- [x] Crash-guarded grading + eager response saving
- [x] CLI runner with retry logic

### Phase 2: Multi-Run Orchestration ✓
- [x] multi_run.py orchestrator
- [x] Per-run logging to files
- [x] Cross-run aggregation (mean ± std)
- [x] Prompt archival in output files (timestamped)

### Phase 3: Free Model Benchmarking ✓
- [x] DeepSeek V4 Flash (paid Go sub, 54.3%)
- [x] MiMo-V2.5 Free (61.0%)
- [x] Run 5 free models ×5 runs via OpenCode Zen API:
  - [x] **Nemotron 3 Ultra Free** — 76.0±2.1% pass rate
  - [x] **DeepSeek V4 Flash Free** — 68.7±3.8%
  - [x] **Ling 3.0 Flash Free** — 66.1±2.1%
  - [x] **Laguna S 2.1 Free** — 29.2±23.9%
  - [x] **North Mini Code Free** — 1.3±1.1%

### Phase 4: Paid Model Benchmarking
- [ ] Create configs for paid OpenCode Zen models:
  - [ ] Qwen3.7 Plus (`qwen-plus-3.7`)
  - [ ] DeepSeek V4 Pro (`deepseek-v4-pro`)
  - [ ] GLM-5.2 (`glm-5.2`)
  - [ ] MiMo-V2.5 (paid tier)
- [ ] Run each paid model ×5 runs  
- [ ] Update results table

### Phase 5: Data Analysis
- [ ] Identify which confusable pairs each model can/cannot handle
- [ ] Analyze whether clean test failures are format issues or genuine errors
- [ ] Compare paid vs free model performance gap
- [ ] Check if latency correlates with accuracy

### Phase 6: Publication
- [ ] Clean up repo, add README
- [ ] Open-source on GitHub
- [ ] Publish results table and analysis
- [ ] Release as evaluation tool for Chinese learning model selection

## 5. Running the Benchmark

```bash
# Single model, single run
source venv/bin/activate
export OPENCODE_API_KEY="sk-..."
python run_benchmark.py --config deepseek_config.yaml --runs 1

# Multi-run (5 runs per config)
python multi_run.py --configs ling_config.yaml nemotron_config.yaml \
    deepseek_free_config.yaml --runs 5 --output results/multi_bench

# Re-aggregate existing runs (e.g., after code changes)
python multi_run.py --aggregate results/multi_5free

# Quick smoke test (2 tests, 1 run)
python run_benchmark.py --config laguna_config.yaml --filter-test 1 35 --runs 1
```

## 6. Current Results Summary (2026-07-24)

| Model | Runs | Pass Rate μ±σ | Avg Score | Notes |
|---|---|---|---|---|
| Nemotron 3 Ultra Free | 5 | 76.0±2.1% | 77.0% | Best overall, consistent |
| DeepSeek V4 Flash Free | 5 | 68.7±3.8% | 76.2% | Close second |
| Ling 3.0 Flash Free | 5 | 66.1±2.1% | 71.2% | Fastest (56s) |
| Laguna S 2.1 Free | 5 | 29.2±23.9% | 36.5% | Wild variance |
| North Mini Code Free | 5 | 1.3±1.1% | 33.9% | Unusable |

**5 dirty tests unsolved by ALL models:** #27 (x/q), #29 (c/q), #37 (c/q), #39 (q/x), #43 (sh/x with context)

**Compound word change in v1.3:** Test data merged individual characters into 8 compounds (明天, 北京, 我们, 老师, 朋友, 早上, 什么, 知道). System prompt now REQUIRES compounds as single entries ("ALWAYS group").

## 7. Known Issues
- Some clean tests fail mysteriously (format quirks, not Chinese errors)
- Laguna has missing test data for some runs (API timeout?)
- "IGNORE ALL ENGLISH WORDS" instruction prevents models from using English context to disambiguate
- North Mini Code produces non-Chinese output (random characters)
