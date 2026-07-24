# Changes made to the pinyin benchmark

## 2026-07-22: Initial project creation

Created the initial benchmark framework with:
- 25 test cases covering basic pinyin-to-character conversion
- Regex-based grader
- Multi-provider support (Featherless, Together, OpenAI, Ollama, vLLM, HF)
- YAML config for benchmark runs

## 2026-07-23: Major refactoring

### Redesigned grader
- Removed brittle per-test regex patterns (150 custom regexes)
- Replaced with generic section-based grader that checks:
  - [Phonetic Sandbox] exists
  - Markdown table with Chinese/Pinyin/English columns exists
  - Table row matches expected chinese/pinyin/english (with flexible matching)
  - Word-by-word literal breakdown section exists
  - Each literal word has chinese character + pinyin + keyword in explanation
- Scoring: phonetics 10%, table_exists 10%, table_content 40%, literal_section 10%, literal_words 30%

### Case-insensitive matching
- Pinyin comparison: trim whitespace, lowercase, strip tone marks via Unicode NFKD normalization
- Chinese comparison: strip spaces for matching merged vs separated characters
- English comparison: strip trailing punctuation, compare against list of acceptable options
- Literal pinyin: extract from parenthesized parts `(pinyin)` only, normalized

### Multiple english options per test
- Changed english field from single string to list of acceptable translations
- Added alternatives for all tests where phrasing could vary
- Added "Do not go", "How is it?", "How about it?", etc.

### Compound word handling
- Split compound words (我们→我+们, 老师→老+师, 朋友→朋+友) in test data literal arrays
- Grader accepts compound characters even if split across multiple bullet lines
- Pinyin extraction from parenthesized parts works across merged compounds (wǒmen→wǒ+men)

### Multi-row table support
- Grader now concatenates adjacent table rows and checks combined values
- Handles case where model splits phrase across multiple table rows

### Punctuation stripping fix
- Expected english options now strip trailing punctuation before comparison
- Fixed: was stripping from model output only, causing mismatch

### Pinyin tone mark fixes
- Added `_strip_pinyin_tones()` using unicodedata.NFKD normalization
- Added `_normalize_pinyin()` for full normalization (tones + non-alnum removal)
- Tone sandhi handled correctly (bú matches bù)

### Changed system prompt
- Replaced "Do NOT correct a syllable — map it as written"
- With: "The pinyin syllables form a coherent Chinese phrase. Use the full context to identify the intended characters. If individual syllables are ambiguous, let the phrase-level meaning guide your mapping."

### Added 20 new corrected_pinyin test cases (version 1.1)
- 10 clean/dirty pairs testing pinyin confusion: q/ch, x/sh, zh/z, c/ch, sh/s
- Both versions map to the same expected characters
- Tests models' ability to use context to correct ambiguous pinyin
- Total tests: 45

### Pronoun normalization
- Added chinese pronoun normalization: 她→他, 妳→你, 牠→他
- Applied in both table check and literal words check
- Added keywords (she, her) to 他 entries

### Retry logic for API errors
- Added retry loop that re-queues tests failing with API errors
- Configurable via retry_failed in config
- Exponential backoff between retry rounds (2s, 4s, 6s, ...)
- Cap of 10 retry rounds
- Per-test max 5 failures before abandoning individual test
- Early termination if >50% of tests fail with API errors (assumes server down)

### Increased max_tokens for Qwen config
- 4096 was causing truncation on complex/long prompts where the sandbox used too many tokens
- Bumped to 6144 to give more room while keeping request times manageable
- Previous attempt at 8192 caused per-request times to become too slow (20 min+ runs)

### Added "you (particle" to test 7 english options
- Some models split 你呢 into two table rows (你/呢 separately)
- Combined english from both rows needs to match

### Updated README with benchmark philosophy
- Added the core distinction: "Can the model infer the user's intended Chinese while respecting the user's input?"
- This clarifies why corrected_pinyin tests exist and what the benchmark actually measures