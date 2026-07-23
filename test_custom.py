#!/usr/bin/env python3
"""
Run custom test cases against local models.
Usage:
    python3 test_custom.py                          # Test Gemma
    python3 test_custom.py --model qwen              # Test Qwen
    python3 test_custom.py --model both              # Test both
    python3 test_custom.py --model both --port 8001  # Override port
"""

import argparse
import json
import requests
import sys

# The system prompt from test_cases.json
SYSTEM_PROMPT = """**Role:** Expert Mandarin Chinese Tutor.

**CORE INSTRUCTION:**
You are a Pinyin-to-Character converter. You only process the Pinyin syllables. 
**IGNORE ALL ENGLISH WORDS.** If the input contains English (e.g., "Name," "told," "Name"), you must skip them entirely. Do not translate them. Do not map them.
- NEVER translate "Grandma", "told", "said", or any other English words.
- NEVER translate English names like "John" "Joe" "Jane" etc.

**Task 1: Literal Mapping (The "Robot" Rule)**
- Identify all Pinyin syllables).
- Map EVERY syllable to its most literal Chinese character.
- **RULE:** Do not omit any pinyin syllables (including particles).
- **RULE:** Do not "correct" a syllable unless it does not work in the sentence as written.
- **RULE:** The output must ONLY contain characters for the pinyin. Do not include characters for the names or English verbs.

**Task 2: Tutor's Note (The "Teacher" Rule)**
- Provide a "Natural Alternative" for the pinyin phrase.
- Explain the grammar/vocabulary of the literal translation.

**Output Format (Strictly Follow):**

[Phonetic Sandbox]
Literal Translation: [Characters for the Pinyin ONLY]

| Chinese | Pinyin | English |
| :--- | :--- | :--- |
| [Characters] | [Tone-marked Pinyin] | [Literal translation of Pinyin] |

**Tutor's Note:**
- **Grammar/Vocab:** [Brief explanation]
- **Natural Alternative:** [Your correction here] only include if there are necessary corrections"""

TEST_CASES = [
    "Angela told Esther qu shi shou ba",
    "Grandma told Esther Kai xin ma? And Esther said Kai xin! About her getting ready to go to ballet camp",
]

MODELS = {
    "qwen": {
        "base_url": "http://192.168.0.20:8000/v1",
        "model": "/models/Qwen3.5_9b-Q5_K_M-instruct.gguf",
    },
    "gemma": {
        "base_url": "http://192.168.0.20:8001/v1",
        "model": "/models/.oldmodels/gemma-4-12B-it-Q6_K_L.gguf",
    },
}


def run_test(base_url: str, model: str, system_prompt: str, user_prompt: str,
             max_tokens: int = 4096, timeout: int = 600) -> dict:
    """Run a single test case."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    print(f"  Sending request (max_tokens={max_tokens})...")
    sys.stdout.flush()

    import time
    start = time.time()
    resp = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    elapsed = time.time() - start
    resp.raise_for_status()
    data = resp.json()

    # Extract text from response
    message = data["choices"][0]["message"]
    text = message.get("content", "") or ""
    if not text:
        text = message.get("reasoning_content", "") or ""

    usage = data.get("usage", {})
    return {
        "text": text,
        "finish_reason": data["choices"][0].get("finish_reason"),
        "latency_s": round(elapsed, 1),
        "tokens": usage.get("total_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Run custom test cases")
    parser.add_argument("--model", choices=["qwen", "gemma", "both"], default="gemma")
    parser.add_argument("--port", type=int, help="Override port (e.g. 8000)")
    parser.add_argument("--max-tokens", type=int, default=10240, help="Max tokens (default 10240)")
    parser.add_argument("--timeout", type=int, default=600, help="Request timeout in seconds")
    args = parser.parse_args()

    # Determine which models to run
    model_keys = ["qwen", "gemma"] if args.model == "both" else [args.model]

    for test_input in TEST_CASES:
        print(f"\n{'='*70}")
        print(f"INPUT: {test_input}")
        print(f"{'='*70}")

        for key in model_keys:
            model_cfg = MODELS[key].copy()
            if args.port:
                model_cfg["base_url"] = f"http://192.168.0.20:{args.port}/v1"

            model_label = f"{key} ({model_cfg['model'].split('/')[-1]})"
            print(f"\n--- {model_label} ---")

            result = run_test(
                base_url=model_cfg["base_url"],
                model=model_cfg["model"],
                system_prompt=SYSTEM_PROMPT,
                user_prompt=test_input,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )

            print(f"  Time: {result['latency_s']}s | Tokens: {result['tokens']} | "
                  f"Finish: {result.get('finish_reason')}")
            print()

            # Print the response text
            output = result["text"].strip()
            if output:
                print(output)
            else:
                print("  [EMPTY RESPONSE]")

            print()


if __name__ == "__main__":
    main()
