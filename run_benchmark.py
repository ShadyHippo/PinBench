#!/usr/bin/env python3
"""
Pinyin-to-Character Benchmark Runner

Usage:
    python run_benchmark.py                    # Run with benchmark_config.yaml
    python run_benchmark.py -c config.yaml     # Run with custom config
    python run_benchmark.py --mock             # Quick test with mock provider
"""

import os
import sys
import json
import yaml
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from runner import RunConfig, BenchmarkConfig
from grader import ResultsAggregator, create_grader_from_test_file
from providers import create_provider


def _resolve_env_vars(obj):
    """Recursively resolve ${VAR_NAME} patterns in strings using env vars."""
    if isinstance(obj, str):
        import re
        def _replace(m):
            var = m.group(1)
            val = os.environ.get(var)
            if val is None:
                raise ValueError(f"Environment variable '{var}' is not set")
            return val
        return re.sub(r'\$\{(\w+)\}', _replace, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj

def load_config(config_path: str) -> RunConfig:
    """Load benchmark configuration from YAML or JSON."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if path.suffix in ('.yaml', '.yml'):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    elif path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    
    # Resolve ${ENV_VAR} patterns
    data = _resolve_env_vars(data)
    
    return RunConfig(**data)


def create_mock_config() -> RunConfig:
    """Create a quick mock config for testing."""
    return RunConfig(
        name="mock_test_run",
        test_file="test_cases.json",
        providers=[
            {"type": "mock", "model": "mock-small-model", "latency_ms": 50},
            {"type": "mock", "model": "mock-medium-model", "latency_ms": 100},
        ],
        runs_per_model=2,
        max_workers=1,
        output_dir="results",
        print_progress=True,
    )


def run_benchmark(config: RunConfig) -> Dict[str, Any]:
    """Run the benchmark with the given configuration."""
    
    print(f"\n{'='*60}")
    print(f"PINYIN BENCHMARK: {config.name}")
    print(f"{'='*60}")
    print(f"Test file: {config.test_file}")
    print(f"Models: {len(config.providers)}")
    print(f"Runs per model: {config.runs_per_model}")
    print(f"Max workers: {config.max_workers}")
    print(f"Output dir: {config.output_dir}")
    print()
    
    # Load test cases
    test_config = BenchmarkConfig.from_json(config.test_file)
    test_cases = test_config.test_cases
    
    # Apply filters
    if config.filter_categories:
        test_cases = [tc for tc in test_cases if tc.category in config.filter_categories]
        print(f"Filtered to categories: {config.filter_categories} ({len(test_cases)} tests)")
    
    if config.filter_test_ids:
        test_cases = [tc for tc in test_cases if tc.id in config.filter_test_ids]
        print(f"Filtered to test IDs: {config.filter_test_ids} ({len(test_cases)} tests)")
    
    if not test_cases:
        print("No test cases after filtering!")
        return {}
    
    print(f"Test cases to run: {len(test_cases)}")
    
    # Create grader
    grader = create_grader_from_test_file(config.test_file)
    
    # Create aggregator
    aggregator = ResultsAggregator()
    
    # Create providers
    providers = []
    for pconfig in config.providers:
        try:
            provider = create_provider(pconfig)
            providers.append(provider)
            print(f"  Created provider: {provider.get_provider_name()}/{provider.get_model_name()}")
        except Exception as e:
            print(f"  ERROR creating provider {pconfig}: {e}")
            if "API key" in str(e) or "api_key" in str(e).lower():
                print("  Hint: Set FEATHERLESS_API_KEY, TOGETHER_API_KEY, or OPENAI_API_KEY environment variable")
    
    if not providers:
        print("No valid providers created!")
        return {}
    
    # System prompt
    system_prompt = config.system_prompt_override or test_config.system_prompt
    
    # Run benchmark
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_requests = len(test_cases) * len(providers) * config.runs_per_model
    completed = 0
    failed = 0
    raw_responses = []
    
    start_time = time.time()
    
    # Run with thread pool
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def run_single_test(test_case, provider, run_idx):
        """Run a single test case."""
        response = None
        try:
            response = provider.generate(system_prompt, test_case.prompt)
            
            if response.error:
                return {
                    "test_case": test_case,
                    "provider": provider,
                    "run_idx": run_idx,
                    "response": response,
                    "error": response.error,
                    "success": False,
                }
            
            # Grade the response
            grading = grader.grade(
                test_id=test_case.id,
                response=response.text,
                model=response.model,
                provider=response.provider,
                latency_ms=response.latency_ms,
                tokens_used=response.tokens_used,
            )
            
            return {
                "test_case": test_case,
                "provider": provider,
                "run_idx": run_idx,
                "response": response,
                "grading": grading,
                "success": True,
            }
        except Exception as e:
            return {
                "test_case": test_case,
                "provider": provider,
                "run_idx": run_idx,
                "response": response,
                "error": str(e),
                "success": False,
            }
    
    # Run test suite with retry loop for API errors
    max_retries = config.retry_failed if config.retry_failed > 0 else 5
    max_retry_multiplier = 5  # Each test can fail up to this many times before abandoning
    
    # Collect all task definitions
    tasks = []
    for provider in providers:
        for test_case in test_cases:
            for run_idx in range(config.runs_per_model):
                tasks.append((test_case, provider, run_idx))
    
    pending = tasks[:]
    retry_count = 0
    fail_count_per_test = {}  # test_id -> fail count
    
    while pending:
        if retry_count >= max_retries:
            print(f"\nReached max retry rounds ({max_retries}). {len(pending)} tests abandoned.")
            for tc, _, ri in pending:
                failed += 1
                print(f"  ABANDONED Test {tc.id}: {tc.title} (run {ri})")
            break
        
        if retry_count > 0:
            backoff = min(retry_count * 2, 30)  # 2s, 4s, 6s, ... 30s max
            print(f"\n--- Retry round {retry_count}/{max_retries} ({len(pending)} remaining, waiting {backoff}s) ---")
            time.sleep(backoff)
        else:
            print(f"\n--- Initial run ({len(pending)} tests) ---")
        
        batch = pending
        pending = []
        batch_failures = 0
        
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = [executor.submit(run_single_test, tc, p, ri) for tc, p, ri in batch]
            
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                tid = result['test_case'].id
                run_idx = result['run_idx']
                
                if result["success"]:
                    grading = result["grading"]
                    aggregator.add_result(grading)
                    
                    status = "✓" if grading.passed else "✗"
                    if config.print_progress:
                        passed_crit = [k for k, v in grading.details.items() if v]
                        failed_crit = [k for k, v in grading.details.items() if not v]
                        print(f"[{completed}] {status} Test {tid}: {grading.test_title}")
                        print(f"       Score: {grading.score:.2f}  |  {grading.latency_ms:.0f}ms")
                        print(f"       PASS: {', '.join(passed_crit) if passed_crit else '(none)'}")
                        print(f"       FAIL: {', '.join(failed_crit) if failed_crit else '(none)'}")
                else:
                    batch_failures += 1
                    fail_count_per_test[tid] = fail_count_per_test.get(tid, 0) + 1
                    
                    if config.print_progress:
                        print(f"[{completed}] ✗ ERROR Test {tid}: {result['error'][:80]}")
                    
                    # Check if this test has failed too many times
                    if fail_count_per_test[tid] >= max_retry_multiplier:
                        failed += 1
                        if config.print_progress:
                            print(f"  Abandoning test {tid} after {max_retry_multiplier} failures")
                    else:
                        pending.append((result['test_case'], result['provider'], run_idx))
                
                # Save raw response eagerly (before grading outcome)
                if config.save_raw_responses and result.get("response") and result["response"].text:
                    resp_data = {
                        "test_id": tid,
                        "test_title": result["test_case"].title,
                        "category": result["test_case"].category,
                        "run_idx": run_idx,
                        "model": getattr(result.get("response"), "model", ""),
                        "provider": getattr(result.get("response"), "provider", ""),
                        "latency_ms": getattr(result.get("response"), "latency_ms", 0),
                        "tokens_used": getattr(result.get("response"), "tokens_used", 0),
                        "prompt": result["test_case"].prompt,
                        "response": result["response"].text,
                        "grading": result.get("grading").to_dict() if result.get("grading") else None,
                    }
                    raw_responses.append(resp_data)
                
                # Verbose output: always show response text if available
                if config.verbose and result.get("response") and result["response"].text:
                    if config.print_progress:
                        print()
                        print(result['test_case'].prompt)
                        print()
                        print(result['response'].text)
                        print()
        
        # If more than 50% of tests in this batch failed with API errors, assume server is down
        batch_size = len(batch)
        if batch_size > 5 and batch_failures > batch_size * 0.5:
            print(f"\n  ⚠ {batch_failures}/{batch_size} tests failed with API errors. Server may be unstable.")
        
        retry_count += 1
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Duration: {elapsed:.1f}s")
    print(f"Total requests: {total_requests}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    
    # Save results
    _save_results(output_dir, run_id, config, aggregator, raw_responses, elapsed, completed, failed)
    
    # Print summary
    aggregator.print_summary()
    
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "elapsed": elapsed,
        "completed": completed,
        "failed": failed,
        "aggregator": aggregator,
    }


def _save_results(output_dir: Path, run_id: str, config: RunConfig,
                  aggregator, raw_responses: List[Dict],
                  elapsed: float, completed: int, failed: int):
    """Save all benchmark results."""
    
    # Summary JSON
    summary = {
        "run_id": run_id,
        "run_name": config.name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "test_file": config.test_file,
            "runs_per_model": config.runs_per_model,
            "providers": [
                {"type": p.get("type", "unknown"), "model": p.get("model", p.get("model_path", "unknown"))}
                for p in config.providers
            ],
        },
        "statistics": {
            "duration_seconds": elapsed,
            "total_requests": completed + failed,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / (completed + failed) if (completed + failed) > 0 else 0,
        },
        "summary": aggregator.summary(),
        "by_model": aggregator.aggregate_by_model(),
        "by_provider": aggregator.aggregate_by_provider(),
        "by_category": aggregator.aggregate_by_category(),
        "by_test": aggregator.aggregate_by_test(),
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Detailed results JSON
    if config.save_results_json:
        detailed = {
            "run_id": run_id,
            "results": [r.to_dict() for r in aggregator.results],
        }
        with open(output_dir / "results.json", 'w') as f:
            json.dump(detailed, f, indent=2)
    
    # CSV export
    if config.save_results_csv:
        aggregator.export_csv(str(output_dir / "results.csv"))
    
    # Raw responses
    if config.save_raw_responses and raw_responses:
        with open(output_dir / "raw_responses.json", 'w') as f:
            json.dump(raw_responses, f, indent=2)
    
    # Config snapshot
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - summary.json (overview)")
    if config.save_results_json:
        print(f"  - results.json (detailed grading)")
    if config.save_results_csv:
        print(f"  - results.csv (for analysis)")
    if config.save_raw_responses:
        print(f"  - raw_responses.json (full model outputs)")
    print(f"  - config.yaml (run configuration)")


def compare_runs(run_dirs: List[str]):
    """Compare multiple benchmark runs."""
    print("\n" + "="*60)
    print("RUN COMPARISON")
    print("="*60)
    
    for run_dir in run_dirs:
        summary_path = Path(run_dir) / "summary.json"
        if not summary_path.exists():
            print(f"  {run_dir}: No summary.json found")
            continue
        
        with open(summary_path) as f:
            data = json.load(f)
        
        print(f"\n{data.get('run_name', 'Unknown')} ({data.get('run_id', 'unknown')})")
        print(f"  Duration: {data['statistics']['duration_seconds']:.1f}s")
        print(f"  Completed: {data['statistics']['completed']}/{data['statistics']['total_requests']}")
        
        if 'summary' in data:
            s = data['summary']
            print(f"  Overall pass rate: {s['pass_rate']*100:.1f}%")
            print(f"  Avg score: {s['avg_score']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Pinyin-to-Character Benchmark Runner")
    parser.add_argument("-c", "--config", default="benchmark_config.yaml", help="Config file path")
    parser.add_argument("--mock", action="store_true", help="Run quick mock benchmark")
    parser.add_argument("--compare", nargs="+", help="Compare previous run directories")
    parser.add_argument("-o", "--output", help="Output directory (overrides config)")
    parser.add_argument("-r", "--runs", type=int, help="Runs per model (overrides config)")
    parser.add_argument("-w", "--workers", type=int, help="Max workers (overrides config)")
    parser.add_argument("--filter-cat", nargs="+", help="Filter by categories")
    parser.add_argument("--filter-test", nargs="+", type=int, help="Filter by test IDs")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output")
    parser.add_argument("--verbose", action="store_true", help="Print full prompt + response per test")
    args = parser.parse_args()
    
    # Handle comparison
    if args.compare:
        compare_runs(args.compare)
        return
    
    # Load config
    if args.mock:
        config = create_mock_config()
        print("Running MOCK benchmark (no API keys needed)")
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            print("Run with --mock for a quick test, or create a config file.")
            print(f"Example config: {Path(__file__).parent / 'benchmark_config.yaml'}")
            sys.exit(1)
        config = load_config(str(config_path))
    
    # Apply CLI overrides
    if args.output:
        config.output_dir = args.output
    if args.runs:
        config.runs_per_model = args.runs
    if args.workers:
        config.max_workers = args.workers
    if args.filter_cat:
        config.filter_categories = args.filter_cat
    if args.filter_test:
        config.filter_test_ids = args.filter_test
    if args.no_progress:
        config.print_progress = False
    if args.verbose:
        config.verbose = True
    
    # Run benchmark
    run_benchmark(config)


if __name__ == "__main__":
    main()