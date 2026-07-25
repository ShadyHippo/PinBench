#!/usr/bin/env python3
"""
Regrade old benchmark raw outputs against the current (v1.3) grader + test criteria.
This lets us compare "old prompt, new grader" vs "new prompt, new grader".

Usage:
    python regrade.py <old_multi_run_dir> <output_dir>
    
Example:
    python regrade.py results/multi_5free results/v1.2_regraded
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grader import create_grader_from_test_file, ResultsAggregator


def find_raw_responses(run_dir: str) -> List[Dict]:
    """Find and load raw_responses.json, searching timestamped subdirs.
    Picks the most recent (reverse alphabetical = latest timestamp)."""
    p = Path(run_dir)
    
    # Direct
    direct = p / "raw_responses.json"
    if direct.exists():
        with open(direct) as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            return data
    
    # Subdirectories (reverse sorted = newest first)
    for child in sorted(p.iterdir(), reverse=True):
        if child.is_dir():
            sp = child / "raw_responses.json"
            if sp.exists():
                with open(sp) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
    
    return []


def load_old_index(multi_dir: str) -> List[Dict]:
    """Load the _index.json from a multi-run directory."""
    index_path = Path(multi_dir) / "_index.json"
    if not index_path.exists():
        # Build index by scanning
        runs = []
        for model_dir in sorted(Path(multi_dir).iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith('_') or model_dir.name.startswith('aggregated'):
                continue
            for run_dir in sorted(model_dir.iterdir()):
                if run_dir.is_dir() and run_dir.name.startswith('run_'):
                    raw = find_raw_responses(str(run_dir))
                    if raw:
                        runs.append({
                            "config_name": model_dir.name,
                            "run_dir": str(run_dir),
                            "test_count": len(raw),
                        })
        return runs
    
    with open(index_path) as f:
        idx = json.load(f)
    return idx.get("runs", [])


def regrade_all(raw_responses: List[Dict], grader) -> tuple:
    """Regrade a list of raw responses using the given grader."""
    aggregator = ResultsAggregator()
    
    for resp in raw_responses:
        grading = grader.grade(
            test_id=resp["test_id"],
            response=resp.get("response", ""),
            model=resp.get("model", ""),
            provider=resp.get("provider", ""),
            latency_ms=resp.get("latency_ms", 0),
            tokens_used=resp.get("tokens_used", 0),
        )
        aggregator.add_result(grading)
    
    return aggregator


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    old_multi_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Create grader from CURRENT test_cases.json (v1.3)
    script_dir = Path(__file__).parent
    test_file = str(script_dir / "test_cases.json")
    grader = create_grader_from_test_file(test_file)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load test data for archival
    with open(test_file) as f:
        test_data = json.load(f)
    
    # System prompt: prefer system_prompt.txt over embedded one
    sp_file = script_dir / "system_prompt.txt"
    if sp_file.exists():
        with open(sp_file) as f:
            system_prompt = f.read().strip()
        system_prompt_source = "system_prompt.txt"
    else:
        system_prompt = test_data.get("system_prompt", "")
        system_prompt_source = test_file
    
    print(f"Regrading old runs from: {old_multi_dir}")
    print(f"Using test file: {test_file}")
    print(f"Output: {output_dir}")
    print()
    
    # Find all runs with raw responses
    runs = load_old_index(old_multi_dir)
    print(f"Found {len(runs)} runs")
    
    # Group by config_name
    by_config: Dict[str, List[Dict]] = {}
    for r in runs:
        name = r["config_name"]
        if name not in by_config:
            by_config[name] = []
        raw = find_raw_responses(r["run_dir"])
        if raw:
            by_config[name].append(raw)
    
    per_config_results = {}
    all_per_test = {}
    
    for config_name in sorted(by_config.keys()):
        raw_batches = by_config[config_name]
        print(f"\n  {config_name}: {len(raw_batches)} runs")
        
        per_run = []
        for batch in raw_batches:
            aggregator = regrade_all(batch, grader)
            per_run.append({
                "summary": aggregator.summary(),
                "raw_count": len(batch),
            })
        
        # Aggregate across runs for this config
        n = len(per_run)
        pass_rates = [r["summary"]["pass_rate"] for r in per_run]
        avg_scores = [r["summary"]["avg_score"] for r in per_run]
        
        pr_mean = sum(pass_rates) / n if n > 0 else 0
        pr_std = (sum((x - pr_mean)**2 for x in pass_rates) / n)**0.5 if n > 1 else 0
        sc_mean = sum(avg_scores) / n if n > 0 else 0
        sc_std = (sum((x - sc_mean)**2 for x in avg_scores) / n)**0.5 if n > 1 else 0
        
        per_config_results[config_name] = {
            "n_runs": n,
            "pass_rate_mean": pr_mean,
            "pass_rate_std": pr_std,
            "avg_score_mean": sc_mean,
            "avg_score_std": sc_std,
            "per_run": per_run,
        }
        
        print(f"    pass_rate: {pr_mean*100:.1f}±{pr_std*100:.1f}%  avg_score: {sc_mean*100:.1f}±{sc_std*100:.1f}%")
    
    # Save regraded results
    bundle = {
        "timestamp": datetime.now().isoformat(),
        "description": f"Regraded runs from {old_multi_dir} using v{test_data.get('version', '?')} grader + criteria",
        "source": old_multi_dir,
        "test_data": {
            "version": test_data.get("version", ""),
            "description": test_data.get("description", ""),
            "system_prompt": system_prompt,
            "system_prompt_source": system_prompt_source,
            "tests": [
                {"id": t["id"], "title": t["title"], "category": t["category"], "prompt": t["prompt"]}
                for t in test_data.get("tests", [])
            ],
        },
        "results": per_config_results,
    }
    
    json_name = f"regraded_{ts}.json"
    with open(out / json_name, "w") as f:
        json.dump(bundle, f, indent=2)
    
    # Also save as latest
    with open(out / "regraded.json", "w") as f:
        json.dump(bundle, f, indent=2)
    
    print(f"\nRegraded results saved to {out / json_name}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  REGRADED RESULTS (old prompt, v{test_data.get('version', '?')} grader)")
    print(f"{'='*70}")
    print(f"\n{'Model':<25} {'Runs':>5} {'Pass Rate':>18} {'Avg Score':>18}")
    print("-"*70)
    for name, data in sorted(per_config_results.items()):
        pr_str = f"{data['pass_rate_mean']*100:.1f}±{data['pass_rate_std']*100:.1f}%"
        sc_str = f"{data['avg_score_mean']*100:.1f}±{data['avg_score_std']*100:.1f}%"
        print(f"{name:<25} {data['n_runs']:>5} {pr_str:>18} {sc_str:>18}")


if __name__ == "__main__":
    main()
