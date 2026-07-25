#!/usr/bin/env python3
"""
Multi-run orchestrator for PinBench.
Runs each config N times, preserves all raw data, aggregates results.

Usage:
    python multi_run.py --configs laguna_config.yaml ling_config.yaml north_config.yaml nemotron_config.yaml deepseek_free_config.yaml --runs 5
    python multi_run.py --configs *.yaml --runs 3
    python multi_run.py --aggregate results/multi_20260724_160000  # re-aggregate existing multi-run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def run_benchmark(config_path: str, output_dir: str, run_log: str) -> Dict:
    """Run a single benchmark, capture output, return timing."""
    cmd = [
        sys.executable, "run_benchmark.py",
        "--config", config_path,
        "--output", output_dir,
    ]
    
    start = time.time()
    with open(run_log, "w") as log_f:
        process = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    elapsed = time.time() - start
    
    return {
        "config": config_path,
        "output_dir": output_dir,
        "log_file": run_log,
        "returncode": process.returncode,
        "elapsed": elapsed,
    }


def find_summary(run_dir: str) -> Optional[str]:
    """Find the summary.json in a run directory (may be in a timestamped subdir)."""
    p = Path(run_dir)
    
    # Direct check
    direct = p / "summary.json"
    if direct.exists():
        return str(direct)
    
    # Check timestamped subdirectories
    for child in sorted(p.iterdir()):
        if child.is_dir():
            sp = child / "summary.json"
            if sp.exists():
                return str(sp)
    
    return None


def load_summary(run_dir: str) -> Optional[Dict]:
    """Load summary.json from a run directory."""
    sp = find_summary(run_dir)
    if sp:
        with open(sp) as f:
            return json.load(f)
    return None


def get_config_name(config_path: str) -> str:
    """Extract a short name from the config path."""
    name = os.path.splitext(os.path.basename(config_path))[0]
    # Remove common suffixes
    for suffix in ["_config", "_free", "_benchmark"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def run_all(configs: List[str], num_runs: int, base_dir: str, parallel: bool = False):
    """Run all configs N times each."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    
    all_results = []  # List[Dict] for each run
    
    for config_path in configs:
        config_name = get_config_name(config_path)
        print(f"\n{'='*70}")
        print(f"  CONFIG: {config_name} ({config_path})")
        print(f"  RUNS:   {num_runs}")
        print(f"{'='*70}")
        
        for run_idx in range(1, num_runs + 1):
            run_label = f"run_{run_idx:03d}"
            run_dir = base / config_name / run_label
            run_dir.mkdir(parents=True, exist_ok=True)
            log_file = run_dir / "run.log"
            
            print(f"\n  [{run_idx}/{num_runs}] Running {config_name}...")
            sys.stdout.flush()
            
            result = run_benchmark(
                config_path=config_path,
                output_dir=str(run_dir),
                run_log=str(log_file),
            )
            
            # Load the summary to check success
            summary = load_summary(str(run_dir))
            if summary:
                pr = summary["summary"].get("pass_rate", -1)
                sc = summary["summary"].get("avg_score", -1)
                dur = summary["statistics"]["duration_seconds"]
                print(f"  [{run_idx}/{num_runs}] {config_name}: "
                      f"pass={pr*100:.1f}%  score={sc:.2%}  "
                      f"duration={dur:.0f}s  log={log_file}")
            else:
                print(f"  [{run_idx}/{num_runs}] {config_name}: "
                      f"FAILED (exit={result['returncode']})  log={log_file}")
            
            all_results.append({
                "config": config_path,
                "config_name": config_name,
                "run_idx": run_idx,
                "run_dir": str(run_dir),
                "summary": summary,
                "returncode": result["returncode"],
            })
    
    # Save master index
    index = {
        "timestamp": datetime.now().isoformat(),
        "base_dir": str(base),
        "runs": [
            {
                "config": r["config"],
                "config_name": r["config_name"],
                "run_idx": r["run_idx"],
                "run_dir": r["run_dir"],
                "returncode": r["returncode"],
                "has_summary": r["summary"] is not None,
            }
            for r in all_results
        ],
    }
    with open(base / "_index.json", "w") as f:
        json.dump(index, f, indent=2)
    
    return all_results


def aggregate(results: List[Dict]) -> Dict:
    """Aggregate results across runs, grouped by config/model."""
    # Group by config_name
    by_config: Dict[str, List[Dict]] = {}
    for r in results:
        if r["summary"] is None:
            continue
        name = r["config_name"]
        if name not in by_config:
            by_config[name] = []
        by_config[name].append(r["summary"])
    
    aggregated = {}
    
    for config_name, summaries in sorted(by_config.items()):
        n = len(summaries)
        
        # Overall stats across runs
        pass_rates = [s["summary"]["pass_rate"] for s in summaries]
        avg_scores = [s["summary"]["avg_score"] for s in summaries]
        durations = [s["statistics"]["duration_seconds"] for s in summaries]
        models = summaries[0]["summary"].get("models", [])
        
        # Per-test aggregation
        all_test_ids = set()
        for s in summaries:
            all_test_ids.update(s.get("by_test", {}).keys())
        
        per_test = {}
        for tid in sorted(all_test_ids, key=lambda x: int(x) if x.isdigit() else x):
            test_runs = []
            for s in summaries:
                bt = s.get("by_test", {})
                if tid in bt:
                    test_runs.append(bt[tid])
            
            if test_runs:
                t_pass_rates = [t["pass_rate"] for t in test_runs]
                t_scores = [t["avg_score"] for t in test_runs]
                t_first = test_runs[0]
                per_test[tid] = {
                    "test_id": t_first.get("test_id", tid),
                    "categories": t_first.get("categories", []),
                    "n_runs": len(test_runs),
                    "pass_rate_mean": sum(t_pass_rates) / len(t_pass_rates),
                    "pass_rate_std": (sum((x - sum(t_pass_rates)/len(t_pass_rates))**2 for x in t_pass_rates) / len(t_pass_rates))**0.5 if len(t_pass_rates) > 1 else 0,
                    "avg_score_mean": sum(t_scores) / len(t_scores),
                    "passes": [t["passed"] for t in test_runs],
                    "counts": [t["count"] for t in test_runs],
                }
        
        # Category aggregation
        all_cats = set()
        for s in summaries:
            all_cats.update(s.get("by_category", {}).keys())
        
        per_cat = {}
        for cat in sorted(all_cats):
            cat_runs = []
            for s in summaries:
                bc = s.get("by_category", {})
                if cat in bc:
                    cat_runs.append(bc[cat])
            if cat_runs:
                c_pass = [c["pass_rate"] for c in cat_runs]
                c_score = [c["avg_score"] for c in cat_runs]
                per_cat[cat] = {
                    "pass_rate_mean": sum(c_pass) / len(c_pass),
                    "pass_rate_std": (sum((x - sum(c_pass)/len(c_pass))**2 for x in c_pass) / len(c_pass))**0.5 if len(c_pass) > 1 else 0,
                    "avg_score_mean": sum(c_score) / len(c_score),
                }
        
        aggregated[config_name] = {
            "n_runs": n,
            "models": models,
            "pass_rate_mean": sum(pass_rates) / n,
            "pass_rate_std": (sum((x - sum(pass_rates)/n)**2 for x in pass_rates) / n)**0.5 if n > 1 else 0,
            "avg_score_mean": sum(avg_scores) / n,
            "avg_score_std": (sum((x - sum(avg_scores)/n)**2 for x in avg_scores) / n)**0.5 if n > 1 else 0,
            "avg_duration": sum(durations) / n,
            "pass_rates": pass_rates,
            "avg_scores": avg_scores,
            "per_test": per_test,
            "per_category": per_cat,
        }
    
    return aggregated


def print_aggregated_table(aggregated: Dict):
    """Print a clean summary table of aggregated results."""
    print("\n" + "="*70)
    print("  AGGREGATED RESULTS")
    print("="*70)
    
    # Model-level overview
    print(f"\n{'Model':<25} {'Runs':>5} {'Pass Rate':>12} {'Avg Score':>12} {'Avg Dur':>10}")
    print("-"*70)
    for name, data in sorted(aggregated.items()):
        pr_mean = data["pass_rate_mean"] * 100
        pr_std = data["pass_rate_std"] * 100
        sc_mean = data["avg_score_mean"] * 100
        sc_std = data["avg_score_std"] * 100
        dur = data["avg_duration"]
        
        pr_str = f"{pr_mean:.1f}±{pr_std:.1f}%" if data["n_runs"] > 1 else f"{pr_mean:.1f}%"
        sc_str = f"{sc_mean:.1f}±{sc_std:.1f}%" if data["n_runs"] > 1 else f"{sc_mean:.1f}%"
        
        print(f"{name:<25} {data['n_runs']:>5} {pr_str:>12} {sc_str:>12} {dur:>7.0f}s")
    
    # Per-test dirty test comparison
    dirty_ids = ["27", "29", "31", "33", "35", "37", "39", "43", "45", "46"]
    
    print(f"\n{'─'*70}")
    print("  DIRTY TEST BREAKDOWN (pass_rate% ± std)")
    print(f"{'─'*70}")
    
    # Header
    header = f"{'Test':<8}"
    for name in sorted(aggregated.keys()):
        header += f" {name[:14]:<16}"
    print(header)
    print("-" * (8 + 16 * len(aggregated)))
    
    for tid in dirty_ids:
        row = f"  {tid:<6}"
        for name in sorted(aggregated.keys()):
            pt = aggregated[name]["per_test"]
            if tid in pt:
                pr = pt[tid]["pass_rate_mean"] * 100
                std = pt[tid]["pass_rate_std"] * 100
                if aggregated[name]["n_runs"] > 1:
                    row += f" {pr:5.0f}±{std:3.0f}%     "
                else:
                    row += f" {pr:5.0f}%          "
            else:
                row += f" {'—':>16}"
        print(row)
    
    # All tests breakdown
    print(f"\n{'─'*70}")
    print("  ALL TEST BREAKDOWN (pass_rate%)")
    print(f"{'─'*70}")
    
    # Collect all test IDs across all models
    all_tids = set()
    for name, data in aggregated.items():
        all_tids.update(data["per_test"].keys())
    all_tids = sorted(all_tids, key=lambda x: int(x) if x.isdigit() else x)
    
    header = f"{'Test':<8}"
    for name in sorted(aggregated.keys()):
        header += f" {name[:14]:<16}"
    print(header)
    print("-" * (8 + 16 * len(aggregated)))
    
    for tid in all_tids:
        row = f"  {tid:<6}"
        for name in sorted(aggregated.keys()):
            pt = aggregated[name]["per_test"]
            if tid in pt:
                pr = pt[tid]["pass_rate_mean"] * 100
                row += f" {pr:5.0f}%          "
            else:
                row += f" {'—':>16}"
        print(row)


def load_test_data(test_file: str = "test_cases.json") -> Dict:
    """Load the test cases and system prompt for archival with results."""
    script_dir = Path(__file__).parent
    path = script_dir / test_file
    if not path.exists():
        return {"error": f"Test file not found: {path}"}
    
    with open(path) as f:
        data = json.load(f)
    
    return {
        "version": data.get("version", ""),
        "description": data.get("description", ""),
        "system_prompt": data.get("system_prompt", ""),
        "tests": [
            {
                "id": t["id"],
                "title": t["title"],
                "category": t["category"],
                "prompt": t["prompt"],
            }
            for t in data.get("tests", [])
        ],
    }


def save_aggregated(aggregated: Dict, output_dir: str):
    """Save aggregated results to JSON and text with test data and timestamp."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load test data (prompts, system prompt) for archival
    test_data = load_test_data()
    
    # Bundle everything together
    bundle = {
        "timestamp": datetime.now().isoformat(),
        "test_data": test_data,
        "results": aggregated,
    }
    
    # Full JSON with timestamp
    json_name = f"aggregated_{ts}.json"
    with open(out / json_name, "w") as f:
        json.dump(bundle, f, indent=2)
    
    # Also save as aggregated.json (latest) for easy programmatic access
    with open(out / "aggregated.json", "w") as f:
        json.dump(bundle, f, indent=2)
    
    # Text summary - capture print output
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_aggregated_table(aggregated)
    sys.stdout = old_stdout
    
    # Also include system prompt and test list in text output
    header = f"""PinBench Multi-Run Results
Generated: {datetime.now().isoformat()}
Test file: test_cases.json v{test_data.get('version', '?')} — {test_data.get('description', '')}
Tests: {len(test_data.get('tests', []))}

System Prompt:
{test_data.get('system_prompt', 'N/A')}

Test Prompts:
"""
    for t in test_data.get("tests", []):
        header += f"  [{t['id']:>2}] {t['title']:<45} | {t['category']:<25} | {t['prompt']}\n"
    
    text_name = f"aggregated_{ts}.txt"
    with open(out / text_name, "w") as f:
        f.write(header)
        f.write("\n\n")
        f.write(buf.getvalue())
    
    # Also save as aggregated.txt (latest)
    with open(out / "aggregated.txt", "w") as f:
        f.write(header)
        f.write("\n\n")
        f.write(buf.getvalue())
    
    print(f"Aggregated results saved to {out / json_name}")
    print(f"Text summary saved to {out / text_name}")
    print(f"(also symlinked as aggregated.json / aggregated.txt for latest)")


def main():
    parser = argparse.ArgumentParser(description="Multi-run PinBench orchestrator")
    parser.add_argument("--configs", nargs="+", help="Config files to run")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per config")
    parser.add_argument("--output", default=None, help="Base output directory (default: results/multi_<timestamp>)")
    parser.add_argument("--aggregate", default=None, help="Re-aggregate existing multi-run directory (skips running)")
    parser.add_argument("--parallel", action="store_true", help="Run configs in parallel (not recommended)")
    
    args = parser.parse_args()
    
    if args.aggregate:
        # Re-aggregate existing runs
        base_dir = args.aggregate
        index_path = Path(base_dir) / "_index.json"
        if not index_path.exists():
            print(f"Error: {index_path} not found. Run without --aggregate first.")
            sys.exit(1)
        
        with open(index_path) as f:
            index = json.load(f)
        
        print(f"Re-aggregating {len(index['runs'])} runs from {base_dir}")
        
        results = []
        for r in index["runs"]:
            summary = load_summary(r["run_dir"])
            results.append({
                "config": r["config"],
                "config_name": r["config_name"],
                "run_idx": r["run_idx"],
                "summary": summary,
            })
        
        aggregated = aggregate(results)
        print_aggregated_table(aggregated)
        save_aggregated(aggregated, base_dir)
        return
    
    if not args.configs:
        parser.print_help()
        sys.exit(1)
    
    # Resolve config paths
    script_dir = Path(__file__).parent
    configs = []
    for c in args.configs:
        p = Path(c)
        if not p.is_absolute():
            p = script_dir / p
        if not p.exists():
            print(f"Config not found: {p}")
            sys.exit(1)
        configs.append(str(p))
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = args.output or str(script_dir / "results" / f"multi_{timestamp}")
    
    print(f"{'='*70}")
    print(f"  PINBENCH MULTI-RUN ORCHESTRATOR")
    print(f"{'='*70}")
    print(f"  Configs: {len(configs)}")
    for c in configs:
        print(f"    - {get_config_name(c)} ({c})")
    print(f"  Runs per config: {args.runs}")
    print(f"  Output: {base_output}")
    print(f"{'='*70}")
    
    results = run_all(configs, args.runs, base_output, args.parallel)
    
    print(f"\n{'='*70}")
    print(f"  ALL RUNS COMPLETE — AGGREGATING...")
    print(f"{'='*70}")
    
    aggregated = aggregate(results)
    print_aggregated_table(aggregated)
    save_aggregated(aggregated, base_output)
    
    print(f"\nDone. All data in: {base_output}")
    print(f"  - <config>/run_001/  (per-run results + raw_responses.json)")
    print(f"  - <config>/run_001/run.log  (terminal output)")
    print(f"  - _index.json  (run index)")
    print(f"  - aggregated.json  (cross-run aggregation)")
    print(f"  - aggregated.txt  (text summary)")


if __name__ == "__main__":
    main()
