#!/usr/bin/env python3
"""Compare v1.2 (regraded) vs v1.3 benchmark results."""
import json

with open('results/v1.2_regraded/regraded.json') as f:
    regraded = json.load(f)

with open('results/v1.3_clean/aggregated.json') as f:
    v3 = json.load(f)

header = f"{'Model':<15} {'Old Prompt (regraded)':>33} {'New Prompt (v1.3)':>33}"
print(header)
print(f"{'':<15} {'Pass Rate':>14} {'Avg Score':>16} {'Pass Rate':>14} {'Avg Score':>16}")
print('-' * 80)

for model in ['deepseek', 'nemotron', 'north', 'ling', 'laguna']:
    rr = regraded['results'].get(model, {})
    v3d = v3['results'].get(model, {})
    
    rr_pr = f"{rr.get('pass_rate_mean', 0)*100:.1f}%"
    rr_sc = f"{rr.get('avg_score_mean', 0)*100:.1f}%"
    
    v3_pr = f"{v3d.get('pass_rate_mean', 0)*100:.1f}%" if v3d.get('pass_rate_mean', 0) > 0 or v3d.get('n_runs', 0) == 0 else "N/A (429)"
    v3_sc = f"{v3d.get('avg_score_mean', 0)*100:.1f}%" if v3d.get('avg_score_mean', 0) > 0 or v3d.get('n_runs', 0) == 0 else "N/A (429)"
    
    print(f"{model:<15} {rr_pr:>14} {rr_sc:>16} {v3_pr:>14} {v3_sc:>16}")

print()
print("Notes:")
print("  - Regraded = old v1.2 prompt graded with current v1.3 grader")
print("  - v1.3 = new compound-requiring prompt + v1.3 grader")
print("  - Ling and Laguna: free tier exhausted (429), no v1.3 new data")
print("  - North: essentially unusable on either prompt")
