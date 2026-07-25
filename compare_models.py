#!/usr/bin/env python3
"""Compare old vs new Gemma/Qwen results."""
import json

runs = {
    "gemma_old": "results/20260724_155934/summary.json",
    "gemma_new": "results/20260725_161733/summary.json",
    "qwen_old": "results/20260724_164553/summary.json",
    "qwen_new": "results/20260725_160537/summary.json",
}

data = {}
for key, path in runs.items():
    with open(path) as f:
        data[key] = json.load(f)

for model, label_old, label_new in [
    ("gemma", "gemma_old", "gemma_new"),
    ("qwen", "qwen_old", "qwen_new"),
]:
    old = data[label_old]
    new = data[label_new]
    
    print(f"{'='*60}")
    print(f"  {model.upper()}")
    print(f"{'='*60}")
    print(f"{'':<20} {'Old (v1.2)':>14} {'New (v2.0)':>14} {'Delta':>10}")
    print(f"{'Pass Rate':<20} {old['summary']['pass_rate']*100:>13.1f}% {new['summary']['pass_rate']*100:>13.1f}% {new['summary']['pass_rate']*100 - old['summary']['pass_rate']*100:>+9.1f}%")
    print(f"{'Avg Score':<20} {old['summary']['avg_score']*100:>13.1f}% {new['summary']['avg_score']*100:>13.1f}% {new['summary']['avg_score']*100 - old['summary']['avg_score']*100:>+9.1f}%")
    print()
    
    bt_old = old.get('by_test', {})
    bt_new = new.get('by_test', {})
    print(f"{'Dirty Test':<12} {'Old':>8} {'New':>8} {'Delta':>8}")
    print("-" * 40)
    for tid in ['27','29','31','33','35','37','39','43','45','46']:
        o = bt_old.get(tid, {})
        n = bt_new.get(tid, {})
        o_pr = o.get('pass_rate', 0)*100 if o else 0
        n_pr = n.get('pass_rate', 0)*100 if n else 0
        print(f"{tid:<12} {o_pr:>7.0f}% {n_pr:>7.0f}% {n_pr-o_pr:>+7.0f}%")
    print()
