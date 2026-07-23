"""
Regex-based grading system for Pinyin-to-Character benchmark.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime


@dataclass
class GradingResult:
    """Result of grading a single test case."""
    test_id: int
    test_title: str
    category: str
    passed: bool
    score: float  # 0.0 - 1.0
    max_score: float = 1.0
    details: Dict[str, bool] = field(default_factory=dict)
    matched_text: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_output: str = ""
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0
    run_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_dict_summary(self) -> dict:
        """Return summary dict without raw output."""
        d = self.to_dict()
        d.pop('raw_output', None)
        d.pop('matched_text', None)
        return d


@dataclass
class GradingCriteria:
    """Criteria for grading a test case."""
    test_id: int
    title: str
    category: str
    patterns: Dict[str, str] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    required_sections: List[str] = field(default_factory=list)
    max_score: float = 1.0
    
    DEFAULT_WEIGHTS = {
        "literal_section": 0.25,
        "table_chinese": 0.20,
        "table_pinyin": 0.20,
        "table_english": 0.15,
        "grammar_vocab": 0.10,
        "natural_alt": 0.10,
    }
    
    DEFAULT_REQUIRED = [
        "literal_section",
        "table_chinese", 
        "table_pinyin",
        "table_english",
        "grammar_vocab",
        "natural_alt",
    ]
    
    def __post_init__(self):
        if not self.weights:
            self.weights = self.DEFAULT_WEIGHTS.copy()
        if not self.required_sections:
            self.required_sections = self.DEFAULT_REQUIRED.copy()
    
    @classmethod
    def from_test_case(cls, test_case: dict) -> "GradingCriteria":
        """Create criteria from a test case dict."""
        patterns = test_case.get("regex_patterns", {})
        return cls(
            test_id=test_case["id"],
            title=test_case["title"],
            category=test_case["category"],
            patterns=patterns,
            weights=test_case.get("weights", {}),
            required_sections=test_case.get("required_sections", []),
            max_score=test_case.get("max_score", 1.0),
        )


class RegexGrader:
    """Regex-based grader for Pinyin-to-Character benchmark."""
    
    def __init__(self, criteria_list: List[GradingCriteria]):
        self.criteria = {c.test_id: c for c in criteria_list}
    
    def grade(self, test_id: int, response: str, **metadata) -> GradingResult:
        """Grade a single response."""
        criteria = self.criteria.get(test_id)
        if not criteria:
            return GradingResult(
                test_id=test_id,
                test_title="Unknown",
                category="unknown",
                passed=False,
                score=0.0,
                errors=[f"No criteria found for test {test_id}"],
                raw_output=response,
                **metadata
            )
        
        details = {}
        matched_text = {}
        errors = []
        warnings = []
        total_score = 0.0
        
        # Check each criterion
        for criterion, pattern in criteria.patterns.items():
            weight = criteria.weights.get(criterion, 0.0)
            
            # Skip non-pattern keys
            if criterion in ("title", "category"):
                continue
            
            try:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    details[criterion] = True
                    matched_text[criterion] = match.group(0)[:200]
                    total_score += weight
                else:
                    details[criterion] = False
                    matched_text[criterion] = ""
                    if criterion in criteria.required_sections:
                        errors.append(f"Missing required section: {criterion} (pattern: {pattern[:80]}...)")
                    else:
                        warnings.append(f"Optional section missing: {criterion}")
            except re.error as e:
                details[criterion] = False
                matched_text[criterion] = ""
                errors.append(f"Invalid regex for {criterion}: {e}")
        
        # Check required sections
        required_missing = [r for r in criteria.required_sections if not details.get(r, False)]
        passed = len(required_missing) == 0 and all(details.get(r, False) for r in criteria.required_sections)
        score = min(total_score / criteria.max_score, 1.0) if criteria.max_score > 0 else 0.0
        
        if required_missing:
            errors.append(f"Required sections missing: {', '.join(required_missing)}")
        
        return GradingResult(
            test_id=test_id,
            test_title=criteria.title,
            category=criteria.category,
            passed=passed,
            score=round(score, 4),
            max_score=criteria.max_score,
            details=details,
            matched_text=matched_text,
            errors=errors,
            warnings=warnings,
            raw_output=response,
            **metadata
        )
    
    def grade_batch(self, responses: List[dict]) -> List[GradingResult]:
        """Grade multiple responses."""
        results = []
        for resp in responses:
            result = self.grade(
                test_id=resp["test_id"],
                response=resp["response"],
                model=resp.get("model", ""),
                provider=resp.get("provider", ""),
                latency_ms=resp.get("latency_ms", 0),
                tokens_used=resp.get("tokens_used", 0),
                run_id=resp.get("run_id", ""),
            )
            results.append(result)
        return results


class ResultsAggregator:
    """Aggregate and analyze grading results across multiple runs/models."""
    
    def __init__(self):
        self.results: List[GradingResult] = []
    
    def add_results(self, results: List[GradingResult]):
        self.results.extend(results)
    
    def add_result(self, result: GradingResult):
        self.results.append(result)
    
    def get_by_model(self, model: str) -> List[GradingResult]:
        return [r for r in self.results if r.model == model]
    
    def get_by_provider(self, provider: str) -> List[GradingResult]:
        return [r for r in self.results if r.provider == provider]
    
    def get_by_test_id(self, test_id: int) -> List[GradingResult]:
        return [r for r in self.results if r.test_id == test_id]
    
    def get_by_category(self, category: str) -> List[GradingResult]:
        return [r for r in self.results if r.category == category]
    
    def get_by_run(self, run_id: str) -> List[GradingResult]:
        return [r for r in self.results if r.run_id == run_id]
    
    def get_models(self) -> list:
        return sorted(set(r.model for r in self.results))
    
    def get_providers(self) -> list:
        return sorted(set(r.provider for r in self.results))
    
    def get_categories(self) -> list:
        return sorted(set(r.category for r in self.results))
    
    def get_run_ids(self) -> list:
        return sorted(set(r.run_id for r in self.results))
    
    def aggregate_by_model(self) -> Dict[str, dict]:
        """Aggregate results by model."""
        agg = {}
        for model in self.get_models():
            model_results = self.get_by_model(model)
            agg[model] = self._compute_aggregate(model_results, model=model)
        return agg
    
    def aggregate_by_provider(self) -> Dict[str, dict]:
        """Aggregate results by provider."""
        agg = {}
        for provider in self.get_providers():
            prov_results = self.get_by_provider(provider)
            agg[provider] = self._compute_aggregate(prov_results, provider=provider)
        return agg
    
    def aggregate_by_category(self) -> Dict[str, dict]:
        """Aggregate results by category."""
        agg = {}
        for cat in self.get_categories():
            cat_results = self.get_by_category(cat)
            agg[cat] = self._compute_aggregate(cat_results, category=cat)
        return agg
    
    def aggregate_by_test(self) -> Dict[int, dict]:
        """Aggregate results by test case (across runs/models)."""
        agg = {}
        for test_id in sorted(set(r.test_id for r in self.results)):
            test_results = self.get_by_test_id(test_id)
            agg[test_id] = self._compute_aggregate(test_results, test_id=test_id)
        return agg
    
    def aggregate_by_run(self) -> Dict[str, dict]:
        """Aggregate results by run."""
        agg = {}
        for run_id in self.get_run_ids():
            run_results = self.get_by_run(run_id)
            agg[run_id] = self._compute_aggregate(run_results, run_id=run_id)
        return agg
    
    def _compute_aggregate(self, results: List[GradingResult], **keys) -> dict:
        """Compute aggregate statistics for a list of results."""
        if not results:
            return {"count": 0, **keys}
        
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        
        scores = [r.score for r in results]
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        tokens = [r.tokens_used for r in results if r.tokens_used > 0]
        
        # Per-criterion pass rates
        criterion_rates = {}
        if results:
            all_criteria = set()
            for r in results:
                all_criteria.update(r.details.keys())
            
            for criterion in all_criteria:
                criterion_passed = sum(1 for r in results if r.details.get(criterion, False))
                criterion_rates[criterion] = criterion_passed / len(results)
        
        # Category breakdown if single category
        categories = set(r.category for r in results)
        
        return {
            **keys,
            "count": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": len(passed) / len(results),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "avg_tokens": sum(tokens) / len(tokens) if tokens else 0,
            "criterion_pass_rates": criterion_rates,
            "categories": list(categories),
            "models": list(set(r.model for r in results)),
            "providers": list(set(r.provider for r in results)),
        }
    
    def summary(self) -> dict:
        """Get overall summary."""
        return self._compute_aggregate(self.results)
    
    def compare_models(self) -> dict:
        """Compare models side by side."""
        comparison = {}
        for model in self.get_models():
            model_results = self.get_by_model(model)
            comparison[model] = self._compute_aggregate(model_results, model=model)
        return comparison
    
    def compare_providers(self) -> dict:
        """Compare providers side by side."""
        comparison = {}
        for provider in self.get_providers():
            prov_results = self.get_by_provider(provider)
            comparison[provider] = self._compute_aggregate(prov_results, provider=provider)
        return comparison
    
    def category_breakdown(self, by_model: bool = False) -> dict:
        """Get category breakdown."""
        if by_model:
            breakdown = {}
            for model in self.get_models():
                model_results = self.get_by_model(model)
                breakdown[model] = {}
                for cat in self.get_categories():
                    cat_results = [r for r in model_results if r.category == cat]
                    breakdown[model][cat] = self._compute_aggregate(cat_results, model=model, category=cat)
            return breakdown
        else:
            breakdown = {}
            for cat in self.get_categories():
                cat_results = self.get_by_category(cat)
                breakdown[cat] = self._compute_aggregate(cat_results, category=cat)
            return breakdown
    
    def find_failures(self, model: str = None, provider: str = None, 
                      category: str = None, test_id: int = None) -> List[GradingResult]:
        """Find failed results with optional filters."""
        results = self.results
        if model:
            results = [r for r in results if r.model == model]
        if provider:
            results = [r for r in results if r.provider == provider]
        if category:
            results = [r for r in results if r.category == category]
        if test_id:
            results = [r for r in results if r.test_id == test_id]
        return [r for r in results if not r.passed]
    
    def export_json(self, filepath: str, include_raw: bool = False):
        """Export results to JSON."""
        data = {
            "summary": self.summary(),
            "by_model": self.aggregate_by_model(),
            "by_provider": self.aggregate_by_provider(),
            "by_category": self.aggregate_by_category(),
            "by_test": self.aggregate_by_test(),
            "by_run": self.aggregate_by_run(),
            "results": [r.to_dict() if include_raw else r.to_dict_summary() for r in self.results],
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def export_csv(self, filepath: str):
        """Export results to CSV."""
        import csv
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if not self.results:
                return
            
            fieldnames = [
                "test_id", "test_title", "category", "model", "provider", 
                "run_id", "passed", "score", "latency_ms", "tokens_used",
                "errors", "warnings", "timestamp"
            ] + [f"criterion_{c}" for c in 
                 sorted(set(c for r in self.results for c in r.details.keys()))]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in self.results:
                row = {
                    "test_id": r.test_id,
                    "test_title": r.test_title,
                    "category": r.category,
                    "model": r.model,
                    "provider": r.provider,
                    "run_id": r.run_id,
                    "passed": r.passed,
                    "score": r.score,
                    "latency_ms": r.latency_ms,
                    "tokens_used": r.tokens_used,
                    "errors": "; ".join(r.errors) if r.errors else "",
                    "warnings": "; ".join(r.warnings) if r.warnings else "",
                    "timestamp": r.timestamp,
                }
                for c in r.details:
                    row[f"criterion_{c}"] = r.details[c]
                writer.writerow(row)
    
    def print_summary(self):
        """Print a summary to console."""
        summary = self.summary()
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests:     {summary['count']}")
        print(f"Passed:          {summary['passed']} ({summary['pass_rate']*100:.1f}%)")
        print(f"Failed:          {summary['failed']}")
        print(f"Avg Score:       {summary['avg_score']:.2%}")
        print(f"Avg Latency:     {summary['avg_latency_ms']:.0f}ms")
        print(f"Avg Tokens:      {summary['avg_tokens']:.0f}")
        print(f"Models:          {', '.join(summary.get('models', []))}")
        print(f"Providers:       {', '.join(summary.get('providers', []))}")
        print(f"Categories:      {', '.join(summary.get('categories', []))}")
        
        # Per-criterion pass rates
        if summary.get('criterion_pass_rates'):
            print(f"\nCriterion Pass Rates:")
            for crit, rate in sorted(summary['criterion_pass_rates'].items()):
                print(f"  {crit:25s}: {rate*100:5.1f}%")
        
        # By model
        if len(self.get_models()) > 1:
            print(f"\nBy Model:")
            for model, agg in self.aggregate_by_model().items():
                print(f"  {model:40s}: {agg['passed']}/{agg['count']} ({agg['pass_rate']*100:.1f}%) score={agg['avg_score']:.2%}")
        
        # By provider
        if len(self.get_providers()) > 1:
            print(f"\nBy Provider:")
            for prov, agg in self.aggregate_by_provider().items():
                print(f"  {prov:20s}: {agg['passed']}/{agg['count']} ({agg['pass_rate']*100:.1f}%) score={agg['avg_score']:.2%}")
        
        # By category
        print(f"\nBy Category:")
        for cat, agg in self.aggregate_by_category().items():
            print(f"  {cat:25s}: {agg['passed']}/{agg['count']} ({agg['pass_rate']*100:.1f}%)")


def create_grader_from_test_file(test_file: str) -> RegexGrader:
    """Create a grader from a test cases JSON file."""
    import json
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    criteria = []
    for tc in data["tests"]:
        criteria.append(GradingCriteria.from_test_case(tc))
    
    return RegexGrader(criteria)


def create_aggregator() -> ResultsAggregator:
    """Create a new results aggregator."""
    return ResultsAggregator()