"""
Grader for Pinyin-to-Character benchmark.
Checks structured model output against expected table + literal breakdown.
"""

import re
import json
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


def _strip_pinyin_tones(s: str) -> str:
    """Strip tone marks from pinyin (ā->a, á->a, ǎ->a, à->a, etc.)."""
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')


def _normalize_pinyin(s: str) -> str:
    """Normalize pinyin: lowercase, strip tones, remove spaces and non-alnum."""
    s = _strip_pinyin_tones(s.lower())
    return ''.join(c for c in s if c.isalnum())


_CHINESE_NORMALIZE = str.maketrans({
    '她': '他',
    '牠': '他',
    '妳': '你',
    '您': '你',
})


def _normalize_chinese(s: str) -> str:
    """Normalize gender-specific pronouns for comparison."""
    return s.translate(_CHINESE_NORMALIZE)


# Known Chinese compounds that the grader should accept as single entries.
# Maps compound -> list of individual characters (for backward compatibility).
_KNOWN_COMPOUNDS = {
    "明天": ["明", "天"],
    "北京": ["北", "京"],
    "我们": ["我", "们"],
    "老师": ["老", "师"],
    "朋友": ["朋", "友"],
    "早上": ["早", "上"],
    "什么": ["什", "么"],
    "知道": ["知", "道"],
    "怎么样": ["怎", "么", "样"],
    "你好": ["你", "好"],
    "早饭": ["早", "饭"],
    "出差": ["出", "差"],
}


def _jaccard_similarity(a: str, b: str) -> float:
    """Token overlap ratio for two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


@dataclass
class GradingResult:
    """Result of grading a single test case."""
    test_id: int
    test_title: str
    category: str
    passed: bool
    score: float
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
        d = self.to_dict()
        d.pop('raw_output', None)
        d.pop('matched_text', None)
        return d


@dataclass
class GradingCriteria:
    """Expected values for grading a single test case."""
    test_id: int
    title: str
    category: str
    table: Dict[str, str]  # {chinese, pinyin, english}
    literal: List[Dict[str, Any]]  # [{chinese, pinyin, keywords}, ...]

    WEIGHTS = {
        "phonetic_sandbox": 0.10,
        "table_exists":     0.10,
        "table_content":    0.40,
        "literal_section":  0.10,
        "literal_words":    0.30,
    }

    REQUIRED = [
        "table_content",
        "literal_words",
    ]

    @classmethod
    def from_test_case(cls, test_case: dict) -> "GradingCriteria":
        return cls(
            test_id=test_case["id"],
            title=test_case["title"],
            category=test_case["category"],
            table=test_case["table"],
            literal=test_case["literal"],
        )


class Grader:
    """Grades model output by checking sections, table, and word-by-word breakdown."""

    TABLE_HEADER_RE = re.compile(
        r'\|\s*Chinese\s*\|\s*Pinyin\s*\|\s*English\s*\|',
        re.IGNORECASE
    )
    TABLE_DATA_ROW_RE = re.compile(
        r'\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|'
    )
    TABLE_SEPARATOR_RE = re.compile(r'\|\s*:?-+:?\s*\|\s*:?-+:?\s*\|\s*:?-+:?\s*\|')

    def __init__(self, criteria_list: List[GradingCriteria]):
        self.criteria = {c.test_id: c for c in criteria_list}

    def grade(self, test_id: int, response: str, **metadata) -> GradingResult:
        criteria = self.criteria.get(test_id)
        if not criteria:
            return GradingResult(
                test_id=test_id, test_title="Unknown", category="unknown",
                passed=False, score=0.0,
                errors=[f"No criteria found for test {test_id}"],
                raw_output=response, **metadata
            )

        try:
            return self._grade(criteria, response, test_id, **metadata)
        except Exception as e:
            return GradingResult(
                test_id=test_id, test_title=criteria.title, category=criteria.category,
                passed=False, score=0.0,
                errors=[f"Grader crash: {e}"],
                raw_output=response, **metadata
            )

    def _grade(self, criteria, response, test_id, **metadata):
        details = {}
        matched_text = {}
        errors = []
        total_score = 0.0

        # 1) Phonetic Sandbox exists
        has_sandbox = "[Phonetic Sandbox]" in response
        details["phonetic_sandbox"] = has_sandbox
        if has_sandbox:
            total_score += criteria.WEIGHTS["phonetic_sandbox"]
        else:
            errors.append("Missing: [Phonetic Sandbox] section")

        # 2) Table exists with correct headers
        header_match = self.TABLE_HEADER_RE.search(response)
        table_exists = bool(header_match)
        details["table_exists"] = table_exists
        if table_exists:
            matched_text["table_exists"] = header_match.group(0)
            total_score += criteria.WEIGHTS["table_exists"]
        else:
            errors.append("Missing: table with 'Chinese | Pinyin | English' headers")

        # 3) Table content matches expected values
        table_content_ok = False
        if table_exists:
            table_content_ok = self._check_table(response, criteria)
        details["table_content"] = table_content_ok
        if table_content_ok:
            total_score += criteria.WEIGHTS["table_content"]
        else:
            errors.append("Table row doesn't match expected chinese/pinyin/english")

        # 4) Literal section exists
        literal_section = self._find_literal_section(response)
        details["literal_section"] = bool(literal_section)
        if literal_section:
            total_score += criteria.WEIGHTS["literal_section"]
        else:
            errors.append("Missing: word-by-word literal breakdown section")

        # 5) Literal words check
        literal_ok = False
        if literal_section:
            literal_ok = self._check_literal_words(literal_section, criteria)
        details["literal_words"] = literal_ok
        if literal_ok:
            total_score += criteria.WEIGHTS["literal_words"]
        else:
            missing = self._find_missing_literal_words(literal_section, criteria) if literal_section else ["(no literal breakdown section found)"]
            errors.append(f"Literal breakdown missing words: {', '.join(missing[:3])}")

        passed = details.get("table_content", False) and details.get("literal_words", False)
        score = min(total_score, 1.0)

        return GradingResult(
            test_id=test_id,
            test_title=criteria.title,
            category=criteria.category,
            passed=passed,
            score=round(score, 4),
            details=details,
            matched_text=matched_text,
            errors=errors,
            raw_output=response,
            **metadata
        )

    def grade_batch(self, responses: List[dict]) -> List[GradingResult]:
        results = []
        for resp in responses:
            results.append(self.grade(
                test_id=resp["test_id"],
                response=resp["response"],
                model=resp.get("model", ""),
                provider=resp.get("provider", ""),
                latency_ms=resp.get("latency_ms", 0),
                tokens_used=resp.get("tokens_used", 0),
                run_id=resp.get("run_id", ""),
            ))
        return results

    def _check_table(self, response: str, criteria: GradingCriteria) -> bool:
        """Check that the table contains the expected chinese, pinyin, english."""
        # Normalize expected values
        expected_chinese = criteria.table["chinese"].strip()
        expected_chinese_norm = _normalize_chinese(expected_chinese).replace(" ", "")
        expected_pinyin = criteria.table["pinyin"].strip().lower()
        expected_pinyin_norm = _normalize_pinyin(expected_pinyin)
        expected_english_opts = criteria.table["english"]
        if isinstance(expected_english_opts, str):
            expected_english_opts = [expected_english_opts]
        expected_english_opts = [e.strip().lower().rstrip('.,!?;:।\u3002\uff0c\uff01\uff1f\uff1b\uff1a') for e in expected_english_opts]

        lines = response.splitlines()
        in_table = False
        # Collect all data rows for multi-row fallback
        all_chinese = []
        all_pinyin = []
        all_english = []

        for line in lines:
            if self.TABLE_HEADER_RE.search(line):
                in_table = True
                continue
            if in_table and self.TABLE_SEPARATOR_RE.search(line):
                continue
            if in_table and self.TABLE_DATA_ROW_RE.search(line):
                m = self.TABLE_DATA_ROW_RE.search(line)
                chinese_col = m.group(1).strip()
                pinyin_col_raw = m.group(2).strip().lower()
                english_col = m.group(3).strip().lower()

                # Strip trailing punctuation
                pinyin_col = pinyin_col_raw.rstrip('.,!?;:।\u3002\uff0c\uff01\uff1f\uff1b\uff1a')
                english_col_clean = english_col.rstrip('.,!?;:।\u3002\uff0c\uff01\uff1f\uff1b\uff1a')

                # Collect for multi-row fallback
                all_chinese.append(chinese_col.strip())
                all_pinyin.append(pinyin_col_raw.strip())
                all_english.append(english_col_clean)

                # Try exact match on this row
                chinese_ok = _normalize_chinese(chinese_col) == expected_chinese_norm
                pinyin_ok = pinyin_col == expected_pinyin
                english_ok = english_col_clean in expected_english_opts
                if chinese_ok and pinyin_ok and english_ok:
                    return True

                # Fuzzy match: remove spaces from chinese
                if not chinese_ok:
                    chinese_ok = _normalize_chinese(chinese_col.replace(" ", "")) == expected_chinese_norm
                # Fuzzy match: normalize pinyin (lowercase, strip tones, remove spaces)
                if not pinyin_ok:
                    pinyin_ok = _normalize_pinyin(pinyin_col) == expected_pinyin_norm
                # Fuzzy match: check english via fallback list
                if not english_ok:
                    # Bidirectional substring check
                    english_ok = any(opt in english_col_clean for opt in expected_english_opts) or \
                                 any(english_col_clean in opt for opt in expected_english_opts)
                # Fuzzy match: token overlap (Jaccard similarity)
                if not english_ok:
                    english_ok = any(_jaccard_similarity(english_col_clean, opt) >= 0.3 for opt in expected_english_opts)
                if chinese_ok and pinyin_ok and english_ok:
                    return True
            elif in_table and line.strip() == "":
                break  # End of table

        # Multi-row fallback: concatenate all rows and check
        if all_chinese:
            combined_chinese = _normalize_chinese("".join(all_chinese).replace(" ", ""))
            combined_pinyin = _normalize_pinyin(" ".join(all_pinyin))
            combined_english = " ".join(all_english)
            english_match = any(opt in combined_english for opt in expected_english_opts) or \
                            any(combined_english in opt for opt in expected_english_opts) or \
                            any(_jaccard_similarity(combined_english, opt) >= 0.3 for opt in expected_english_opts)
            if (expected_chinese_norm in combined_chinese and
                expected_pinyin_norm in combined_pinyin and
                english_match):
                return True

        return False

    def _find_literal_section(self, response: str) -> Optional[str]:
        """Find the word-by-word breakdown section after the table."""
        # Look for common section headers
        section_headers = [
            "Word-by-word:", "Word breakdown:", "Literal breakdown:",
            "Word by word:", "Literal translation:", "Breakdown:",
        ]
        # Search from the bottom of the output (after the table)
        lines = response.splitlines()
        for i, line in enumerate(lines):
            for header in section_headers:
                if line.strip().lower().startswith(header.lower()):
                    remaining = "\n".join(lines[i:])
                    return remaining
        return None

    def _check_literal_words(self, section: str, criteria: GradingCriteria) -> bool:
        """Check that each literal word has chinese + pinyin + at least one keyword.
        Accepts compound words (e.g. 明天) or individual characters (明 + 天)."""
        section_lower = section.lower()
        section_norm = _normalize_chinese(section)
        # Extract all pinyins from parenthesized parts of bullet lines
        section_pinyins = _normalize_pinyin(" ".join(re.findall(r'\(([^)]+)\)', section)))
        missing = 0
        for word in criteria.literal:
            ch = word["chinese"]
            ch_norm = _normalize_chinese(ch)
            py = word["pinyin"].lower()
            kws = [k.lower() for k in word["keywords"]]

            # Check chinese (with pronoun normalization)
            has_chinese = ch in section or ch_norm in section_norm
            # Fallback: if this is a compound, check if its individual chars appear
            if not has_chinese and ch in _KNOWN_COMPOUNDS:
                chars = _KNOWN_COMPOUNDS[ch]
                has_chinese = all(c in section or _normalize_chinese(c) in section_norm for c in chars)
            # Fallback for single chars that might be part of a compound in output
            if not has_chinese and len(ch) == 1:
                # Check if this char appears within any known compound in the section
                for comp, chars in _KNOWN_COMPOUNDS.items():
                    if ch in chars and comp in section:
                        has_chinese = True
                        break
            if not has_chinese and len(ch) > 1:
                has_chinese = all(c in section or _normalize_chinese(c) in section_norm for c in ch)

            # Check pinyin (extracted from parenthesized parts)
            has_pinyin = _normalize_pinyin(py) in section_pinyins
            # Fallback: if compound pinyin not found, check individual char pinyins
            if not has_pinyin and ch in _KNOWN_COMPOUNDS:
                chars = _KNOWN_COMPOUNDS[ch]
                # Build expected individual pinyins for this compound's chars
                # by looking up the expected literal criteria for those chars
                char_pinyins = []
                for c in chars:
                    for w in criteria.literal:
                        if w["chinese"] == c:
                            char_pinyins.append(_normalize_pinyin(w["pinyin"].lower()))
                has_pinyin = all(cp in section_pinyins for cp in char_pinyins)

            # Check keyword (case-insensitive)
            has_keyword = any(kw in section_lower for kw in kws)
            # Fallback: check keywords from individual characters of this compound
            if not has_keyword and ch in _KNOWN_COMPOUNDS:
                chars = _KNOWN_COMPOUNDS[ch]
                for c in chars:
                    for w in criteria.literal:
                        if w["chinese"] == c:
                            if any(kw.lower() in section_lower for kw in w["keywords"]):
                                has_keyword = True
                                break
                    if has_keyword:
                        break

            if not (has_chinese and has_pinyin and has_keyword):
                missing += 1

        return missing == 0

    def _find_missing_literal_words(self, section: str, criteria: GradingCriteria) -> List[str]:
        """Return labels for words missing from the literal breakdown."""
        if not section:
            return ["(no literal breakdown section found)"]
        section_lower = section.lower()
        section_norm = _normalize_chinese(section)
        section_pinyins = _normalize_pinyin(" ".join(re.findall(r'\(([^)]+)\)', section)))
        missing = []
        for word in criteria.literal:
            ch = word["chinese"]
            ch_norm = _normalize_chinese(ch)
            py = word["pinyin"].lower()
            kws = [k.lower() for k in word["keywords"]]

            parts_ok = []
            has_chinese = ch in section or ch_norm in section_norm
            # Compound fallback
            if not has_chinese and ch in _KNOWN_COMPOUNDS:
                chars = _KNOWN_COMPOUNDS[ch]
                has_chinese = all(c in section or _normalize_chinese(c) in section_norm for c in chars)
            if not has_chinese and len(ch) == 1:
                for comp, chars in _KNOWN_COMPOUNDS.items():
                    if ch in chars and comp in section:
                        has_chinese = True
                        break
            if not has_chinese and len(ch) > 1:
                has_chinese = all(c in section or _normalize_chinese(c) in section_norm for c in ch)
            if not has_chinese:
                parts_ok.append("chinese")

            has_pinyin = _normalize_pinyin(py) in section_pinyins
            if not has_pinyin and ch in _KNOWN_COMPOUNDS:
                chars = _KNOWN_COMPOUNDS[ch]
                char_pinyins = []
                for c in chars:
                    for w in criteria.literal:
                        if w["chinese"] == c:
                            char_pinyins.append(_normalize_pinyin(w["pinyin"].lower()))
                has_pinyin = all(cp in section_pinyins for cp in char_pinyins)
            if not has_pinyin:
                parts_ok.append("pinyin")

            has_keyword = any(kw in section_lower for kw in kws)
            if not has_keyword and ch in _KNOWN_COMPOUNDS:
                chars = _KNOWN_COMPOUNDS[ch]
                for c in chars:
                    for w in criteria.literal:
                        if w["chinese"] == c:
                            if any(kw.lower() in section_lower for kw in w["keywords"]):
                                has_keyword = True
                                break
                    if has_keyword:
                        break
            if not has_keyword:
                parts_ok.append(f"keyword ({'/'.join(kws)})")

            if parts_ok:
                missing.append(f"{ch} ({py}) missing {', '.join(parts_ok)}")

        return missing


def create_grader_from_test_file(test_file: str) -> Grader:
    """Create a grader from a test cases JSON file."""
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    criteria = []
    for tc in data["tests"]:
        criteria.append(GradingCriteria.from_test_case(tc))

    return Grader(criteria)


@dataclass
class ResultsAggregator:
    """Aggregate and analyze grading results across multiple runs/models."""

    results: List[GradingResult] = field(default_factory=list)

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
        agg = {}
        for model in self.get_models():
            model_results = self.get_by_model(model)
            agg[model] = self._compute_aggregate(model_results, model=model)
        return agg

    def aggregate_by_provider(self) -> Dict[str, dict]:
        agg = {}
        for provider in self.get_providers():
            prov_results = self.get_by_provider(provider)
            agg[provider] = self._compute_aggregate(prov_results, provider=provider)
        return agg

    def aggregate_by_category(self) -> Dict[str, dict]:
        agg = {}
        for cat in self.get_categories():
            cat_results = self.get_by_category(cat)
            agg[cat] = self._compute_aggregate(cat_results, category=cat)
        return agg

    def aggregate_by_test(self) -> Dict[int, dict]:
        agg = {}
        for test_id in sorted(set(r.test_id for r in self.results)):
            test_results = self.get_by_test_id(test_id)
            agg[test_id] = self._compute_aggregate(test_results, test_id=test_id)
        return agg

    def aggregate_by_run(self) -> Dict[str, dict]:
        agg = {}
        for run_id in self.get_run_ids():
            run_results = self.get_by_run(run_id)
            agg[run_id] = self._compute_aggregate(run_results, run_id=run_id)
        return agg

    def _compute_aggregate(self, results: List[GradingResult], **keys) -> dict:
        if not results:
            return {
                **keys,
                "count": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "avg_latency_ms": 0.0,
                "avg_tokens": 0,
                "criterion_pass_rates": {},
                "categories": [],
                "models": [],
                "providers": [],
            }

        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]

        scores = [r.score for r in results]
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        tokens = [r.tokens_used for r in results if r.tokens_used > 0]

        criterion_rates = {}
        if results:
            all_criteria = set()
            for r in results:
                all_criteria.update(r.details.keys())

            for criterion in all_criteria:
                criterion_passed = sum(1 for r in results if r.details.get(criterion, False))
                criterion_rates[criterion] = criterion_passed / len(results)

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
            "categories": list(set(r.category for r in self.results)),
            "models": list(set(r.model for r in self.results)),
            "providers": list(set(r.provider for r in self.results)),
        }

    def summary(self) -> dict:
        return self._compute_aggregate(self.results)

    def export_json(self, filepath: str, include_raw: bool = False):
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

        if summary.get('criterion_pass_rates'):
            print(f"\nCriterion Pass Rates:")
            for crit, rate in sorted(summary['criterion_pass_rates'].items()):
                print(f"  {crit:25s}: {rate*100:5.1f}%")

        if len(self.get_models()) > 1:
            print(f"\nBy Model:")
            for model, agg in self.aggregate_by_model().items():
                print(f"  {model:40s}: {agg['passed']}/{agg['count']} ({agg['pass_rate']*100:.1f}%) score={agg['avg_score']:.2%}")

        if len(self.get_providers()) > 1:
            print(f"\nBy Provider:")
            for prov, agg in self.aggregate_by_provider().items():
                print(f"  {prov:20s}: {agg['passed']}/{agg['count']} ({agg['pass_rate']*100:.1f}%) score={agg['avg_score']:.2%}")

        print(f"\nBy Category:")
        for cat, agg in self.aggregate_by_category().items():
            print(f"  {cat:25s}: {agg['passed']}/{agg['count']} ({agg['pass_rate']*100:.1f}%)")


def create_aggregator() -> ResultsAggregator:
    """Create a new results aggregator."""
    return ResultsAggregator()
