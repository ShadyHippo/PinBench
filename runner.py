"""
Data types used by the benchmark runner.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TestCase:
    """Single test case for the benchmark."""
    id: int
    title: str
    category: str
    prompt: str
    table: Dict[str, str]
    literal: List[Dict[str, Any]]

    @classmethod
    def from_dict(cls, data: dict) -> "TestCase":
        return cls(**data)


@dataclass
class BenchmarkConfig:
    """Configuration loaded from test_cases.json."""
    test_cases: List[TestCase]
    system_prompt: str

    @classmethod
    def from_json(cls, filepath: str) -> "BenchmarkConfig":
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        test_cases = [TestCase.from_dict(tc) for tc in data["tests"]]
        return cls(test_cases=test_cases, system_prompt=data["system_prompt"])


@dataclass
class RunConfig:
    """Configuration for a benchmark run from a YAML/JSON config file."""
    name: str
    test_file: str = "test_cases.json"
    providers: List[Dict[str, Any]] = field(default_factory=list)
    runs_per_model: int = 1
    max_workers: int = 4
    output_dir: str = "results"
    save_raw_responses: bool = True
    save_responses_json: bool = True
    save_results_json: bool = True
    save_results_csv: bool = True
    print_progress: bool = True
    timeout_per_request: int = 300
    retry_failed: int = 0
    filter_categories: Optional[List[str]] = None
    filter_test_ids: Optional[List[int]] = None
    system_prompt_override: Optional[str] = None
    provider_kwargs: Dict[str, Any] = field(default_factory=dict)
    verbose: bool = False
    
    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: str) -> "RunConfig":
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)