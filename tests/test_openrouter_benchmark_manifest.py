from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = PROJECT_ROOT / "benchmarks" / "openrouter" / "validate_manifest.py"


def _validator_module():
    spec = importlib.util.spec_from_file_location("openrouter_benchmark_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openrouter_benchmark_manifest_is_offline_valid():
    validator = _validator_module()
    assert validator.validate(validator.load_manifest()) == []
