from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "voice" / "live_benchmark.py"


def _load():
    spec = importlib.util.spec_from_file_location("voice_live_benchmark", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wer_and_critical_semantic_equivalents():
    module = _load()
    assert module.wer("Привет, ёлка!", "привет елка") == 0
    assert module.critical_hit("23 октября", "Встретимся двадцать третьего октября")
    assert module.critical_hit("350000", "Бюджет триста пятьдесят тысяч")
    assert module.wer(
        "Двадцать третьего октября в восемнадцать сорок, триста пятьдесят тысяч.",
        "23 октября в 18:40, 350 тысяч.",
    ) == 0


def test_budget_is_durable_and_fail_closed(tmp_path):
    module = _load()
    path = tmp_path / "ledger.json"
    ledger = module.BudgetLedger(path, 1.0, 0.05)
    ledger.reserve(request_id="a", estimated_usd=0.04, route="p:m")
    ledger.settle("a", status="settled", observed_usd=0.01)
    reloaded = module.BudgetLedger(path, 1.0, 0.05)
    assert reloaded.reserved_total == pytest.approx(0.04)
    assert json.loads(path.read_text())["reservations"][0]["observed_usd"] == 0.01
    with pytest.raises(module.BudgetError):
        reloaded.reserve(request_id="b", estimated_usd=0.051, route="p:m")


def test_product_trace_has_no_content_fields(tmp_path):
    module = _load()
    path = tmp_path / "trace.jsonl"
    module._trace(path, request_id="r", purpose="stt", provider="p", model="m",
                  route_meta={"language": "ru"}, input_count=10, output_count=3,
                  latency_ms=5, status="success", cost_usd=0.001)
    record = json.loads(path.read_text())
    assert not ({"audio", "transcript", "text", "prompt", "credential", "path"} & record.keys())
    assert record["retry_count"] == 0


def test_stt_candidate_stops_after_route_auth_failure(tmp_path, monkeypatch):
    module = _load()
    module.ROOT = tmp_path
    audio = tmp_path / "case.ogg"
    audio.write_bytes(b"ogg")
    manifest = {
        "stt_candidates": [{
            "id": "google", "provider": "google_cloud", "model": "chirp_3",
            "language": "ru-RU", "project_id": "p", "region": "eu",
            "public_price_usd_per_minute": 0.01,
        }],
        "stt_cases": [
            {"id": "one", "text": "one", "critical": []},
            {"id": "two", "text": "two", "critical": []},
        ],
    }
    index = {"entries": [
        {"case_id": "one", "telegram_derivative": {"path": "case.ogg", "duration_seconds": 1}},
        {"case_id": "two", "telegram_derivative": {"path": "case.ogg", "duration_seconds": 1}},
    ]}
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        request = module.httpx.Request("POST", "https://example.test")
        response = module.httpx.Response(401, request=request)
        raise module.httpx.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setattr(module, "_stt_call", fail)
    ledger = module.BudgetLedger(tmp_path / "ledger.json", 1.0, 0.05)
    module.run_stt(manifest, index, tmp_path / "out", ledger, {})
    assert len(calls) == 1
    rows = (tmp_path / "out/stt_results.jsonl").read_text().splitlines()
    assert len(rows) == 1
