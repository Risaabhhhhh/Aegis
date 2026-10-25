import json
import pytest
from base import Request, ContextChunk, Response, RequestProfile, EngineResult
from engines.base import BaseEngine
from ameva import weight_tables
from routing.execution_policy import build_action_plan, determine_skipped_engines
import aegis


class DummyEngine(BaseEngine):
    def __init__(self, name: str, score: float = 0.8):
        self.name = name
        self.score = score

    def evaluate(self, request, context, response, profile):
        return EngineResult(
            status="SUCCESS",
            score=self.score,
            findings=[],
            evidence=[],
            metadata={"engine_version": "1.0.0"},
            logs=[f"Dummy {self.name} ran"],
        )


@pytest.fixture(autouse=True)
def configure_weights(monkeypatch):
    monkeypatch.setitem(weight_tables.DEFAULT_WEIGHTS, "groundedness", 40.0)
    monkeypatch.setitem(weight_tables.DEFAULT_WEIGHTS, "dummy_hallucination", 35.0)
    monkeypatch.setitem(weight_tables.DEFAULT_WEIGHTS, "dummy_injection", 25.0)


def test_build_action_plan_excludes_groundedness_when_no_context():
    profile = RequestProfile(context_available=False)
    plan = build_action_plan(
        profile,
        request_id="test-no-ctx",
        selected_engines=["groundedness", "dummy_hallucination", "dummy_injection"],
    )

    assert "groundedness" not in plan.selected_engines
    assert set(plan.selected_engines) == {"dummy_hallucination", "dummy_injection"}
    assert "groundedness" not in plan.weights
    assert plan.weights == {
        "dummy_hallucination": 35.0,
        "dummy_injection": 25.0,
    }


def test_build_action_plan_includes_groundedness_when_context_available():
    profile = RequestProfile(context_available=True)
    plan = build_action_plan(
        profile,
        request_id="test-with-ctx",
        selected_engines=["groundedness", "dummy_hallucination", "dummy_injection"],
    )

    assert "groundedness" in plan.selected_engines
    assert set(plan.selected_engines) == {"groundedness", "dummy_hallucination", "dummy_injection"}
    assert plan.weights == {
        "groundedness": 40.0,
        "dummy_hallucination": 35.0,
        "dummy_injection": 25.0,
    }


def test_no_context_groundedness_skipped_and_coverage_stays_100_percent():
    req = Request(text="Tell me about quantum computing")
    ctx = []  # No context available
    res = Response(text="Quantum computing uses qubits.")

    dummy_groundedness = DummyEngine("groundedness", score=0.9)
    dummy_hallucination = DummyEngine("dummy_hallucination", score=0.8)
    dummy_injection = DummyEngine("dummy_injection", score=0.6)

    engines = [dummy_groundedness, dummy_hallucination, dummy_injection]
    result = aegis.evaluate(req, ctx, res, engines=engines)

    # 1. Groundedness must be skipped with reason 'missing_context'
    assert result.skipped_engines == {"groundedness": "missing_context"}
    assert "groundedness" not in result.engine_results
    assert "groundedness" not in result.failed_engines

    # 2. Both other engines ran and succeeded
    assert result.engine_results["dummy_hallucination"].status == "SUCCESS"
    assert result.engine_results["dummy_injection"].status == "SUCCESS"

    # 3. Coverage must stay 100% (1.0) because all selected engines succeeded
    assert result.coverage == 1.0

    # 4. Aggregate score is renormalized over the selected engines (35 and 25 weights)
    expected_score = (0.8 * 35.0 + 0.6 * 25.0) / (35.0 + 25.0)
    assert abs(result.aggregate_score - expected_score) < 1e-6
    assert result.status == "success"


def test_with_context_groundedness_not_skipped_and_evaluates_normally():
    req = Request(text="What is the capital of France?")
    ctx = [ContextChunk(text="Paris is the capital of France.", source_id="doc-1")]
    res = Response(text="Paris is the capital of France.")

    dummy_groundedness = DummyEngine("groundedness", score=0.95)
    dummy_hallucination = DummyEngine("dummy_hallucination", score=0.8)
    dummy_injection = DummyEngine("dummy_injection", score=0.7)

    engines = [dummy_groundedness, dummy_hallucination, dummy_injection]
    result = aegis.evaluate(req, ctx, res, engines=engines)

    # Groundedness is NOT skipped
    assert result.skipped_engines == {}
    assert "groundedness" in result.engine_results
    assert result.engine_results["groundedness"].status == "SUCCESS"
    assert result.engine_results["groundedness"].score == 0.95

    # All three engines succeeded -> coverage 1.0
    assert result.coverage == 1.0
    expected_score = (0.95 * 40.0 + 0.8 * 35.0 + 0.7 * 25.0) / (40.0 + 35.0 + 25.0)
    assert abs(result.aggregate_score - expected_score) < 1e-6
    assert result.status == "success"


def test_skipped_engines_serialization_in_to_dict_and_to_json():
    req = Request(text="Hello world")
    ctx = []
    res = Response(text="Hi!")

    dummy_groundedness = DummyEngine("groundedness", score=1.0)
    dummy_other = DummyEngine("dummy_hallucination", score=0.9)

    result = aegis.evaluate(req, ctx, res, engines=[dummy_groundedness, dummy_other])

    # Assert skipped_engines field on AegisResult
    assert result.skipped_engines == {"groundedness": "missing_context"}

    # Assert to_dict serialization contains skipped_engines
    d = result.to_dict()
    assert "skipped_engines" in d
    assert d["skipped_engines"] == {"groundedness": "missing_context"}

    # Assert to_json roundtrips with skipped_engines
    json_str = result.to_json()
    parsed = json.loads(json_str)
    assert parsed["skipped_engines"] == {"groundedness": "missing_context"}
