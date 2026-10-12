import itertools
import pytest
from base import Request, ContextChunk, Response, RequestProfile, EngineResult
from engines.base import BaseEngine
from ameva import weight_tables
from routing.execution_policy import build_action_plan
import aegis


class FixedOutputDummyEngine(BaseEngine):
    def __init__(self, name: str, score: float | None = 0.5, status: str = "SUCCESS", failure_reason: str | None = None, raise_error: bool = False):
        self.name = name
        self.score = score
        self.status = status
        self.failure_reason = failure_reason
        self.raise_error = raise_error

    def evaluate(self, request, context, response, profile):
        if self.raise_error:
            raise RuntimeError(f"Engine {self.name} crashed intentionally!")
        return EngineResult(
            status=self.status,
            score=self.score,
            findings=[],
            evidence=[],
            metadata={"engine_version": "1.0.0", "engine_id": self.name},
            logs=[f"Dummy {self.name} executed with status {self.status}"],
            failure_reason=self.failure_reason,
        )


@pytest.fixture(autouse=True)
def setup_dummy_weights(monkeypatch):
    weights = {
        "dummy_a": 40.0,
        "dummy_b": 35.0,
        "dummy_c": 25.0,
    }
    for k, v in weights.items():
        monkeypatch.setitem(weight_tables.DEFAULT_WEIGHTS, k, v)


def test_two_of_three_succeeding_combinations_and_renormalization():
    req = Request(text="test")
    ctx = [ContextChunk(text="context")]
    res = Response(text="response")

    # Combination 1: A and B succeed, C fails
    eng_a = FixedOutputDummyEngine("dummy_a", score=0.8, status="SUCCESS")
    eng_b = FixedOutputDummyEngine("dummy_b", score=0.6, status="SUCCESS")
    eng_c = FixedOutputDummyEngine("dummy_c", score=None, status="FAILED", failure_reason="internal_error")

    result1 = aegis.evaluate(req, ctx, res, engines=[eng_a, eng_b, eng_c])
    expected_agg1 = (0.8 * 40.0 + 0.6 * 35.0) / (40.0 + 35.0)
    expected_cov1 = (40.0 + 35.0) / 100.0

    assert abs(result1.aggregate_score - expected_agg1) < 1e-6
    assert abs(result1.coverage - expected_cov1) < 1e-6
    assert result1.failed_engines == ["dummy_c"]

    # Reverse order for Combination 1 to verify order-independence
    result1_rev = aegis.evaluate(req, ctx, res, engines=[eng_c, eng_b, eng_a])
    assert abs(result1_rev.aggregate_score - expected_agg1) < 1e-6
    assert abs(result1_rev.coverage - expected_cov1) < 1e-6

    # Combination 2: A and C succeed, B fails
    eng_b_fail = FixedOutputDummyEngine("dummy_b", score=None, status="FAILED", failure_reason="internal_error")
    eng_c_succ = FixedOutputDummyEngine("dummy_c", score=0.4, status="SUCCESS")

    result2 = aegis.evaluate(req, ctx, res, engines=[eng_a, eng_b_fail, eng_c_succ])
    expected_agg2 = (0.8 * 40.0 + 0.4 * 25.0) / (40.0 + 25.0)
    expected_cov2 = (40.0 + 25.0) / 100.0

    assert abs(result2.aggregate_score - expected_agg2) < 1e-6
    assert abs(result2.coverage - expected_cov2) < 1e-6
    assert result2.failed_engines == ["dummy_b"]

    # Reverse order for Combination 2
    result2_rev = aegis.evaluate(req, ctx, res, engines=[eng_c_succ, eng_b_fail, eng_a])
    assert abs(result2_rev.aggregate_score - expected_agg2) < 1e-6
    assert abs(result2_rev.coverage - expected_cov2) < 1e-6


def test_all_three_succeed():
    req = Request(text="test")
    ctx = [ContextChunk(text="context")]
    res = Response(text="response")

    eng_a = FixedOutputDummyEngine("dummy_a", score=0.8)
    eng_b = FixedOutputDummyEngine("dummy_b", score=0.6)
    eng_c = FixedOutputDummyEngine("dummy_c", score=0.4)

    result = aegis.evaluate(req, ctx, res, engines=[eng_a, eng_b, eng_c])
    expected_agg = (0.8 * 40.0 + 0.6 * 35.0 + 0.4 * 25.0) / 100.0

    assert abs(result.aggregate_score - expected_agg) < 1e-6
    assert result.coverage == 1.0
    assert result.status == "success"
    assert result.failed_engines == []


def test_all_three_fail():
    req = Request(text="test")
    ctx = [ContextChunk(text="context")]
    res = Response(text="response")

    eng_a = FixedOutputDummyEngine("dummy_a", score=None, status="FAILED", failure_reason="internal_error")
    eng_b = FixedOutputDummyEngine("dummy_b", score=None, status="FAILED", failure_reason="internal_error")
    eng_c = FixedOutputDummyEngine("dummy_c", score=None, status="FAILED", failure_reason="internal_error")

    result = aegis.evaluate(req, ctx, res, engines=[eng_a, eng_b, eng_c])

    assert result.aggregate_score is None
    assert result.coverage == 0.0
    assert result.status == "all_engines_failed"
    assert sorted(result.failed_engines) == ["dummy_a", "dummy_b", "dummy_c"]


def test_out_of_range_score_sanitized_to_failed():
    req = Request(text="test")
    ctx = [ContextChunk(text="context")]
    res = Response(text="response")

    # eng_c returns invalid score 1.5
    eng_a = FixedOutputDummyEngine("dummy_a", score=0.8, status="SUCCESS")
    eng_b = FixedOutputDummyEngine("dummy_b", score=0.6, status="SUCCESS")
    eng_c = FixedOutputDummyEngine("dummy_c", score=1.5, status="SUCCESS")

    result = aegis.evaluate(req, ctx, res, engines=[eng_a, eng_b, eng_c])

    # sanitize_results turns eng_c into FAILED
    c_res = result.engine_results["dummy_c"]
    assert c_res.status == "FAILED"
    assert c_res.score is None
    assert c_res.failure_reason == "invalid_score_range"
    assert c_res.metadata["invalid_raw_score"] == 1.5
    assert "dummy_c" in result.failed_engines

    # eng_a and eng_b are unaffected
    assert result.engine_results["dummy_a"].status == "SUCCESS"
    assert result.engine_results["dummy_b"].status == "SUCCESS"

    expected_agg = (0.8 * 40.0 + 0.6 * 35.0) / (40.0 + 35.0)
    assert abs(result.aggregate_score - expected_agg) < 1e-6
    assert abs(result.coverage - (75.0 / 100.0)) < 1e-6


def test_action_plan_weights_scoped_to_registered_engines():
    profile = RequestProfile()
    plan = build_action_plan(
        profile,
        request_id="plan-scoped-1",
        selected_engines=["dummy_a", "dummy_b", "dummy_c"],
    )

    assert set(plan.selected_engines) == {"dummy_a", "dummy_b", "dummy_c"}
    assert plan.weights == {
        "dummy_a": 40.0,
        "dummy_b": 35.0,
        "dummy_c": 25.0,
    }


def test_engine_raising_exception_does_not_stop_others():
    req = Request(text="test")
    ctx = [ContextChunk(text="context")]
    res = Response(text="response")

    eng_a = FixedOutputDummyEngine("dummy_a", score=0.8, status="SUCCESS")
    eng_b = FixedOutputDummyEngine("dummy_b", raise_error=True)
    eng_c = FixedOutputDummyEngine("dummy_c", score=0.4, status="SUCCESS")

    result = aegis.evaluate(req, ctx, res, engines=[eng_a, eng_b, eng_c])

    # eng_b failed cleanly due to internal_error exception catch
    b_res = result.engine_results["dummy_b"]
    assert b_res.status == "FAILED"
    assert b_res.score is None
    assert b_res.failure_reason == "internal_error"
    assert "dummy_b" in result.failed_engines

    # eng_a and eng_c succeeded unaffected
    assert result.engine_results["dummy_a"].status == "SUCCESS"
    assert result.engine_results["dummy_a"].score == 0.8
    assert result.engine_results["dummy_c"].status == "SUCCESS"
    assert result.engine_results["dummy_c"].score == 0.4

    expected_agg = (0.8 * 40.0 + 0.4 * 25.0) / (40.0 + 25.0)
    assert abs(result.aggregate_score - expected_agg) < 1e-6


def test_coverage_and_aggregate_identical_regardless_of_order():
    req = Request(text="test")
    ctx = [ContextChunk(text="context")]
    res = Response(text="response")

    base_engines = [
        FixedOutputDummyEngine("dummy_a", score=0.8),
        FixedOutputDummyEngine("dummy_b", score=0.6),
        FixedOutputDummyEngine("dummy_c", score=0.4),
    ]

    expected_agg = (0.8 * 40.0 + 0.6 * 35.0 + 0.4 * 25.0) / 100.0
    expected_cov = 1.0

    for perm in itertools.permutations(base_engines):
        result = aegis.evaluate(req, ctx, res, engines=list(perm))
        assert abs(result.aggregate_score - expected_agg) < 1e-9
        assert abs(result.coverage - expected_cov) < 1e-9
