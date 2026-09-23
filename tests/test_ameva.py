import pytest
from base import ActionPlan, EngineResult
from ameva import aggregator, coverage

def test_single_engine_invariant():
    plan = ActionPlan(
        request_id="test-1",
        selected_engines=["groundedness", "hallucination", "prompt_injection"],
        weights={
            "groundedness": 40.0,
            "hallucination": 35.0,
            "prompt_injection": 25.0
        }
    )
    
    results = {
        "groundedness": EngineResult(status="SUCCESS", score=0.90, findings=[], evidence=[], metadata={}, logs=[]),
        "hallucination": EngineResult(status="FAILED", score=None, findings=[], evidence=[], metadata={}, logs=[], failure_reason="timeout"),
        "prompt_injection": EngineResult(status="FAILED", score=None, findings=[], evidence=[], metadata={}, logs=[], failure_reason="error")
    }
    
    agg, status = aggregator.calculate_aggregate(plan, results)
    cov = coverage.calculate_coverage(plan, results)
    
    assert agg == 0.90
    assert status == "success"
    assert cov == 40.0 / 100.0

def test_renormalization_on_partial_failure():
    plan = ActionPlan(
        request_id="test-2",
        selected_engines=["groundedness", "hallucination", "prompt_injection"],
        weights={
            "groundedness": 40.0,
            "hallucination": 35.0,
            "prompt_injection": 25.0
        }
    )
    
    results = {
        "groundedness": EngineResult(status="SUCCESS", score=0.90, findings=[], evidence=[], metadata={}, logs=[]),
        "hallucination": EngineResult(status="SUCCESS", score=0.70, findings=[], evidence=[], metadata={}, logs=[]),
        "prompt_injection": EngineResult(status="FAILED", score=None, findings=[], evidence=[], metadata={}, logs=[], failure_reason="error")
    }
    
    agg, status = aggregator.calculate_aggregate(plan, results)
    cov = coverage.calculate_coverage(plan, results)
    
    expected_agg = (0.90 * 40.0 + 0.70 * 35.0) / (40.0 + 35.0)
    
    assert abs(agg - expected_agg) < 1e-5
    assert status == "success"
    assert cov == 75.0 / 100.0

def test_zero_engines_edge_case():
    plan = ActionPlan(
        request_id="test-3",
        selected_engines=[],
        weights={
            "groundedness": 40.0,
            "hallucination": 35.0,
            "prompt_injection": 25.0
        }
    )
    
    results = {}
    
    agg, status = aggregator.calculate_aggregate(plan, results)
    cov = coverage.calculate_coverage(plan, results)
    
    assert agg is None
    assert status == "no_engines_selected"
    assert cov == 0.0

def test_all_engines_failed_edge_case():
    plan = ActionPlan(
        request_id="test-4",
        selected_engines=["groundedness", "hallucination"],
        weights={
            "groundedness": 40.0,
            "hallucination": 35.0,
        }
    )
    
    results = {
        "groundedness": EngineResult(status="FAILED", score=None, findings=[], evidence=[], metadata={}, logs=[], failure_reason="error"),
        "hallucination": EngineResult(status="FAILED", score=None, findings=[], evidence=[], metadata={}, logs=[], failure_reason="error")
    }
    
    agg, status = aggregator.calculate_aggregate(plan, results)
    cov = coverage.calculate_coverage(plan, results)
    
    assert agg is None
    assert status == "all_engines_failed"
    assert cov == 0.0
