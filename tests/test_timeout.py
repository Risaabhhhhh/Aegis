import time
import pytest
from base import Request, ContextChunk, Response, EngineResult
from engines.base import BaseEngine
from ameva import weight_tables
import aegis


class DummySleepingEngine(BaseEngine):
    name = "dummy_sleep"

    def evaluate(self, request, context, response, profile):
        time.sleep(2.0)
        return EngineResult(
            status="SUCCESS",
            score=1.0,
            findings=[],
            evidence=[],
            metadata={"engine_version": "1.0"},
            logs=[],
        )


class DummyFastEngine(BaseEngine):
    name = "dummy_fast"

    def evaluate(self, request, context, response, profile):
        return EngineResult(
            status="SUCCESS",
            score=0.9,
            findings=[],
            evidence=[],
            metadata={"engine_version": "1.0"},
            logs=[],
        )


def test_engine_timeout_enforced_and_other_engines_continue(monkeypatch):
    # Configure weights for the dummy engines
    monkeypatch.setitem(weight_tables.DEFAULT_WEIGHTS, "dummy_sleep", 50.0)
    monkeypatch.setitem(weight_tables.DEFAULT_WEIGHTS, "dummy_fast", 50.0)

    sleep_engine = DummySleepingEngine()
    fast_engine = DummyFastEngine()

    req = Request(text="test query")
    ctx = [ContextChunk(text="sample context")]
    res = Response(text="sample response")

    # Set timeout to 0.1s so test finishes rapidly
    result = aegis.evaluate(
        request=req,
        context=ctx,
        response=res,
        engine_timeout=0.1,
        engines=[sleep_engine, fast_engine],
    )

    # 1. Sleeping engine timed out
    sleep_res = result.engine_results["dummy_sleep"]
    assert sleep_res.status == "FAILED"
    assert sleep_res.failure_reason == "timeout"
    assert sleep_res.score is None
    assert "dummy_sleep" in result.failed_engines

    # 2. Fast engine still ran and succeeded
    fast_res = result.engine_results["dummy_fast"]
    assert fast_res.status == "SUCCESS"
    assert fast_res.score == 0.9
    assert "dummy_fast" not in result.failed_engines

    # 3. Aggregation properly renormalized around the succeeding engine
    assert result.aggregate_score == 0.9
    assert result.coverage == 0.5
