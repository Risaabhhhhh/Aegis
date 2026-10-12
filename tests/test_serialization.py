import json
import pytest
from base import EngineResult, Request, ContextChunk, Response
from engines.base import BaseEngine
import aegis


class DummySerializableEngine(BaseEngine):
    name = "dummy_serializable"

    def evaluate(self, request, context, response, profile):
        return EngineResult(
            status="SUCCESS",
            score=0.95,
            findings=["Finding A", "Finding B"],
            evidence=[{"source": "chunk-1", "match": "exact"}],
            metadata={"engine_version": "1.0.0", "model": "test-v1"},
            logs=["Log 1", "Log 2"],
            failure_reason=None,
        )


def test_aegis_result_to_dict_and_to_json(monkeypatch):
    from ameva import weight_tables
    monkeypatch.setitem(weight_tables.DEFAULT_WEIGHTS, "dummy_serializable", 100.0)

    engine = DummySerializableEngine()
    req = Request(text="test query")
    ctx = [ContextChunk(text="some context", source_id="src-1")]
    res = Response(text="some answer")

    result = aegis.evaluate(req, ctx, res, engines=[engine])

    # 1. Assert request_id exists and is non-empty
    assert result.request_id is not None
    assert len(result.request_id) > 0

    # 2. Assert to_dict structure
    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["request_id"] == result.request_id
    assert d["aggregate_score"] == 0.95
    assert d["coverage"] == 1.0
    assert d["failed_engines"] == []
    assert d["status"] == "success"

    # Check EngineResult serialization has all fields
    assert "dummy_serializable" in d["engine_results"]
    eng_d = d["engine_results"]["dummy_serializable"]
    assert eng_d["status"] == "SUCCESS"
    assert eng_d["score"] == 0.95
    assert eng_d["findings"] == ["Finding A", "Finding B"]
    assert eng_d["evidence"] == [{"source": "chunk-1", "match": "exact"}]
    assert eng_d["metadata"] == {"engine_version": "1.0.0", "model": "test-v1"}
    assert eng_d["logs"] == ["Log 1", "Log 2"]
    assert eng_d["failure_reason"] is None

    # 3. Assert to_json produces valid JSON that round-trips
    json_str = result.to_json(indent=2)
    parsed = json.loads(json_str)
    assert parsed == d
