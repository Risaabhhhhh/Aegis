import pytest
from base import Request, ContextChunk, Response, RequestProfile, EngineResult
from engines.base import BaseEngine
from engines.registry import REGISTERED_ENGINES


class DummyConformanceSuccessEngine(BaseEngine):
    name = "dummy_success"

    def evaluate(self, request, context, response, profile):
        return EngineResult(
            status="SUCCESS",
            score=0.75,
            findings=["Finding 1"],
            evidence=[{"item": "evidence"}],
            metadata={"engine_version": "1.0.0", "model": "dummy-model-v1"},
            logs=["Evaluated dummy successfully"],
        )


class DummyConformanceRaisingEngine(BaseEngine):
    name = "dummy_raising"

    def evaluate(self, request, context, response, profile):
        raise RuntimeError("Engine evaluation exploded unexpectedly!")


class DummyConformanceFailingEngine(BaseEngine):
    name = "dummy_failing"

    def validate(self, request, context, response, profile):
        return "invalid_input"

    def evaluate(self, request, context, response, profile):
        return EngineResult(
            status="FAILED",
            score=None,
            findings=[],
            evidence=[],
            metadata={"engine_version": "1.0.0"},
            logs=[],
            failure_reason="invalid_input",
        )


ALL_TEST_ENGINES = list(REGISTERED_ENGINES) + [
    DummyConformanceSuccessEngine(),
    DummyConformanceRaisingEngine(),
    DummyConformanceFailingEngine(),
]


def assert_engine_result_contract(result: EngineResult):
    assert isinstance(result, EngineResult), f"Expected EngineResult, got {type(result)}"
    assert isinstance(result.findings, list), "findings must be a list"
    assert isinstance(result.evidence, list), "evidence must be a list"
    assert isinstance(result.logs, list), "logs must be a list"
    assert isinstance(result.metadata, dict), "metadata must be a dict"
    assert result.status in ("SUCCESS", "FAILED"), f"Invalid status: {result.status}"

    if result.status == "FAILED":
        assert result.score is None, f"Failed engine must have score None, got {result.score}"
        assert result.failure_reason is not None, "Failed engine must have failure_reason set"
    elif result.status == "SUCCESS":
        assert isinstance(result.score, float), f"Success engine score must be float, got {type(result.score)}"
        assert 0.0 <= result.score <= 1.0, f"Success engine score must be in [0, 1], got {result.score}"
        assert "engine_version" in result.metadata, "Success engine metadata must include 'engine_version'"


@pytest.mark.parametrize("engine", ALL_TEST_ENGINES, ids=lambda e: getattr(e, "name", str(e)))
def test_engine_conformance_normal_input(engine):
    req = Request(text="What is the capital of France?")
    ctx = [ContextChunk(text="Paris is the capital of France.", source_id="src-1")]
    res = Response(text="The capital of France is Paris.")
    profile = RequestProfile(context_available=True)

    result = engine.run(req, ctx, res, profile)
    assert_engine_result_contract(result)


@pytest.mark.parametrize("engine", ALL_TEST_ENGINES, ids=lambda e: getattr(e, "name", str(e)))
def test_engine_conformance_empty_response(engine):
    req = Request(text="What is the capital of France?")
    ctx = [ContextChunk(text="Paris is the capital of France.", source_id="src-1")]
    profile = RequestProfile(context_available=True)

    for empty_text in ["", "    ", "\n\t"]:
        res = Response(text=empty_text)
        result = engine.run(req, ctx, res, profile)
        assert_engine_result_contract(result)


@pytest.mark.parametrize("engine", ALL_TEST_ENGINES, ids=lambda e: getattr(e, "name", str(e)))
def test_engine_conformance_empty_context(engine):
    req = Request(text="What is the capital of France?")
    ctx = []
    res = Response(text="Paris is the capital of France.")
    profile = RequestProfile(context_available=False)

    result = engine.run(req, ctx, res, profile)
    assert_engine_result_contract(result)


@pytest.mark.parametrize("engine", ALL_TEST_ENGINES, ids=lambda e: getattr(e, "name", str(e)))
def test_engine_conformance_empty_both(engine):
    req = Request(text="What is the capital of France?")
    ctx = []
    res = Response(text="")
    profile = RequestProfile(context_available=False)

    result = engine.run(req, ctx, res, profile)
    assert_engine_result_contract(result)


def test_raising_engine_never_crashes_run():
    engine = DummyConformanceRaisingEngine()
    req = Request(text="hello")
    ctx = [ContextChunk(text="world")]
    res = Response(text="test")
    profile = RequestProfile()

    # run() must catch and return EngineResult with failure_reason='internal_error'
    result = engine.run(req, ctx, res, profile)
    assert_engine_result_contract(result)
    assert result.status == "FAILED"
    assert result.score is None
    assert result.failure_reason == "internal_error"
