import pytest
from engines.base import BaseEngine
from engines.registry import validate_registry, REGISTERED_ENGINES
from routing.execution_policy import build_action_plan
from base import Request, Context, Response, RequestProfile, EngineResult


class DummyEngine(BaseEngine):
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, request, context, response, profile):
        return EngineResult(
            status="SUCCESS",
            score=1.0,
            findings=[],
            evidence=[],
            metadata={"engine_version": "1.0"},
            logs=[],
        )


def test_registered_engines_valid():
    """Ensure the default REGISTERED_ENGINES passes validation without errors."""
    validate_registry(REGISTERED_ENGINES)


def test_duplicate_engine_names_raises():
    engines = [
        DummyEngine("groundedness"),
        DummyEngine("groundedness"),
    ]
    with pytest.raises(ValueError, match="Duplicate engine name"):
        validate_registry(engines)


@pytest.mark.parametrize("bad_name", ["", "   ", None])
def test_empty_engine_name_raises(bad_name):
    engines = [
        DummyEngine(bad_name),
    ]
    with pytest.raises(ValueError, match="empty or invalid name"):
        validate_registry(engines)


def test_missing_weight_in_registry_raises():
    engines = [
        DummyEngine("unknown_engine_without_weight"),
    ]
    with pytest.raises(ValueError, match="no configured weight in ameva.weight_tables.DEFAULT_WEIGHTS"):
        validate_registry(engines)


def test_build_action_plan_missing_weight_raises():
    profile = RequestProfile()
    with pytest.raises(ValueError, match="no weight configured in ameva.weight_tables.DEFAULT_WEIGHTS"):
        build_action_plan(profile, "req-123", selected_engines=["unweighted_engine"])
