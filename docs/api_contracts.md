# API Contracts

If you're about to change a field here, stop and ask whether it's actually required.

## Core Types (`base.py`)

### `Request`
```python
@dataclass
class Request:
    text: str
    metadata: dict
```

### `ContextChunk`
```python
@dataclass
class ContextChunk:
    text: str
    source_id: str | None
```
`Context` is `list[ContextChunk]`.

### `Response`
```python
@dataclass
class Response:
    text: str
    metadata: dict
```

### `RequestProfile`
```python
@dataclass
class RequestProfile:
    task_type: str | None
    domain: str | None
    risk_level: str | None
    context_available: bool
    metadata: dict
```

### `EngineResult`
```python
@dataclass
class EngineResult:
    status: Literal["SUCCESS", "FAILED"]
    score: float | None
    findings: list[str]
    evidence: list[dict]
    metadata: dict
    logs: list[str]
    failure_reason: Literal["invalid_input", "timeout", "internal_error", "missing_context", "empty_response", "invalid_score_range"] | None
```

### `ActionPlan`
```python
@dataclass
class ActionPlan:
    request_id: str
    selected_engines: list[str]
    weights: dict[str, float]
```

### `AdapterResponse` (`adapters/base.py`)
```python
@dataclass
class AdapterResponse:
    text: str
    latency_ms: float
    raw: dict
```

## Data Schemas

Evaluation datasets (such as `groundedness_seeds.json`) must conform to the [`eval_example_schema.json`](../data/schemas/eval_example_schema.json) standard structure.

## Engine Conformance Suite

Every new engine implementation **must** pass the reusable conformance suite in [`tests/test_engine_contract.py`](../tests/test_engine_contract.py).
The suite enforces the following contract rules:
1. `run()` must always return an `EngineResult` and must never raise any exception (even when `evaluate()` encounters an error or raises).
2. Status `"FAILED"` requires `score` to be `None` and `failure_reason` to be set to a valid literal reason.
3. Status `"SUCCESS"` requires `score` to be a `float` in `[0.0, 1.0]`.
4. `findings`, `evidence`, and `logs` must always be `list` instances; `metadata` must always be a `dict`.
5. Empty response (`""`, whitespace) and empty context (`[]`) must be handled gracefully without crashing.
6. Successful engine evaluations must include `"engine_version"` in their `metadata` dictionary.

## Engine Metadata Convention

Every engine's `metadata` dictionary must include:
- `"engine_version"`: Version string (e.g., `"0.1.0"` or `"1.0.0"`).
- Engine-specific model and configuration identifiers (e.g. `"model"`, `"threshold"`, tokenizer configs).

## Adapter Usage Policy

Engines that do **NOT** use an LLM must **never** touch the `adapters/` layer. Any engine that requires an external LLM call must implement the adapter first (inheriting from `BaseAdapter` in `adapters/base.py`), and requires an explicit PR note documenting the external dependency and model behavior.

