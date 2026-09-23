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
    failure_reason: str | None
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
