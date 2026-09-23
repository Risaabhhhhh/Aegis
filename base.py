from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Request:
    text: str
    metadata: dict = field(default_factory=dict)

@dataclass
class ContextChunk:
    text: str
    source_id: str | None = None

Context = list[ContextChunk]

@dataclass
class Response:
    text: str
    metadata: dict = field(default_factory=dict)

@dataclass
class RequestProfile:
    task_type: str | None = None
    domain: str | None = None
    risk_level: str | None = None
    context_available: bool = False
    metadata: dict = field(default_factory=dict)

@dataclass
class EngineResult:
    status: Literal["SUCCESS", "FAILED"]
    score: float | None
    findings: list[str]
    evidence: list[dict]
    metadata: dict
    logs: list[str]
    failure_reason: str | None = None

@dataclass
class ActionPlan:
    request_id: str
    selected_engines: list[str]
    weights: dict[str, float]
