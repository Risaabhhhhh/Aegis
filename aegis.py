import copy
import concurrent.futures
import json
import uuid
from dataclasses import dataclass, field
from base import Request, ContextChunk, Context, Response, RequestProfile, EngineResult, ActionPlan
from routing import request_understanding, execution_policy
from ameva import aggregator, coverage
from engines.registry import REGISTERED_ENGINES
from engines.base import BaseEngine

DEFAULT_ENGINE_TIMEOUT: float = 30.0

@dataclass
class AegisResult:
    aggregate_score: float | None
    coverage: float
    failed_engines: list[str]
    engine_results: dict[str, EngineResult]
    status: str
    request_id: str = ""
    skipped_engines: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "aggregate_score": self.aggregate_score,
            "coverage": self.coverage,
            "failed_engines": list(self.failed_engines),
            "skipped_engines": dict(self.skipped_engines),
            "status": self.status,
            "engine_results": {
                name: {
                    "status": res.status,
                    "score": res.score,
                    "findings": list(res.findings),
                    "evidence": list(res.evidence),
                    "metadata": dict(res.metadata),
                    "logs": list(res.logs),
                    "failure_reason": res.failure_reason,
                }
                for name, res in self.engine_results.items()
            },
        }

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

def sanitize_results(results: dict[str, EngineResult]) -> dict[str, EngineResult]:
    """Returns a NEW dict. Any SUCCESS result with score outside [0,1]
    becomes a new EngineResult with status=FAILED, score=None,
    failure_reason='invalid_score_range', and the ORIGINAL invalid value
    preserved at metadata['invalid_raw_score']. Never mutates inputs."""
    sanitized = {}
    for name, res in results.items():
        if res.status == "SUCCESS" and res.score is not None and not (0.0 <= res.score <= 1.0):
            new_res = copy.deepcopy(res)
            new_res.status = "FAILED"
            new_res.metadata["invalid_raw_score"] = new_res.score
            new_res.score = None
            new_res.failure_reason = "invalid_score_range"
            sanitized[name] = new_res
        else:
            sanitized[name] = res
    return sanitized

def evaluate(
    request: Request,
    context: Context,
    response: Response,
    engine_timeout: float = DEFAULT_ENGINE_TIMEOUT,
    engines: list[BaseEngine] | None = None,
) -> AegisResult:
    request_id = str(uuid.uuid4())
    
    # 1. Request Understanding (Routing)
    profile = request_understanding.build_request_profile(request, context, response)
    target_engines = engines if engines is not None else REGISTERED_ENGINES
    candidate_names = [e.name for e in target_engines]
    skipped_engines = execution_policy.determine_skipped_engines(profile, candidate_names)
    plan = execution_policy.build_action_plan(
        profile,
        request_id,
        selected_engines=candidate_names,
    )
    
    # 2. Run Engines
    results: dict[str, EngineResult] = {}
    
    for engine in target_engines:
        if engine.name in plan.selected_engines:
            timeout = getattr(engine, "timeout", engine_timeout)
            if timeout is None:
                timeout = engine_timeout

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(engine.run, request, context, response, profile)
            try:
                res = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                res = EngineResult(
                    status="FAILED",
                    score=None,
                    findings=[],
                    evidence=[],
                    metadata={},
                    logs=[f"Engine '{engine.name}' timed out after {timeout}s"],
                    failure_reason="timeout",
                )
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                res = EngineResult(
                    status="FAILED",
                    score=None,
                    findings=[],
                    evidence=[],
                    metadata={},
                    logs=[str(e)],
                    failure_reason="internal_error",
                )
                executor.shutdown(wait=False)
            else:
                executor.shutdown(wait=False)

            results[engine.name] = res
                
    # 3. Aggregation (AMEVA)
    results = sanitize_results(results)
    failed_engines = [name for name, res in results.items() if res.status == "FAILED"]
    aggregate_score, status = aggregator.calculate_aggregate(plan, results)
    coverage_score = coverage.calculate_coverage(plan, results)
    
    return AegisResult(
        request_id=request_id,
        aggregate_score=aggregate_score,
        coverage=coverage_score,
        failed_engines=failed_engines,
        skipped_engines=skipped_engines,
        engine_results=results,
        status=status,
    )
