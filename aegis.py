import uuid
from dataclasses import dataclass
from base import Request, ContextChunk, Context, Response, RequestProfile, EngineResult, ActionPlan
from routing import request_understanding, execution_policy
from ameva import aggregator, coverage
from engines.registry import REGISTERED_ENGINES

@dataclass
class AegisResult:
    aggregate_score: float | None
    coverage: float
    failed_engines: list[str]
    engine_results: dict[str, EngineResult]
    status: str

def evaluate(request: Request, context: Context, response: Response) -> AegisResult:
    request_id = str(uuid.uuid4())
    
    # 1. Request Understanding (Routing)
    profile = request_understanding.build_request_profile(request, context, response)
    plan = execution_policy.build_action_plan(profile, request_id)
    
    # 2. Run Engines
    results: dict[str, EngineResult] = {}
    failed_engines = []
    
    for engine in REGISTERED_ENGINES:
        if engine.name in plan.selected_engines:
            res = engine.run(request, context, response, profile)
            results[engine.name] = res
            if res.status == "FAILED":
                failed_engines.append(engine.name)
                
    # 3. Aggregation (AMEVA)
    aggregate_score, status = aggregator.calculate_aggregate(plan, results)
    coverage_score = coverage.calculate_coverage(plan, results)
    
    return AegisResult(
        aggregate_score=aggregate_score,
        coverage=coverage_score,
        failed_engines=failed_engines,
        engine_results=results,
        status=status
    )
