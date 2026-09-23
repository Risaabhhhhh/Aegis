from base import ActionPlan, EngineResult

def calculate_aggregate(plan: ActionPlan, results: dict[str, EngineResult]) -> tuple[float | None, str]:
    if not plan.selected_engines:
        return None, "no_engines_selected"
        
    total_weight = 0.0
    weighted_score_sum = 0.0
    
    for engine_name in plan.selected_engines:
        res = results.get(engine_name)
        if not res:
            continue
            
        if res.status == "SUCCESS" and res.score is not None:
            weight = plan.weights.get(engine_name, 0.0)
            weighted_score_sum += res.score * weight
            total_weight += weight
            
    if total_weight == 0.0:
        return None, "all_engines_failed"
        
    return (weighted_score_sum / total_weight), "success"
