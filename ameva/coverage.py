from base import ActionPlan, EngineResult

def calculate_coverage(plan: ActionPlan, results: dict[str, EngineResult]) -> float:
    if not plan.selected_engines:
        return 0.0
        
    total_weight = 0.0
    success_weight = 0.0
    
    for engine_name in plan.selected_engines:
        weight = plan.weights.get(engine_name, 0.0)
        total_weight += weight
        
        res = results.get(engine_name)
        if res and res.status == "SUCCESS":
            success_weight += weight
            
    if total_weight == 0.0:
        return 0.0
        
    return (success_weight / total_weight)
