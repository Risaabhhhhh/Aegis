from base import RequestProfile, ActionPlan
from engines.registry import REGISTERED_ENGINES
from ameva import weight_tables

def build_action_plan(profile: RequestProfile, request_id: str) -> ActionPlan:
    selected = [e.name for e in REGISTERED_ENGINES]
    return ActionPlan(
        request_id=request_id,
        selected_engines=selected,
        weights={k: v for k, v in weight_tables.DEFAULT_WEIGHTS.items() if k in selected},
    )
