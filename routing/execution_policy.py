from base import RequestProfile, ActionPlan
from engines.registry import REGISTERED_ENGINES
from ameva import weight_tables

def build_action_plan(
    profile: RequestProfile,
    request_id: str,
    selected_engines: list[str] | None = None,
) -> ActionPlan:
    if selected_engines is None:
        selected_engines = [e.name for e in REGISTERED_ENGINES]

    weights: dict[str, float] = {}
    for name in selected_engines:
        if name not in weight_tables.DEFAULT_WEIGHTS:
            raise ValueError(
                f"Selected engine '{name}' has no weight configured in ameva.weight_tables.DEFAULT_WEIGHTS"
            )
        weights[name] = weight_tables.DEFAULT_WEIGHTS[name]

    return ActionPlan(
        request_id=request_id,
        selected_engines=selected_engines,
        weights=weights,
    )
