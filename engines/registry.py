from engines.base import BaseEngine
from engines.groundedness import GroundednessEngine
# from engines.hallucination import HallucinationEngine
# from engines.prompt_injection import PromptInjectionEngine
from ameva import weight_tables


def validate_registry(engines: list[BaseEngine] | None = None) -> None:
    """Validate that every engine in the registry meets requirements:
    - Non-empty unique name
    - Registered weight in ameva.weight_tables.DEFAULT_WEIGHTS
    Raises ValueError on violation.
    """
    if engines is None:
        engines = REGISTERED_ENGINES

    seen_names = set()
    for engine in engines:
        name = getattr(engine, "name", None)
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(f"Engine {engine!r} has an empty or invalid name: {name!r}")
        if name in seen_names:
            raise ValueError(f"Duplicate engine name found in registry: {name!r}")
        seen_names.add(name)
        if name not in weight_tables.DEFAULT_WEIGHTS:
            raise ValueError(
                f"Engine {name!r} has no configured weight in ameva.weight_tables.DEFAULT_WEIGHTS"
            )


# =====================================================================
# Engine Registry
# Teammates: uncomment or add only your own engine line below.
# =====================================================================
REGISTERED_ENGINES: list[BaseEngine] = [
    GroundednessEngine(),
    # HallucinationEngine(),
    # PromptInjectionEngine(),
]

# Fail loudly at startup if registry is invalid
validate_registry(REGISTERED_ENGINES)

