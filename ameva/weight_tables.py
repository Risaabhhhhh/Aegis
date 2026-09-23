from base import RequestProfile

DEFAULT_WEIGHTS: dict[str, float] = {
    "groundedness": 40.0,
    "hallucination": 35.0,
    "prompt_injection": 25.0,
}

def get_weights(profile: RequestProfile | None = None) -> dict[str, float]:
    return DEFAULT_WEIGHTS  # v1 ignores profile on purpose
