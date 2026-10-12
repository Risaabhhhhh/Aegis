"""
Specific research anchor:
- Prompt Injection: a fine-tuned classifier approach rather than an LLM-judge, per Raina et al. EMNLP 2024 on LLM-judge adversarial vulnerability.

Benchmarks to validate against:
- Lakera's PINT, Open-Prompt-Injection, NotInject
"""
from base import Request, Context, Response, RequestProfile, EngineResult
from engines.base import BaseEngine

class PromptInjectionEngine(BaseEngine):
    name = "prompt_injection"
    
    def __init__(self):
        pass
        
    def validate(self, request: Request, context: Context, response: Response, profile: RequestProfile) -> str | None:
        return None
        
    def evaluate(self, request: Request, context: Context, response: Response, profile: RequestProfile) -> EngineResult:
        raise NotImplementedError("Prompt Injection logic to be written by assigned teammate.")
