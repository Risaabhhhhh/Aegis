from abc import ABC, abstractmethod

from base import Request, Context, Response, RequestProfile, EngineResult

class BaseEngine(ABC):
    name: str

    def validate(self, request: Request, context: Context, response: Response, profile: RequestProfile) -> str | None:
        """Return a failure_reason string if this engine cannot run; else None."""
        return None

    @abstractmethod
    def evaluate(self, request: Request, context: Context, response: Response, profile: RequestProfile) -> EngineResult:
        ...

    def run(self, request: Request, context: Context, response: Response, profile: RequestProfile) -> EngineResult:
        reason = self.validate(request, context, response, profile)
        if reason:
            return EngineResult(
                status="FAILED", 
                score=None, 
                findings=[], 
                evidence=[], 
                metadata={}, 
                logs=[], 
                failure_reason=reason
            )
        try:
            return self.evaluate(request, context, response, profile)
        except Exception as e:
            return EngineResult(
                status="FAILED", 
                score=None, 
                findings=[], 
                evidence=[], 
                metadata={}, 
                logs=[str(e)], 
                failure_reason="internal_error"
            )
