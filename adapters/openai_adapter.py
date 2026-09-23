from adapters.base import BaseAdapter, AdapterResponse

class OpenAIAdapter(BaseAdapter):
    def complete(self, prompt: str, **kwargs) -> AdapterResponse:
        # TODO: Implement actual OpenAI call
        return AdapterResponse(text="", latency_ms=0.0, raw={})
