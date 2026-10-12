"""OpenAI Adapter Stub.

NOTE ON ADAPTER USAGE:
- Engines that do NOT use an LLM (e.g. cross-encoder or NLI based engines like Groundedness)
  must NEVER touch the adapters layer.
- Any engine that requires an LLM must implement the adapter first and include an explicit PR note.
"""
from adapters.base import BaseAdapter, AdapterResponse


class OpenAIAdapter(BaseAdapter):
    """Stub adapter for OpenAI LLM calls.

    Engines that do NOT use an LLM never touch this adapter. Any engine that
    requires an external LLM call must implement this adapter first and include
    a PR note.
    """

    def complete(self, prompt: str, **kwargs) -> AdapterResponse:
        # TODO: Implement actual OpenAI call
        return AdapterResponse(text="", latency_ms=0.0, raw={})

