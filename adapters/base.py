from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class AdapterResponse:
    text: str
    latency_ms: float
    raw: dict  # provider's raw response, for debugging

class BaseAdapter(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> AdapterResponse:
        ...
