import torch
from base import Request, Context, Response, RequestProfile, EngineResult
from engines.base import BaseEngine

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


class PromptInjectionEngine(BaseEngine):
    name = "prompt_injection"

    def __init__(
        self,
        model_name: str = "deepset/deberta-v3-base-injection",
        threshold: float = 0.65,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self._tokenizer = None
        self._model = None
        self._injection_idx = None

    @property
    def model(self):
        if self._model is None:
            if AutoTokenizer is None or AutoModelForSequenceClassification is None:
                raise ImportError("transformers is required for PromptInjectionEngine")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            for idx, label in self._model.config.id2label.items():
                if "inject" in label.lower():
                    self._injection_idx = idx
                    break
            if self._injection_idx is None:
                self._injection_idx = 1
        return self._model

    def validate(self, request: Request, context: Context, response: Response, profile: RequestProfile) -> str | None:
        if not response.text or not response.text.strip():
            return "empty_response"
        return None

    def predict_injection_probability(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        _ = self.model
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            return float(probs[self._injection_idx])

    def evaluate(self, request: Request, context: Context, response: Response, profile: RequestProfile) -> EngineResult:
        inputs_to_check = []
        if request and request.text and request.text.strip():
            inputs_to_check.append(("request", request.text))
        if response and response.text and response.text.strip():
            inputs_to_check.append(("response", response.text))
        if context:
            for idx, chunk in enumerate(context):
                if chunk and chunk.text and chunk.text.strip():
                    src_id = chunk.source_id if chunk.source_id else f"chunk_{idx}"
                    inputs_to_check.append((src_id, chunk.text))

        findings: list[str] = []
        evidence: list[dict] = []
        max_injection_prob = 0.0

        for source, text in inputs_to_check:
            prob = self.predict_injection_probability(text)
            if prob > max_injection_prob:
                max_injection_prob = prob

            preview = text[:100]
            evidence.append({
                "source": source,
                "text_preview": preview,
                "injection_probability": prob,
            })

            if prob > self.threshold:
                findings.append(f"Prompt injection detected in {source} (prob: {prob:.4f}): \"{preview}\"")

        score = max(0.0, min(1.0, 1.0 - max_injection_prob))

        return EngineResult(
            status="SUCCESS",
            score=score,
            findings=findings,
            evidence=evidence,
            metadata={
                "engine_version": "0.1.0",
                "model": self.model_name,
                "threshold": self.threshold,
            },
            logs=[
                f"Checked {len(inputs_to_check)} inputs for prompt injection.",
                f"Max injection probability: {max_injection_prob:.4f}. Score: {score:.4f}.",
            ],
        )
