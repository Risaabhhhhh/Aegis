import re
import scipy.special
from base import Request, Context, Response, RequestProfile, EngineResult
from engines.base import BaseEngine
from engines.common import split_claims

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

# Regex patterns matching specific falsifiable assertions
SPECIFIC_PATTERNS = [
    # Numbers and numeric quantities
    r"\b\d+([.,]\d+)?\b",
    # Percentages
    r"\b\d+(\.\d+)?%",
    r"\bpercent(age)?\b",
    # Currencies
    r"[\$€£₹]\s*\d+",
    r"\b\d+\s*(usd|eur|inr|gbp|dollars?|euros?|rupees?|pounds?)\b",
    # Units of measurement
    r"\b\d+\s*(mg|kg|g|km|miles?|meters?|cm|mm|hours?|minutes?|seconds?|days?|weeks?|months?|years?|billion|million|trillion|°c|°f)\b",
    # Dates, years, months, quarters
    r"\b(18|19|20)\d{2}\b",
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\bQ[1-4]\b",
    # Citations, studies, and empirical assertions
    r"\baccording to\b",
    r"\bstudies show\b",
    r"\bstudy by\b",
    r"\bstudy found\b",
    r"\bresearchers? at\b",
    r"\breported by\b",
    r"\bet al\.?\b",
    r"\bpublished in\b",
    r"\bfound that\b",
    r"\bdata from\b",
    r"\bsurvey by\b",
    r"\bclinical trials?\b",
    # Major organizations & agency acronyms
    r"\b(WHO|NASA|FDA|SEC|CDC|RBI|EU|UN|NIH|IMF|UNESCO)\b",
    # Titles and named individuals
    r"\b(Dr|Prof|Mr|Mrs|Ms|President|Minister|CEO)\.?\s+[A-Z][a-z]+",
]


def contains_specific_assertion(sentence: str) -> bool:
    """Check if a claim contains a specific falsifiable assertion (number, date, percentage, named person/org, or citation)."""
    for pattern in SPECIFIC_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            return True
    return False


class HallucinationEngine(BaseEngine):
    name = "hallucination"

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self._entailment_idx = None

    @property
    def model(self):
        if self._model is None:
            if CrossEncoder is None:
                raise ImportError("sentence-transformers is required for HallucinationEngine NLI evaluation")
            self._model = CrossEncoder(self.model_name)
            id2label = self._model.config.id2label
            for idx, label in id2label.items():
                if label.lower() == "entailment":
                    self._entailment_idx = idx
                    break
            if self._entailment_idx is None:
                self._entailment_idx = 1
        return self._model

    def validate(self, request: Request, context: Context, response: Response, profile: RequestProfile) -> str | None:
        if not response.text or not response.text.strip():
            return "empty_response"
        return None

    def evaluate(self, request: Request, context: Context, response: Response, profile: RequestProfile) -> EngineResult:
        sentences = split_claims(response.text)
        if not sentences:
            sentences = [response.text]

        total_claims = len(sentences)
        findings: list[str] = []
        evidence: list[dict] = []

        claims_info = []
        for sentence in sentences:
            is_specific = contains_specific_assertion(sentence)
            claim_type = "specific" if is_specific else "general"
            claims_info.append((sentence, is_specific, claim_type))

        specific_claims = [c for c in claims_info if c[1]]
        total_specific = len(specific_claims)

        if context:
            # Context available: run NLI against context chunks
            supported_specific = 0
            for sentence, is_specific, claim_type in claims_info:
                best_prob = -1.0
                best_chunk_id = None

                for idx, chunk in enumerate(context):
                    score = self.model.predict([(chunk.text, sentence)])[0]
                    if hasattr(score, "__len__"):
                        prob = float(scipy.special.softmax(score)[self._entailment_idx])
                    else:
                        prob = float(scipy.special.expit(score))

                    if prob > best_prob:
                        best_prob = prob
                        best_chunk_id = chunk.source_id if chunk.source_id else f"chunk_{idx}"

                is_supported = (best_prob >= self.threshold)
                if is_specific:
                    if is_supported:
                        supported_specific += 1
                    else:
                        findings.append(f"Claim not grounded in context (max prob: {best_prob:.2f}): \"{sentence}\"")

                evidence.append({
                    "sentence": sentence,
                    "type": claim_type,
                    "entailment_probability": best_prob,
                    "source_id": best_chunk_id,
                })

            score = (supported_specific / total_specific) if total_specific > 0 else 1.0
            logs = [
                f"Evaluated {total_claims} claims ({total_specific} specific, {total_claims - total_specific} general).",
                f"Supported specific claims: {supported_specific}/{total_specific} with context.",
            ]
        else:
            # Context is NOT available: flag every specific claim as unverifiable
            flagged_specific = 0
            for sentence, is_specific, claim_type in claims_info:
                if is_specific:
                    flagged_specific += 1
                    findings.append(f"Claim contains unverifiable specific assertion without context: \"{sentence}\"")

                evidence.append({
                    "sentence": sentence,
                    "type": claim_type,
                })

            score = 1.0 - (flagged_specific / total_claims) if total_claims > 0 else 1.0
            logs = [
                f"Evaluated {total_claims} claims without context.",
                f"Unverifiable specific claims flagged: {flagged_specific}/{total_claims}.",
            ]

        # Clamp score to [0.0, 1.0]
        score = max(0.0, min(1.0, float(score)))

        return EngineResult(
            status="SUCCESS",
            score=score,
            findings=findings,
            evidence=evidence,
            metadata={
                "engine_version": "0.1.0",
                "approach": "factscore_style_v0",
                "model": self.model_name if context else None,
                "threshold": self.threshold if context else None,
            },
            logs=logs,
        )
