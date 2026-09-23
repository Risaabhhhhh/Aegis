import scipy.special
from base import Request, Context, Response, RequestProfile, EngineResult
from engines.base import BaseEngine

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None
    
try:
    import nltk
except ImportError:
    nltk = None

class GroundednessEngine(BaseEngine):
    name = "groundedness"
    
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", threshold: float = 0.5):
        if CrossEncoder is None:
            raise ImportError("sentence-transformers is required for GroundednessEngine")
        if nltk is None:
            raise ImportError("nltk is required for GroundednessEngine")
            
        self.model_name = model_name
        self.threshold = threshold
        self._model = None
        self._entailment_idx = None
        
    @property
    def model(self):
        if self._model is None:
            # Lazy load the model on first use
            self._model = CrossEncoder(self.model_name)
            
            # Find which index corresponds to 'entailment'
            id2label = self._model.config.id2label
            for idx, label in id2label.items():
                if label.lower() == "entailment":
                    self._entailment_idx = idx
                    break
                    
            if self._entailment_idx is None:
                self._entailment_idx = 1 # Fallback
                
        return self._model
        
    def evaluate(self, request: Request, context: Context, response: Response, profile: RequestProfile) -> EngineResult:
        if not context:
            return EngineResult(
                status="FAILED",
                score=None,
                findings=["No context provided for groundedness evaluation."],
                evidence=[],
                metadata={},
                logs=["Skipping groundedness due to empty context."],
                failure_reason="empty_context"
            )
            
        # Proper sentence tokenization
        sentences = nltk.sent_tokenize(response.text)
        if not sentences:
            sentences = [response.text]
            
        findings = []
        evidence = []
        supported_claims = 0
        
        for sentence in sentences:
            best_prob = -1.0
            best_chunk_id = None
            
            # Score against each chunk separately
            for idx, chunk in enumerate(context):
                score = self.model.predict([(chunk.text, sentence)])[0] # predict returns shape (1, num_classes)
                
                # Handling if the model outputs logits or probs
                if hasattr(score, '__len__'):
                    prob = scipy.special.softmax(score)[self._entailment_idx]
                else:
                    prob = scipy.special.expit(score)
                    
                prob_float = float(prob)
                if prob_float > best_prob:
                    best_prob = prob_float
                    best_chunk_id = chunk.source_id if chunk.source_id else f"chunk_{idx}"
                    
            # Check threshold
            if best_prob >= self.threshold:
                supported_claims += 1
            else:
                findings.append(f"Sentence lacks grounding in context (max prob: {best_prob:.2f}): \"{sentence}\"")
                
            evidence.append({
                "sentence": sentence,
                "entailment_probability": best_prob,
                "source_id": best_chunk_id
            })
            
        total_claims = len(sentences)
        score_percentage = (supported_claims / total_claims) * 100.0 if total_claims > 0 else 0.0
        
        status = "FAILED" if supported_claims < total_claims else "SUCCESS"
        
        return EngineResult(
            status=status,
            score=score_percentage,
            findings=findings,
            evidence=evidence,
            metadata={
                "model": self.model_name,
                "threshold": self.threshold
            },
            logs=[
                f"Evaluated {total_claims} claims.",
                f"Supported: {supported_claims}/{total_claims} ({(supported_claims/total_claims)*100:.1f}%)"
            ]
        )
