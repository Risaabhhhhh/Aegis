try:
    import nltk
except ImportError:
    nltk = None


def split_claims(text: str) -> list[str]:
    """Split text into individual claims/sentences using nltk sent_tokenize,
    with fallback to [text] if nltk is unavailable or produces no sentences.
    """
    if not text or not text.strip():
        return []
    if nltk is not None:
        try:
            sentences = nltk.sent_tokenize(text)
            if sentences:
                return sentences
        except Exception:
            pass
    return [text]
