try:
    import nltk
except ImportError:
    nltk = None


def split_claims(text: str) -> list[str]:
    """Split text into individual claims/sentences using nltk sent_tokenize.
    
    Must not silently degrade: if nltk or its punkt/punkt_tab data is unavailable,
    raises a clear error directing the user to download the required NLTK resources.
    """
    if not text or not text.strip():
        return []

    if nltk is None:
        raise RuntimeError(
            "NLTK is not installed. Install requirements and run 'python -m nltk.downloader punkt punkt_tab'."
        )

    try:
        sentences = nltk.sent_tokenize(text)
        if sentences:
            return sentences
        return [text]
    except LookupError as e:
        raise LookupError(
            f"NLTK tokenizer data is unavailable: {e}. "
            "Run 'python -m nltk.downloader punkt punkt_tab' to install required resources."
        ) from e

