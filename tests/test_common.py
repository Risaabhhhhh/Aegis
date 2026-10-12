import pytest
from engines.common import split_claims


def test_split_claims_multiple_sentences():
    text = "The sky is blue. Grass is green! Is the sun hot?"
    claims = split_claims(text)
    assert len(claims) == 3
    assert claims[0] == "The sky is blue."
    assert claims[1] == "Grass is green!"
    assert claims[2] == "Is the sun hot?"


def test_split_claims_single_sentence():
    text = "Only one sentence here."
    claims = split_claims(text)
    assert claims == ["Only one sentence here."]


@pytest.mark.parametrize("empty_input", ["", "   ", "\n\t  "])
def test_split_claims_empty_or_whitespace(empty_input):
    claims = split_claims(empty_input)
    assert claims == []


def test_split_claims_raises_when_nltk_missing(monkeypatch):
    import engines.common as common
    monkeypatch.setattr(common, "nltk", None)
    text = "Text needing tokenization."
    with pytest.raises(RuntimeError, match="NLTK is not installed"):
        common.split_claims(text)


def test_split_claims_raises_when_punkt_data_missing(monkeypatch):
    import engines.common as common
    def mock_sent_tokenize(text):
        raise LookupError("Resource punkt_tab not found.")

    monkeypatch.setattr(common.nltk, "sent_tokenize", mock_sent_tokenize)
    text = "Text needing tokenization when punkt is absent."
    with pytest.raises(LookupError, match="NLTK tokenizer data is unavailable"):
        common.split_claims(text)

