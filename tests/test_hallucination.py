import json
import os
import pytest
from engines.hallucination import HallucinationEngine
from base import Request, ContextChunk, Response, RequestProfile

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SEEDS_FILE = os.path.join(FIXTURES_DIR, "hallucination_seeds.json")


@pytest.fixture(scope="module")
def hallucination_engine():
    return HallucinationEngine()


@pytest.fixture
def seed_data():
    with open(SEEDS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_hallucination_seeds(hallucination_engine, seed_data):
    correct = 0
    total = len(seed_data["examples"])
    failed_seeds = []

    for item in seed_data["examples"]:
        req = Request(text="test")
        ctx = [ContextChunk(text=c["text"], source_id=c.get("source_id")) for c in item.get("context", [])]
        res = Response(text=item["response"])
        profile = RequestProfile(context_available=len(ctx) > 0)

        result = hallucination_engine.run(req, ctx, res, profile)
        expected_label = item["expected_label"]

        if expected_label == "clean":
            is_correct = (result.status == "SUCCESS" and result.score == 1.0)
        elif expected_label == "hallucinated":
            is_correct = (result.status == "SUCCESS" and result.score == 0.0)
        elif expected_label == "partial":
            is_correct = (result.status == "SUCCESS" and 0.0 < result.score < 1.0)
        else:
            is_correct = False

        if is_correct:
            correct += 1
        else:
            failed_seeds.append(
                f"ID {item['id']}: expected label {expected_label}, got score {result.score} (status: {result.status}). "
                f"Findings: {result.findings}. Notes: {item.get('notes')}"
            )

    accuracy = correct / total
    print(f"Hallucination Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    assert not failed_seeds, (
        f"Hallucination seed test failed on {len(failed_seeds)}/{total} seeds. Details:\n"
        + "\n".join(failed_seeds)
    )


def test_empty_response(hallucination_engine):
    req = Request(text="test")
    ctx = [ContextChunk(text="some context")]
    res = Response(text="   ")
    profile = RequestProfile(context_available=True)

    result = hallucination_engine.run(req, ctx, res, profile)

    assert result.status == "FAILED"
    assert result.failure_reason == "empty_response"
    assert result.score is None


def test_empty_context_no_specific_claims_scores_one(hallucination_engine):
    """Edge case: With empty context, a purely general response with no specific claims must score 1.0."""
    req = Request(text="What is wisdom?")
    ctx = []
    res = Response(text="Wisdom is the ability to think and act using knowledge and common sense.")
    profile = RequestProfile(context_available=False)

    result = hallucination_engine.run(req, ctx, res, profile)

    assert result.status == "SUCCESS"
    assert result.score == 1.0
    assert result.findings == []


def test_empty_context_unverifiable_specific_claims_penalized(hallucination_engine):
    """Edge case: With empty context, specific falsifiable claims are flagged and lower the score."""
    req = Request(text="What happened in 2024?")
    ctx = []
    res = Response(text="In 2024, the company made $500 million in revenue. In addition, 95% of users renewed.")
    profile = RequestProfile(context_available=False)

    result = hallucination_engine.run(req, ctx, res, profile)

    assert result.status == "SUCCESS"
    assert result.score == 0.0
    assert len(result.findings) == 2
    assert all("unverifiable_specific" in f or "unverifiable" in f for f in result.findings)
