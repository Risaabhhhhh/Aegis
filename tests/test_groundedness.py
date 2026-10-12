import json
import os
import pytest
from engines.groundedness import GroundednessEngine
from base import Request, ContextChunk, Response, RequestProfile

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SEEDS_FILE = os.path.join(FIXTURES_DIR, "groundedness_seeds.json")

@pytest.fixture(scope="module")
def groundedness_engine():
    # This might take a bit to download the model the first time
    return GroundednessEngine()

@pytest.fixture
def seed_data():
    with open(SEEDS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def test_groundedness_seeds(groundedness_engine, seed_data):
    correct = 0
    total = len(seed_data["examples"])
    failed_seeds = []
    
    for item in seed_data["examples"]:
        req = Request(text="test")
        ctx = [ContextChunk(text=c["text"], source_id=c.get("source_id")) for c in item["context"]]
        res = Response(text=item["response"])
        profile = RequestProfile()
        
        result = groundedness_engine.run(req, ctx, res, profile)
        expected_score = 1.0 if item["expected_label"] == "entailment" else 0.0
        
        if result.status == "SUCCESS" and result.score == expected_score:
            correct += 1
        else:
            failed_seeds.append(
                f"ID {item['id']}: expected score {expected_score}, got {result.score} (status: {result.status}). "
                f"Notes: {item['notes']}"
            )
            
    accuracy = correct / total
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    if failed_seeds:
        print("Failed seeds:")
        for fs in failed_seeds:
            print(f"  - {fs}")
            
    assert accuracy >= 0.8, f"Groundedness accuracy too low: {accuracy*100:.2f}%"

def test_empty_context(groundedness_engine):
    req = Request(text="test")
    ctx = []
    res = Response(text="hello world")
    profile = RequestProfile()
    
    result = groundedness_engine.run(req, ctx, res, profile)
    
    assert result.status == "FAILED"
    assert result.failure_reason == "missing_context"
    assert result.score is None

def test_empty_response(groundedness_engine):
    req = Request(text="test")
    ctx = [ContextChunk(text="some context")]
    res = Response(text="   ")
    profile = RequestProfile()
    
    result = groundedness_engine.run(req, ctx, res, profile)
    
    assert result.status == "FAILED"
    assert result.failure_reason == "empty_response"
    assert result.score is None

def test_partial_grounding_score(groundedness_engine):
    req = Request(text="test")
    ctx = [ContextChunk(text="The Eiffel Tower is located in Paris, France.", source_id="doc-loc-1")]
    res = Response(text="The Eiffel Tower is in Paris. It was built in 1999.")
    profile = RequestProfile()
    
    result = groundedness_engine.run(req, ctx, res, profile)
    
    assert result.status == "SUCCESS"
    assert result.score == 0.5
