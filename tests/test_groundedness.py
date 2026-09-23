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
    total = len(seed_data)
    failed_seeds = []
    
    for item in seed_data:
        req = Request(text="test")
        ctx = [ContextChunk(text=item["context"])]
        res = Response(text=item["response"])
        profile = RequestProfile()
        
        result = groundedness_engine.run(req, ctx, res, profile)
        
        if result.status == item["expected_status"]:
            correct += 1
        else:
            failed_seeds.append(
                f"ID {item['id']}: expected {item['expected_status']}, got {result.status}. "
                f"Score: {result.score:.2f}. "
                f"Desc: {item['description']}"
            )
            
    accuracy = correct / total
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    if failed_seeds:
        print("Failed seeds:")
        for fs in failed_seeds:
            print(f"  - {fs}")
            
    # We expect a high accuracy baseline, say >= 80% on this synthetic set.
    assert accuracy >= 0.8, f"Groundedness accuracy too low: {accuracy*100:.2f}%"

def test_empty_context(groundedness_engine):
    req = Request(text="test")
    ctx = []
    res = Response(text="hello world")
    profile = RequestProfile()
    
    result = groundedness_engine.run(req, ctx, res, profile)
    
    assert result.status == "FAILED"
    assert result.failure_reason == "empty_context"
    assert result.score is None
