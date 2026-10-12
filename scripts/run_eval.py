import json
import argparse
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base import Request, ContextChunk, Response, RequestProfile
from engines.registry import REGISTERED_ENGINES

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on a seed set.")
    parser.add_argument("--engine", required=True, help="Engine name (e.g., groundedness)")
    parser.add_argument("--file", required=True, help="Path to seed JSON file")
    parser.add_argument("--output", default=None, help="Optional output JSON path for per-example results")
    args = parser.parse_args()
    
    engine = next((e for e in REGISTERED_ENGINES if e.name == args.engine), None)
    if not engine:
        print(f"Error: Engine '{args.engine}' not found in registry.", file=sys.stderr)
        sys.exit(1)
        
    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    correct = 0
    total = len(data.get("examples", []))
    misclassified = []
    per_example_results = []
    
    for item in data.get("examples", []):
        req = Request(text="test")
        ctx = [ContextChunk(text=c["text"], source_id=c.get("source_id")) for c in item.get("context", [])]
        res = Response(text=item["response"])
        profile = RequestProfile()
        
        result = engine.run(req, ctx, res, profile)
        
        expected = item["expected_label"]
        expected_status = "SUCCESS" if expected in ["entailment", "SUCCESS"] else "FAILED"
        is_correct = (result.status == expected_status)
        
        if is_correct:
            correct += 1
        else:
            misclassified.append({
                "id": item.get("id"),
                "expected": expected_status,
                "got": result.status,
                "score": result.score,
                "notes": item.get("notes")
            })
            
        per_example_results.append({
            "id": item.get("id"),
            "expected_label": expected,
            "expected_status": expected_status,
            "status": result.status,
            "score": result.score,
            "failure_reason": result.failure_reason,
            "findings": result.findings,
            "evidence": result.evidence,
            "metadata": result.metadata,
            "logs": result.logs,
            "is_correct": is_correct,
            "notes": item.get("notes"),
        })
            
    accuracy = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"Engine: {engine.name}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    if misclassified:
        print("\nMisclassified Examples:")
        for mc in misclassified:
            print(f" - ID: {mc['id']} | Expected: {mc['expected']} | Got: {mc['got']} | Score: {mc['score']}")
            print(f"   Notes: {mc['notes']}")

    # Determine output file path
    if args.output:
        output_path = args.output
    else:
        base_dir = os.path.dirname(args.file)
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        output_path = os.path.join(base_dir, f"{base_name}_{args.engine}_results.json")

    results_data = {
        "engine": engine.name,
        "input_file": args.file,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "misclassified_count": len(misclassified),
        "examples": per_example_results,
    }

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(results_data, out_f, indent=2)

    print(f"\nPer-example results written to: {output_path}")

if __name__ == "__main__":
    main()
