import json
import uuid
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Append labeled example to seed JSON file.")
    parser.add_argument("--file", required=True, help="Path to seed JSON file")
    parser.add_argument("--response", required=True, help="The model response")
    parser.add_argument("--expected-label", required=True, help="Expected label")
    parser.add_argument("--notes", default="", help="Notes about the example")
    parser.add_argument("--context-text", action="append", default=[], help="Context text chunk(s)")
    
    args = parser.parse_args()
    
    if not args.expected_label:
        print("Error: expected_label must be non-empty", file=sys.stderr)
        sys.exit(1)
        
    context = [{"text": ctx, "source_id": f"ctx-{i+1}"} for i, ctx in enumerate(args.context_text)]
    
    example = {
        "id": str(uuid.uuid4())[:8],
        "context": context,
        "response": args.response,
        "expected_label": args.expected_label,
        "notes": args.notes
    }
    
    data = {"examples": []}
    if os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
    if "examples" not in data:
        data["examples"] = []
        
    data["examples"].append(example)
    
    with open(args.file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print(f"Added example {example['id']} to {args.file}")

if __name__ == "__main__":
    main()
