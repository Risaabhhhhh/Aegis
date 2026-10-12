"""Demo script running aegis.evaluate() on a hard-coded example."""
import json
from base import Request, ContextChunk, Response
import aegis


def main():
    # Hard-coded sample evaluation query
    request = Request(text="Verify patient dosage guideline for Paracetamol.")
    context = [
        ContextChunk(
            text="Paracetamol should not be taken in doses exceeding 4000mg per day for adults, as this raises the risk of liver damage.",
            source_id="doc-med-guideline-1",
        )
    ]
    response = Response(
        text="Adults should not exceed 4000mg of paracetamol daily due to risk of liver damage."
    )

    print("Running Aegis evaluation with currently registered engines...\n")
    result = aegis.evaluate(request, context, response)

    print("=== Aegis Evaluation Result (JSON) ===")
    print(result.to_json(indent=2))


if __name__ == "__main__":
    main()
