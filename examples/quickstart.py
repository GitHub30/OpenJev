"""Answer several typed questions about one state in a single forward pass.

    python examples/quickstart.py                       # deterministic mock backend
    python examples/quickstart.py Qwen/Qwen2.5-1.5B-Instruct
"""

import json
import sys

from openjev import SystemOneEngine, load_backend

model = sys.argv[1] if len(sys.argv) > 1 else "mock"
engine = SystemOneEngine(backend=load_backend(model), model_name=f"openjev/{model}")

response = engine.system_one(
    state="Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP.",
    questions={
        "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Charges, refunds and invoices",
                "integrations": "Connecting third-party services such as Stripe or Slack",
                "account": "Login, password and profile problems",
                "other": None,
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is the customer's problem?",
            "criteria": ["Cosmetic issue", "Broken feature, but a workaround exists", "Blocking issue with no workaround"],
        },
    },
)

print(json.dumps(response.model_dump(), indent=2))

team = response.answers["team"]
if team.confidence < 0.5:
    print("\n-> low confidence: route to a human")
else:
    print(f"\n-> route to {team.choice} (confidence {team.confidence:.2f})")
