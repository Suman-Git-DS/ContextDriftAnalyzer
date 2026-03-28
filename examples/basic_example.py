"""Example: Basic drift tracking with DriftTracker.

No extra dependencies needed:
    pip install context-decay-drift
"""

from context_decay_drift import DriftTracker, FewShotExample

tracker = DriftTracker(
    system_prompt=(
        "You are a Python programming tutor. Help students learn Python concepts "
        "including variables, functions, loops, classes, and error handling. "
        "Always provide code examples and explain step by step."
    ),
    few_shot_examples=[
        FewShotExample(
            user="What is a variable?",
            assistant="A variable stores data. Example: x = 5",
        ),
    ],
    mode="always",
    decay_rate=0.93,
    max_summary_sessions=3,
)

# Simulate a multi-turn conversation
conversations = [
    ("How do I use loops?", "In Python, for loops iterate: for i in range(10): print(i). While loops run until a condition is false."),
    ("What about functions?", "Functions use def: def greet(name): return f'Hello {name}'. Functions organize code into reusable blocks."),
    ("Tell me about movies", "The latest blockbuster has amazing effects. Critics gave it 4/5 stars."),
    ("What's for dinner?", "Try grilled salmon with vegetables. Season with lemon and herbs."),
    ("Any travel tips?", "Pack light, use public transport, book early for better rates."),
]

print("Turn | Score | Verdict    | Explanation")
print("-" * 90)

for user_msg, assistant_msg in conversations:
    result = tracker.record_turn(user_msg, assistant_msg)
    if result.drift:
        print(
            f"  {result.drift.turn_number:2d} | "
            f"{result.drift.score:5.1f} | "
            f"{result.drift.verdict.value:10s} | "
            f"{result.explanation[:60] if result.explanation else ''}"
        )

# End session and see managed context
report = tracker.end_session()
print(f"\n--- Session ended (score: {report.drift.score:.1f}) ---")
print(f"\nManaged context for next session:\n{tracker.get_managed_context()}")
