"""
examples/quickstart.py
======================

Simulates a multi-session banking app using context-drift-analyzer.

Session 1: Acme Bank assistant — on-topic banking questions (establish context)
Session 2: Same app, still on topic — should show low drift
Session 3: Topic has drifted — off-topic questions, drift should spike

Run:
    python examples/quickstart.py

After each session a .session_memory file is written.
Run `context-drift-analyzer status` to inspect it.
Run `context-drift-analyzer history` to see the full drift trail.
"""

import time
from anthropic import AnthropicFoundry
from context_drift_analyzer import wrap, FewShotExample

MEMORY_FILE = ".quickstart_session_memory"

client = AnthropicFoundry(
    api_key="<YOUR-API-KEY>",
    base_url="<YOUR-ENDPOINT-URL>",
)

SYSTEM_PROMPT = (
    "You are a banking assistant for Acme Bank. "
    "Help customers with savings accounts, credit cards, loans, "
    "and account inquiries. Always provide accurate financial information "
    "and guide customers to the right banking products. "
    "If a user asks something unrelated to banking, respond with: "
    "'This is off-topic and I may not have relevant information. "
    "I can help you with banking products, accounts, loans, and credit cards.'"
)

MODEL = "claude-opus-4-5"


def run_session(tracked, session_label, system, turns, check=False):
    print(f"\n{'=' * 70}")
    print(f"  {session_label}")
    print(f"  Session #{tracked._tracker.session_number} | "
          f"Exchanges so far: {tracked._tracker.turn_count}")
    print(f"{'=' * 70}\n")

    for question in turns:
        response = tracked.messages.create(
            model=MODEL,
            system=system,
            messages=[{"role": "user", "content": question}],
            max_tokens=200,
        )

        score = response._drift.score
        verdict = response._drift.verdict.value
        reply = response.content[0].text[:100]

        print(f"  User:    {question}")
        print(f"  Bot:     {reply}...")
        print(f"  Drift:   {score:.1f}/100 ({verdict})")
        print(f"  Why:     {response._drift_explanation[:80]}")
        print()
        time.sleep(0.5)  # avoid rate limits

    if check:
        print("  " + "-" * 60)
        print("  DRIFT CHECK:")
        report = tracked.drift_check()
        print(f"    Score:       {report.drift.score:.1f}/100")
        print(f"    Verdict:     {report.drift.verdict.value}")
        print(f"    Explanation: {report.explanation[:80]}")
        print(f"    Effective:   {report.drift.is_effective}")
        print(f"    Needs reset: {report.drift.needs_reset}")
        print("  " + "-" * 60)

    # End session — summarize, persist, start fresh
    end_report = tracked.end_session()
    if end_report:
        print(f"\n  SESSION ENDED — Final score: {end_report.drift.score:.1f}/100")
        print(f"  Summary saved to {MEMORY_FILE}")


# ── Create the tracked client (persists across all sessions) ─────────
tracked = wrap(
    client,
    system_prompt=SYSTEM_PROMPT,
    few_shot_examples=[
        FewShotExample(
            user="What interest rate do your savings accounts offer?",
            assistant="Our standard savings account offers 4.5% APY. "
                      "Premium savings offers 5.1% APY for balances over $10,000.",
        ),
        FewShotExample(
            user="How do I apply for a credit card?",
            assistant="You can apply online at acmebank.com/cards or visit any branch. "
                      "You'll need your ID, proof of income, and SSN.",
        ),
    ],
    mode="always",
    persist=True,
    persist_path=MEMORY_FILE,
    max_summary_sessions=3,
)

# ── Session 1: Establish context — on-topic banking questions ────────
run_session(
    tracked,
    session_label="SESSION 1: On-topic banking questions (establishing context)",
    system=SYSTEM_PROMPT,
    turns=[
        "What are the requirements for a home loan at Acme Bank?",
        "Can I refinance my existing mortgage?",
        "What's the difference between a checking and savings account?",
        "How do I set up automatic bill payments?",
        "What credit cards do you offer for small businesses?",
    ],
    check=False,  # no check yet — just establishing context
)

# ── Session 2: Still on topic — low drift expected ───────────────────
run_session(
    tracked,
    session_label="SESSION 2: Still on topic (low drift expected)",
    system=SYSTEM_PROMPT,
    turns=[
        "What are your fixed deposit rates for 1 year?",
        "Can I open a joint account with my spouse?",
        "How do I report a lost debit card?",
    ],
    check=True,  # check — should show low drift
)

# ── Session 3: Drifted — off-topic questions ─────────────────────────
run_session(
    tracked,
    session_label="SESSION 3: Off-topic questions (drift should spike)",
    system=SYSTEM_PROMPT,
    turns=[
        "What's a good pasta carbonara recipe?",
        "Tell me about the best hiking trails in Colorado.",
        "Who won the FIFA World Cup in 2022?",
    ],
    check=True,  # check — should show significant drift
)

# ── Final summary ────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  DONE — All 3 sessions completed.")
print(f"{'=' * 70}")
print(f"\nInspect the session memory:")
print(f"  context-drift-analyzer status  --file {MEMORY_FILE}")
print(f"  context-drift-analyzer history --file {MEMORY_FILE}")
print(f"  context-drift-analyzer reset   --file {MEMORY_FILE}  (to clean up)")
print(f"\nOr view the raw JSON:")
print(f"  cat {MEMORY_FILE}")
