"""
Persistence Test — Run this AFTER demo_banking_chatbot.py
==========================================================
This script proves that session memory persists across runs.
It loads the .demo_session_memory file from the previous run,
shows what was saved, then continues with new questions and
appends to the same memory file.

Usage:
    1. First run:  python examples/demo_banking_chatbot.py
    2. Then run:   python examples/demo_persistence_test.py
    3. Run again:  python examples/demo_persistence_test.py  (to see it accumulate)
"""

import json
import os

from anthropic import AnthropicFoundry
from context_drift_analyzer import wrap, FewShotExample

# ── Azure Foundry config (same as demo) ──────────────────────────────
endpoint = <YOUR-ENDPOINT-URL> 
deployment_name = <YOUR-DEPLOYMENT-NAME/ MODEL-NAME>
api_key = "<YOUR-API-KEY>"  # <-- paste your key here

MEMORY_FILE = ".demo_session_memory"

# ── Step 1: Show what's already in memory ─────────────────────────────
print("=" * 80)
print("  PERSISTENCE TEST — context-drift-analyzer")
print("=" * 80)
print()

if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE) as f:
        data = json.load(f)
    print(f"EXISTING MEMORY FOUND ({MEMORY_FILE}):")
    print(f"  Session count:  {data.get('session_count', 0)}")
    print(f"  Context frozen: {data.get('context_frozen', False)}")
    print(f"  Sessions:       {len(data.get('sessions', []))}")
    print()

    sessions = data.get("sessions", [])
    for s in sessions:
        sn = s.get("session_number", "?")
        status = s.get("status", "?")
        exchanges = s.get("exchanges", [])
        summary = s.get("summary")
        score = s.get("final_drift_score")
        print(f"  Session {sn} [{status}] — {len(exchanges)} exchanges")
        if score is not None:
            print(f"    Final drift: {score}/100")
        if summary:
            print(f"    Summary: {summary[:100]}")
        if exchanges:
            print(f"    Last 3 exchanges:")
            for e in exchanges[-3:]:
                print(f"      Q{e.get('exchange','?')}: {e.get('user','')[:50]}")
                print(f"      A{e.get('exchange','?')}: {e.get('assistant','')[:50]}")
                print(f"        Score: {e.get('score','?')}/100 ({e.get('verdict','?')})")
        print()
else:
    print(f"NO MEMORY FILE FOUND at {MEMORY_FILE}")
    print("Run demo_banking_chatbot.py first, or this will start fresh.")
    print()

# ── Step 2: Create tracked client (loads existing memory) ────────────
SYSTEM_PROMPT = (
    "You are a banking assistant for Acme Bank. "
    "Help customers with savings accounts, credit cards, loans, "
    "and account inquiries. Always provide accurate financial information "
    "and guide customers to the right banking products. "
    "If a user asks something unrelated to banking, respond with: "
    "'This is off-topic and I may not have relevant information. "
    "I can help you with banking products, accounts, loans, and credit cards.'"
)

client = AnthropicFoundry(
    api_key=api_key,
    base_url=endpoint,
)

tracked = wrap(
    client,
    system_prompt=SYSTEM_PROMPT,
    few_shot_examples=[
        FewShotExample(
            user="What interest rate do your savings accounts offer?",
            assistant="Our standard savings account offers 4.5% APY. "
                      "Premium savings offers 5.1% APY for balances over $10,000.",
        ),
    ],
    mode="always",
    persist=True,
    persist_path=MEMORY_FILE,
    max_summary_sessions=3,
)

# ── Step 3: Show the managed context (includes past session summaries) ─
print("-" * 80)
print("MANAGED CONTEXT (original prompt + past session summaries):")
print("-" * 80)
ctx = tracked.get_managed_context()
print(ctx[:600])
if len(ctx) > 600:
    print(f"... ({len(ctx)} chars total)")
print()

# ── Step 4: New conversation turns ───────────────────────────────────
new_turns = [
    "What are your fixed deposit rates for 1 year?",
    "Can I open a joint account with my spouse?",
    "How do I set up automatic loan repayments?",
]

print("-" * 80)
print("NEW SESSION — Asking 3 new banking questions:")
print("-" * 80)
print()

for question in new_turns:
    response = tracked.messages.create(
        model=deployment_name,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
        max_tokens=300,
    )

    assistant_text = response.content[0].text
    score = response._drift.score
    verdict = response._drift.verdict.value

    print(f"  User:    {question}")
    print(f"  Bot:     {assistant_text[:100]}...")
    print(f"  Drift:   {score:.1f}/100 ({verdict})")
    print(f"  Explain: {response._drift_explanation[:80]}")
    print()

# ── Step 5: End session and save ──────────────────────────────────────
print("-" * 80)
end_report = tracked.end_session()
print(f"SESSION ENDED — Score: {end_report.drift.score:.1f}/100")
print()

# ── Step 6: Show updated memory ──────────────────────────────────────
with open(MEMORY_FILE) as f:
    updated = json.load(f)

print("UPDATED MEMORY:")
print(f"  Session count: {updated.get('session_count', 0)}")
print(f"  Sessions:      {len(updated.get('sessions', []))}")

sessions = updated.get("sessions", [])
for s in sessions:
    sn = s.get("session_number", "?")
    status = s.get("status", "?")
    exchanges = s.get("exchanges", [])
    summary = s.get("summary")
    score = s.get("final_drift_score")
    print(f"    Session {sn} [{status}] — {len(exchanges)} exchanges, drift: {score}")
    if summary:
        print(f"      Summary: {summary[:100]}")

print()
print("Run this script again to see sessions accumulate!")
print(f"Or inspect the file directly: cat {MEMORY_FILE}")
