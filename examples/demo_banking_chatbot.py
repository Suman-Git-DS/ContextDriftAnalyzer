"""
Banking Chatbot Drift Demo
===========================
Watch how a banking assistant's context drifts as the conversation
goes off-topic — and how context-drift-analyzer detects it in real time.

Usage:
    python examples/demo_banking_chatbot.py
"""

from anthropic import AnthropicFoundry
from context_drift_analyzer import wrap, FewShotExample

# ── Azure Foundry config ──────────────────────────────────────────────
endpoint = <YOUR-ENDPOINT-URL>  
deployment_name = <YOUR-DEPLOYMENT-NAME/ MODEL-NAME>
api_key = "<YOUR-API-KEY>"  # <-- paste your key here

client = AnthropicFoundry(
    api_key=api_key,
    base_url=endpoint,
)

# ── Wrap the client with drift tracking ───────────────────────────────
SYSTEM_PROMPT = (
    "You are a banking assistant for Acme Bank. "
    "Help customers with savings accounts, credit cards, loans, "
    "and account inquiries. Always provide accurate financial information "
    "and guide customers to the right banking products. "
    "If a user asks something unrelated to banking, respond with: "
    "'This is off-topic and I may not have relevant information. "
    "I can help you with banking products, accounts, loans, and credit cards.'"
)

tracked = wrap(
    client,
    system_prompt=SYSTEM_PROMPT,
    few_shot_examples=[
        FewShotExample(
            user="What interest rate do your savings accounts offer?",
            assistant="Our standard savings account offers 4.5% APY. "
                      "Premium savings offers 5.1% APY for balances over $10,000. "
                      "Both are FDIC insured up to $250,000.",
        ),
        FewShotExample(
            user="How do I apply for a credit card?",
            assistant="You can apply online at acmebank.com/cards or visit any branch. "
                      "You'll need a government-issued ID, proof of income, and your SSN. "
                      "Approval typically takes 1-2 business days.",
        ),
    ],
    mode="always",
    persist=True,
    persist_path=".demo_session_memory",
    max_summary_sessions=3,
)

# ── Conversation turns: on-topic → off-topic ──────────────────────────
turns = [
    # On-topic banking questions
    "What are the requirements for a home loan at Acme Bank?",
    "Can I refinance my existing mortgage? What rates do you offer?",
    "What's the difference between a checking and savings account?",
    # Gradual drift
    "I'm planning a vacation, should I use my Acme Travel credit card abroad?",
    # Off-topic
    "What's a good pasta carbonara recipe?",
    "Tell me about the best hiking trails in Colorado.",
    "Who won the FIFA World Cup in 2022?",
]

# ── Run the conversation ──────────────────────────────────────────────
print("=" * 90)
print("  BANKING CHATBOT DRIFT DEMO — context-drift-analyzer v0.5.0")
print("=" * 90)
print()
print(f"System Prompt: {SYSTEM_PROMPT[:80]}...")
print()
print("-" * 90)
print(f"{'Turn':>4} | {'Score':>5} | {'Verdict':<10} | {'User Question':<50}")
print("-" * 90)

for question in turns:
    response = tracked.messages.create(
        model=deployment_name,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
        max_tokens=300,
    )

    # Extract response text
    assistant_text = response.content[0].text

    # Drift info is attached to every response automatically
    score = response._drift.score
    verdict = response._drift.verdict.value
    explanation = response._drift_explanation

    print(f"  {response._drift.turn_number:2d}  | {score:5.1f} | {verdict:<10} | {question[:50]}")
    print(f"       |       |            | Reply: {assistant_text[:70]}...")
    print(f"       |       |            | Why: {explanation[:70]}")
    print()

# ── On-demand drift check ─────────────────────────────────────────────
print("-" * 90)
print("ON-DEMAND DRIFT CHECK:")
report = tracked.drift_check()
print(f"  Score:       {report.drift.score:.1f}/100")
print(f"  Verdict:     {report.drift.verdict.value}")
print(f"  Explanation: {report.explanation}")
print(f"  Total turns: {report.total_turns}")
print(f"  Token est:   ~{report.context_token_estimate} tokens")

# ── Show managed context ──────────────────────────────────────────────
print()
print("-" * 90)
print("MANAGED CONTEXT (what you'd send as system message next session):")
print("-" * 90)
ctx = tracked.get_managed_context()
print(ctx[:500])
if len(ctx) > 500:
    print(f"  ... ({len(ctx)} chars total)")

# ── End session ───────────────────────────────────────────────────────
print()
print("-" * 90)
end_report = tracked.end_session()
print(f"SESSION ENDED — Final score: {end_report.drift.score:.1f}/100")
print(f"Session summary saved to .demo_session_memory")
print()
print("Run this script again to see how session summaries carry over!")
