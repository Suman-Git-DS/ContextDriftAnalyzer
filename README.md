<p align="center">
  <h1 align="center">context-drift-analyzer</h1>
  <p align="center">
    Detect, explain, and <strong>solve</strong> context drift in LLM conversations across sessions.
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/context-drift-analyzer/"><img alt="PyPI" src="https://img.shields.io/pypi/v/context-drift-analyzer?v=1"></a>
  <a href="https://pypi.org/project/context-drift-analyzer/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/context-drift-analyzer?v=1"></a>
  <a href="https://github.com/Suman-Git-DS/ContextDriftAnalyzer/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Suman-Git-DS/ContextDriftAnalyzer?v=1"></a>
</p>

---

## The Problem

LLM-powered chatbots lose focus over long conversations. A banking assistant that starts by explaining savings accounts ends up discussing travel tips after a few sessions — with no visibility into *when* this happens or *why*.

Most tools only **detect** drift. This package **detects, explains, and solves** it:

| Capability | What it does |
|-----------|-------------|
| **Detect** | Drift score (0-100) via semantic embeddings |
| **Explain** | 1-2 line human-readable reason for drift |
| **Solve** | Context management: original context + session summaries, kept within token budget |
| **Persist** | `.session_memory` file tracks drift across restarts and deploys |

## How It Works

Imagine a banking assistant chatbot:

```
Session 1 (Turn 1-2):  Score 92  [FRESH]      "Context well-preserved. Responses align with banking instructions."
Session 2 (Turn 3-6):  Score 76  [MILD]       "Mild drift: core topics still present (savings, accounts, interest)."
Session 3 (Turn 7-12): Score 48  [SEVERE]     "Significant drift: now focused on travel, restaurants, recipes."
Session 4 (Turn 13+):  Score 22  [CRITICAL]   "Critical drift: conversation departed from banking purpose."
                                                ↑ recommend reset
```

## Installation

```bash
# Core + Sentence Transformers (recommended — free, local, semantic)
pip install context-drift-analyzer[semantic]

# Core only (zero dependencies — keyword/TF strategies, or bring your own embedder)
pip install context-drift-analyzer

# Everything (semantic + OpenAI + Anthropic embedding support)
pip install context-drift-analyzer[all]
```

## Quick Start — Drop-in Client Wrapper

The easiest way to add drift tracking. Wrap your existing LLM client and use it exactly as before — drift scores are attached to every response automatically.

### OpenAI

```python
from openai import OpenAI
from context_drift_analyzer import wrap

client = OpenAI()
tracked = wrap(client, system_prompt="You are a banking assistant specializing in savings accounts, credit cards, and loans. Always provide accurate financial information.")

# Use exactly like the original client
response = tracked.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a banking assistant."},
        {"role": "user", "content": "What savings accounts do you offer?"},
    ],
)

# Drift score is attached to the response
print(response._drift.score)        # 88.5
print(response._drift_explanation)   # "Context well-preserved..."

# On-demand check
report = tracked.drift_check()
```

### Anthropic

```python
from anthropic import Anthropic
from context_drift_analyzer import wrap

client = Anthropic()
tracked = wrap(client, system_prompt="You are a banking assistant specializing in savings accounts, credit cards, and loans.")

response = tracked.messages.create(
    model="claude-haiku-4-5-20251001",
    system="You are a banking assistant.",
    messages=[{"role": "user", "content": "Tell me about your credit card options."}],
    max_tokens=200,
)

print(response._drift.score)  # Drift score attached!
report = tracked.drift_check()
```

## Quick Start — Direct Tracker

For more control, use `DriftTracker` directly in your pipeline:

```python
from context_drift_analyzer import DriftTracker, FewShotExample

tracker = DriftTracker(
    system_prompt="You are a banking assistant for Acme Bank. Help customers with savings accounts, credit cards, loans, and account inquiries. Always provide accurate financial information.",
    few_shot_examples=[
        FewShotExample(
            user="What interest rate do your savings accounts offer?",
            assistant="Our standard savings account offers 4.5% APY. Premium savings offers 5.1% APY for balances over $10,000."
        ),
        FewShotExample(
            user="How do I apply for a credit card?",
            assistant="You can apply online at acmebank.com/cards or visit any branch. You'll need your ID, proof of income, and SSN. Approval typically takes 1-2 business days."
        ),
    ],
    mode="always",              # "always" or "ondemand"
    persist=True,               # save to .session_memory file
    max_summary_sessions=3,     # keep last 3 session summaries
)

# After each LLM call in your pipeline:
result = tracker.record_turn(
    user_message="What are the requirements for a home loan?",
    assistant_response="For a home loan at Acme Bank, you'll need a credit score of 620+, proof of income, 2 years of tax returns, and a down payment of at least 3.5% for FHA loans or 20% for conventional loans."
)

print(f"Score:       {result.drift.score:.1f}/100")      # 87.3/100
print(f"Verdict:     {result.drift.verdict.value}")       # "mild"
print(f"Explanation: {result.explanation}")                # "Mild drift: core topics present (loans, credit, banking)."
print(f"Effective:   {result.drift.is_effective}")         # True
print(f"Needs reset: {result.drift.needs_reset}")          # False

# Get managed context (original + session summaries) for your LLM
system_message = tracker.get_managed_context()

# End session — summarizes and preserves for next time
tracker.end_session()
```

## On-Demand vs Always-On Mode

Choose when drift scoring happens:

```python
# Always-on: scores every turn (default)
# Good for: monitoring dashboards, alerting
tracker = DriftTracker(system_prompt="You are a banking assistant.", mode="always")
result = tracker.record_turn(user_msg, assistant_msg)
print(result.drift.score)  # computed automatically

# On-demand: scores only when you ask
# Good for: production pipelines where you check periodically
tracker = DriftTracker(system_prompt="You are a banking assistant.", mode="ondemand")
tracker.record_turn(user_msg, assistant_msg)  # no scoring overhead
tracker.record_turn(user_msg2, assistant_msg2)

report = tracker.check()  # explicitly request drift check
print(report.drift.score)
print(report.explanation)
```

## Context Management (The Solution)

Most drift tools stop at detection. This package actually **solves** the problem by managing the context window intelligently:

```
┌──────────────────────────────────────────────────────┐
│              Managed Context Window                   │
├──────────────────────────────────────────────────────┤
│  [ALWAYS] Original System Prompt                     │
│  "You are a banking assistant for Acme Bank..."      │
│  [ALWAYS] Few-Shot Examples                          │
│  "Q: What interest rate? A: 4.5% APY..."            │
├──────────────────────────────────────────────────────┤
│  [AUTO] Session 1 Summary: "Customer asked about     │
│         savings accounts and CD rates."              │
│  [AUTO] Session 2 Summary: "Discussed home loan      │
│         requirements and mortgage pre-approval."     │
│  [AUTO] Session 3 Summary: "Helped with credit card  │
│         dispute and fraud alert process."            │
├──────────────────────────────────────────────────────┤
│  [LIVE] Current Conversation Turns                   │
└──────────────────────────────────────────────────────┘
```

**How it works:**
1. The original context (system prompt + few-shots) is **always preserved** — never truncated
2. At the end of each session, the conversation is **summarized** into 2-3 compact sentences
3. You configure how many past session summaries to keep (default: 3)
4. The managed context = original + summaries — use this as your system message
5. Old summaries are automatically dropped when `max_summary_sessions` is exceeded

```python
tracker = DriftTracker(
    system_prompt="You are a banking assistant for Acme Bank.",
    max_summary_sessions=3,
    summarize_fn=my_llm_summarizer,  # optional: use an LLM to summarize
)

# After each session:
tracker.end_session()

# Use this as your system message — includes original context + session summaries
system_message = tracker.get_managed_context()
```

### Custom Summarization (LLM-Powered)

By default, summaries use simple extractive logic (first + last sentences). For production, provide an LLM-based summarizer:

```python
from openai import OpenAI
client = OpenAI()

def llm_summarize(session_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize this banking conversation in 2-3 sentences. Focus on products discussed and customer needs."},
            {"role": "user", "content": session_text},
        ],
        max_tokens=100,
    )
    return response.choices[0].message.content

tracker = DriftTracker(
    system_prompt="You are a banking assistant.",
    summarize_fn=llm_summarize,
)
```

### Context Control

```python
# Freeze context — prevent any modifications to session history
tracker.freeze_context()

# Unfreeze to allow changes again
tracker.unfreeze_context()

# Clear all session summaries (original context preserved)
tracker.clear_history()

# Full reset — clears everything including .session_memory file
tracker.reset()
```

## Drift Explanation

Every drift score comes with a human-readable explanation of **why** drift occurred:

```python
# Banking assistant getting asked about cooking
result = tracker.record_turn(
    "What's a good pasta recipe?",
    "Try pasta carbonara with eggs, parmesan, pancetta, and black pepper..."
)
print(result.explanation)
# "Significant drift: conversation has moved away from original purpose.
#  Now focused on: carbonara, pasta, recipe, cooking."
```

Explanations are generated locally (no API calls) by default. You can plug in your own explainer:

```python
def llm_explain(original_context: str, recent_text: str, score: float) -> str:
    # Call an LLM to explain the drift
    ...

tracker = DriftTracker(system_prompt="...", explain_fn=llm_explain)
```

## Persistence (.session_memory File)

Enable persistence to track drift **across restarts and deploys**:

```python
tracker = DriftTracker(
    system_prompt="You are a banking assistant.",
    persist=True,
    persist_path=".session_memory",  # default
)
```

The `.session_memory` file is a plain JSON file stored locally:

| Field | Description |
|-------|-------------|
| `original_context` | The full initial context (system prompt + few-shots) |
| `session_summaries` | List of past session summaries |
| `session_count` | Total number of sessions |
| `total_turns` | Cumulative turn count |
| `context_frozen` | Whether context is frozen |
| `drift_history` | List of `{turn, session, score, verdict, explanation}` entries |
| `last_response_text` | Most recent response text |

> **Note:** Add `.session_memory` to your `.gitignore`. Do not commit it — it may contain content from your conversations.

## Embedding Strategies

Choose how drift is measured:

### Sentence Transformers (Recommended — Free, Local)

```python
from context_drift_analyzer.strategies.sentence_transformer import SentenceTransformerStrategy

tracker = DriftTracker(
    system_prompt="You are a banking assistant.",
    strategies=[SentenceTransformerStrategy(model_name="all-MiniLM-L6-v2")],
)
```

Models: `all-MiniLM-L6-v2` (80MB, fast), `all-mpnet-base-v2` (420MB, best quality), `paraphrase-MiniLM-L3-v2` (60MB, fastest).

### OpenAI Embeddings (Paid API)

```python
from openai import OpenAI
from context_drift_analyzer.strategies.openai_embedding import OpenAIEmbeddingStrategy

client = OpenAI()
tracker = DriftTracker(
    system_prompt="You are a banking assistant.",
    strategies=[OpenAIEmbeddingStrategy(client=client, model="text-embedding-3-small")],
)
```

### Bring Your Own Embedder

```python
from context_drift_analyzer.strategies.callable_embedding import CallableEmbeddingStrategy

def my_embedder(text: str) -> list[float]:
    # Cohere, Voyage, Google, custom model, etc.
    ...

tracker = DriftTracker(
    system_prompt="You are a banking assistant.",
    strategies=[CallableEmbeddingStrategy(embed_fn=my_embedder, strategy_name="cohere")],
)
```

### Keyword + Token Overlap (Zero Dependencies)

The default strategies when no embedding backend is installed:

```python
# No extra install needed — uses keyword hit-rate + TF cosine similarity
tracker = DriftTracker(system_prompt="You are a banking assistant.")
```

### Composite (Mix Multiple Strategies)

```python
from context_drift_analyzer.strategies.composite import CompositeStrategy
from context_drift_analyzer.strategies.sentence_transformer import SentenceTransformerStrategy
from context_drift_analyzer.strategies.keyword import KeywordStrategy

tracker = DriftTracker(
    system_prompt="You are a banking assistant.",
    strategies=[
        CompositeStrategy(
            strategies=[SentenceTransformerStrategy(), KeywordStrategy()],
            weights=[0.8, 0.2],  # 80% semantic, 20% keyword
        )
    ],
)
```

## CLI

```bash
# Show session memory status
context-drift-analyzer status
context-drift-analyzer status --file /path/to/.session_memory

# Show drift history
context-drift-analyzer history
context-drift-analyzer history --last 10

# Delete session memory
context-drift-analyzer reset

# Freeze/unfreeze context
context-drift-analyzer freeze
context-drift-analyzer unfreeze
```

## Drift Score Reference

| Score | Verdict | Meaning | Action |
|-------|---------|---------|--------|
| 90-100 | `FRESH` | Context well-preserved | None needed |
| 75-89 | `MILD` | Minor drift | Monitor |
| 55-74 | `MODERATE` | Noticeable drift | Consider intervention |
| 35-54 | `SEVERE` | Significant drift | Reset recommended |
| 0-34 | `CRITICAL` | Context largely lost | Reset required |

## Under the Hood

Here is exactly what happens when you call `tracker.record_turn()`:

```
1. USER MESSAGE recorded in Session
              ↓
2. ASSISTANT RESPONSE stripped of markdown formatting
   (code blocks, headers, bold, links removed to avoid false-positive drift)
              ↓
3. Cleaned response recorded in Session
              ↓
4. STRATEGY SCORING (if mode="always"):
   a. The initial context (system prompt + few-shots) is embedded → reference vector
      (cached after first call — never re-computed)
   b. Recent assistant responses (last N turns) are embedded → current vector
   c. Cosine similarity(reference, current) → raw score (0-1)
   d. Calibrated scaling [0, 0.55] → [0, 100] for meaningful scores
   e. Exponential decay applied: raw_score × decay_rate^(turns/2)
   f. Clamped to 0-100 → final drift score
              ↓
5. EXPLANATION generated:
   - Score-based analysis with topic comparison
   - Shared and divergent topics identified
   - 1-2 sentence explanation produced (locally, no API calls)
              ↓
6. PERSISTENCE (if enabled):
   - Drift entry appended to .session_memory drift_history
   - Session metadata updated
              ↓
7. TURN RESULT returned with:
   - drift score + verdict
   - explanation
   - managed context string
```

**When you call `tracker.end_session()`:**

```
1. Final drift score computed
2. Session text SUMMARIZED (extractive or LLM-based)
3. Summary added to ContextManager (capped at max_summary_sessions)
4. Session turns CLEARED
5. Session counter incremented
6. State persisted to .session_memory
7. Next session starts fresh with original context + summaries intact
```

## Cost and Latency

| Strategy | Cost | Latency per Turn | Install Size | Quality |
|----------|------|-------------------|-------------|---------|
| Keyword + Token Overlap (default) | **Free** | **<1ms** | **0 MB** | Basic (lexical) |
| Sentence Transformers (`all-MiniLM-L6-v2`) | **Free** | ~20-50ms (CPU) | ~80 MB | Good (semantic) |
| Sentence Transformers (`all-mpnet-base-v2`) | **Free** | ~50-100ms (CPU) | ~420 MB | Best (semantic) |
| OpenAI `text-embedding-3-small` | ~$0.02/1M tokens | ~100-200ms (API) | ~1 MB | Excellent |
| OpenAI `text-embedding-3-large` | ~$0.13/1M tokens | ~100-200ms (API) | ~1 MB | Best (API) |
| Custom callable | Varies | Varies | Varies | You decide |

## Configuration Reference

```python
DriftTracker(
    system_prompt="...",                     # Required: your system instructions
    few_shot_examples=[...],                 # Optional: FewShotExample pairs
    mode="always",                           # "always" or "ondemand"
    strategies=[...],                        # Optional: custom strategies
    decay_rate=0.95,                         # 0-1, lower = faster decay
    window_size=5,                           # recent turns to evaluate (0 = all)
    persist=False,                           # save to .session_memory
    persist_path=".session_memory",          # file path
    max_summary_sessions=3,                  # past session summaries to keep
    summarize_fn=None,                       # custom summarizer (str) -> str
    explain_fn=None,                         # custom explainer (str, str, float) -> str
    strip_md=True,                           # strip markdown before embedding
    frozen=False,                            # freeze context (no modifications)
)
```

## Project Structure

```
src/context_drift_analyzer/
  tracker.py               # DriftTracker — main entry point
  wrap.py                  # Drop-in client wrapper (OpenAI, Anthropic)
  core/
    analyzer.py            # Drift analysis engine
    scorer.py              # DriftScore, DriftVerdict
    session.py             # Session, Turn, FewShotExample
  context/
    manager.py             # Context window management + session summaries
    explainer.py           # Drift explanation generator
  persistence/
    session_memory.py      # .session_memory file read/write
  strategies/
    embedding_base.py      # Base class for embedding strategies
    sentence_transformer.py    # HuggingFace sentence-transformers
    openai_embedding.py    # OpenAI embedding API
    callable_embedding.py  # Bring-your-own embedder
    keyword.py             # Keyword hit-rate (lexical)
    token_overlap.py       # TF cosine similarity (lexical)
    composite.py           # Weighted multi-strategy combiner
  cli/
    main.py                # CLI tool (status/history/reset/freeze)
  utils/
    text.py                # Tokenization, TF vectors
    markdown.py            # Markdown stripping
tests/                     # 188 tests
examples/                  # Ready-to-run examples (banking chatbot)
```

## Running Tests

```bash
git clone https://github.com/Suman-Git-DS/ContextDriftAnalyzer.git
cd ContextDriftAnalyzer
pip install -e ".[dev]"
pytest tests/ -v
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [PyPI Package](https://pypi.org/project/context-drift-analyzer/)
- [GitHub Repository](https://github.com/Suman-Git-DS/ContextDriftAnalyzer)
- [Issue Tracker](https://github.com/Suman-Git-DS/ContextDriftAnalyzer/issues)
- [Changelog](CHANGELOG.md)
