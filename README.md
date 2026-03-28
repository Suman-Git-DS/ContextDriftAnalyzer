<p align="center">
  <h1 align="center">context-decay-drift</h1>
  <p align="center">
    Detect, explain, and <strong>solve</strong> context drift in LLM conversations across sessions.
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/context-decay-drift/"><img alt="PyPI" src="https://img.shields.io/pypi/v/context-decay-drift"></a>
  <a href="https://pypi.org/project/context-decay-drift/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/context-decay-drift"></a>
  <a href="https://github.com/Suman-Git-DS/ContextDecayDrift/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Suman-Git-DS/ContextDecayDrift"></a>
</p>

---

## The Problem

LLM-powered chatbots lose focus over long conversations. After several sessions, the model "forgets" its system prompt and few-shot examples, leading to off-topic responses, reduced accuracy, and poor user experience — with no visibility into *when* this happens or *why*.

Most tools only **detect** drift. This package **detects, explains, and solves** it:

| Capability | What it does |
|-----------|-------------|
| **Detect** | Drift score (0-100) via semantic embeddings |
| **Explain** | 1-2 line human-readable reason for drift |
| **Solve** | Context management: original context + session summaries, kept within token budget |
| **Persist** | `.session_memory` file tracks drift across restarts and deploys |

## How It Works

```
Session 1 (Turn 1-2):  Score 92  [FRESH]      "Context well-preserved."
Session 2 (Turn 3-6):  Score 76  [MILD]       "Mild drift: key topics fading — loops, classes."
Session 3 (Turn 7-12): Score 53  [SEVERE]      "Only 35% of original keywords present."
Session 4 (Turn 13+):  Score 28  [CRITICAL]    "Conversation departed from original purpose."
                                                ↑ recommend reset
```

## Installation

```bash
# Core + Sentence Transformers (recommended — free, local, semantic)
pip install context-decay-drift[semantic]

# Core only (zero dependencies — keyword/TF strategies, or bring your own embedder)
pip install context-decay-drift

# Everything (semantic + OpenAI + Anthropic embedding support)
pip install context-decay-drift[all]
```

## Quick Start

```python
from context_decay_drift import DriftTracker, FewShotExample

tracker = DriftTracker(
    system_prompt="You are a Python programming tutor. Always provide code examples.",
    few_shot_examples=[
        FewShotExample(user="What is a variable?", assistant="A variable stores data. Example: x = 5"),
    ],
    mode="always",              # "always" or "ondemand"
    persist=True,               # save to .session_memory file
    max_summary_sessions=3,     # keep last 3 session summaries
)

# After each LLM call in your pipeline:
result = tracker.record_turn(
    user_message="How do loops work?",
    assistant_response="Use for loops: for i in range(10): print(i)"
)

print(f"Score:       {result.drift.score:.1f}/100")      # 85.2/100
print(f"Verdict:     {result.drift.verdict.value}")       # "mild"
print(f"Explanation: {result.explanation}")                # "Mild drift: ..."
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
tracker = DriftTracker(system_prompt="...", mode="always")
result = tracker.record_turn(user_msg, assistant_msg)
print(result.drift.score)  # computed automatically

# On-demand: scores only when you ask
# Good for: production pipelines where you check periodically
tracker = DriftTracker(system_prompt="...", mode="ondemand")
tracker.record_turn(user_msg, assistant_msg)  # no scoring overhead
tracker.record_turn(user_msg2, assistant_msg2)

report = tracker.check()  # explicitly request drift check
print(report.drift.score)
print(report.explanation)
```

## Context Management (The Solution)

Most drift tools stop at detection. This package actually **solves** the problem by managing the context window intelligently:

```
┌──────────────────────────────────────────────┐
│           Managed Context Window              │
├──────────────────────────────────────────────┤
│  [ALWAYS] Original System Prompt             │
│  [ALWAYS] Few-Shot Examples                  │
├──────────────────────────────────────────────┤
│  [AUTO] Session 1 Summary (2-3 sentences)    │
│  [AUTO] Session 2 Summary (2-3 sentences)    │
│  [AUTO] Session 3 Summary (2-3 sentences)    │
├──────────────────────────────────────────────┤
│  [LIVE] Current Conversation Turns           │
└──────────────────────────────────────────────┘
```

**How it works:**
1. The original context (system prompt + few-shots) is **always preserved** — never truncated
2. At the end of each session, the conversation is **summarized** into 2-3 compact sentences
3. You configure how many past session summaries to keep (default: 3)
4. The managed context = original + summaries — use this as your system message
5. Old summaries are automatically dropped when `max_summary_sessions` is exceeded

```python
tracker = DriftTracker(
    system_prompt="You are a Python tutor.",
    max_summary_sessions=3,       # keep last 3 session summaries
    summarize_fn=my_llm_summarizer,  # optional: use an LLM to summarize (see below)
)

# After each session:
tracker.end_session()

# Use this as your system message — it includes original context + session summaries
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
            {"role": "system", "content": "Summarize this conversation in 2-3 sentences. Focus on key topics discussed."},
            {"role": "user", "content": session_text},
        ],
        max_tokens=100,
    )
    return response.choices[0].message.content

tracker = DriftTracker(
    system_prompt="...",
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
result = tracker.record_turn("What's for dinner?", "Try pasta carbonara...")
print(result.explanation)
# "Significant drift: only 25% of original context keywords present.
#  Conversation shifted toward: carbonara, pasta, recipe."
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
    system_prompt="...",
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
from context_decay_drift.strategies.sentence_transformer import SentenceTransformerStrategy

tracker = DriftTracker(
    system_prompt="...",
    strategies=[SentenceTransformerStrategy(model_name="all-MiniLM-L6-v2")],
)
```

Models: `all-MiniLM-L6-v2` (80MB, fast), `all-mpnet-base-v2` (420MB, best quality), `paraphrase-MiniLM-L3-v2` (60MB, fastest).

### OpenAI Embeddings (Paid API)

```python
from openai import OpenAI
from context_decay_drift.strategies.openai_embedding import OpenAIEmbeddingStrategy

client = OpenAI()
tracker = DriftTracker(
    system_prompt="...",
    strategies=[OpenAIEmbeddingStrategy(client=client, model="text-embedding-3-small")],
)
```

### Bring Your Own Embedder

```python
from context_decay_drift.strategies.callable_embedding import CallableEmbeddingStrategy

def my_embedder(text: str) -> list[float]:
    # Cohere, Voyage, Google, custom model, etc.
    ...

tracker = DriftTracker(
    system_prompt="...",
    strategies=[CallableEmbeddingStrategy(embed_fn=my_embedder, strategy_name="cohere")],
)
```

### Keyword + Token Overlap (Zero Dependencies)

The default strategies when no embedding backend is installed:

```python
# No extra install needed — uses keyword hit-rate + TF cosine similarity
tracker = DriftTracker(system_prompt="...")
```

### Composite (Mix Multiple Strategies)

```python
from context_decay_drift.strategies.composite import CompositeStrategy
from context_decay_drift.strategies.sentence_transformer import SentenceTransformerStrategy
from context_decay_drift.strategies.keyword import KeywordStrategy

tracker = DriftTracker(
    system_prompt="...",
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
context-decay-drift status
context-decay-drift status --file /path/to/.session_memory

# Show drift history
context-decay-drift history
context-decay-drift history --last 10

# Delete session memory
context-decay-drift reset

# Freeze/unfreeze context
context-decay-drift freeze
context-decay-drift unfreeze
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
   d. Exponential decay applied: raw_score × decay_rate^(turns/2)
   e. Clamped to 0-100 → final drift score
              ↓
5. EXPLANATION generated:
   - Keywords from original context vs response are compared
   - Missing/new topics identified
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

**Session summarization** (optional):
- Default extractive: **Free, <1ms**
- LLM-based (e.g., GPT-4o-mini): ~$0.15/1M tokens, ~500ms per session end

**Context management overhead:** Zero. It's just string concatenation.

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
src/context_decay_drift/
  tracker.py               # DriftTracker — main entry point
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
tests/                     # 164 tests
examples/                  # Ready-to-run examples
```

## Running Tests

```bash
git clone https://github.com/Suman-Git-DS/ContextDecayDrift.git
cd ContextDecayDrift
pip install -e ".[dev]"
pytest tests/ -v
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [PyPI Package](https://pypi.org/project/context-decay-drift/)
- [GitHub Repository](https://github.com/Suman-Git-DS/ContextDecayDrift)
- [Issue Tracker](https://github.com/Suman-Git-DS/ContextDecayDrift/issues)
