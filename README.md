# Ramp Assistant

A multi-agent technical onboarding tool built with **LangChain**, **LangGraph**, **LangSmith**, and **Claude**.

Helps new technical sellers ramp on complex products by extracting key concepts from product documentation, generating realistic customer conversation scenarios, and evaluating responses with coaching feedback.

## Why This Exists

Most onboarding programs dump content on new hires and call it done. Completion ≠ competency. A new seller who watched 40 hours of product videos still freezes when a VP of Infrastructure asks "why shouldn't I just use open source?"

This tool applies an **80/20 learning model** — 20% concept extraction, 80% scenario practice — so new sellers practice the conversations they'll actually have, not memorize slides they'll forget.

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Agent 1:       │     │  Agent 2:            │     │  Agent 3:           │
│  Concept        │────▶│  Scenario            │────▶│  Response           │
│  Extractor      │     │  Generator           │     │  Evaluator          │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘
     LangChain               LangChain                    LangChain
         │                       │                            │
         └───────────────────────┴────────────────────────────┘
                          LangGraph (orchestration)
                          LangSmith (tracing & observability)
                          Claude (LLM)
```

**Agent 1 — Concept Extractor:** Reads product documentation and identifies the 5-8 concepts a new seller needs for credible customer conversations. Not a feature list — the concepts that win or lose trust in a live room.

**Agent 2 — Scenario Generator:** Creates realistic buyer conversation scenarios across difficulty levels. Each scenario has a buyer persona, business context, and a specific question or objection the seller must handle. Designed to feel like a real call, not a quiz.

**Agent 3 — Response Evaluator:** Coaches the seller on their response. Evaluates technical accuracy, buyer awareness, and conversation quality. Provides specific gaps, strengths, and a model response showing how a senior practitioner would handle it. No generic praise — direct, actionable feedback.

## Quick Start

```bash
# Clone and install
git clone https://github.com/celine-valentine/langchain-ramp-assistant.git
cd langchain-ramp-assistant
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your ANTHROPIC_API_KEY and LANGCHAIN_API_KEY

# Run
python main.py
```

## Usage

```
$ python main.py

  Product name: HashiCorp Vault
  Paste product docs (or press Enter for sample): [Enter]

  ── KEY CONCEPTS EXTRACTED ──
  1. [FOUNDATIONAL] Dynamic Secrets
     On-demand credential generation with automatic TTL-based revocation.
     Why it matters: Buyers ask "how is this different from a password vault?"

  ── PRACTICE SCENARIOS ──
  Scenario 1: "We already use AWS Secrets Manager"
  Buyer: VP of Platform Engineering
  Context: Mid-evaluation, leaning toward native cloud tooling...

  ── YOUR TURN ──
  How would you respond to this buyer?
  > ...

  ── COACHING FEEDBACK ──
  Score: 7/10
  Strengths: Correctly positioned multi-cloud value
  Gaps: Missed the dynamic secrets angle — that's the differentiator
  Coaching tip: Lead with what changes for the developer, not the security team
```

## Observability

All agent runs are traced in **LangSmith**. Each session shows:
- Token usage per agent
- Latency breakdown across the pipeline
- Full prompt/response history for debugging
- Evaluation scoring trends over time

## Design Principles

- **Practice > content.** The tool generates scenarios, not slides.
- **Coaching > grading.** Feedback is specific and actionable, not a letter grade.
- **Buyer-centric.** Every scenario starts from what the buyer cares about, not what the product does.
- **Product-agnostic.** Swap the docs, get new scenarios. Works for any technical product.

## Built With

- [LangChain](https://langchain.com) — LLM application framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Multi-agent orchestration
- [LangSmith](https://smith.langchain.com) — Observability and tracing
- [Claude](https://anthropic.com) — Anthropic's LLM

## Author

**Celine Valentine** — [celinevalentine.com](https://celinevalentine.com)

Field readiness leader, presales practitioner, and learning science researcher. Writing about what separates technical sellers who close from those who demo.
