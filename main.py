"""
Ramp Assistant — Multi-agent technical onboarding tool.

Built with LangChain, LangGraph, LangSmith, and Claude.

Usage:
    python main.py

Extracts key concepts from product docs, generates realistic
customer conversation scenarios, and evaluates your responses
with coaching feedback.
"""

import json
import os

from dotenv import load_dotenv

from ramp_assistant.graph import build_evaluation_graph, build_ramp_graph

load_dotenv()

# ── Sample product docs (replace with any product) ──────────────────────

SAMPLE_DOCS = """
HashiCorp Vault is a secrets management and data protection platform.
It provides a unified interface to any secret, while providing tight
access control and recording a detailed audit log.

Key capabilities:
- Secrets Management: Centrally store, access, and deploy secrets across
  applications, systems, and infrastructure. Vault handles leasing,
  key revocation, key rolling, and auditing.
- Dynamic Secrets: Generate secrets on-demand for systems like AWS, SQL
  databases, and more. Vault can automatically revoke these secrets
  after a configurable TTL.
- Data Encryption: Encrypt and decrypt data without storing it.
  Encryption as a Service lets developers encrypt data at the
  application layer without managing encryption keys.
- Identity-based Access: Authenticate and authorize users and machines
  using trusted identity providers. Vault ties authentication to
  authorization policies for fine-grained access control.
- PKI & Certificate Management: Generate and manage X.509 certificates
  for TLS, mTLS, and code signing. Automated certificate lifecycle
  management eliminates manual processes.

Common use cases:
- Replacing hardcoded credentials in application configs
- Automating database credential rotation
- Encrypting sensitive data at rest and in transit
- Managing SSH access across cloud infrastructure
- Zero trust security architecture implementation
"""


def print_header(text: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_concepts(concepts: list[dict]) -> None:
    """Display extracted concepts."""
    for i, c in enumerate(concepts, 1):
        difficulty = c.get("difficulty", "unknown").upper()
        print(f"  {i}. [{difficulty}] {c['name']}")
        print(f"     {c['description']}")
        print(f"     Why it matters: {c.get('why_it_matters', 'N/A')}")
        print()


def print_scenario(scenario: dict) -> None:
    """Display a practice scenario."""
    print(f"  Buyer: {scenario.get('buyer_persona', 'Unknown')}")
    print(f"  Difficulty: {scenario.get('difficulty', 'Unknown').upper()}")
    print(f"  Context: {scenario.get('context', '')}")
    print(f"\n  Buyer says: \"{scenario.get('buyer_question', '')}\"")
    print()


def print_evaluation(evaluation: dict) -> None:
    """Display coaching evaluation."""
    score = evaluation.get("score", "?")
    print(f"  Score: {score}/10")
    print(f"\n  Technical Accuracy: {evaluation.get('technical_accuracy', '')}")
    print(f"  Buyer Awareness: {evaluation.get('buyer_awareness', '')}")
    print(f"  Conversation Quality: {evaluation.get('conversation_quality', '')}")

    strengths = evaluation.get("strengths", [])
    if strengths:
        print(f"\n  Strengths:")
        for s in strengths:
            print(f"    + {s}")

    gaps = evaluation.get("gaps", [])
    if gaps:
        print(f"\n  Gaps:")
        for g in gaps:
            print(f"    - {g}")

    tip = evaluation.get("coaching_tip", "")
    if tip:
        print(f"\n  Coaching tip: {tip}")

    model = evaluation.get("model_response", "")
    if model:
        print(f"\n  How a senior SE would handle this:")
        print(f"  \"{model}\"")


def main():
    """Run the ramp assistant interactively."""

    print_header("RAMP ASSISTANT — Technical Onboarding Coach")
    print("  Built with LangChain + LangGraph + Claude")
    print("  Traces visible in LangSmith\n")

    product_name = input("  Product name (or press Enter for 'HashiCorp Vault'): ").strip()
    if not product_name:
        product_name = "HashiCorp Vault"

    product_docs = SAMPLE_DOCS
    custom_docs = input("  Paste product docs (or press Enter to use sample): ").strip()
    if custom_docs:
        product_docs = custom_docs

    # ── Phase 1: Extract concepts and generate scenarios ─────────────

    print_header("PHASE 1: Analyzing Product & Generating Scenarios")
    print("  Running concept extraction → scenario generation pipeline...")

    ramp_graph = build_ramp_graph()

    initial_state = {
        "product_docs": product_docs,
        "product_name": product_name,
        "key_concepts": [],
        "scenarios": [],
        "user_response": "",
        "evaluation": {},
    }

    result = ramp_graph.invoke(initial_state)

    print_header("KEY CONCEPTS EXTRACTED")
    print_concepts(result["key_concepts"])

    print_header("PRACTICE SCENARIOS")
    for i, scenario in enumerate(result["scenarios"], 1):
        print(f"  --- Scenario {i}: {scenario.get('title', 'Untitled')} ---")
        print_scenario(scenario)

    # ── Phase 2: User practices a scenario ───────────────────────────

    print_header("YOUR TURN — Practice Time")

    scenario_count = len(result["scenarios"])
    if scenario_count == 0:
        print("  No scenarios generated. Exiting.")
        return

    choice = input(f"  Pick a scenario (1-{scenario_count}): ").strip()
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= scenario_count:
            idx = 0
    except ValueError:
        idx = 0

    chosen = result["scenarios"][idx]
    print(f"\n  --- {chosen.get('title', 'Scenario')} ---")
    print_scenario(chosen)

    print("  How would you respond to this buyer?")
    print("  (Type your response, then press Enter twice to submit)\n")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    user_response = "\n".join(lines)

    # ── Phase 3: Evaluate response ───────────────────────────────────

    print_header("PHASE 2: Evaluating Your Response")
    print("  Running coaching evaluation...")

    eval_graph = build_evaluation_graph()

    eval_state = {
        "product_docs": product_docs,
        "product_name": product_name,
        "key_concepts": result["key_concepts"],
        "scenarios": [chosen],
        "user_response": user_response,
        "evaluation": {},
    }

    eval_result = eval_graph.invoke(eval_state)

    print_header("COACHING FEEDBACK")
    print_evaluation(eval_result["evaluation"])

    print(f"\n{'='*60}")
    print("  Session complete. Traces available in LangSmith.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
