"""Three-agent ramp assistant using LangChain and Claude."""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from .state import RampState

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)
parser = JsonOutputParser()


def extract_concepts(state: RampState) -> dict:
    """Agent 1: Extract key technical concepts from product documentation."""

    messages = [
        SystemMessage(content="""You are a technical curriculum designer.
Given product documentation, extract the key concepts a new technical seller
needs to understand to have credible customer conversations.

Return JSON array:
[
  {
    "name": "concept name",
    "description": "one-sentence explanation",
    "difficulty": "foundational|intermediate|advanced",
    "why_it_matters": "why a seller needs to know this for customer conversations"
  }
]

Extract 5-8 concepts. Focus on what matters for technical sales conversations,
not exhaustive product knowledge. Return ONLY valid JSON."""),
        HumanMessage(content=f"""Product: {state['product_name']}

Documentation:
{state['product_docs'][:8000]}"""),
    ]

    response = llm.invoke(messages)
    concepts = parser.parse(response.content)

    return {"key_concepts": concepts}


def generate_scenarios(state: RampState) -> dict:
    """Agent 2: Generate practice scenarios from extracted concepts."""

    concepts_text = json.dumps(state["key_concepts"], indent=2)

    messages = [
        SystemMessage(content="""You are a sales engineering coach.
Given key product concepts, generate realistic customer conversation scenarios
that a new technical seller would encounter in their first 90 days.

Each scenario should simulate a real buyer interaction — discovery calls,
technical deep dives, objection handling, or architecture discussions.

Return JSON array:
[
  {
    "title": "scenario title",
    "buyer_persona": "who the buyer is (e.g., Platform Engineer, CISO, VP Infra)",
    "context": "2-3 sentences setting up the situation",
    "buyer_question": "the specific question or objection the buyer raises",
    "concepts_tested": ["which concepts from the list this tests"],
    "difficulty": "foundational|intermediate|advanced"
  }
]

Generate 3-5 scenarios across different difficulty levels.
Make them feel like real conversations, not textbook questions.
Return ONLY valid JSON."""),
        HumanMessage(content=f"""Product: {state['product_name']}

Key concepts to build scenarios from:
{concepts_text}"""),
    ]

    response = llm.invoke(messages)
    scenarios = parser.parse(response.content)

    return {"scenarios": scenarios}


def evaluate_response(state: RampState) -> dict:
    """Agent 3: Evaluate user's response to a scenario and provide coaching."""

    scenario = state["scenarios"][0] if state["scenarios"] else {}
    concepts_text = json.dumps(state["key_concepts"], indent=2)

    messages = [
        SystemMessage(content="""You are a senior solutions engineer and coach.
A new technical seller just responded to a practice scenario. Evaluate their
response based on:

1. Technical accuracy — did they get the product concepts right?
2. Buyer awareness — did they address what the buyer actually cares about?
3. Conversation quality — did they sound like a trusted advisor or a script reader?

Return JSON:
{
  "score": 1-10,
  "technical_accuracy": "assessment of product knowledge",
  "buyer_awareness": "assessment of how well they read the buyer",
  "conversation_quality": "assessment of delivery and confidence",
  "strengths": ["what they did well"],
  "gaps": ["what they missed or got wrong"],
  "coaching_tip": "one specific thing to practice next",
  "model_response": "how a senior SE would handle this scenario (2-3 sentences)"
}

Be direct and specific. Generic praise is useless.
Return ONLY valid JSON."""),
        HumanMessage(content=f"""Product: {state['product_name']}

Available concepts:
{concepts_text}

Scenario:
{json.dumps(scenario, indent=2)}

Seller's response:
{state['user_response']}"""),
    ]

    response = llm.invoke(messages)
    evaluation = parser.parse(response.content)

    return {"evaluation": evaluation}
