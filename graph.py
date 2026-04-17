"""
graph.py — Pipeline Orchestrator
Chains all 4 agents in order using plain function calls (no LangGraph).

Pipeline flow:
  difficulty_dict
      → run_analyzer_agent   → state["problems"]
      → run_retriever_agent  → state["principles"]
      → recommend_agent       → state["recommendations"]
      → generate_report      → markdown string  (returned, not written to state)
"""

import sys
import os

# Ensure src/ is on the path so agent imports resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agents.analyzer  import run_analyzer_agent
from agents.retriever import run_retriever_agent
from agents.recommend import recommend_agent
from agents.reporter  import generate_report


def run_pipeline(difficulty_dict: dict) -> str:
    """
    Run the full 4-agent assessment pipeline.

    Args:
        difficulty_dict: e.g. {"Easy": 10, "Medium": 20, "Hard": 70, "total": 100}

    Returns:
        A Markdown-formatted assessment quality report string.
    """
    # Build initial state
    state = {
        "difficulty": difficulty_dict,
    }

    # Agent 1 — Analyze difficulty and identify problems
    print("▶ Running Agent 1: Analyzer...")
    state = run_analyzer_agent(state)
    print(f"   Problems found: {len(state.get('problems', []))}")

    # Agent 2 — Retrieve relevant pedagogical principles
    print("▶ Running Agent 2: Retriever...")
    state = run_retriever_agent(state)
    print(f"   Principles retrieved: {len(state.get('principles', []))}")

    # Agent 3 — Generate recommendations
    print("▶ Running Agent 3: Recommender...")
    state = recommend_agent(state)
    print(f"   Recommendations generated: {len(state.get('recommendations', []))}")

    # Agent 4 — Generate final report (returns markdown string, not written to state)
    print("▶ Running Agent 4: Reporter...")
    report = generate_report(state)
    print("   Report generated.\n")

    return report


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_difficulty = {
        "Easy":   10,
        "Medium": 20,
        "Hard":   70,
        "total":  100,
    }

    report = run_pipeline(test_difficulty)
    print(report)
