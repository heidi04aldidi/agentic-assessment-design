import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents.analyzer import run_analyzer_agent
from agents.retriever import run_retriever_agent

state = {
    "difficulty": {
        "Easy": 45,
        "Medium": 25,
        "Hard": 30,
        "total": 100
    }
}

# Agent 1 runs first
agent_1 = run_analyzer_agent(state)
print("Agent 1 Problems:", agent_1["problems"])

# Agent 2 runs second
agent_2 = run_retriever_agent(state)
print("\nAgent 2 Principles:")
for p in agent_2["principles"]:
    print(f"  • {p}")