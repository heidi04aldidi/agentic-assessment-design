try:
    from rag.retriever import retrieve_relevant_principles
except ImportError:
    def retrieve_relevant_principles(problems, top_k=3):
        return [
            "A well-balanced exam should have 30% Easy, 40% Medium, 30% Hard questions.",
            "Bloom's Taxonomy suggests evaluating recall, understanding, and application.",
            "Assessments should begin with easier questions to build student confidence.",
        ]


def run_retriever_agent(state: dict) -> dict:
    problems = state.get("problems", [])

    if not problems:
        state["principles"] = ["No problems identified — exam appears well-balanced."]
        return state

    principles = retrieve_relevant_principles(problems, top_k=3)
    state["principles"] = principles

    print("Agent 2 — Retrieved Principles:")
    for p in principles:
        print(f"  → {p}")

    return state
