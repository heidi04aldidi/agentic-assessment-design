import os

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


def _fallback_recommendations(problems: list, state: dict = None) -> list:
    """Rule-based recommendations when LLM is unavailable (based on actual distribution)."""
    recs = []
    
    diff = state.get("difficulty", {}) if state else {}
    total = diff.get("total", 1)
    easy_pct = (diff.get("Easy", 0) / total) * 100
    med_pct = (diff.get("Medium", 0) / total) * 100
    hard_pct = (diff.get("Hard", 0) / total) * 100

    if hard_pct > 35:
        recs.append(f"1. Reduce Hard questions to ~30% (currently {hard_pct:.0f}%) to alleviate student anxiety.")
    elif hard_pct < 25:
        recs.append("1. Add a few more complex questions to adequately challenge top-performing students.")
    else:
        recs.append("1. Current Hard ratio is good, maintain it.")

    if med_pct < 35:
        recs.append("2. Add 1-2 more Medium difficulty questions on core topics to strengthen conceptual coverage.")
    elif med_pct > 50:
        recs.append("2. Convert some Medium questions into Easy questions for a better warm-up.")
    else:
        recs.append("2. Consider adding questions on related foundational topics like error handling.")

    if easy_pct < 25:
        recs.append("3. Incorporate more Easy recall questions at the beginning of the exam to build student confidence.")
    elif easy_pct > 40:
        recs.append("3. Reduce the amount of Easy questions to ensure the exam remains challenging.")
    else:
        recs.append("3. Present difficulty structure follows standard pedagogical guidelines well.")

    return recs[:3]


def recommend_agent(state: dict) -> dict:
    problems = state.get("problems", [])
    principles = state.get("principles", [])
    topic_analysis = state.get("topic_analysis", {})

    topic_text_lines = []
    if topic_analysis:
        for topic, data in topic_analysis.items():
            topic_text_lines.append(f"- {topic}: Avg Score {data.get('score', 0)} ({data.get('difficulty', 'Unknown')})")
    topic_text = "\n".join(topic_text_lines) if topic_text_lines else "None provided"

    prompt = f"""
        You are an expert in educational assessment design.
        Problems identified in the exam:
        {chr(10).join(f"- {p}" for p in problems)}
        Relevant teaching principles:
        {chr(10).join(f"- {r}" for r in principles)}
        
        Topic-Level Performance Data:
        {topic_text}

        Task:
        Give exactly 3 clear, actionable recommendations to improve the exam.
        Rules:
        - Be concise
        - Be specific! Link recommendations directly to specific topics where students are struggling based on the performance data.
        - Output as a numbered list"""

    try:
        if not _GROQ_AVAILABLE:
            raise RuntimeError("groq not installed")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip().split("\n")
        recommendations = [line.strip() for line in result if line.strip()]
        print("   (LLM recommendations generated via Groq)")

    except Exception as e:
        print(f"   ⚠ LLM unavailable ({type(e).__name__}), using rule-based fallback.")
        recommendations = _fallback_recommendations(problems, state)

    state["recommendations"] = recommendations
    return state