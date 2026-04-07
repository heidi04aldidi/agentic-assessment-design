import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return None

load_dotenv()

try:
    import google.genai as genai
except ImportError:
    genai = None


def call_llm(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


def analyze_difficulty(difficulty_json, use_llm=False):
    problems = []
    easy   = difficulty_json.get("Easy", 0)
    medium = difficulty_json.get("Medium", 0)
    hard   = difficulty_json.get("Hard", 0)
    total  = difficulty_json.get("total", 1)

    easy_pct   = (easy   / total) * 100
    medium_pct = (medium / total) * 100
    hard_pct   = (hard   / total) * 100

    if hard_pct > 50:
        problems.append(f"{hard_pct:.1f}% questions are Hard — exam is too difficult")
    if easy_pct < 15:
        problems.append("Very few Easy questions — students may lack confidence building")
    if easy_pct > 60:
        problems.append("Too many Easy questions — exam may not challenge students enough")
    if not (25 <= easy_pct <= 35 and 35 <= medium_pct <= 45 and 25 <= hard_pct <= 35):
        problems.append("Difficulty distribution is imbalanced (not close to 30-40-30 ideal)")
    if medium_pct < 20:
        problems.append("Too few Medium questions — lack of conceptual depth")
    if easy == 0 or medium == 0 or hard == 0:
        problems.append("One or more difficulty levels missing — poor exam structure")

    if use_llm:
        prompt = f"""
Analyze this exam difficulty distribution:
Easy: {easy_pct:.1f}%
Medium: {medium_pct:.1f}%
Hard: {hard_pct:.1f}%
Identify problems in this exam. Return 3-5 short bullet points.
"""
        llm_output = call_llm(prompt)
        if llm_output:
            problems.append("LLM Insights:")
            problems.append(llm_output.strip())

    return problems


def run_analyzer_agent(state: dict) -> dict:
    difficulty_json = state.get("difficulty", {})
    problems = analyze_difficulty(difficulty_json)
    state["problems"] = problems
    return state
