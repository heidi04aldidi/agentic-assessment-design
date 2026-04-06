# src/agents/analyzer.py
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

# OPTIONAL: LLM usage (can be skipped initially)
def call_llm(prompt):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(prompt)
    return response.text


def analyze_difficulty(difficulty_json, use_llm=False, api_key=None):
    """
    Agent 1: Analyzer

    Input:
        difficulty_json (dict)

    Output:
        list of detected problems
    """

    problems = []

    # Extract values safely
    easy = difficulty_json.get("Easy", 0)
    medium = difficulty_json.get("Medium", 0)
    hard = difficulty_json.get("Hard", 0)
    total = difficulty_json.get("total", 1)

    # Convert to percentages
    easy_pct = (easy / total) * 100
    medium_pct = (medium / total) * 100
    hard_pct = (hard / total) * 100

    # ──────────────────────────────
    # RULE-BASED ANALYSIS
    # ──────────────────────────────

    # 1. Too many hard questions
    if hard_pct > 50:
        problems.append(f"{hard_pct:.1f}% questions are Hard — exam is too difficult")

    # 2. Too few easy questions
    if easy_pct < 15:
        problems.append("Very few Easy questions — students may lack confidence building")

    # 3. Too many easy questions
    if easy_pct > 60:
        problems.append("Too many Easy questions — exam may not challenge students enough")

    # 4. Imbalance detection
    if not (25 <= easy_pct <= 35 and 35 <= medium_pct <= 45 and 25 <= hard_pct <= 35):
        problems.append("Difficulty distribution is imbalanced (not close to 30-40-30 ideal)")

    # 5. Too few medium questions
    if medium_pct < 20:
        problems.append("Too few Medium questions — lack of conceptual depth")

    # 6. Edge case: no variation
    if easy == 0 or medium == 0 or hard == 0:
        problems.append("One or more difficulty levels missing — poor exam structure")

    # ──────────────────────────────
    # OPTIONAL: LLM Enhancement
    # ──────────────────────────────
    if use_llm and api_key:
        prompt = f"""
        Analyze this exam difficulty distribution:

        Easy: {easy_pct:.1f}%
        Medium: {medium_pct:.1f}%
        Hard: {hard_pct:.1f}%

        Identify problems in this exam. Return 3–5 short bullet points.
        """

        llm_output = call_llm(prompt, api_key)

        if llm_output:
            problems.append("LLM Insights:")
            problems.append(llm_output.strip())

    return problems

def test_api_connection():
    api_key = os.getenv("GEMINI_API_KEY")

    print("Loaded API Key:", api_key)

    if not api_key:
        print("API key not found!")
        return

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        response = model.generate_content("Say hello in one sentence.")
        print("API Response:", response.text)

    except Exception as e:
        print("API Error:", str(e))

if __name__ == "__main__":
    test_api_connection()