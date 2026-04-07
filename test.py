from src.agents.analyzer import analyze_difficulty

# Simulate your app output
test_input = {
    "Easy": 45,
    "Medium": 25,
    "Hard": 30,
    "total": 100
}

problems = analyze_difficulty(test_input)

print("Detected Problems:")
for p in problems:
    print("-", p)