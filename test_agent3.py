from src.agents.recommend import recommend_agent

state = {
    "problems": [
        "70% of questions are Hard — exam is too difficult",
        "No Easy questions present to build student confidence",
        "Python loops topic dominates 80% of the exam"
    ],
    "principles": [
        "Exams with more than 60% Hard questions cause student anxiety and reduce learning outcomes.",
        "A well-balanced exam should have 30% Easy, 40% Medium, and 30% Hard questions based on Bloom's Taxonomy.",
        "Easy questions build student confidence and should never be completely absent."
    ]

}
final = recommend_agent(state)
print("Final recommendations:", final["recommendations"])