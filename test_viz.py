import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

df = pd.read_csv('/Users/mac/Downloads/questions.csv')
print(df.columns)
if "Score" in df.columns:
    print("Score min/max:", df["Score"].min(), df["Score"].max())
    fig, ax = plt.subplots(figsize=(8, 4))
    df["Score"].clip(-5, 100).hist(bins=40, ax=ax, color="#764ba2", edgecolor="white")
    plt.savefig('test_hist.png')
    print("Saved test_hist.png")

    q_low = df["Score"].quantile(0.33)
    q_high = df["Score"].quantile(0.66)
    df["Difficulty"] = pd.cut(
        df["Score"],
        bins=[-np.inf, q_low, q_high, np.inf],
        labels=["Hard", "Medium", "Easy"],
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="Score", by="Difficulty", ax=ax)
    plt.savefig('test_boxplot.png')
    print("Saved test_boxplot.png")
