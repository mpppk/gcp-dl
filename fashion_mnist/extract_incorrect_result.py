import pandas as pd

df = pd.read_csv("results.csv")
df[df["actual"] != df["predict"]].to_csv("incorrect.csv", index=False)