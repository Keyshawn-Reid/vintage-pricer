import pandas as pd

df = pd.read_csv("data/raw/harley_raw.csv")

print(df.shape)
print(df.head())
print(df.describe())