import pandas as pd

# load dataset
df = pd.read_csv("data/dataset.csv")

# print data
print("First 5 rows:\n")
print(df.head())

print("\nColumn names:\n")
print(df.columns)