import pandas as pd 

df = pd.read_csv("data/raw/lending_club_loan_two.csv")

df.info()

df.head()

df.describe(include="all")

df.isnull().sum().sort_values(ascending=False)

df["loan_status"].value_counts()

print(df["loan_status"].value_counts())
print(df.isnull().sum().sort_values(ascending=False))