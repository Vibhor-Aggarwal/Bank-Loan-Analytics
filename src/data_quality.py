import pandas as pd
df = pd.read_csv("data/raw/loans_full_schema.csv")

print("="*70)
print("DATA QUALITY REPORT")
print("="*70)

duplicates = df.duplicated().sum()
print("\nduplicate rows")
print(duplicates)

missing = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Percentage" : (df.isnull().sum()/len(df))*100
})

missing = missing.sort_values(
    by="Percentage",
    ascending=False
)

print('\nMissing Values Report')
print(missing)

print("\nData Types")
print(df.dtypes)

missing.to_csv(
    "reports/missing_value_report.csv"
)

print("\nReport Saved Successfully")