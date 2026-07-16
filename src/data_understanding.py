import pandas as pd

print("=" * 50)
print("BANK LOAN ANALYTICS PROJECT")
print("=" * 50) 

df = pd.read_csv("data/raw/loans_full_schema.csv")
print("\nData Loaded Successfully")
print("Dataset Information : ")
print(df.info())

print("\n" + "=" *50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

missing_percentage = (df.isnull().sum()/len(df)) * 100
print("=" *50)
print("MISSING PERCENTAGE ")
print("="*50)
print(missing_percentage)

print("=" *60)
print("STATISTICAL SUMMARY")
print("="*60)
print(df.describe())

print("="*50)
print("DATASET DIMENSIONS")
print("="*50)
rows, columns = df.shape

print(f"Total records : {rows}")
print(f"Total Features : {columns}")


print("="*60)
print("DATATYPE COUNT")
print("="*60)
print(df.dtypes.value_counts())

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = df.select_dtypes(include=["object"]).columns

print("\nnumerical columns")
print(numerical_columns)

print("\ncategorical columns")
print(categorical_columns)

for column in categorical_columns:
    print(f"\n{column}")
    print(df[column].nunique())