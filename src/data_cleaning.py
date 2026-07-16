import pandas as pd 

print("="*70)
print("BANKING LOAN ANALYTICS : DATA CLEANING")
print("="*70)

def load_data():
    """Load the loan dataset"""
    print("="*60)
    print("Loading the dataset. . .")
    print("="*60)
    
    df = pd.read_csv("data/raw/loans_full_schema.csv")
    
    print(f"\nData Loaded Successfully, Dataset shape: {df.shape}")
    return df


def check_duplicates(df):
    """Check for duplicate rows and remove them if present."""
    print("\n" + "="*60)
    print("DUPLICATE CHECK")
    print("="*60)

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate Rows: {duplicate_count}")

    if duplicate_count>0:
        df = df.drop_duplicates()
        print("Duplicates removed successfully")
    else:
        print("No duplicate rows found")

    return df


def generate_missing_value_report(df):
    """Generate and save a missing value report."""

    print("\n"+"="*60)
    print("MISSING VALUE REPORT")
    print("="*60)

    missing_report = pd.DataFrame({
        "Missing Value":df.isnull().sum(),
        "Percentage":(df.isnull().sum()/len(df))*100
    })

    missing_report = missing_report.sort_values(
        by="Percentage",
        ascending=False
    )

    print(missing_report)

    missing_report.to_csv(
        "reports/missing_value_report.csv",
        index=True
    )

    print("\n Missing value report saved")
    return missing_report


def clean_employee_columns(df):
    """Clean employee related columns"""

    print("\n" + "="*60)
    print("CLEANING EMPLOYEE COLUMNS")
    print("="*60)

    #Fill missing employee titles
    df["emp_title"] = df["emp_title"].fillna("Unknown")

    #Fill missing employment length with mode
    emp_length_mode = df["emp_length"].mode()[0]
    df["emp_length"] = df["emp_length"].fillna(emp_length_mode)

    print("Employee columns cleaned")

    return df


def clean_remaining_columns(df):
    """Clean the remaining columns with missing values."""

    print("\n" + "="*60)
    print("CLEANING REMAINING COLUMNS")
    print("="*60)

     # Remove unnecessary index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("Unnamed: 0 column removed")
    else:
        print("No unnecessary index column found")

    #Fill missing count values
    df["num_accounts_120d_past_due"] = df["num_accounts_120d_past_due"].fillna(0)

    #fill debt to income with median
    median_dti = df["debt_to_income"].median()
    df["debt_to_income"] = df["debt_to_income"].fillna(median_dti)

    print("Remaining columns cleaned")
    return df


def save_dataset(df):
    """Save cleaned dataset"""

    print("\n" + "="*60)
    print("\nSaving cleaned dataset . . .")
    print("="*60)

    df.to_csv(
        "data/processed/cleaned_data.csv",
        index=False
    )

    print("cleaned dataset saved")


def final_data_quality_check(df):
    """Perform final data quality checks"""

    print("\n"+"="*60)
    print("FINAL DATA QUALITY CHECK")
    print("="*60)

    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]

    if remaining_missing.empty:
        print("no remaining missing values")
    else:
        print("Remaining missing values: ")
        print(remaining_missing)

    print(f"Dataset Shape: {df.shape}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")


def main():
    df = load_data()

    df = check_duplicates(df)

    generate_missing_value_report(df)

    df = clean_employee_columns(df)

    df = clean_remaining_columns(df)

    save_dataset(df)

    final_data_quality_check(df)

   

if __name__ == "__main__":
    main()