import os
import pandas as pd

print("=" * 70)
print("LENDING CLUB LOAN DEFAULT PREDICTION : DATA CLEANING")
print("=" * 70)


def print_section(title):
    """Print formatted section headings."""

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_data():
    """Load the Lending Club dataset."""

    print_section("LOADING DATASET")

    df = pd.read_csv("data/raw/lending_club_loan_two.csv")

    print(f"\nDataset Loaded Successfully")
    print(f"Dataset Shape: {df.shape}")

    return df


def check_duplicates(df):
    """Check for duplicate rows and remove them."""

    print_section("DUPLICATE CHECK")

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate Rows: {duplicate_count}")

    if duplicate_count > 0:
        df = df.drop_duplicates()
        print("Duplicates removed successfully.")
    else:
        print("No duplicate rows found.")

    return df


def generate_missing_value_report(df):
    """Generate and save missing value report."""

    print_section("MISSING VALUE REPORT")

    missing_report = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Percentage": (df.isnull().sum() / len(df)) * 100
    })

    missing_report = missing_report.sort_values(
        by="Percentage",
        ascending=False
    )

    print(missing_report)

    os.makedirs("reports", exist_ok=True)

    missing_report.to_csv(
        "reports/missing_value_report.csv",
        index=True
    )

    print("\nMissing Value Report Saved Successfully.")

    return missing_report


def clean_dataset(df):
    """Clean the Lending Club dataset."""

    print_section("DATA CLEANING")

    # ----------------------------------------------------
    # Remove unnecessary column
    # ----------------------------------------------------

    df = df.drop(columns=["title"])

    print("Dropped column: title")

    # ----------------------------------------------------
    # Employee Title
    # ----------------------------------------------------

    df["emp_title"] = df["emp_title"].fillna("Unknown")

    print("Filled missing emp_title with 'Unknown'")

    # ----------------------------------------------------
    # Employee Length
    # ----------------------------------------------------

    df["emp_length"] = (
        df["emp_length"]
        .str.replace("years", "", regex=False)
        .str.replace("year", "", regex=False)
        .str.replace("\\+", "", regex=True)
        .str.replace("< 1", "0", regex=False)
        .str.replace("n/a", "", regex=False)
        .str.strip()
    )

    df["emp_length"] = pd.to_numeric(
        df["emp_length"],
        errors="coerce"
    )

    median_emp = df["emp_length"].median()

    df["emp_length"] = df["emp_length"].fillna(median_emp)

    print("Converted emp_length to numeric")

    # ----------------------------------------------------
    # Revolving Utilization
    # ----------------------------------------------------

    median_revol = df["revol_util"].median()

    df["revol_util"] = df["revol_util"].fillna(
        median_revol
    )

    print("Filled revol_util missing values")

    # ----------------------------------------------------
    # Public Bankruptcies
    # ----------------------------------------------------

    df["pub_rec_bankruptcies"] = (
        df["pub_rec_bankruptcies"]
        .fillna(0)
    )

    print("Filled pub_rec_bankruptcies with 0")

    # ----------------------------------------------------
    # Mortgage Accounts
    # ----------------------------------------------------

    median_mort = df["mort_acc"].median()

    df["mort_acc"] = df["mort_acc"].fillna(
        median_mort
    )

    print("Filled mort_acc missing values")

    return df


def final_data_quality_check(df):
    """Perform final data quality checks."""

    print_section("FINAL DATA QUALITY CHECK")

    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[
        remaining_missing > 0
    ]

    if remaining_missing.empty:
        print("No Remaining Missing Values.")
    else:
        print("Remaining Missing Values:")
        print(remaining_missing)

    print(f"\nDataset Shape: {df.shape}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")


def save_dataset(df):
    """Save cleaned dataset."""

    print_section("SAVING CLEANED DATASET")

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    df.to_csv(
        "data/processed/cleaned_data.csv",
        index=False
    )

    print("Cleaned Dataset Saved Successfully.")


def main():

    df = load_data()

    df = check_duplicates(df)

    generate_missing_value_report(df)

    df = clean_dataset(df)

    final_data_quality_check(df)

    save_dataset(df)


if __name__ == "__main__":
    main()