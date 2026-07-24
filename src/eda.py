import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_data():
    """Load the cleaned loan dataset"""

    print("\n" + "=" * 60)
    print("LOADING CLEANED DATASET")
    print("=" * 60)

    df = pd.read_csv("data/processed/cleaned_data.csv")

    print(f"Dataset loaded successfully, Dataset shape: {df.shape}")

    return df


def dataset_overview(df):
    """Display basic information about the dataset."""

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMemory Usage:")

    memory = df.memory_usage(deep=True).sum() / (1024 * 1024)

    print(f"{memory:.2f} MB")


def identify_column_types(df):
    """Identify numerical and categorical columns."""

    print("\n" + "=" * 60)
    print("COLUMN TYPE SUMMARY")
    print("=" * 60)

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    print(f"Numerical Columns ({len(numerical_columns)}):")

    print(list(numerical_columns))

    print()

    print(f"Categorical Columns ({len(categorical_columns)}):")

    print(list(categorical_columns))


def analyse_target_distribution(df):
    """Analyse loan default distribution."""

    print("\n" + "=" * 60)
    print("TARGET VARIABLE DISTRIBUTION")
    print("=" * 60)

    df["loan_status"] = df["loan_status"].astype(str)

    plt.figure(figsize=(8,6))

    ax = sns.countplot(
        data=df,
        x="loan_status"
    )

    for bar in ax.patches:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            int(height),
            ha="center",
            va="bottom"
        )

    plt.title("Loan Status Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Number of Loans")

    os.makedirs("images", exist_ok=True)

    plt.savefig(
        "images/loan_status_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nLoan Status Counts:")

    print(df["loan_status"].value_counts())

    default_rate = (
        df["loan_status"].value_counts(normalize=True) * 100
    )

    print("\nPercentage Distribution:")

    print(default_rate.round(2))

def analyse_loan_amount(df):
    """Analyse loan amount distribution."""

    print("\n" + "=" * 60)
    print("LOAN AMOUNT DISTRIBUTION")
    print("=" * 60)

    plt.figure(figsize=(10,6))

    sns.histplot(
        data=df,
        x="loan_amnt",
        bins=40,
        kde=True
    )

    plt.title("Distribution of Loan Amount")

    plt.xlabel("Loan Amount ($)")

    plt.ylabel("Number of Loans")

    plt.savefig(
        "images/loan_amount_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(df["loan_amnt"].describe())


def analyse_loan_amount_boxplot(df):
    """Analyse loan amount using boxplot."""

    print("\n" + "=" * 60)
    print("LOAN AMOUNT BOXPLOT")
    print("=" * 60)

    plt.figure(figsize=(10,3))

    sns.boxplot(
        x=df["loan_amnt"]
    )

    plt.title("Loan Amount Boxplot")

    plt.savefig(
        "images/loan_amount_boxplot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def analyse_interest_rate(df):
    """Analyse interest rate distribution."""

    print("\n" + "=" * 60)
    print("INTEREST RATE DISTRIBUTION")
    print("=" * 60)

    plt.figure(figsize=(10,6))

    sns.histplot(
        data=df,
        x="int_rate",
        bins=40,
        kde=True
    )

    plt.title("Interest Rate Distribution")

    plt.xlabel("Interest Rate")

    plt.ylabel("Number of Loans")

    plt.savefig(
        "images/interest_rate_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(df["int_rate"].describe())



def analyse_annual_income(df):
    """Analyse annual income distribution."""

    print("\n" + "=" * 60)
    print("ANNUAL INCOME DISTRIBUTION")
    print("=" * 60)

    plt.figure(figsize=(10,6))

    sns.histplot(
        data=df,
        x="annual_inc",
        bins=40
    )

    plt.xscale("log")

    plt.title("Annual Income Distribution (Log Scale)")

    plt.xlabel("Annual Income")

    plt.ylabel("Number of Borrowers")

    plt.savefig(
        "images/annual_income_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(df["annual_inc"].describe())

def analyse_dti(df):
    """Analyse Debt To Income Ratio"""

    print("\n" + "=" * 60)
    print("DEBT TO INCOME RATIO DISTRIBUTION")
    print("=" * 60)

    plt.figure(figsize=(10,6))

    sns.histplot(
        data=df,
        x="dti",
        bins=40,
        kde=True
    )

    plt.title("Debt To Income Ratio Distribution")
    plt.xlabel("DTI")
    plt.ylabel("Number of Borrowers")

    plt.savefig(
        "images/dti_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(df["dti"].describe())


def analyse_installment(df):
    """Analyse installment distribution"""

    print("\n" + "=" * 60)
    print("INSTALLMENT DISTRIBUTION")
    print("=" * 60)

    plt.figure(figsize=(10,6))

    sns.histplot(
        data=df,
        x="installment",
        bins=40,
        kde=True
    )

    plt.title("Monthly Installment Distribution")
    plt.xlabel("Monthly Installment")
    plt.ylabel("Number of Loans")

    plt.savefig(
        "images/installment_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(df["installment"].describe())


def analyse_grade(df):
    """Analyse loan grades"""

    print("\n" + "=" * 60)
    print("GRADE DISTRIBUTION")
    print("=" * 60)

    plt.figure(figsize=(10,6))

    ax = sns.countplot(
        data=df,
        x="grade",
        order=sorted(df["grade"].unique())
    )

    for bar in ax.patches:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            int(height),
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.title("Loan Grade Distribution")
    plt.xlabel("Grade")
    plt.ylabel("Number of Loans")

    plt.savefig(
        "images/grade_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(df["grade"].value_counts().sort_index())


def analyse_home_ownership(df):
    """Analyse Home Ownership"""

    print("\n" + "=" * 60)
    print("HOME OWNERSHIP DISTRIBUTION")
    print("=" * 60)

    plt.figure(figsize=(10,6))

    sns.countplot(
        data=df,
        y="home_ownership",
        order=df["home_ownership"].value_counts().index
    )

    plt.title("Home Ownership Distribution")
    plt.xlabel("Count")
    plt.ylabel("Home Ownership")

    plt.savefig(
        "images/home_ownership_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(df["home_ownership"].value_counts())


def analyse_purpose(df):
    """Analyse Loan Purpose"""

    print("\n" + "=" * 60)
    print("LOAN PURPOSE DISTRIBUTION")
    print("=" * 60)

    plt.figure(figsize=(12,7))

    sns.countplot(
        data=df,
        y="purpose",
        order=df["purpose"].value_counts().index
    )

    plt.title("Loan Purpose Distribution")
    plt.xlabel("Count")
    plt.ylabel("Purpose")

    plt.savefig(
        "images/purpose_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(df["purpose"].value_counts())


def analyse_term(df):
    """Analyse Loan Term"""

    print("\n" + "=" * 60)
    print("LOAN TERM DISTRIBUTION")
    print("=" * 60)

    plt.figure(figsize=(6,5))

    sns.countplot(
        data=df,
        x="term"
    )

    plt.title("Loan Term Distribution")
    plt.xlabel("Loan Term")
    plt.ylabel("Count")

    plt.savefig(
        "images/term_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(df["term"].value_counts())


def analyse_verification_status(df):
    """Analyse Verification Status"""

    print("\n" + "=" * 60)
    print("VERIFICATION STATUS")
    print("=" * 60)

    plt.figure(figsize=(8,5))

    sns.countplot(
        data=df,
        x="verification_status"
    )

    plt.title("Verification Status")
    plt.xticks(rotation=20)

    plt.savefig(
        "images/verification_status.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(df["verification_status"].value_counts())


def analyse_application_type(df):
    """Analyse Application Type"""

    print("\n" + "=" * 60)
    print("APPLICATION TYPE")
    print("=" * 60)

    plt.figure(figsize=(5,5))

    sns.countplot(
        data=df,
        x="application_type"
    )

    plt.title("Application Type")

    plt.savefig(
        "images/application_type.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(df["application_type"].value_counts())


def analyse_initial_list_status(df):
    """Analyse Initial Listing Status"""

    print("\n" + "=" * 60)
    print("INITIAL LIST STATUS")
    print("=" * 60)

    plt.figure(figsize=(5,5))

    sns.countplot(
        data=df,
        x="initial_list_status"
    )

    plt.title("Initial List Status")

    plt.savefig(
        "images/initial_list_status.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(df["initial_list_status"].value_counts())


def analyse_correlation(df):
    """Analyse correlations between numerical features."""

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    correlation_matrix = df.select_dtypes(include="number").corr()

    loan_corr = correlation_matrix["loan_amnt"].sort_values(
        ascending=False
    )

    print("\nCorrelation with Loan Amount")
    print(loan_corr)

    print("\nHighest Positive Correlations")

    print(
        correlation_matrix.unstack()
        .sort_values(ascending=False)
        .drop_duplicates()
        .head(15)
    )


def analyse_correlation_heatmap(df):
    """Display correlation heatmap."""

    print("\n" + "=" * 60)
    print("CORRELATION HEATMAP")
    print("=" * 60)

    numerical_df = df.select_dtypes(include="number")

    correlation_matrix = numerical_df.corr()

    plt.figure(figsize=(14,10))

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Heatmap")

    plt.savefig(
        "images/correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def analyse_default_vs_grade(df):
    """Analyse default rate by grade."""

    print("\n" + "=" * 60)
    print("DEFAULT RATE BY GRADE")
    print("=" * 60)

    temp = (
        df.groupby("grade")["loan_status"]
        .value_counts(normalize=True)
        .unstack()
    )

    temp["Charged Off"].plot(
        kind="bar",
        figsize=(8,5)
    )

    plt.title("Default Rate by Grade")
    plt.ylabel("Default Rate")

    plt.savefig(
        "images/default_rate_grade.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(temp)


def analyse_default_vs_home_ownership(df):
    """Analyse default rate by home ownership."""

    print("\n" + "=" * 60)
    print("DEFAULT RATE BY HOME OWNERSHIP")
    print("=" * 60)

    temp = (
        df.groupby("home_ownership")["loan_status"]
        .value_counts(normalize=True)
        .unstack()
    )

    temp["Charged Off"].sort_values().plot(
        kind="barh",
        figsize=(8,6)
    )

    plt.title("Default Rate by Home Ownership")

    plt.savefig(
        "images/default_home_ownership.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(temp)


def analyse_default_vs_purpose(df):
    """Analyse default rate by loan purpose."""

    print("\n" + "=" * 60)
    print("DEFAULT RATE BY PURPOSE")
    print("=" * 60)

    temp = (
        df.groupby("purpose")["loan_status"]
        .value_counts(normalize=True)
        .unstack()
    )

    temp["Charged Off"].sort_values().plot(
        kind="barh",
        figsize=(10,7)
    )

    plt.title("Default Rate by Loan Purpose")

    plt.savefig(
        "images/default_purpose.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(temp)


def main():

    os.makedirs("images", exist_ok=True)

    df = load_data()

    dataset_overview(df)

    identify_column_types(df)

    analyse_target_distribution(df)

    analyse_loan_amount(df)

    analyse_loan_amount_boxplot(df)

    analyse_interest_rate(df)

    analyse_annual_income(df)

    analyse_dti(df)

    analyse_installment(df)

    analyse_grade(df)

    analyse_home_ownership(df)

    analyse_purpose(df)

    analyse_term(df)

    analyse_verification_status(df)

    analyse_application_type(df)

    analyse_initial_list_status(df)

    analyse_correlation(df)

    analyse_correlation_heatmap(df)

    analyse_default_vs_grade(df)

    analyse_default_vs_home_ownership(df)

    analyse_default_vs_purpose(df)


if __name__ == "__main__":
    main()