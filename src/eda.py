import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

def load_data():
    """Load the cleaned loan dataset"""

    print("\n" + "="*60)
    print("LOADING CLEANED DATASET")
    print("=" * 60)

    df = pd.read_csv("data/processed/cleaned_data.csv")

    print(f"Dataset loaded successfully, Dataset shape: {df.shape}")

    return df


def dataset_overview(df):
    """Display basic information about the dataset."""

    print("\n" + "=" *60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn Names: ")
    print(df.columns)

    print("\nData Types: ")
    print(df.dtypes)

    print("\nMemory Usage: ")
    memory = df.memory_usage(deep = True).sum() / (1024 * 1024)
    print(f"{memory:.2f} MB")


def identify_column_types(df):
    "Identify numerical and categorical columns."

    print("\n" + "=" * 60)
    print("COLUMN TYPE SUMMARY")
    print("=" * 60)

    numerical_columns = df.select_dtypes(include=["int64","float64"]).columns
    categorical_columns = df.select_dtypes(include=["object"]).columns

    print(f"Numerical Columns ({len(numerical_columns)}): ")
    print(list(numerical_columns))

    print()

    print(f"Categorical Columns ({len(categorical_columns)}): ")
    print(list(categorical_columns))


def analyse_loan_amount(df):
    """Analyse the distrbution of loan amounts"""

    print("\n" + "="*60)
    print("LOAN AMOUNT DISTRIBUTION")
    print("=" * 60)

    plt.figure(figsize=(10,6))

    sns.histplot(
        data=df,
        x="loan_amount",
        bins = 40,
        kde=True
    )

    plt.title("Distribution of Loan Amount")
    plt.xlabel("Loan Amount")
    plt.ylabel("No. of Loans")

    plt.savefig(
        "images/loan_amount_distribution.png",
        dpi=300,
        bbox_inches = "tight"
    )

    plt.show()


def analyse_loan_amount_boxplot(df):
    """Analyse loan amount using a box plot"""

    print("\n" + "="*60)
    print("LOAN AMOUNT BOXPLOT")
    print("="*60)

    plt.figure(figsize=(10,3))

    sns.boxplot(
        x = df["loan_amount"]
    )

    plt.title("Box Plot of Loan Amount")
    
    plt.savefig(
        "images/loan_amount_boxplot.png",
        dpi=300,
        bbox_inches = "tight"
    )

    plt.show()


def analyse_loan_grade(df):
    """Analyse the distribution of loan grades"""

    print("\n" + "=" *60)
    print("LOAN GRADE DISTRIBUTION")
    print("=" * 60)

    plt.figure(figsize=(10, 6))

    ax = sns.countplot(
            data = df,
            x = "grade",
            order = sorted(df["grade"].unique())
        )
    
    for bar in ax.patches:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            int(height),
            ha = "center",
            va = "bottom"
        )

    plt.title("Distribution of loan grades")
    plt.xlabel("Loan Grade")
    plt.ylabel("No. of loans")
    
    plt.savefig(
        "images/loan_grade_distribution.png",
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.show()

    print("\nLoan Grade Counts: ")
    print(df["grade"].value_counts().sort_index())


def analyse_income_vs_loan(df):
    """Analyse relationship between annual income and loan amount."""

    print("\n" + "=" *60)
    print("ANNUAL INCOME VS LOAN AMOUNT")
    print("=" * 60)

    plt.figure(figsize=(10,6))

    sns.scatterplot(
        data=df,
        x="annual_income",
        y="loan_amount",
        alpha = 0.6
    )
    
    plt.xscale("log")

    plt.title("Annual Income vs Loan Amount")
    plt.xlabel("Annual Income")
    plt.ylabel("Loan Amount")

    plt.savefig(
        "images/income_vs_loan_amount.png",
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.show()


def analyse_correlation(df):
    """Analyse correlation between annual income and loan amount"""

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    correlation = df["annual_income"].corr(df["loan_amount"])

    print(f"Correlation between Annual Income and Loan Amount: {correlation:.3f}")


def analyse_correlation_heatmap(df):
    """Display correlation heatmap for numerical features"""

    print("\n" + "=" *60)
    print("CORRELATION HEATMAP")
    print("=" * 60)

    numerical_df = df.select_dtypes(include="number")

    correlation_matrix = numerical_df.corr()

    plt.figure(figsize=(14,10))

    sns.heatmap(
        correlation_matrix,
        cmap = "coolwarm",
        center = 0
    )

    plt.title("CORRELATION HEATMAP")

    plt.savefig(
        "images/correlation_heatmap.png",
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.show()

def main():
    df = load_data()
    dataset_overview(df)
    identify_column_types(df)
    analyse_loan_amount(df)
    analyse_loan_amount_boxplot(df)
    analyse_loan_grade(df)
    analyse_income_vs_loan(df)
    analyse_correlation(df)
    analyse_correlation_heatmap(df)


if __name__ == "__main__":
    main()

