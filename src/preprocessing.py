import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from typing import Tuple

INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_DIR = "data/processed/"
TARGET_COLUMN = "loan_risk"


GRADE_MAPPING = {
    "A":0,
    "B":1,
    "C":2,
    "D":3,
    "E":4,
    "F":5,
    "G":6
}


SUBGRADE_MAPPING = {}
index = 0

for grade in ["A","B","C","D","E","F","G"]:
    for number in range(1,6):
        SUBGRADE_MAPPING[f"{grade}{number}"] = index
        index += 1


RISK_MAPPING = {
    "Fully Paid":0,
    "Charged Off":1
}


NOMINAL_COLUMNS = [
    "term",
    "home_ownership",
    "verification_status",
    "purpose",
    "initial_list_status",
    "application_type"
]


DROP_COLUMNS = [
    "emp_title",
    "address"
]


def load_data(file_path:str)->pd.DataFrame:
    """
    Load cleaned dataset.
    """

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    return df


def create_target(df:pd.DataFrame)->pd.DataFrame:
    """
    Create binary target.

    0 = Fully Paid
    1 = Charged Off
    """

    df = df[df["loan_status"].isin(RISK_MAPPING.keys())]

    df["loan_risk"] = df["loan_status"].map(RISK_MAPPING)

    print("\nTarget created successfully.")
    print(df["loan_risk"].value_counts())

    return df


def feature_engineering(df:pd.DataFrame)->pd.DataFrame:
    """
    Perform feature engineering.
    """

    print("\nPerforming feature engineering...")

    # Employment Length already numeric

    # Earliest Credit Line
    df["earliest_cr_year"] = (
        df["earliest_cr_line"]
        .str[-4:]
        .astype(int)
    )

    df = df.drop(columns=["earliest_cr_line"])

    # Issue Month

    df["issue_month"] = (
        pd.to_datetime(
            df["issue_d"],
            format="%b-%Y"
        ).dt.month
    )

    df["issue_year"] = (
        pd.to_datetime(
            df["issue_d"],
            format="%b-%Y"
        ).dt.year
    )

    df = df.drop(columns=["issue_d"])

    print("Feature engineering completed.")

    return df


def split_features_target(
    df:pd.DataFrame,
    target_column:str
)->Tuple[pd.DataFrame,pd.Series]:
    """
    Split features and target.
    """

    x = df.drop(
        columns=[
            target_column,
            "loan_status"
        ]
    )

    y = df[target_column]

    print(f"Features shape: {x.shape}")
    print(f"Target shape: {y.shape}")

    return x,y


def train_test_split_data(
    x,
    y,
    test_size,
    random_state
):
    """
    Train Test Split
    """

    x_train,x_test,y_train,y_test = train_test_split(

        x,
        y,

        test_size=test_size,

        random_state=random_state,

        stratify=y
    )

    print(f"Training Set: {x_train.shape}")
    print(f"Testing Set: {x_test.shape}")

    return x_train,x_test,y_train,y_test


def encode_ordinal_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame
):
    """
    Encode ordinal categorical features.
    """

    x_train["grade"] = x_train["grade"].map(GRADE_MAPPING).astype(int)
    x_test["grade"] = x_test["grade"].map(GRADE_MAPPING).astype(int)

    x_train["sub_grade"] = x_train["sub_grade"].map(SUBGRADE_MAPPING).astype(int)
    x_test["sub_grade"] = x_test["sub_grade"].map(SUBGRADE_MAPPING).astype(int)

    print("Ordinal features encoded successfully.")

    return x_train, x_test


def encode_nominal_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame
):
    """
    One Hot Encode nominal features.
    """

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    encoder.fit(
        x_train[NOMINAL_COLUMNS]
    )

    train_encoded = encoder.transform(
        x_train[NOMINAL_COLUMNS]
    )

    test_encoded = encoder.transform(
        x_test[NOMINAL_COLUMNS]
    )

    encoded_columns = encoder.get_feature_names_out(
        NOMINAL_COLUMNS
    )

    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoded_columns,
        index=x_train.index
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=encoded_columns,
        index=x_test.index
    )

    x_train = x_train.drop(columns=NOMINAL_COLUMNS)
    x_test = x_test.drop(columns=NOMINAL_COLUMNS)

    x_train = pd.concat(
        [x_train, train_encoded_df],
        axis=1
    )

    x_test = pd.concat(
        [x_test, test_encoded_df],
        axis=1
    )

    print("Nominal features encoded successfully.")
    print(f"Training Shape: {x_train.shape}")
    print(f"Testing Shape: {x_test.shape}")

    return x_train, x_test


def get_numerical_columns(
    x: pd.DataFrame
):
    """
    Identify numerical columns for scaling.
    """

    numerical_columns = x.select_dtypes(
        include=["int64","float64"]
    ).columns.tolist()

    for column in ["grade","sub_grade"]:
        if column in numerical_columns:
            numerical_columns.remove(column)

    return numerical_columns


def scale_numerical_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    numerical_columns: list
):
    """
    Scale numerical features.
    """

    scaler = StandardScaler()

    scaler.fit(
        x_train[numerical_columns]
    )

    x_train_scaled = pd.DataFrame(
        scaler.transform(
            x_train[numerical_columns]
        ),
        columns=numerical_columns,
        index=x_train.index
    )

    x_test_scaled = pd.DataFrame(
        scaler.transform(
            x_test[numerical_columns]
        ),
        columns=numerical_columns,
        index=x_test.index
    )

    x_train[numerical_columns] = x_train_scaled
    x_test[numerical_columns] = x_test_scaled

    print("Numerical features scaled successfully.")
    print(f"Scaled {len(numerical_columns)} columns.")

    return x_train, x_test

def save_processed_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str
):
    """
    Save processed datasets.
    """

    x_train.to_csv(
        f"{output_dir}/x_train.csv",
        index=False
    )

    x_test.to_csv(
        f"{output_dir}/x_test.csv",
        index=False
    )

    y_train.to_csv(
        f"{output_dir}/y_train.csv",
        index=False
    )

    y_test.to_csv(
        f"{output_dir}/y_test.csv",
        index=False
    )

    print("Processed datasets saved successfully.")


def main():

    # Load cleaned dataset
    df = load_data(INPUT_PATH)

    # Create binary target
    df = create_target(df)

    # Feature Engineering
    df = feature_engineering(df)

    # Split features and target
    x, y = split_features_target(
        df,
        TARGET_COLUMN
    )

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split_data(
        x,
        y,
        test_size=0.20,
        random_state=42
    )

    # Drop unnecessary columns
    x_train = x_train.drop(columns=DROP_COLUMNS)
    x_test = x_test.drop(columns=DROP_COLUMNS)

    # Identify numerical columns
    numerical_columns = get_numerical_columns(
        x_train
    )

    # Encode ordinal features
    x_train, x_test = encode_ordinal_features(
        x_train,
        x_test
    )

    # Encode nominal features
    x_train, x_test = encode_nominal_features(
        x_train,
        x_test
    )

    # Scale numerical features
    x_train, x_test = scale_numerical_features(
        x_train,
        x_test,
        numerical_columns
    )

    # Save processed datasets
    save_processed_data(
        x_train,
        x_test,
        y_train,
        y_test,
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()