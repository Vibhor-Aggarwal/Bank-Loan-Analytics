import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler

from typing import Tuple

INPUT_PATH = "data/processed/cleaned_data.csv"
OUTPUT_DIR = "data/processed/"
TARGET_COLUMN = "loan_status"

GRADE_MAPPING = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
}

SUBGRADE_MAPPING = {}
index=0
for grade in ["A","B","C","D","E","F","G"]:
    for number in range(1,6):
        SUBGRADE_MAPPING[f"{grade}{number}"] = index
        index = index + 1


NOMINAL_COLUMNS = [
    "homeownership",
    "verified_income",
    "loan_purpose",
    "application_type",
    "state",
]

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the cleaned dataset
    
    Args:
        file_path(str): path to the cleaned dataset
        
    Returns:
        pd.DataFrame: loaded dataframe.
    """

    df = pd.read_csv(file_path)

    print(f"Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    return df


def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate the features and target variable.

    Args: 
        df(pd.DataFrame): cleaned dataset.
        target_column(str): Name of the target column.

    Returns:
        tuple: Features(x) and target(y).
    """

    x = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"Features shape: {x.shape}")
    print(f"Target shape: {y.shape}")

    return x, y


def train_test_split_data(x: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    """
    Split the dataset into training and testing sets.

    Args: 
        x (pd.DataFrame): Feature matrix,
        y (pd.Series): Target Variable,
        test_size (float): Proportion of data for testing,
        random_state (int): Random seed for shuffling.

    Returns:
        tuple: x_train, x_test, y_train, y_test
    """

    x_train, x_test, y_train, y_test = train_test_split(
        x, 
        y,
        test_size = test_size,
        random_state = random_state,
        stratify = y
    )

    print(f"Training Set: {x_train.shape}")
    print(f"Testing Set: {x_test.shape}")

    return x_train, x_test, y_train, y_test


def encode_ordinal_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame
):
    """Encode ordinal categorical features using predefined mappings.
    
    Args:
        x_train (pd.DataFrame): Training Features.
        x_test (pd.DataFrame): Testing Features.
        
    Returns:
        tuple: Encoded training and testing features."""
    
    #encode loan grade
    x_train["grade"] = x_train["grade"].map(GRADE_MAPPING).astype(int)
    x_test["grade"] = x_test["grade"].map(GRADE_MAPPING).astype(int)

    #encode loan sub grade 
    x_train["sub_grade"] = x_train["sub_grade"].map(SUBGRADE_MAPPING).astype(int)
    x_test["sub_grade"] = x_test["sub_grade"].map(SUBGRADE_MAPPING).astype(int)

    print("Ordinal features encoded successfully.")

    return x_train, x_test


def encode_nominal_features(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame
):
    """One hot encode nominal categorical features.
    
    Args:
        x_train(pd.DataFrame): Training features.
        x_test(pd.DataFrame): Testing Features.
        
    Returns:
        tuple: encoded training and testing features.
    """

    #Initialise One Hot Encoder
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    #Fit only on training data
    encoder.fit(x_train[NOMINAL_COLUMNS])

    #Transform training and testing data
    train_encoded = encoder.transform(x_train[NOMINAL_COLUMNS])
    test_encoded = encoder.transform(x_test[NOMINAL_COLUMNS])

    #Get names of newly created columns
    encoded_columns = encoder.get_feature_names_out(NOMINAL_COLUMNS)

    #Convert numpy arrays back to dataframe
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns = encoded_columns,
        index=x_train.index
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns = encoded_columns,
        index = x_test.index
    )

    #Remove original categorical columns 
    x_train = x_train.drop(columns=NOMINAL_COLUMNS)
    x_test = x_test.drop(columns=NOMINAL_COLUMNS)

    #combine numerical columns with encoded columns 
    x_train = pd.concat(
        [x_train, train_encoded_df],
        axis = 1
    )

    x_test = pd.concat(
        [x_test, test_encoded_df],
        axis =1 
    )

    print("Nominal features encoded successfully.")
    print(f"Training Shape: {x_train.shape}")
    print(f"Testing shape: {x_test.shape}")

    return x_train, x_test


def get_numerical_columns(
        x: pd.DataFrame
) -> list:
    """
    Identify continuous numerical columns that require scaling.

    Args:
        X (pd.DataFrame): Feature dataset.

    Returns:
        list: Names of numerical columns to scale.
    """

    numerical_columns = x.select_dtypes(include=["int64","float64"]).columns.tolist()

    #remove ordinal columns
    for column in ["grade","sub_grade"]:
        if column in numerical_columns:
            numerical_columns.remove(column)
    
    return numerical_columns


def scale_numerical_features(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        numerical_columns: list
):
    """Scale continuous numerical features using StandardScaler
    
    Args:
        X_train (pd.DataFrame): Training feature set.
        X_test (pd.DataFrame): Testing feature set.

    Returns:
        tuple: Scaled X_train and X_test.
    """

    scaler = StandardScaler()

    #learn scaling parameters from training data
    scaler.fit(x_train[numerical_columns])

    #apply scaling
    x_train_scaled = pd.DataFrame(
        scaler.transform(x_train[numerical_columns]),
        columns=numerical_columns,
        index=x_train.index
    )

    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test[numerical_columns]),
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
    Save processed train and test datasets.

    Args:
        x_train (pd.DataFrame): Training features.
        x_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
        output_dir (str): Output directory.
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

    #load cleaned dataset
    df = load_data(INPUT_PATH)

    #split features and target
    x, y = split_features_target(df, TARGET_COLUMN)

    #train-test split
    x_train, x_test, y_train, y_test = train_test_split_data(
        x, 
        y, 
        test_size=0.2,
        random_state=42
    )

    #identify numerical columns
    numerical_columns = get_numerical_columns(x_train)

    #encode ordinal features
    x_train, x_test = encode_ordinal_features(
        x_train,
        x_test
    )

    #encode nominal features 
    x_train, x_test = encode_nominal_features(
        x_train,
        x_test
    )

    #scale numerical features
    x_train, x_test = scale_numerical_features(
        x_train,
        x_test,
        numerical_columns
    )

    #save processed datasets
    save_processed_data(
        x_train,
        x_test,
        y_train,
        y_test,
        OUTPUT_DIR
    )

if __name__ == "__main__":
    main()