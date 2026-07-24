import os
import joblib
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def load_processed_data(data_dir: str):
    """
    Load the processed training and testing datasets.

    Args:
        data_dir (str): Path to the processed data directory.

    Returns:
        tuple:
            x_train,
            x_test,
            y_train,
            y_test
    """

    x_train = pd.read_csv(
        os.path.join(data_dir, "x_train.csv")
    )

    x_test = pd.read_csv(
        os.path.join(data_dir, "x_test.csv")
    )

    y_train = pd.read_csv(
        os.path.join(data_dir, "y_train.csv")
    ).squeeze("columns")

    y_test = pd.read_csv(
        os.path.join(data_dir, "y_test.csv")
    ).squeeze("columns")

    print("Processed datasets loaded successfully.")
    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return x_train, x_test, y_train, y_test


def create_pipeline():
    """
    Create Random Forest pipeline with SMOTE.
    """

    pipeline = Pipeline(

        steps=[

            (
                "smote",
                SMOTE(
                    random_state=42
                )
            ),

            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced"
                )
            )

        ]

    )

    return pipeline


def create_parameter_grid():
    """
    Create Random Forest parameter search space.
    """

    param_grid = {

        "model__n_estimators": [
            100,
            200
        ],

        "model__max_depth": [
            5,
            10,
            20
        ],

        "model__min_samples_split": [
            2,
            5,
            10
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4
        ],

        "model__max_features": [
            "sqrt",
            "log2"
        ]

    }

    return param_grid

def create_search(
    pipeline,
    param_grid
):
    """
    Create Randomized Search object.
    """

    search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=param_grid,

        n_iter=5,

        scoring="f1",

        cv=3,

        random_state=42,

        n_jobs=-1,

        verbose=2

    )

    return search


def train_search(
    search,
    x_train,
    y_train
):
    """
    Perform hyperparameter tuning.
    """

    search.fit(
        x_train,
        y_train
    )

    print("\nHyperparameter tuning completed.")

    print("\nBest Parameters:")
    print(search.best_params_)

    print(
        f"\nBest Cross Validation F1 Score: {search.best_score_:.4f}"
    )

    return search


def evaluate_model(
    model,
    x_test,
    y_test
):
    """
    Evaluate model on multiple probability thresholds.
    """

    probabilities = model.predict_proba(
        x_test
    )[:,1]

    thresholds = [
        0.50,
        0.40,
        0.30,
        0.20,
        0.10
    ]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        results.append({

            "Threshold": threshold,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1,

            "ROC-AUC": roc_auc

        })

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)

    print(results_df)

    best_threshold = results_df.sort_values(
        by="F1 Score",
        ascending=False
    ).iloc[0]["Threshold"]

    print(f"\nBest Threshold: {best_threshold}")

    final_predictions = (
        probabilities >= best_threshold
    ).astype(int)

    print("\n")
    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    print(
        confusion_matrix(
            y_test,
            final_predictions
        )
    )

    print("\n")
    print("=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)

    print(
        classification_report(
            y_test,
            final_predictions,
            target_names=[
                "Healthy",
                "Risky"
            ],
            zero_division=0
        )
    )

    return results_df

def save_model(
    model,
    results_df
):
    """
    Save the tuned model and threshold comparison.
    """

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    joblib.dump(
        model,
        "models/best_random_forest.pkl"
    )

    results_df.to_csv(
        "reports/hyperparameter_results.csv",
        index=False
    )

    print("\n")
    print("=" * 80)
    print("FILES GENERATED")
    print("=" * 80)

    print("✓ models/best_random_forest.pkl")
    print("✓ reports/hyperparameter_results.csv")


def main():

    DATA_DIR = "data/processed"

    x_train, x_test, y_train, y_test = load_processed_data(
        DATA_DIR
    )

    pipeline = create_pipeline()

    param_grid = create_parameter_grid()

    search = create_search(
        pipeline,
        param_grid
    )

    search = train_search(
        search,
        x_train,
        y_train
    )

    best_model = search.best_estimator_

    results_df = evaluate_model(
        best_model,
        x_test,
        y_test
    )

    save_model(
        best_model,
        results_df
    )

if __name__ == "__main__":
    main()