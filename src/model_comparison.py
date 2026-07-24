import os
import joblib
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def create_models():
    """
    Create and return classification models.
    """

    models = {

        "Logistic Regression": Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", LogisticRegression(
                random_state=42,
                class_weight="balanced",
                max_iter=3000
            ))
        ]),

        "Decision Tree": Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced"
            ))
        ]),

        "Random Forest": Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", RandomForestClassifier(
                random_state=42,
                class_weight="balanced",
                n_estimators=200
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", GradientBoostingClassifier(
                random_state=42
            ))
        ])
    }

    return models


def load_processed_data(data_dir: str):
    """
    Load the processed training and testing datasets.
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


def evaluate_model(
    y_test,
    predictions,
    probabilities
):
    """
    Calculate evaluation metrics.
    """

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

    return {

        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc

    }

def main():

    DATA_DIR = "data/processed"

    x_train, x_test, y_train, y_test = load_processed_data(
        DATA_DIR
    )

    models = create_models()

    results = []

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    thresholds = [
        0.50,
        0.40,
        0.30,
        0.20,
        0.10
    ]

    for name, model in models.items():

        print("\n" + "="*70)
        print(f"TRAINING : {name}")
        print("="*70)

        model.fit(
            x_train,
            y_train
        )

        joblib.dump(
            model,
            f"models/{name.lower().replace(' ','_')}.pkl"
        )

        probabilities = model.predict_proba(
            x_test
        )[:,1]

        for threshold in thresholds:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            metrics = evaluate_model(
                y_test,
                predictions,
                probabilities
            )

            metrics["Model"] = name
            metrics["Threshold"] = threshold

            results.append(
                metrics
            )

        print(f"{name} completed successfully.")

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by=["F1 Score"],
        ascending=False
    )

    result_df.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    print("\n")
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(result_df)

    print("\n")
    print("=" * 80)
    print("TOP 10 RESULTS")
    print("=" * 80)

    print(result_df.head(10))

    print("\n")
    print("=" * 80)
    print("FILES GENERATED")
    print("=" * 80)

    print("✓ reports/model_comparison.csv")

    for name in models.keys():
        print(f"✓ models/{name.lower().replace(' ','_')}.pkl")


if __name__ == "__main__":
    main()