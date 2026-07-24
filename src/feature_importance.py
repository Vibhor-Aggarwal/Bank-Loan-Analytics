"""
Feature Importance Analysis Script

This script loads the best trained Random Forest model,
extracts feature importance scores, generates a ranked report,
and visualizes the top contributing features.

Author: Vibhor Aggarwal
Project: Loan Portfolio Status Prediction
"""

import os
import joblib
import pandas as pd 
import matplotlib.pyplot as plt 

MODEL_PATH = "models/best_random_forest.pkl"

TRAIN_FEATURES_PATH = "data/processed/x_train.csv"

REPORTS_DIR = "reports"

IMAGES_DIR = "images"

FEATURE_IMPORTANCE_REPORT = os.path.join(
    REPORTS_DIR,
    "feature_importance.csv"
)

FEATURE_IMPORTANCE_PLOT = os.path.join(
    IMAGES_DIR,
    "feature_importance.png"
)

def load_model(model_path: str):
    """
    Load the trained model from disk.

    Args:
        model_path (str):
            Path to the saved model file.

    Returns:
        object:
            Loaded trained model or pipeline.
    """

    print("=" * 60)
    print("Loading trained model...")
    print("=" * 60)

    model = joblib.load(model_path)

    print("Model loaded successfully.\n")

    return model

def load_feature_names(train_features_path: str) -> list:
    """
    Load the processed training dataset and extract feature names.

    Args:
        train_features_path (str):
            Path to processed training features.

    Returns:
        list:
            List of feature names.
    """

    print("Loading processed training features...")

    x_train = pd.read_csv(train_features_path)

    print(f"Number of features: {x_train.shape[1]}\n")

    return list(x_train.columns)

def extract_feature_importance(model, feature_names):
    """
    Extract feature importance scores from the trained model.

    Supports both:
    - RandomForestClassifier
    - Pipeline containing RandomForestClassifier

    Args:
        model:
            Trained model or pipeline.

        feature_names (list):
            Feature names.

    Returns:
        pandas.DataFrame:
            Ranked feature importance table.
    """

    print("Extracting feature importance...")

    # If the saved object is a Pipeline
    if hasattr(model, "named_steps"):

        random_forest = model.named_steps["model"]

    else:

        random_forest = model

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": random_forest.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    print("Feature importance extracted successfully.\n")

    return importance_df

def save_feature_importance_report(
    importance_df: pd.DataFrame,
    report_path: str
) -> None:
    """
    Save the feature importance report as a CSV file.

    Args:
        importance_df (pd.DataFrame):
            DataFrame containing feature importance scores.

        report_path (str):
            Path where the report will be saved.

    Returns:
        None
    """

    print("Saving feature importance report...")

    # Create the reports directory if it does not exist
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Add ranking column
    importance_df.insert(
        0,
        "Rank",
        range(1, len(importance_df) + 1)
    )

    # Save report
    importance_df.to_csv(report_path, index=False)

    print(f"Report saved to: {report_path}\n")

def plot_feature_importance(
    importance_df: pd.DataFrame,
    image_path: str,
    top_n: int = 20
) -> None:
    """
    Plot and save the top N most important features.

    Args:
        importance_df (pd.DataFrame):
            Feature importance DataFrame.

        image_path (str):
            Path to save the plot.

        top_n (int):
            Number of top features to display.

    Returns:
        None
    """

    print(f"Generating Top {top_n} Feature Importance plot...")

    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    top_features = (
        importance_df
        .head(top_n)
        .sort_values(by="Importance")
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.grid(
        axis="x",
        linestyle="--",
        alpha=0.6
    )

    plt.title(f"Top {top_n} Most Important Features")

    plt.xlabel("Feature Importance")

    plt.ylabel("Features")

    plt.tight_layout()

    plt.savefig(image_path, dpi=300)

    plt.close()

    print(f"Plot saved to: {image_path}\n")

def main():
    """
    Execute the complete feature importance analysis workflow.
    """

    model = load_model(MODEL_PATH)

    feature_names = load_feature_names(TRAIN_FEATURES_PATH)

    importance_df = extract_feature_importance(
        model,
        feature_names
    )

    save_feature_importance_report(
        importance_df,
        FEATURE_IMPORTANCE_REPORT
    )

    plot_feature_importance(
        importance_df,
        FEATURE_IMPORTANCE_PLOT
    )

    print("=" * 60)
    print("Feature Importance Analysis Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()