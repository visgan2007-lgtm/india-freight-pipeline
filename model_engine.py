"""
model_engine.py
==============================================================================
Machine Learning Engine: Shipment Delay Classification
------------------------------------------------------------------------------
Builds a production-style Scikit-Learn Pipeline that predicts whether a
shipment will be delayed (`is_delayed`) based on route, fleet, and
environmental features.

Architecture:
    ColumnTransformer
        ├── StandardScaler   -> numerical features
        └── OneHotEncoder    -> categorical features
    GradientBoostingClassifier

Returns fitted pipeline + evaluation artifacts (report, confusion matrix)
so app.py can consume them directly for visualization.
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

NUMERICAL_FEATURES = [
    "route_distance_km",
    "payload_tons",
    "scheduled_hours",
    "avg_speed_kmh",
]

CATEGORICAL_FEATURES = [
    "fleet_type",
    "season_condition",
    "origin_hub",
    "destination_hub",
]

TARGET_COLUMN = "is_delayed"


@dataclass
class ModelArtifacts:
    """Bundles every object app.py needs after training in one clean handle."""
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    accuracy: float
    classification_report_str: str
    classification_report_dict: dict
    confusion_matrix: np.ndarray
    test_indices: pd.Index


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Constructs the ColumnTransformer: scales numeric columns and
    one-hot-encodes categorical columns."""
    return ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), NUMERICAL_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_model_pipeline() -> Pipeline:
    """Wires the preprocessing ColumnTransformer to a GradientBoostingClassifier
    inside a single Scikit-Learn Pipeline object."""
    preprocessor = build_preprocessing_pipeline()

    classifier = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=3,
        subsample=0.9,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    return pipeline


def train_and_evaluate(df: pd.DataFrame) -> ModelArtifacts:
    """
    Full train/evaluate cycle:
        1. Stratified train/test split on `is_delayed` to preserve class ratio.
        2. Fit the ColumnTransformer + GradientBoostingClassifier pipeline.
        3. Generate predictions, classification report, and confusion matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Must already be cleaned + feature-engineered (output of
        pipeline_utils.run_pipeline).

    Returns
    -------
    ModelArtifacts dataclass bundling the fitted pipeline and all evaluation
    outputs required for downstream reporting/visualization.
    """
    feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,  # preserves the original delayed/on-time class distribution
    )

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_str = classification_report(
        y_test, y_pred, target_names=["On-Time", "Delayed"], digits=3
    )
    report_dict = classification_report(
        y_test, y_pred, target_names=["On-Time", "Delayed"], digits=3, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 70)
    print(" GRADIENT BOOSTING CLASSIFIER — SHIPMENT DELAY PREDICTION")
    print("=" * 70)
    print(f"\nTest-set Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(report_str)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm)
    print("=" * 70)

    return ModelArtifacts(
        pipeline=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        accuracy=accuracy,
        classification_report_str=report_str,
        classification_report_dict=report_dict,
        confusion_matrix=cm,
        test_indices=X_test.index,
    )


if __name__ == "__main__":
    from data_generator import generate_shipment_telemetry
    from pipeline_utils import run_pipeline

    raw = generate_shipment_telemetry(3000)
    processed = run_pipeline(raw)
    artifacts = train_and_evaluate(processed)
