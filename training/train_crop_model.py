"""
Crop Prediction Training Script
Trains a Random Forest Classifier on soil/climate features to predict optimal crops.
Saves trained model and accuracy metrics.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "crop_recommendation.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.pkl")


def load_data():
    """Load and prepare crop recommendation dataset."""
    df = pd.read_csv(DATASET_PATH)
    print(f"[INFO] Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"[INFO] Crops: {df['label'].nunique()} classes")
    print(f"[INFO] Crop labels: {sorted(df['label'].unique())}")
    return df


def prepare_features(df):
    """Split data into features and labels."""
    X = df[["N", "P", "K", "temperature", "humidity", "pH", "rainfall"]].values
    y = df["label"].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le


def train_model(X_train, y_train):
    """Train Random Forest Classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, X_train, y_train, le):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "test_accuracy": round(test_accuracy, 4),
        "train_accuracy": round(train_accuracy, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importances": model.feature_importances_.tolist(),
        "feature_names": ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"],
        "n_classes": len(le.classes_),
        "class_names": le.classes_.tolist(),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
    }
    
    return metrics


def save_artifacts(model, le, metrics):
    """Save model, encoder, and metrics to disk."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[SAVED] Model -> {MODEL_PATH}")
    
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"[SAVED] Label Encoder -> {ENCODER_PATH}")
    
    with open(METRICS_PATH, "wb") as f:
        pickle.dump(metrics, f)
    print(f"[SAVED] Metrics -> {METRICS_PATH}")


def main():
    print("=" * 60)
    print("  CROP PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load and prepare data
    df = load_data()
    X, y, le = prepare_features(df)
    
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train model
    print("\n[INFO] Training Random Forest Classifier...")
    model = train_model(X_train, y_train)
    
    # Evaluate
    print("\n[INFO] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, X_train, y_train, le)
    
    # Print results
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Training Accuracy:  {metrics['train_accuracy'] * 100:.2f}%")
    print(f"  Testing Accuracy:   {metrics['test_accuracy'] * 100:.2f}%")
    print(f"  CV Mean Accuracy:   {metrics['cv_mean'] * 100:.2f}% (+/- {metrics['cv_std'] * 100:.2f}%)")
    print(f"  Total Classes:      {metrics['n_classes']}")
    print(f"{'=' * 60}")
    
    # Feature importances
    print("\n  Feature Importances:")
    for name, imp in zip(metrics["feature_names"], metrics["feature_importances"]):
        bar = "#" * int(imp * 50)
        print(f"    {name:>12}: {imp:.4f} {bar}")
    
    # Save artifacts
    print(f"\n[INFO] Saving artifacts...")
    save_artifacts(model, le, metrics)
    print("\n[DONE] Training complete!")


if __name__ == "__main__":
    main()
