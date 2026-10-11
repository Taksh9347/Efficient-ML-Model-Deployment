"""
Model Training and Serialization Script.
Trains a classification model on the Breast Cancer dataset (scikit-learn)
and exports both Joblib (.joblib) and ONNX (.onnx) serialized artifacts to models/.
"""

import os

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_export():
    """
    1. Loads dataset.
    2. Splits into train/test sets.
    3. Fits classifier.
    4. Evaluates accuracy.
    5. Saves models/cancer_model.joblib and models/cancer_model.onnx.
    """
    print("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Model test accuracy: {score:.4f}")

    os.makedirs("models", exist_ok=True)

    # 1. Export Joblib artifact
    joblib_path = "models/cancer_model.joblib"
    joblib.dump(model, joblib_path)
    print(f"Exported joblib artifact to: {joblib_path}")

    # 2. Export ONNX artifact
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        onnx_path = "models/cancer_model.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"Exported ONNX artifact to: {onnx_path}")
    except ImportError:
        print("skl2onnx not installed; skipping ONNX export (run `pip install skl2onnx`).")


if __name__ == "__main__":
    train_and_export()
