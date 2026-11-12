"""
Inference Engine and Model Loader.
Provides unified abstraction for loading and executing inference using
either ONNX Runtime (primary) or Scikit-Learn Joblib (fallback).
"""

import os
from typing import Any

import numpy as np

from app.config import settings


class ModelManager:
    """Manages model artifact lifecycle, memory buffers, and prediction execution."""

    def __init__(self, model_path: str = "models/cancer_model.onnx"):
        self.model_path = model_path
        self.session: Any = None
        self.is_onnx = model_path.endswith(".onnx")
        self.version = "1.0.0"
        self.backend = "uninitialized"

    def load_model(self) -> None:
        """
        Loads the serialized model artifact into RAM.
        Initializes onnxruntime.InferenceSession (primary) or joblib.load (fallback).
        """
        # Try ONNX first if requested and file exists
        if self.is_onnx and os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(self.model_path)
                self.backend = "onnx"
                return
            except (RuntimeError, ValueError, OSError, ImportError) as e:
                print(f"Failed to load ONNX model ({e}). Falling back to Joblib.")

        # Fallback to Joblib
        fallback_path = getattr(settings, "FALLBACK_MODEL_PATH", "models/cancer_model.joblib")
        target_path = fallback_path if os.path.exists(fallback_path) else self.model_path
        if os.path.exists(target_path):
            import joblib
            self.session = joblib.load(target_path)
            self.is_onnx = False
            self.backend = "joblib"
            return

        raise FileNotFoundError(
            f"No valid model artifact found at '{self.model_path}' or fallback '{fallback_path}'. "
            f"Please run 'python scripts/train.py' first."
        )

    def unload_model(self) -> None:
        """Frees model resources upon shutdown."""
        self.session = None
        self.backend = "uninitialized"

    def _prepare_vector(self, features: Any) -> np.ndarray:
        """Extracts 30-feature vector from CancerFeatures, dict, or list."""
        if hasattr(features, "to_features_list"):
            vec = features.to_features_list()
        elif isinstance(features, dict):
            # Extract or use defaults
            from app.core.schemas import CancerFeatures
            cf = CancerFeatures(**features)
            vec = cf.to_features_list()
        elif isinstance(features, (list, tuple)):
            vec = list(features)
        else:
            vec = list(features)

        return np.array(vec, dtype=np.float32)

    def predict(self, features: Any) -> tuple[int, str, float]:
        """
        Executes single prediction.
        Returns:
            Tuple of (prediction_class, diagnosis_string, probability_score)
        """
        if self.session is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        vec = self._prepare_vector(features).reshape(1, -1)

        if self.is_onnx:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: vec})
            # outputs[0] is array of labels [1]
            # outputs[1] is list of dicts [{0: prob0, 1: prob1}]
            pred = int(outputs[0][0])
            prob_dict = outputs[1][0]
            prob = float(prob_dict.get(pred, max(prob_dict.values())))
        else:
            preds = self.session.predict(vec)
            probas = self.session.predict_proba(vec)
            pred = int(preds[0])
            prob = float(probas[0][pred])

        # In Wisconsin dataset: 0 = Malignant, 1 = Benign
        diagnosis = "Benign" if pred == 1 else "Malignant"
        return pred, diagnosis, prob

    def predict_batch(self, features_list: list[Any]) -> list[tuple[int, str, float]]:
        """
        Executes vectorized batch prediction for high-throughput workloads.
        """
        if self.session is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        vectors = [self._prepare_vector(f) for f in features_list]
        mat = np.array(vectors, dtype=np.float32)

        results: list[tuple[int, str, float]] = []
        if self.is_onnx:
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: mat})
            labels = outputs[0]
            prob_dicts = outputs[1]
            for i in range(len(features_list)):
                pred = int(labels[i])
                prob = float(prob_dicts[i].get(pred, max(prob_dicts[i].values())))
                diag = "Benign" if pred == 1 else "Malignant"
                results.append((pred, diag, prob))
        else:
            preds = self.session.predict(mat)
            probas = self.session.predict_proba(mat)
            for i in range(len(features_list)):
                pred = int(preds[i])
                prob = float(probas[i][pred])
                diag = "Benign" if pred == 1 else "Malignant"
                results.append((pred, diag, prob))

        return results
