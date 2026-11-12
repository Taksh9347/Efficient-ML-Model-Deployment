"""
Unit Tests for Inference Engine and Model Outputs.
Validates output tensor shapes, probability bounds [0.0, 1.0], and class labels.
"""



def test_model_manager_interface():
    """Validates that ModelManager defines required prediction methods."""
    from app.core.inference import ModelManager

    manager = ModelManager("models/cancer_model.onnx")
    assert hasattr(manager, "load_model")
    assert hasattr(manager, "unload_model")
    assert hasattr(manager, "predict")
    assert hasattr(manager, "predict_batch")


def test_dummy_prediction_contract():
    """Verifies that predicted classes are binary (0 or 1) and probabilities are between 0 and 1."""
    from app.core.inference import ModelManager

    manager = ModelManager("models/cancer_model.onnx")
    manager.load_model()

    # Test with sample 30-dimensional feature array
    sample = [17.99, 10.38, 122.8, 1001.0, 0.1184] + [0.1] * 25
    pred, diag, prob = manager.predict(sample)

    assert pred in (0, 1)
    assert diag in ("Benign", "Malignant")
    assert 0.0 <= prob <= 1.0
