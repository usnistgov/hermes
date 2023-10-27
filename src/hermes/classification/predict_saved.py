import numpy as np
import tensorflow as tf

from hermes.base import Analysis


class SavedModel(Analysis):
    """Base class for saved model."""

    def __init__(self, model_path: str):
        """Initialize SavedModel."""
        self.model_path = model_path
        self.model = tf.saved_model.load(self.model_path)

    def predict(self, v: np.ndarray) -> tuple:
        """Predict y from loaded model."""
        return self.model.compiled_predict_y(v)
