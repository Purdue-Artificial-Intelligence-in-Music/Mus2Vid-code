import os
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVR
from src.emotion.model.util import MODEL_DIR, MODEL_EXT


class EmotionRegressor():
    """A Support Vector Regressor (SVR) for predicting valence and
    arousal values given the pre-extracted features of an audio file.
    
    EmotionRegressor can be used to predict both valence and
    arousal values from an audio file. Separate instances of EmotionRegressor
    should be used for valence and arousal.
    """

    def __init__(self, epsilon=0.2) -> None:
        """Initialize regressor with option to set epsilon value."""
        self.svr = SVR(epsilon=epsilon)

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> None:
        """Fit model to provided set of features and targets."""
        self.svr.fit(features, targets)

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict and return valence and arousal values."""
        return self.svr.predict(inputs)

    def save(self, filename: str) -> None:
        """Save the current model to a file."""
        if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)
        joblib.dump(self.svr, f"{MODEL_DIR}/{filename}.{MODEL_EXT}")

    def load(self, filename: str) -> None:
        """Load a model from a file."""
        self.svr = joblib.load(f"{MODEL_DIR}/{filename}.{MODEL_EXT}") 


if __name__ == "__main__":
    pass
