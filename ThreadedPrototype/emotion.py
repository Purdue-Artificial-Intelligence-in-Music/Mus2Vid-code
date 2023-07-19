import numpy as np
import os
import tensorflow as tf
import threading
import pandas as pd
import joblib
from sklearn.svm import SVR

MODEL_DIR = "."
MODEL_EXT = "model"
SELECTOR_EXT = "selector"
FEATURES_DIR = "."

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

'''
This class is a thread class that predicts the genre of input notes in real time.
'''

class EmotionClassificationThread(threading.Thread):
    # Import model
    
    """
    This function is called when a GenrePredictorThread is created. It sets the BasicPitchThread to grab MIDI data from.
    Parameters:
        name: the name of the thread
        BP_Thread: a reference to the BasicPitchThread to use
    Returns: nothing
    """ 
    def __init__(self, name, SM_Thread):
        super(EmotionClassificationThread, self).__init__()
        self.name = name
        self.stop_request = False
        self.SM_Thread = SM_Thread
        self.emo_values = None
        self.valence_regressor = EmotionRegressor()
        self.valence_regressor.load("valence_regressor")
        self.arousal_regressor = EmotionRegressor()
        self.arousal_regressor.load("arousal_regressor")
        self.opensmile_valence_selector = joblib.load(f"{FEATURES_DIR}/opensmile_valence.{SELECTOR_EXT}")
        self.opensmile_arousal_selector = joblib.load(f"{FEATURES_DIR}/opensmile_arousal.{SELECTOR_EXT}")

    
    """
    When the thread is started, this function is called which repeatedly grabs the most recent
    MIDI data from the BasicPitchThread, predicts its genre, and stores it in the data field.
    Parameters: nothing
    Returns: nothing
    """
    def run(self):
        while not self.stop_request:
            if not (self.SM_Thread is None or self.SM_Thread.data is None):
                smile_features = self.SM_Thread.data

                opensmile_valence_features = self.opensmile_valence_selector.transform(smile_features)
                opensmile_arousal_features = self.opensmile_arousal_selector.transform(smile_features)

                self.emo_values = (self.valence_regressor.predict(opensmile_valence_features)[0], self.arousal_regressor.predict(opensmile_arousal_features)[0])