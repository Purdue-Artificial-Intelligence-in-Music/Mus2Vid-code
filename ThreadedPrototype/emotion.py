import numpy as np
import tensorflow as tf
import threading
import pandas as pd

from src.emotion.model.regressor import EmotionRegressor
from src.emotion.features.best import get_best_opensmile_features

def get_va_values(opensmile_features: pd.DataFrame) -> tuple[float, float]:
    """Process audio at given filepath and return valence and arousal values.

    Parameters
    ----------
    audio_filepath
        Filepath relative to repository root.

    Returns
    -------
    valence: float
        A float between 1 and 9.
    arousal: float
        A float between 1 and 9.
    """
    valence_regressor = EmotionRegressor()
    arousal_regressor = EmotionRegressor()
    valence_regressor.load("valence_regressor")
    arousal_regressor.load("arousal_regressor")

    opensmile_valence_features, opensmile_arousal_features = get_best_opensmile_features(opensmile_features)

    valence = valence_regressor.predict(opensmile_valence_features)[0]
    arousal = arousal_regressor.predict(opensmile_arousal_features)[0]

    return valence, arousal

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
        self.SM_Thread = SM_Thread
        self.valence = None
        self.arousal = None
    
    """
    When the thread is started, this function is called which repeatedly grabs the most recent
    MIDI data from the BasicPitchThread, predicts its genre, and stores it in the data field.
    Parameters: nothing
    Returns: nothing
    """
    def run(self):
        while (self.is_alive()):
            smile_features = self.SM_Thread.data
            if not smile_features is None:
                (self.valence, self.arousal) = get_va_values(smile_features)