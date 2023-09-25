import numpy as np
import time
import os
import tensorflow as tf
from tensorflow import keras
import threading
import pandas as pd
import joblib

MODEL_DIR = "."
MODEL_EXT = "keras"
SELECTOR_EXT = "selector"
FEATURES_DIR = "."

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
        self.valence_selector = joblib.load(f"{FEATURES_DIR}/opensmile_valence.{SELECTOR_EXT}")
        self.arousal_selector = joblib.load(f"{FEATURES_DIR}/opensmile_arousal.{SELECTOR_EXT}")
        self.valence_regressor = keras.models.load_model(f"{MODEL_DIR}/valence.{MODEL_EXT}")
        self.valence_regressor = keras.models.load_model(f"{MODEL_DIR}/arousal.{MODEL_EXT}")
    
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
                va_feats = self.valence_selector.transform(smile_features)
                ar_feats = self.arousal_selector.transform(smile_features)

                self.emo_values = (self.valence_regressor.predict(va_feats)[0], self.arousal_regressor.predict(ar_feats)[0])

class EmotionClassificationThreadSPA(threading.Thread):
    # Import model
    
    """
    This function is called when a GenrePredictorThread is created. It sets the BasicPitchThread to grab MIDI data from.
    Parameters:
        name: the name of the thread
        BP_Thread: a reference to the BasicPitchThread to use
    Returns: nothing
    """ 
    def __init__(self, name, SPA_Thread):
        super(EmotionClassificationThreadSPA, self).__init__()
        self.name = name
        self.stop_request = False
        self.SPA_Thread = SPA_Thread
        self.emo_values = None
        self.valence_selector = joblib.load(f"{FEATURES_DIR}/opensmile_valence.{SELECTOR_EXT}")
        self.arousal_selector = joblib.load(f"{FEATURES_DIR}/opensmile_arousal.{SELECTOR_EXT}")
        self.valence_regressor = keras.models.load_model(f"{MODEL_DIR}/valence.{MODEL_EXT}")
        self.valence_regressor = keras.models.load_model(f"{MODEL_DIR}/arousal.{MODEL_EXT}")

    
    """
    When the thread is started, this function is called which repeatedly grabs the most recent
    MIDI data from the BasicPitchThread, predicts its genre, and stores it in the data field.
    Parameters: nothing
    Returns: nothing
    """
    def run(self):
        while not self.stop_request:
            if not (self.SPA_Thread is None or self.SPA_Thread.data is None):
                _, smile_features = self.SPA_Thread.data
                va_feats = self.valence_selector.transform(smile_features)
                ar_feats = self.arousal_selector.transform(smile_features)
            
                #if not (opensmile_arousal_features is None or opensmile_valence_features is None):
                #    print("Emo data found at thread level")

                v_val = self.valence_regressor.predict(va_feats)[0]
                a_val = self.arousal_regressor.predict(ar_feats)[0]

                print(f"Valence: %.2f, Arousal: %.2f" % (v_val, a_val))
                
                #if v_val is None or a_val is None:
                #    print("prediction bad")

                self.emo_values = (v_val, a_val)

                if (self.emo_values is None):
                    print("bad data val")

                # print(self.emo_values)
            time.sleep(0.2)