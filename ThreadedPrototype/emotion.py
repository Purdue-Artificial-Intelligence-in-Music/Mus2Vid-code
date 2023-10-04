import numpy as np
import time
import os
import tensorflow as tf
from tensorflow import keras
import threading
import joblib
from keras import backend as K


def custom_activation(x):
    return (K.sigmoid(x) * 8) + 1


MODEL_DIR = "utils"
BOUNDED = ""
MODEL_EXT = "model"
SELECTOR_EXT = "selector"
FEATURES_DIR = "utils"


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
        self.average_count = 0
        self.average = [0.0, 0.0]
        self.valence_selector = joblib.load(f"{FEATURES_DIR}/svm_valence.{SELECTOR_EXT}")
        self.arousal_selector = joblib.load(f"{FEATURES_DIR}/svm_arousal.{SELECTOR_EXT}")

        # self.valence_regressor = joblib.load(f"{FEATURES_DIR}/valence_svm.{MODEL_EXT}")
        # self.arousal_regressor = joblib.load(f"{FEATURES_DIR}/arousal_svm.{MODEL_EXT}")

        self.valence_regressor = keras.models.load_model(f"{MODEL_DIR}/fcnn_valence{BOUNDED}.{MODEL_EXT}") #, custom_objects={'custom_activation':custom_activation})
        self.arousal_regressor = keras.models.load_model(f"{MODEL_DIR}/fcnn_arousal{BOUNDED}.{MODEL_EXT}") #, custom_objects={'custom_activation':custom_activation})

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

                # if not (opensmile_arousal_features is None or opensmile_valence_features is None):
                #    print("Emo data found at thread level")
                # print(va_feats)

                v_val = self.valence_regressor.predict(va_feats)
                a_val = self.arousal_regressor.predict(ar_feats)

                self.average[0] = ((self.average_count) / (self.average_count + 1) * self.average[0]
                                   + v_val / (self.average_count + 1))
                self.average[1] = ((self.average_count) / (self.average_count + 1) * self.average[1]
                                   + v_val / (self.average_count + 1))
                self.average_count += 1

                print(f"Valence: %.2f, Arousal: %.2f" % (v_val, a_val))
                print(f"Average Valence: %.2f, Arousal: %.2f" % (self.average[0], self.average[1]))

                # if v_val is None or a_val is None:
                #    print("prediction bad")

                self.emo_values = (v_val, a_val)

                if (self.emo_values is None):
                    print("bad data val")

                # print(self.emo_values)
            time.sleep(0.2)
