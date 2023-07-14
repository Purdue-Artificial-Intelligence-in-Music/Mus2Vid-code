import numpy as np
import pretty_midi
import tensorflow as tf
import opensmile
import joblib

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
        self.emotion_output = None
    
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
                # self.emotion_output = predict(smile_features)