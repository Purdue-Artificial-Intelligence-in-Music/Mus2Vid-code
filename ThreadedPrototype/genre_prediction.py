import numpy as np
import pretty_midi
import tensorflow as tf
import opensmile
import joblib
import threading

def get_subgenre(num):
    genre_list = ['20th Century', 'Romantic', 'Classical', 'Baroque']
    return genre_list[num]

'''
This class is a thread class that predicts the genre of input notes in real time.
'''

class GenrePredictorThread(threading.Thread):
    genre_model = tf.keras.models.load_model('.\model.h5')
    
    """
    This function is called when a GenrePredictorThread is created. It sets the BasicPitchThread to grab MIDI data from.
    Parameters:
        name: the name of the thread
        BP_Thread: a reference to the BasicPitchThread to use
    Returns: nothing
    """ 
    def __init__(self, name, MF_Thread, SM_Thread):
        super(GenrePredictorThread, self).__init__()
        self.MF_Thread = MF_Thread
        self.SM_Thread = SM_Thread
        self.genre_output = None
    
    """
    When the thread is started, this function is called which repeatedly grabs the most recent
    MIDI data from the BasicPitchThread, predicts its genre, and stores it in the data field.
    Parameters: nothing
    Returns: nothing
    """
    def run(self):
        while type(BP_Thread.data) == NoneType:
            time.sleep(0.2)
        while(self.is_alive()):
            smile_features = self.SM_Thread.data
            midi_features = self.MF_Thread.midi_features
            
            audio_features = np.concatenate((smile_features, midi_features), axis=1)
            audio_features = selector.transform(audio_features)
            
            subgenre_num = model.predict(audio_features)
            self.genre_output = get_subgenre(np.argmax(subgenre_num))
            