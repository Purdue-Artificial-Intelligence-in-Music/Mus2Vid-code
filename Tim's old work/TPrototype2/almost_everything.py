import multiprocessing
import threading
from emotion import *
from features import *
from genre_prediction import *
from image_generation import *
from prompting import *

class AlmostEverythingThread(multiprocessing.Process):
    def __init__(self, name):
        super(AlmostEverythingThread, self).__init__()
        self.name = name
        self.SPA_Thread = SinglePyAudioThread(name = 'SPA_Thread', 
                                        starting_chunk_size = 1024)                            
        self.MF_Thread = ModifiedMIDIFeatureThread(name = 'MF_Thread',
                                    SinglePyAudioThread = self.SPA_Thread)
        self.Emo_Thread = ModifiedEmotionClassificationThread(name = 'Emo_Thread',
                                            SPA_Thread = self.SPA_Thread)
        self.GP_Thread = ModifiedGenrePredictorThread(name = 'GP_Thread',
                                        SPA_Thread = self.SPA_Thread, 
                                        MF_Thread = self.MF_Thread)
        self.Prompt_Thread = PromptGenerationThread(name = 'Prompt_Thread',
                                            genre_thread = self.GP_Thread,
                                            emotion_thread = self.Emo_Thread)
        print("All threads init'ed")

        self.stop_request = False

    def run(self):
        self.SPA_Thread.start()
        print("============== SPA started")
        self.MF_Thread.start()
        print("============== MF started")
        self.Emo_Thread.start()
        print("============== Emo started")
        self.GP_Thread.start()
        print("============== GP started")
        self.Prompt_Thread.start()
        print("============== Prompt started")
        while not self.stop_request:
            print(self.Prompt_Thread.prompt)
            time.sleep(0.5)

    def set_stop(self):
        self.SPA_Thread.stop_request = True
        self.MF_Thread.stop_request = True
        self.GP_Thread.stop_request = True
        self.Emo_Thread.stop_request = True
        self.Prompt_Thread.stop_request = True