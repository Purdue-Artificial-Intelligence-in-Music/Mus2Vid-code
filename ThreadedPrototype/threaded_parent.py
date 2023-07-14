import sys
from emotion import *
from features import *
from genre_prediction import *
from image_generation import *
from prompting import *
import time

STARTING_CHUNK = 1024

def display_images(pipe):
    for i in range(len(pipe[0])):
        image = pipe.images[i]
        image.show()

def main():
    BP_Thread = BasicPitchThread(name = 'BP_Thread', 
                                 starting_chunk_size = STARTING_CHUNK)
    BP_Thread.start()
    print("BP initialized")
    SM_Thread = SmileThread(name = 'SM_Thread', 
                            starting_chunk_size = STARTING_CHUNK)
    SM_Thread.start()
    print("SM initialized")
    MF_Thread = MIDIFeatureThread(name = 'MF_Thread',
                                  BP_Thread = BP_Thread)
    MF_Thread.start()
    print("MF initialized")
    GP_Thread = GenrePredictorThread(name = 'GP_Thread',
                                     SM_Thread = SM_Thread, 
                                     MF_Thread = MF_Thread)
    GP_Thread.start()
    print("GP initialized")
    Emo_Thread = EmotionClassificationThread(name = 'Emo_Thread',
                                         SM_Thread = SM_Thread)
    Emo_Thread.start()
    print("Emo initialized")
    Prompt_Thread = PromptGenerationThread(name = 'Prompt_Thread',
                                           genre_thread = GP_Thread,
                                           emotion_thread = Emo_Thread)
    Prompt_Thread.start()
    while True:
        print(Prompt_Thread.prompt)
        time.sleep(1)
    """
    print("Prompt initialized")
    Img_Thread = ImageGenerationThread(name = 'Img_Thread',
                                      Prompt_Thread = Prompt_Thread)
    Img_Thread.start()
    print("Img initialized")
    """
    #Img_Thread.join()
    

if __name__ == "__main__":
    main()