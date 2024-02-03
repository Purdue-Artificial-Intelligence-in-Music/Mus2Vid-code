from emotion import *
from features import *
from genre_prediction import *
from image_generation import *
from prompting import *
import time
import os

STARTING_CHUNK = 1024

def display_images(pipe):
    for i in range(len(pipe)):
        image = pipe[i]
        image.save(f"image_output_cache/image%d.png" % int(round(time.time() * 10, 1)))

def main():
    try:
        dir = 'image_output_cache'
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
        SPA_Thread = SinglePyAudioThread(name = 'SPA_Thread', 
                                        starting_chunk_size = STARTING_CHUNK)                            
        MF_Thread = ModifiedMIDIFeatureThread(name = 'MF_Thread',
                                    BP_Thread = SPA_Thread)
        Emo_Thread = ModifiedEmotionClassificationThread(name = 'Emo_Thread',
                                            SM_Thread = SPA_Thread)
        GP_Thread = ModifiedGenrePredictorThread(name = 'GP_Thread',
                                        SM_Thread = SPA_Thread, 
                                        MF_Thread = MF_Thread)
        
        Prompt_Thread = PromptGenerationThread(name = 'Prompt_Thread',
                                            genre_thread = None,
                                            emotion_thread = None)
        
        Img_Thread = ImageGenerationThread(name = 'Img_Thread',
                                        Prompt_Thread = None,
                                        display_func=display_images)
        print("All threads init'ed")
        SPA_Thread.start()
        print("============== SPA started")
        MF_Thread.start()
        print("============== MF started")
        Emo_Thread.start()
        print("============== Emo started")
        GP_Thread.start()
        print("============== GP started")
        Prompt_Thread.start()
        print("============== Prompt started")
        Img_Thread.start()
        print("============== Img started")
        Img_Thread.join()
        while True:
            print("Hi")
            if not (BP_Thread is None or BP_Thread.data is None):
                print(BP_Thread.data)
            if not (SM_Thread is None or SM_Thread.data is None):
                print(SM_Thread.data)
            time.sleep(0.5)
    except KeyboardInterrupt:
        SPA_Thread.stop_request = True
        MF_Thread.stop_request = True
        GP_Thread.stop_request = True
        Emo_Thread.stop_request = True
        Prompt_Thread.stop_request = True
        Img_Thread.stop_request = True
        

if __name__ == "__main__":
    main()