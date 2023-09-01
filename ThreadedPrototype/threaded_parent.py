from emotion import *
from features_2 import *
from genre_prediction_2 import *
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
        SPA_Thread = SinglePyAudioThread(name = "SPA_Thread", starting_chunk_size=STARTING_CHUNK)
        MMF_Thread = ModifiedMIDIFeatureThread(name = "MMF_Thread", SinglePyAudioThread=SPA_Thread)
        Emo_Thread = EmotionClassificationThreadSPA(name = 'Emo_Thread',
                                            SPA_Thread = SPA_Thread)
        GP_Thread = ModifiedGenrePredictorThread(name = 'GP_Thread',
                                        MF_Thread = MMF_Thread, 
                                        SPA_Thread=SPA_Thread)
        
        Prompt_Thread = PromptGenerationThread(name = 'Prompt_Thread',
                                            genre_thread = GP_Thread,
                                            emotion_thread = Emo_Thread)
        
        Img_Thread = ImageGenerationThread(name = 'Img_Thread',
                                           seed=1028,
                                        Prompt_Thread = Prompt_Thread,
                                        display_func=display_images)
        print("All threads init'ed")
        SPA_Thread.start()
        print("============== SPA started")
        MMF_Thread.start()
        print("============== MMF started")
        Emo_Thread.start()
        print("============== Emo started")
        GP_Thread.start()
        print("============== GP started")
        Prompt_Thread.start()
        print("============== Prompt started")
        Img_Thread.start()
        print("============== Img started")
        while True:
            print("\n\n")
            print(Prompt_Thread.prompt)
            time.sleep(0.5)
    except KeyboardInterrupt:
        SPA_Thread.stop_request = True
        MMF_Thread.stop_request = True
        GP_Thread.stop_request = True
        Emo_Thread.stop_request = True
        Prompt_Thread.stop_request = True
        Img_Thread.stop_request = True
        

if __name__ == "__main__":
    main()