import sys
from utils.features import *
from utils.diffusion import *
from utils.prompting import *
from utils.upscaling import *

STARTING_CHUNK = 1024

def display_images(pipe):
    for i in range(len(pipe[0])):
        image = pipe.images[i]
        image.show()

def main():
    BP_Thread = BasicPitchThread(name = 'BP_Thread', 
                                 starting_chunk_size = STARTING_CHUNK)
    BP_Thread.start()
    SM_Thread = SmileThread(name = 'SM_Thread', 
                            starting_chunk_size = STARTING_CHUNK)
    SM_Thread.start()
    MF_Thread = MIDIFeatureThread(name = 'MF_Thread',
                                  BP_Thread = BP_Thread)
    MF_Thread.start()
    GP_Thread = GenrePredictorThread(name = 'GP_Thread'
                                     SM_Thread = SM_Thread, 
                                     MF_Thread = MF_Thread)
    GP_Thread.start()
    Emo_Thread = EmotionGenerationThread(name = 'Emo_Thread',
                                         SM_Thread = SM_Thread)
    Emo_Thread.start()
    Prompt_Thread = PromptGenerationThread(name = 'Prompt_Thread',
                                           genre_thread = GP_Thread,
                                           emotion_thread = Emo_Thread)
    Prompt_Thread.start()
    Img_Thread = ImageGenerationThread(name = 'Img_Thread',
                                      Prompt_Thread = Prompt_Thread)
    Img_Thread.start()
    

if __name__ == "__main__":
    main()