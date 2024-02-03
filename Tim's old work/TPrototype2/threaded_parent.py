from almost_everything import *
from image_generation import *
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
        AE_Thread = AlmostEverythingThread(name = 'AE_Thread')
        # Img_Thread = ModifiedImageGenerationThread(name = 'Img_Thread',
                                        #AE_Thread= None,
                                        #display_func=display_images)
        AE_Thread.start()
        AE_Thread.join()
        print("============== AE started")
        #Img_Thread.start()
        #print("============== Img started")
        #Img_Thread.join()

    except KeyboardInterrupt:
        AE_Thread.set_stop()
        #Img_Thread.stop_request = True
        

if __name__ == "__main__":
    main()