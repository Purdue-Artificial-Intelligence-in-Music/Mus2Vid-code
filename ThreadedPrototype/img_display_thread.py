import threading
import cv2
import time
import numpy as np

"""
This class is a thread class that generates prompts procedurally in real time.
"""

class ImageDisplayThread(threading.Thread):

    """
    This function is called when a PromptGenerationThread is created.
    Parameters:
        name: the name of the thread
    Returns: nothing
    """

    def __init__(self, name, Prompt_Thread, Img_Thread, window_name = "Image", static_dur = 1200, blend_frames = 60, blend_mspf = 1):
        super(ImageDisplayThread, self).__init__()
        self.name = name
        self.prompt_thread = Prompt_Thread
        self.image_thread = Img_Thread
        self.window_name = window_name
        self.past_image = None
        self.current_image = None
        self.static_dur = static_dur
        self.blend_frames = blend_frames
        self.blend_mspf = blend_mspf
        self.stop_request = False
    
    def get_image(self):
        self.past_image = self.current_image
        if self.prompt_thread.prompt == "Black screen":
            self.current_image = np.zeros((1024,1024,3))
        else:
            self.current_image = self.image_thread.output
    
    """
    When the thread is started, this function is called which repeatedly displays new images.
    Parameters: nothing
    Returns: nothing
    """

    def run(self):
        while not self.stop_request and (self.prompt_thread is None or self.prompt_thread.prompt is None or self.image_thread is None or self.image_thread.output is None):
            cv2.imshow(self.window_name, np.zeros((1024,1024,3)))
            cv2.waitKey(self.static_dur)
        self.get_image()
        cv2.imshow(self.window_name, self.current_image)
        cv2.waitKey(self.static_dur)
        while not self.stop_request: 
            self.get_image()
            self.current_image = self.image_thread.output
            for i in range(1,self.blend_frames):
                dst = cv2.addWeighted(self.past_image, 1 - (i / float(self.blend_frames)), self.current_image, i / float(self.blend_frames), 0.0)
                cv2.imshow(self.window_name, dst)
                cv2.waitKey(self.blend_mspf)
            cv2.imshow(self.window_name, self.current_image)
            cv2.waitKey(self.static_dur)
        cv2.destroyAllWindows()
