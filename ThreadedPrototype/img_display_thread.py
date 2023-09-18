import threading
import cv2
import time

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

    def __init__(self, name, Img_Thread, window_name = "Image", static_dur = 300, blend_frames = 120, blend_mspf = 0):
        super(ImageDisplayThread, self).__init__()
        self.name = name
        self.image_thread = Img_Thread
        self.window_name = window_name
        self.past_image = None
        self.current_image = None
        self.static_dur = static_dur
        self.blend_frames = blend_frames
        self.blend_mspf = blend_mspf
        self.stop_request = False
    
    """
    When the thread is started, this function is called which repeatedly generates new prompts.
    Parameters: nothing
    Returns: nothing
    """

    def run(self):
        while not self.stop_request and (self.image_thread is None or self.image_thread.output is None):
            time.sleep(self.static_dur / 1000)
        cv2.imshow(self.window_name, self.image_thread.output)
        self.current_image = self.image_thread.output
        cv2.waitKey(self.static_dur)
        while not self.stop_request: 
            self.past_image = self.current_image
            self.current_image = self.image_thread.output
            for i in range(1,self.blend_frames):
                dst = cv2.addWeighted(self.past_image, 1 - (i / float(self.blend_frames)), self.current_image, i / float(self.blend_frames), 0.0)
                cv2.imshow(self.window_name, dst)
                cv2.waitKey(self.blend_mspf)
            cv2.imshow(self.window_name, self.current_image)
            cv2.waitKey(self.static_dur)
        cv2.destroyAllWindows()
