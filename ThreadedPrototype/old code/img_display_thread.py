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

    def __init__(self, name, Prompt_Thread, Img_Thread,
                 window_name="Image",
                 static_dur=1200,
                 blend_frames=50,
                 blend_mspf=1,
                 font=cv2.FONT_HERSHEY_SIMPLEX):
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

        self.font = font

    def get_image(self):
        self.past_image = self.current_image
        self.current_image = np.ascontiguousarray(self.image_thread.output)
        self.current_image = np.float32(self.current_image)
        self.current_image /= np.float32(256.0)
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
        if self.prompt_thread.prompt != "Black screen":
            prompt_split = self.prompt_thread.prompt.split("\n")
            i = 0
            for p in prompt_split:
                # org
                org = (50, 50 + i)
                # fontScale
                fontScale = 0.7
                # White color in BGR
                color = (255, 255, 255)
                # Line thickness of 2 px
                thickness = 2

                # Using cv2.putText() method
                self.current_image = cv2.putText(self.current_image, p, org, self.font,
                                                 fontScale, color, thickness, cv2.LINE_AA)

                i += 30

    """
    When the thread is started, this function is called which repeatedly displays new images.
    Parameters: nothing
    Returns: nothing
    """

    def run(self):
        while not self.stop_request and (
                self.prompt_thread is None or self.prompt_thread.prompt is None or self.image_thread is None or self.image_thread.output is None):
            time.sleep(0.5)
        self.get_image()
        cv2.imshow(self.window_name, self.current_image)
        cv2.waitKey(self.static_dur)
        while not self.stop_request:
            self.get_image()
            for i in range(1, self.blend_frames):
                dst = cv2.addWeighted(self.past_image, 1 - (i / float(self.blend_frames)), self.current_image,
                                      i / float(self.blend_frames), 0.0, dtype=cv2.CV_32F)
                cv2.imshow(self.window_name, dst)
                cv2.waitKey(self.blend_mspf)
            cv2.imshow(self.window_name, self.current_image)
            cv2.waitKey(self.static_dur)
        cv2.destroyAllWindows()
