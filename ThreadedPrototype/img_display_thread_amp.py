import math
import threading
import cv2
import time
import numpy as np

"""
This class is a thread class that generates prompts procedurally in real time.
"""


class ImageDisplayThreadWithAmpTracking(threading.Thread):
    """
    This function is called when a PromptGenerationThread is created.
    Parameters:
        name: the name of the thread
    Returns: nothing
    """

    def __init__(self, name, Prompt_Thread, Img_Thread, SPA_Thread,
                 window_name="Image",
                 static_dur=5,
                 blend_time=1,
                 bloom_decay_time=0.5,
                 bloom_threshold=-0.5,
                 font=cv2.FONT_HERSHEY_TRIPLEX):
        super(ImageDisplayThreadWithAmpTracking, self).__init__()
        self.name = name
        self.blank_image = np.zeros((1024, 1024, 3))
        self.prompt_thread = Prompt_Thread
        self.image_thread = Img_Thread
        self.SPA_Thread = SPA_Thread
        self.window_name = window_name
        self.past_image = self.blank_image
        self.current_image = self.blank_image
        self.output_image = self.blank_image
        self.static_dur = static_dur
        self.blend_time = blend_time
        self.stop_request = False

        self.font = font

        self.blending = False
        self.blend_i = 1

        self.time_last_change = time.time()
        self.blend_time_reset = False
        self.time_blend_start = time.time()

        self.time_prompt_update = time.time()
        self.stored_prompt = []

        self.bloom_decay_time = bloom_decay_time
        self.bloom_threshold = bloom_threshold
        self.time_last_bloom = time.time()
        self.bloom_val = 0.0

    def get_image(self):
        if self.blending is False:
            if time.time() - self.time_last_change > self.static_dur:
                self.time_last_change = time.time()
                self.past_image = self.current_image

                if self.prompt_thread.prompt == "Black screen":
                    self.current_image = self.blank_image
                else:
                    # Conversion of color space
                    self.current_image = np.ascontiguousarray(self.image_thread.output)
                    self.current_image = np.float32(self.current_image)
                    self.current_image /= np.float32(256.0)
                    self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)

                self.output_image = self.past_image
                self.blending = True
                self.blend_time_reset = False
            else:
                self.output_image = self.current_image
        else:
            if self.blend_time_reset is False:
                self.blend_time_reset = True
                self.time_blend_start = time.time()

            self.blend_i = min(1.0, (time.time() - self.time_blend_start) / self.blend_time)
            self.output_image = cv2.addWeighted(self.past_image, 1 - self.blend_i,
                                                self.current_image, self.blend_i,
                                                0.0, dtype=cv2.CV_32F)
            if self.blend_i == 1:
                self.blending = False

        if self.prompt_thread.prompt != "Black screen":
            # Audio beats
            if self.SPA_Thread.buffer_index > 501:
                audio = self.SPA_Thread.get_last_samples(500)
                audio = audio.astype(np.int32)
                rms = np.sqrt(np.mean(audio ** 2))
                dB_val = np.log10((rms + 1) / 32768.0)
                dB_val = min(dB_val, 0)
                dB_val = (dB_val / abs(dB_val)) * math.pow(dB_val, 2)
                # print(dB_val)
                if dB_val > self.bloom_threshold:
                    bloom_val = (-1 * dB_val / 0.6)
                    if bloom_val > self.bloom_val:
                        self.bloom_val = bloom_val
                        self.time_last_bloom = time.time()

            # Apply bloom
            decay_mult = max(0.0, 1 - ((time.time() - self.time_last_bloom) / self.bloom_decay_time))
            self.output_image *= 1 - (self.bloom_val * decay_mult)
            self.output_image += self.bloom_val * decay_mult

            if (time.time() - self.time_prompt_update) > 1:
                # Text prompt
                word_split = self.prompt_thread.prompt.split(" ")

                # Split prompt into multiple lines
                curr_str = ""
                prompt_split = []
                for word in word_split:
                    curr_str += word
                    curr_str += " "
                    if len(curr_str) > 60:
                        prompt_split.append(curr_str)
                        curr_str = ""
                prompt_split.append(curr_str)
                self.stored_prompt = prompt_split
                self.time_prompt_update = time.time()

            # Print text
            i = 0
            for p in self.stored_prompt:
                # org
                org = (50, 50 + i)
                # fontScale
                fontScale = 0.7
                # White color in BGR
                color = (255, 255, 255)
                # Line thickness of 2 px
                thickness = 1

                # Using cv2.putText() method
                self.output_image = cv2.putText(self.output_image, p, org, self.font,
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
        while not self.stop_request:
            self.get_image()
            cv2.imshow(self.window_name, self.output_image)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
