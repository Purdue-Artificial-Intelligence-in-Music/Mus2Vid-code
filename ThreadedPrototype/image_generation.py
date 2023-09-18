from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
stable_diffusion_model_id = "stabilityai/stable-diffusion-2-1-base"
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import numpy
from PIL import Image
import threading
import prompting
import time

import sys
sys.path.append("../")
from src.image.StableDiffusion.Diffusion import get_pipe
from src.image.Real_ESRGAN.upscale import get_upsampler

DEFAULT_NEGATIVE_PROMPT = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy, low resolution, cropped, beginner, amateur, oversaturated"

'''
This class is a thread class that generates images procedurally in real time.
'''

class ImageGenerationThread(threading.Thread):
    
    """
    This function is called when a StableDiffusionThread is created.
    Parameters:
        name: the name of the thread
        prompt: the prompt to use when generating images
        modulus: the amount to increment the seed by when generating images
    Returns: nothing
    """ 
    def __init__(self, 
                 name, 
                 Prompt_Thread = None,
                 seed = None, 
                 inference = 10, 
                 guidance_scale = 7.5,
                 imgs_per_prompt = 1,
                 upsampler_model_str = 'x2',
                 upsampler = None,
                 display_func = None):
        super(ImageGenerationThread, self).__init__()
        self.name = name
        self.pipe = get_pipe()
        self.seed = seed
        self.Prompt_Thread = Prompt_Thread
        self.negative_prompt = DEFAULT_NEGATIVE_PROMPT
        self.inference = inference
        self.guidance_scale = guidance_scale
        self.imgs_per_prompt = imgs_per_prompt

        if seed is None:
            self.generator = torch.Generator("cuda")
        else:
            self.generator = torch.Generator("cuda").manual_seed(seed)
        
        self.upsampler_model_str = upsampler_model_str
        self.upsampler = upsampler
        if self.upsampler == None:
            self.upsampler = get_upsampler(upsampler_model_str)
        self.output = None
        self.display_func = display_func

        self.stop_request = False
      
    
    def set_negative_prompt(self, prompt):
        self.negative_prompt = prompt
    
    """
    When the thread is started, this function is called which repeatedly generates new Stable Diffusion images.
    Parameters: nothing
    Returns: nothing
    """
    def run(self):
        while not self.stop_request:
            if not self.Prompt_Thread is None:
                prompt = self.Prompt_Thread.prompt
                if not (prompt is None or prompt == ""):
                    images = self.pipe(
                        str(prompt),
                        negative_prompt= self.negative_prompt,
                        num_inference_steps = self.inference,
                        guidance_scale = self.guidance_scale,
                        num_images_per_prompt = self.imgs_per_prompt,
                        generator= self.generator
                    ).images
                    for image in images:
                        if not image is None:
                            img = numpy.array(image)
                        sr_image, _ = self.upsampler.enhance(img)
                        #self.output = [Image.fromarray(sr_image)] 
                        self.output = sr_image
                        if not self.display_func is None:
                            self.display_func(self.output)
            else:
                print("No prompt or thread")
                self.output = None
            time.sleep(0.2)
                

def main():
    p_thr = prompting.PromptGenerationThread(name = "p_thr", genre_thread=None, emotion_thread=None)
    #p_thr.start()      
    thr = ImageGenerationThread(name = "thr", Prompt_Thread=p_thr)
    thr.start()
                    
if __name__ == "__main__":
    main()