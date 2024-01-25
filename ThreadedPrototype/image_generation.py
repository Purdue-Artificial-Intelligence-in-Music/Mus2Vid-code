from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import numpy
from PIL import Image
import threading
import prompting
import time
import numpy as np

DEFAULT_NEGATIVE_PROMPT = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy, low resolution, cropped, beginner, amateur, oversaturated"
MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
IMG2IMG_ID = "runwayml/stable-diffusion-v1-5"


def get_pipe(img2img):
    """
    Initializes a stable diffusion pipeline
    Parameters:

    Returns:
        pipe: pipeline object
    """
    if (img2img):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(IMG2IMG_ID, torch_dtype=torch.float16)
    else:
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, revision="fp16")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    return pipe


def get_upsampler(model_str='x2'):
    """    
    Downloads the real-ESRGAN model into an upsampler object

    Parameters:
        model_str (str): x2 or x4 default: x2
        
    Returns:
    upsampler object
    """
    if (model_str == 'x2'):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
    elif (model_str == 'x4'):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    model_path = load_file_from_url(url=file_url)
    upsampler = RealESRGANer(scale=netscale, model_path=model_path, model=model, half=True)
    return upsampler


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
                 Prompt_Thread,
                 audio_thread,
                 blank_image=np.zeros((1024, 1024, 3)) + .65,
                 # adding .65 was experimentally decided to provide neutral lighting
                 seed=None,
                 img2img=True,
                 strength=.8,  # for img2img
                 inference=10,
                 guidance_scale=7.5,
                 imgs_per_prompt=1,
                 upsampler_model_str='x2',
                 upsampler=None,
                 display_func=None):
        super(ImageGenerationThread, self).__init__()
        self.name = name
        self.img2img = img2img
        self.pipe = get_pipe(self.img2img)
        self.seed = seed
        self.strength = strength
        self.Prompt_Thread = Prompt_Thread
        self.negative_prompt = DEFAULT_NEGATIVE_PROMPT
        self.inference = inference
        self.guidance_scale = guidance_scale
        self.imgs_per_prompt = imgs_per_prompt
        self.audio_thread = audio_thread
        self.blank_image = blank_image

        if seed is None:
            self.generator = torch.Generator("cuda")
        else:
            self.generator = torch.Generator("cuda").manual_seed(seed)

        self.upsampler_model_str = upsampler_model_str
        self.upsampler = upsampler
        if self.upsampler == None:
            self.upsampler = get_upsampler(upsampler_model_str)
        self.output = blank_image
        self.uninit = True
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
            if not self.Prompt_Thread is None and not (
                    self.Prompt_Thread.prompt is None or self.Prompt_Thread.prompt == "" or self.Prompt_Thread.prompt == "Blank screen"):

                # get prompt
                prompt = self.Prompt_Thread.prompt

                # generate image
                if self.img2img:  # send previous image as an arg
                    print(type(self.output == self.blank_image))
                    print(self.output == self.blank_image)
                    if self.uninit:  # if no previous image, use blank image and minimum image influence
                        image_strength = 1  # To base image only off of prompt
                        self.uninit = False
                    else:  # previous image exists
                        image_strength = self.strength

                    images = self.pipe(
                        prompt=str(prompt),
                        negative_prompt=self.negative_prompt,
                        num_inference_steps=self.inference,
                        guidance_scale=self.guidance_scale,
                        num_images_per_prompt=self.imgs_per_prompt,
                        generator=self.generator,
                        image=self.output,
                        strength=image_strength
                    ).images

                else:  # no img-to-img diffusion or no previous image
                    images = self.pipe(
                        prompt=str(prompt),
                        negative_prompt=self.negative_prompt,
                        num_inference_steps=self.inference,
                        guidance_scale=self.guidance_scale,
                        num_images_per_prompt=self.imgs_per_prompt,
                        generator=self.generator
                    ).images

                for image in images:
                    if not image is None:
                        img = numpy.array(image)
                    sr_image, _ = self.upsampler.enhance(img)
                    # self.output = [Image.fromarray(sr_image)]
                    self.output = sr_image
                    if not self.display_func is None:
                        self.display_func(self.output)
            else:
                print("No prompt or thread")
                self.output = self.blank_image
            time.sleep(1)


def main():
    p_thr = prompting.PromptGenerationThread(name="p_thr", genre_thread=None, emotion_thread=None)
    # p_thr.start()
    thr = ImageGenerationThread(name="thr", Prompt_Thread=p_thr)
    thr.start()


if __name__ == "__main__":
    main()
