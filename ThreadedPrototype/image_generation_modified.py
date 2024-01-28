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
    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(IMG2IMG_ID, torch_dtype=torch.float16)
    normal_pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, revision="fp16")

    img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(img2img_pipe.scheduler.config)
    normal_pipe.scheduler = DPMSolverMultistepScheduler.from_config(normal_pipe.scheduler.config)
    img2img_pipe = img2img_pipe.to("cuda")
    normal_pipe = normal_pipe.to("cuda")

    return img2img_pipe, normal_pipe


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
    if model_str == 'x2' or model_str == 'x4':
        model_path = load_file_from_url(url=file_url)
        upsampler = RealESRGANer(scale=netscale, model_path=model_path, model=model, half=True)
    else:
        upsampler = None
    return upsampler

def dummy(images, **kwargs):
    return images, False


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
                 strength=1,  # for img2img
                 inference=10,
                 guidance_scale=7.5,
                 imgs_per_prompt=1,
                 upsampler_model_str='x2',
                 upsampler=None,
                 display_func=None):
        super(ImageGenerationThread, self).__init__()
        self.name = name
        self.img2img = img2img
        self.pipes = get_pipe(self.img2img)
        self.pipe = self.pipes[0]
        self.img2img_pipe = self.pipes[0]
        self.normal_pipe = self.pipes[1]
        self.seed = seed
        self.strength = strength
        self.Prompt_Thread = Prompt_Thread
        self.negative_prompt = DEFAULT_NEGATIVE_PROMPT
        self.inference = inference
        self.guidance_scale = guidance_scale
        self.imgs_per_prompt = imgs_per_prompt
        self.audio_thread = audio_thread
        self.blank_image = blank_image

        self.pipe.safety_checker = dummy

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
        self.blank = False
        self.display_func = display_func

        self.use_img2img_pipe = True

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
                    self.Prompt_Thread.prompt is None or self.Prompt_Thread.prompt == "" or self.Prompt_Thread.prompt == "Black screen"):

                # get prompt
                prompt = self.Prompt_Thread.prompt

                # generate image
                if not self.blank or not (self.output == self.blank_image).all():  # send previous image as an arg
                    images = self.img2img_pipe(
                        prompt=str(prompt),
                        negative_prompt=self.negative_prompt,
                        num_inference_steps=self.inference,
                        guidance_scale=self.guidance_scale,
                        num_images_per_prompt=self.imgs_per_prompt,
                        generator=self.generator,
                        image=self.output,
                        strength=self.strength
                    ).images
                    self.use_img2img_pipe = True

                else:  # no img-to-img diffusion or no previous image
                    images = self.normal_pipe(
                        prompt=str(prompt),
                        negative_prompt=self.negative_prompt,
                        num_inference_steps=self.inference,
                        guidance_scale=self.guidance_scale,
                        num_images_per_prompt=self.imgs_per_prompt,
                        generator=self.generator
                    ).images
                    self.use_img2img_pipe = False

                for image in images:
                    if not image is None:
                        img = numpy.array(image)
                    else:
                        img = self.blank_image
                    if not self.use_img2img_pipe and (not self.blank or not (self.output == self.blank_image).all()) and self.upsampler is not None:
                        sr_image, _ = self.upsampler.enhance(img)
                    else:
                        sr_image = img
                    # self.output = [Image.fromarray(sr_image)]
                    self.output = sr_image
                    self.blank = False
                    if not self.display_func is None:
                        self.display_func(self.output)

            else:
                print("No prompt or thread")
                self.output = self.blank_image
                self.blank = True
            time.sleep(1)


def main():
    p_thr = prompting.PromptGenerationThread(name="p_thr", genre_thread=None, emotion_thread=None)
    # p_thr.start()
    thr = ImageGenerationThread(name="thr", Prompt_Thread=p_thr)
    thr.start()


if __name__ == "__main__":
    main()
