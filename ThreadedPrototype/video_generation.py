# from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
import torch
# from realesrgan import RealESRGANer
# from basicsr.archs.rrdbnet_arch import RRDBNet
# from basicsr.utils.download_util import load_file_from_url
import numpy
from PIL import Image
import threading
import prompting
import time
import numpy as np


from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoencoderTiny
import cv2
from torch.nn import functional as F
from train_log.RIFE_HDv3 import Model
from onediff.infer_compiler import oneflow_compile
import random


IMG2IMG_ID = "stabilityai/sdxl-turbo"
VAE_ID = "madebyollin/taesdxl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    # print("availale")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def rife(img0,img1,n,model):
    # start_time = time.time()
    img0, img1,h,w = process(img0,img1)
    img_list = [img0, img1]

    for i in range(n):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
            tmp.append(img1)
            img_list = tmp

    final=[]

    for i in range(len(img_list)-1):
        k= (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        a= Image.fromarray(k)
        final.append(a)
    # print("Rife Time: ", time.time() - start_time , " seconds for ", n, " number of images")
    return final

def image2image(image,prompt,pipe,strength): #1 step img2img with turbo models
    # start_time = time.time()
    i =1 / strength
    if i % 1 != 0:
        i = int(i) + 1
    else:
        i = int(i)
    # print("img2img Time: ", time.time() - start_time , " seconds")
    return pipe(prompt, image=image, num_inference_steps=i, strength=strength, guidance_scale=1).images[0]

def process(img0,img1):
    # start_time = time.time()
    img0 = np.array(img0)
    img1 = np.array(img1)

    img0 = cv2.resize(img0, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img1 = cv2.resize(img1, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)
    # print("Process Time: ", time.time() - start_time , " seconds")
    return img0, img1,h,w

def zoom(image, zoom_factor):
    # start_time = time.time()
    original_width, original_height = image.size
    new_width = int(original_width * zoom_factor)
    new_height = int(original_height * zoom_factor)

    zoomed_image = image.resize((new_width, new_height))
    left = max((new_width - original_width) // 2, 0)
    top = max((new_height - original_height) // 2, 0)
    right = min((new_width + original_width) // 2, new_width)
    bottom = min((new_height + original_height) // 2, new_height)

    cropped_image = zoomed_image.crop((left, top, right, bottom))
    # print("Zoom Time: ", time.time() - start_time , " seconds")

    return cropped_image

def add_noise(img,level):
    # start_time = time.time()
    img_gray = np.array(img)
    noise = np.random.normal(0, level, img_gray.shape)
    img_noised = img_gray + noise
    img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)

    i = Image.fromarray(img_noised)
    # print("Noise Time: ", time.time() - start_time , " seconds")
    return Image.fromarray(img_noised)


def get_pipe():
    pipe = AutoPipelineForImage2Image.from_pretrained(IMG2IMG_ID, torch_dtype=torch.float16, variant="fp16")
    pipe.unet = oneflow_compile(pipe.unet)
    pipe.vae = AutoencoderTiny.from_pretrained(VAE_ID, torch_dtype=torch.float16)
    pipe.to("cuda")
    return pipe

def get_interpolation():
    model = Model()
    model.device()
    model.load_model('train_log')
    model.eval()
    return model


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
                 blank_image=np.zeros((512, 512, 3)),
                #  seed=None,
                #  img2img=False,
                 strength=.8,  # for img2img
                #  inference=10,
                 guidance_scale=1,
                #  imgs_per_prompt=1,
                 display_func=None,
                 noise = 20,
                 scale = 1,
                 n_interpolation = 1):
        super(ImageGenerationThread, self).__init__()
        self.name = name
        # self.img2img = img2img
        self.pipe = get_pipe()
        self.interpolation_model = get_interpolation()
        # self.seed = seed
        self.strength = strength
        self.Prompt_Thread = Prompt_Thread
        # self.negative_prompt = DEFAULT_NEGATIVE_PROMPT
        # self.inference = inference
        self.guidance_scale = guidance_scale
        # self.imgs_per_prompt = imgs_per_prompt
        self.audio_thread = audio_thread
        self.blank_image = blank_image
        self.prev = blank_image
        self.noise = noise
        self.scale = scale
        self.n_interpolation = n_interpolation

        # if seed is None:
        #     self.generator = torch.Generator("cuda")
        # else:
        #     self.generator = torch.Generator("cuda").manual_seed(seed)

        self.output = blank_image
        self.uninit = True
        self.display_func = display_func

        self.stop_request = False

    def compile_warm(self):
        img1 = np.zeros((512, 512, 3))
        img2 = image2image(img1,"any picture",self.pipe,0.2)
        img_list = rife(self.blank_image,img2,2,self.interpolation_model)



    """
    When the thread is started, this function is called which repeatedly generates new Stable Diffusion images.
    Parameters: nothing
    Returns: nothing
    """

    def run(self):
        self.compile_warm(self)
        while not self.stop_request:
            if not self.Prompt_Thread is None and not (
                    self.Prompt_Thread.prompt is None or self.Prompt_Thread.prompt == "" or self.Prompt_Thread.prompt == "Blank screen"):

                # get prompt and models
                prompt = self.Prompt_Thread.prompt
                model = self.interpolation_model
                pipe = self.pipe

                image = image2image(add_noise(self.prev,self.noise),prompt,pipe,self.strength)
                image = zoom(image, self.scale)
                img_list = rife(self.prev, image,self.n_interpolation,model)
                self.prev = image
                for img in img_list:
                    self.output = img
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
