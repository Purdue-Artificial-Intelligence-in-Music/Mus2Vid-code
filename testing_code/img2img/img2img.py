import requests
import torch
import numpy as np
import cv2

from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

def get_upsampler(model_str = 'x2'):
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
    upsampler = RealESRGANer(scale=netscale,model_path=model_path,model=model,half=True)
    return upsampler

upsampler = get_upsampler()

prompt = "An futuristic urban setting at night while it's raining"
last_img = np.zeros((1024, 1024, 3)) + .65
# Image.fromarray(last_img).save("seed.png")

last_img = pipe(prompt=prompt, image=last_img, strength=1, guidance_scale=7.5, num_inference_steps=10).images
img = last_img[0]
img = np.array(img)
# img, _ = upsampler.enhance(img)
img = np.ascontiguousarray(img)
img = np.float32(img)
img /= np.float32(256.0)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
wid = img.shape[1]
hgt = img.shape[0]
print(str(wid) + "x" + str(hgt)) # 2048 x 2048
cv2.imshow("image", img)
cv2.waitKey()

next_img = pipe(prompt=prompt, image=last_img, strength=.8, guidance_scale=7.5, num_inference_steps=30).images
next_img = next_img[0]
next_img = np.array(next_img)
# next_img, _ = upsampler.enhance(next_img)
next_img = np.ascontiguousarray(next_img)
next_img = np.float32(next_img)
next_img /= np.float32(256.0)
next_img = cv2.cvtColor(next_img, cv2.COLOR_RGB2BGR)

output_image = cv2.addWeighted(img, 0, next_img, 1, 0.0, dtype=cv2.CV_32F)
cv2.imshow('image', output_image)
cv2.waitKey(0)