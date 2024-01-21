import requests
import torch
import numpy as np

from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

prompt = "A fantasy landscape, trending on artstation, very dark lighting"
last_img = np.zeros((1024, 1024, 3)) + .65
# Image.fromarray(last_img).save("seed.png")

last_img = pipe(prompt=prompt, image=last_img, strength=1, guidance_scale=7.5, num_inference_steps=30).images
last_img[0].save("strength.png")

for i in range (0,2):
    last_img = pipe(prompt=prompt, image=last_img, strength=0.2, guidance_scale=7.5, num_inference_steps=30).images
    last_img[0].save(str(i) + ".png")