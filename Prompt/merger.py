import Emotion_Prompt
import modifier_selector
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

## function for different type of genre with date
def get_genre(subgenre):
    prompt = ''
    if subgenre == 'Baroque':
        prompt = 'Baroque Classical Music, 1600-1750'
    elif subgenre ==  'Classical':
        prompt = 'Classical music, 1750-1830'
    elif subgenre == 'Romantic':
        prompt = 'Romantic Classical Music, 1830-1920'
    elif subgenre ==  '20th Century':
        prompt = 'Classical Music, 1900-2000'

    return prompt

## connect all the text into one prompt to send to SD 

def merger(genre, emotion):
    modify = modifier_selector.perspectiveRandom()
    return genre,emotion,modify

## the specific genre and emotion 
subgenre = ['Baroque','Classical', 'Romantic','20th Century']
no = random.randint(0,3)
genre = get_genre(subgenre[no])
emotion = Emotion_Prompt.emotion_from_value(0.5)

## send to SD and display images 

result = merger(genre, emotion)
modify_str = [', '.join(item) if isinstance(item, list) else item for item in result]
Prompt = ', '.join(str(item) for item in modify_str)
print(Prompt)


stable_diffusion_model_id = "runwayml/stable-diffusion-v1-5"

def dummy(images, **kwargs):
    return images, False

def display_images(pipe):
    for i in range(len(pipe[0])):
        image = pipe.images[i]
        display(image)

pipe = StableDiffusionPipeline.from_pretrained(
    stable_diffusion_model_id, 
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.safety_checker = dummy # Warning: enables NSFW images

def get_pic(prompt, inference = 50, guidance_scale = 7.5, num_images_per_prompt = 1, seed = 1):
    
    generator_list = []
    for i in range(num_images_per_prompt):
        generator_list.append(torch.Generator("cuda").manual_seed(seed + i))

    return pipe(
        prompt,
        negative_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy",
        num_inference_steps = inference,
        guidance_scale = guidance_scale,
        num_images_per_prompt = num_images_per_prompt,
        generator = generator_list
    )

image = get_pic(Prompt, inference = 150, num_images_per_prompt = 5, seed = 1298)
display_images(image)
