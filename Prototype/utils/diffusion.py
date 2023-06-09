from diffusers import StableDiffusionPipeline
stable_diffusion_model_id = "runwayml/stable-diffusion-v1-5"
import torch

def get_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(stable_diffusion_model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

def get_pic(prompt, inference = 50, guidance_scale = 7.5, num_images_per_prompt = 3, seed = 0):
    """
    Creates an image using stable diffusion pipeline
    Parameters:
        prompt: string of the prompt
        inference: number of inference step, around 50 for a high quality image
        guidance scale: a way to increase the adherence to the conditional signal that guides the generation (text, in this case) as well as overall sample quality
    Returns:
        output: stable diffusion pipeline output
    """
    generator_list = []
    for i in range(num_images_per_prompt):
        generator_list.append(torch.Generator("cuda").manual_seed(seed + i))

    pipe = get_pipe()
    return pipe(
        prompt,
        num_inference_steps = inference,
        guidance_scale = guidance_scale,
        num_images_per_prompt = num_images_per_prompt,
        generator = generator_list
    )

def display_images(pipe):
    for i in range(len(pipe[0])):
        image = pipe.images[i]
        image.show()